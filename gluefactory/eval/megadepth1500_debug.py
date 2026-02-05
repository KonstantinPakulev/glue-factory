"""
MegaDepth1500 evaluation with debug outputs.

Extended version that:
1. Limits processing to first max_pairs
2. Saves intermediate outputs for debugging
3. Enables dense outputs in SuperPoint
"""

import logging
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH, EVAL_PATH
from ..utils.export_predictions_debug import export_predictions_debug
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import (
    eval_matches_depth,
    eval_matches_epipolar,
    eval_poses,
    eval_relative_pose_robust,
)

logger = logging.getLogger(__name__)


class MegaDepth1500DebugPipeline(EvalPipeline):
    """MegaDepth1500 evaluation pipeline with debug capabilities."""

    default_conf = {
        "data": {
            "name": "posed_images",
            "root": "",
            "image_dir": "{scene}/images",
            "depth_dir": "{scene}/depths",
            "views": "{scene}/views.txt",
            "view_groups": "{scene}/pairs.txt",
            "depth_format": "h5",
            "scene_list": ["megadepth1500"],
            "preprocessing": {
                "side": "long",
            },
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": 1.0,
        },
        "debug": {
            "max_pairs": None,  # None = process all pairs
            "enable_dense_outputs": True,  # Enable dense descriptors in SuperPoint
        },
    }

    export_keys = [
        # "image0",
        # "image1",
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
        "descriptors0",
        "descriptors1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    ]
    optional_export_keys = [
        # "dense_descriptors0",
        # "dense_descriptors1",
    ]

    def _init(self, conf):
        if not (DATA_PATH / "megadepth1500").exists():
            logger.info("Downloading the MegaDepth-1500 dataset.")
            url = "https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip"
            zip_path = DATA_PATH / url.rsplit("/", 1)[-1]
            zip_path.parent.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(url, zip_path)
            with zipfile.ZipFile(zip_path) as fid:
                fid.extractall(DATA_PATH)
            zip_path.unlink()

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = get_dataset(data_conf["name"])(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
        pred_file = experiment_dir / "predictions.h5"

        # Check if we should enable debug mode
        debug_conf = self.conf.get("debug", {})
        max_pairs = debug_conf.get("max_pairs", None)
        enable_dense = debug_conf.get("enable_dense_outputs", True)

        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)

            # Enable dense outputs in SuperPoint extractor if requested
            if enable_dense and hasattr(model, "extractor"):
                if hasattr(model.extractor, "conf"):
                    print(f"Enabling dense outputs in extractor...")
                    import omegaconf
                    with omegaconf.read_write(model.extractor.conf):
                        with omegaconf.open_dict(model.extractor.conf):
                            model.extractor.conf["dense_outputs"] = True
                            # Also check if sparse_outputs needs to be enabled
                            if "sparse_outputs" in model.extractor.conf:
                                model.extractor.conf["sparse_outputs"] = True

            if max_pairs is not None:
                print(f"Processing first {max_pairs} pairs only")

            export_predictions_debug(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
                max_pairs=max_pairs,
            )
        else:
            print(f"Using existing predictions: {pred_file}")

        return pred_file

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        conf = self.conf.eval
        results = defaultdict(list)
        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )
        pose_results = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()

        # Limit evaluation if max_pairs is set
        debug_conf = self.conf.get("debug", {})
        max_pairs = debug_conf.get("max_pairs", None)

        # Store per-pair results for debugging
        per_pair_results = []

        for i, data in enumerate(tqdm(loader)):
            if max_pairs is not None and i >= max_pairs:
                print(f"\nReached max_pairs limit ({max_pairs}), stopping evaluation.")
                break

            pred = cache_loader(data)
            # add custom evaluations here
            results_i = eval_matches_epipolar(data, pred)
            if "depth" in data["view0"].keys():
                results_i.update(eval_matches_depth(data, pred))

            # Store pose results for the primary threshold
            primary_th = test_thresholds[0] if test_thresholds else conf.ransac_th
            pose_results_i = eval_relative_pose_robust(
                data,
                pred,
                {"estimator": conf.estimator, "ransac_th": primary_th},
            )
            results_i.update(pose_results_i)

            # Evaluate all thresholds for aggregation
            for th in test_thresholds:
                pose_results_th = eval_relative_pose_robust(
                    data,
                    pred,
                    {"estimator": conf.estimator, "ransac_th": th},
                )
                [pose_results[th][k].append(v) for k, v in pose_results_th.items()]

            # we also store the names for later reference
            results_i["names"] = data["name"][0]
            if "scene" in data.keys():
                results_i["scenes"] = data["scene"][0]

            # Save per-pair result
            per_pair_results.append({
                "name": data["name"][0],
                "results": {k: v for k, v in results_i.items() if k not in ["names", "scenes"]}
            })

            for k, v in results_i.items():
                results[k].append(v)

        # summarize results as a dict[str, float]
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.mean(arr), 3)

        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=[5, 10, 20], key="rel_pose_error"
        )
        results = {**results, **pose_results[best_th]}
        summaries = {
            **summaries,
            **best_pose_results,
        }

        figures = {
            "pose_recall": plot_cumulative(
                {self.conf.eval.estimator: results["rel_pose_error"]},
                [0, 30],
                unit="°",
                title="Pose ",
            )
        }

        # Save per-pair results to HDF5
        self._save_per_pair_results(per_pair_results, pred_file)

        return summaries, figures, results

    def _save_per_pair_results(self, per_pair_results, pred_file):
        """Save per-pair evaluation results to HDF5 file."""
        import h5py

        eval_file = pred_file.parent / "evaluation_results.h5"
        print(f"\nSaving per-pair evaluation results to {eval_file}")

        with h5py.File(eval_file, "w") as f:
            for pair_result in per_pair_results:
                pair_name = pair_result["name"]
                results = pair_result["results"]

                # Create group for this pair
                pair_group = f.create_group(pair_name)

                # Save all numeric results
                for key, value in results.items():
                    if isinstance(value, (int, float, np.number)):
                        pair_group.create_dataset(key, data=value)
                    elif isinstance(value, torch.Tensor):
                        pair_group.create_dataset(key, data=value.cpu().numpy())

        print(f"Saved {len(per_pair_results)} pair results")


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    dataset_name = "megadepth1500"
    parser = get_eval_parser()
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Maximum number of pairs to process (default: all)",
    )
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(MegaDepth1500DebugPipeline.default_conf)

    # Add debug configuration from args
    if args.max_pairs is not None:
        debug_conf = {
            "max_pairs": args.max_pairs,
            "enable_dense_outputs": True,
        }
        default_conf["debug"] = debug_conf

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    # Add debug suffix to experiment name if using debug mode
    if args.max_pairs is not None:
        name = f"{name}_first{args.max_pairs}"

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = MegaDepth1500DebugPipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    pprint(s)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
