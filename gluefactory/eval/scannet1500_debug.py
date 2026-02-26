"""
ScanNet1500 evaluation with debug outputs.

Extended version that:
1. Limits processing to first max_pairs
2. Saves intermediate outputs for debugging
3. Supports threshold sweep (ransac_th=-1)
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
from .utils import eval_matches_epipolar, eval_poses, eval_relative_pose_robust

logger = logging.getLogger(__name__)


class ScanNet1500DebugPipeline(EvalPipeline):
    """ScanNet1500 evaluation pipeline with debug capabilities."""

    default_conf = {
        "data": {
            "name": "image_pairs",
            "pairs": "scannet1500/pairs_calibrated.txt",
            "root": "scannet1500/",
            "extra_data": "relative_pose",
            "preprocessing": {
                "side": "long",
            },
            "num_workers": 14,
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": "poselib",  # poselib for consistency with Summertime pipeline
            "ransac_th": 1.0,  # -1 runs a sweep and selects the best
        },
        "debug": {
            "max_pairs": None,  # None = process all pairs
            "enable_dense_outputs": True,
        },
    }

    export_keys = [
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
    optional_export_keys = []

    def _init(self, conf):
        if not (DATA_PATH / "scannet1500").exists():
            logger.info("Downloading the ScanNet-1500 dataset.")
            url = "https://cvg-data.inf.ethz.ch/scannet/scannet1500.zip"
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

        debug_conf = self.conf.get("debug", {})
        max_pairs = debug_conf.get("max_pairs", None)
        enable_dense = debug_conf.get("enable_dense_outputs", True)

        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)

            if enable_dense and hasattr(model, "extractor"):
                if hasattr(model.extractor, "conf"):
                    print(f"Enabling dense outputs in extractor...")
                    import omegaconf
                    with omegaconf.read_write(model.extractor.conf):
                        with omegaconf.open_dict(model.extractor.conf):
                            model.extractor.conf["dense_outputs"] = True
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

        debug_conf = self.conf.get("debug", {})
        max_pairs = debug_conf.get("max_pairs", None)

        per_pair_results = []

        for i, data in enumerate(tqdm(loader)):
            if max_pairs is not None and i >= max_pairs:
                print(f"\nReached max_pairs limit ({max_pairs}), stopping evaluation.")
                break

            pred = cache_loader(data)
            results_i = eval_matches_epipolar(data, pred)

            primary_th = test_thresholds[0] if test_thresholds else conf.ransac_th
            pose_results_i = eval_relative_pose_robust(
                data,
                pred,
                {"estimator": conf.estimator, "ransac_th": primary_th},
            )
            results_i.update(pose_results_i)

            for th in test_thresholds:
                pose_results_th = eval_relative_pose_robust(
                    data,
                    pred,
                    {"estimator": conf.estimator, "ransac_th": th},
                )
                [pose_results[th][k].append(v) for k, v in pose_results_th.items()]

            results_i["names"] = data["name"][0]

            per_pair_results.append({
                "name": data["name"][0],
                "results": {k: v for k, v in results_i.items() if k not in ["names"]}
            })

            for k, v in results_i.items():
                results[k].append(v)

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
            "best_ransac_th": best_th,
        }

        figures = {
            "pose_recall": plot_cumulative(
                {self.conf.eval.estimator: results["rel_pose_error"]},
                [0, 30],
                unit="°",
                title="Pose ",
            )
        }

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

                pair_group = f.create_group(pair_name)

                for key, value in results.items():
                    if isinstance(value, (int, float, np.number)):
                        pair_group.create_dataset(key, data=value)
                    elif isinstance(value, torch.Tensor):
                        pair_group.create_dataset(key, data=value.cpu().numpy())

        print(f"Saved {len(per_pair_results)} pair results")


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    dataset_name = "scannet1500"
    parser = get_eval_parser()
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Maximum number of pairs to process (default: all)",
    )
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(ScanNet1500DebugPipeline.default_conf)

    if args.max_pairs is not None:
        debug_conf = {
            "max_pairs": args.max_pairs,
            "enable_dense_outputs": True,
        }
        default_conf["debug"] = debug_conf

    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    if args.max_pairs is not None:
        name = f"{name}_first{args.max_pairs}"

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = ScanNet1500DebugPipeline(conf)
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
