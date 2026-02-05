"""
Export predictions with debug outputs for intermediate results.

Extended version of export_predictions.py that:
1. Limits processing to first max_pairs
2. Saves intermediate outputs: grayscale images, score maps, descriptor maps
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .tensor import batch_to_device


@torch.no_grad()
def export_predictions_debug(
    loader,
    model,
    output_file,
    as_half=False,
    keys="*",
    callback_fn=None,
    optional_keys=[],
    max_pairs=None
):
    """Export predictions with optional debug outputs.

    Args:
        loader: Data loader
        model: Model to run
        output_file: Path to output h5 file
        as_half: Save as float16
        keys: Keys to export ("*" or list)
        callback_fn: Optional callback function
        optional_keys: Optional keys to export
        max_pairs: Maximum number of pairs to process (None = all)
        debug_output_dir: Directory to save debug outputs (None = no debug)
    """
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    hfile = h5py.File(str(output_file), "w")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    pair_count = 0
    for data_ in tqdm(loader):
        # Check if we've reached max_pairs
        if max_pairs is not None and pair_count >= max_pairs:
            print(f"\nReached max_pairs limit ({max_pairs}), stopping.")
            break

        data = batch_to_device(data_, device, non_blocking=True)

        # Run model
        pred = model(data)

        if callback_fn is not None:
            pred = {**callback_fn(pred, data), **pred}
        if keys != "*":
            if len(set(keys) - set(pred.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
            pred = {k: v for k, v in pred.items() if k in keys + optional_keys}
        assert len(pred) > 0

        # renormalization
        for k in pred.keys():
            if k.startswith("keypoints"):
                idx = k.replace("keypoints", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("lines"):
                idx = k.replace("lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("orig_lines"):
                idx = k.replace("orig_lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]

        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        
        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)
        try:
            name = data["name"][0]
            grp = hfile.create_group(name)
            for k, v in pred.items():
                grp.create_dataset(k, data=v)
        except RuntimeError:
            continue

        del pred
        pair_count += 1

    hfile.close()
    return output_file
