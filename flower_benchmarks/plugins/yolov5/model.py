# flower_benchmarks/plugins/yolov5/model.py
"""
Helpers to interop between Flower model state_dicts and YOLOv5 checkpoint files.
We keep things minimal: we create a YOLO-style checkpoint dict with a 'model' key
so YOLO's CLI/train.py can load it and continue training.
"""

import torch
import os

YoloSizeToPretrained = {
    "n": "yolov5n.pt",
    "s": "yolov5s.pt",
    "m": "yolov5m.pt",
    "l": "yolov5l.pt",
    "x": "yolov5x.pt",
}

def save_state_dict_as_yolo_checkpoint(state_dict: dict, out_path: str):
    """
    Wrap a PyTorch state_dict into a YOLO-friendly checkpoint dict and save.
    YOLO typically expects ckpt['model'] or a full checkpoint; this creates ckpt['model'].
    """
    ckpt = {"model": state_dict}
    torch.save(ckpt, out_path)

def load_yolo_checkpoint_as_state_dict(ckpt_path: str) -> dict:
    """
    Load a YOLO .pt checkpoint (best.pt / last.pt) saved by YOLO train.py and
    return a state_dict suitable for ArrayRecord conversion.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # YOLO checkpoints sometimes store 'model' (state_dict) or the whole model object.
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        # fallback: maybe entire state_dict is at top-level
        # many YOLO checkpoints have 'model' as the converted state
        # If not, try to extract keys that look like param tensors
        # (best-effort)
        candidate = {k: v for k, v in ckpt.items() if hasattr(v, "shape")}
        if candidate:
            return candidate
    # If ckpt is already a state_dict-like mapping
    return ckpt
