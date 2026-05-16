"""Quick kwarg validation — does NOT train, just checks Ultralytics accepts all params."""
import sys
from pathlib import Path

# Mimic the same kwargs used in train_detector.py
train_kwargs = {
    "data":    "/app/data/yolo/skylogic_annotated.yaml",
    "epochs":  1,
    "imgsz":   416,
    "batch":   4,
    "device":  "cpu",
    "amp":     False,
    "half":    False,
    "workers": 0,
    "lr0":  0.01,
    "lrf":  0.01,
    "optimizer": "SGD",
    "momentum":  0.937,
    "weight_decay": 5e-4,
    "warmup_epochs": 3.0,
    "cls_pw":          2.0,
    "fl_gamma":        1.5,
    "label_smoothing": 0.0,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "mosaic":      1.0,
    "mixup":       0.05,
    "copy_paste":  0.1,
    "degrees":     5.0,
    "translate":   0.1,
    "scale":       0.5,
    "fliplr":      0.5,
    "flipud":      0.0,
    "hsv_h":       0.015,
    "hsv_s":       0.7,
    "hsv_v":       0.4,
    "patience":    15,
    "save":        True,
    "save_period": 5,
    "exist_ok":    True,
    "project": "/app/models/yolo",
    "name":    "test_kwarg",
    "plots":   True,
    "verbose": True,
}

from ultralytics import YOLO
import ultralytics
print(f"Ultralytics version: {ultralytics.__version__}")

# Check annotated yaml exists
annotated_yaml = Path("/app/data/yolo/skylogic_annotated.yaml")
if not annotated_yaml.exists():
    print("NOTE: skylogic_annotated.yaml not yet built (run with --annotated-only first)")
    train_kwargs["data"] = "/app/data/yolo/skylogic.yaml"

m = YOLO("yolov8n.pt")
print("yolov8n.pt loaded OK")

# Use YOLO's internal cfg checker without training
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
cfg = get_cfg(DEFAULT_CFG, train_kwargs)
print("All kwargs validated by Ultralytics get_cfg ✓")
print(f"  amp        = {cfg.amp}")
print(f"  half       = {cfg.half}")
print(f"  fl_gamma   = {cfg.fl_gamma}")
print(f"  cls_pw     = {cfg.cls_pw}")
print(f"  copy_paste = {cfg.copy_paste}")
print(f"  mixup      = {cfg.mixup}")
print("KWARGS OK — safe to train")
