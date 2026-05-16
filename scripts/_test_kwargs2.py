"""Validate corrected train kwargs for Ultralytics 8.4.42."""
from pathlib import Path
from ultralytics.cfg import get_cfg, DEFAULT_CFG

train_kwargs = {
    "data":    "/app/data/yolo/skylogic_annotated.yaml",
    "epochs":  50, "imgsz": 416, "batch": 4, "device": "cpu",
    "amp":     False,
    "half":    False,
    "workers": 0,
    "lr0": 0.01, "lrf": 0.01,
    "optimizer": "SGD", "momentum": 0.937, "weight_decay": 5e-4,
    "warmup_epochs": 3.0,
    # Class imbalance — valid in 8.4.42
    "cls_pw":  2.0,
    "cls":     1.5,
    "box":     7.5,
    "dfl":     1.5,
    # Augmentation
    "mosaic": 1.0, "close_mosaic": 0,
    "mixup": 0.05, "copy_paste": 0.1,
    "degrees": 5.0, "translate": 0.1, "scale": 0.5,
    "fliplr": 0.5, "flipud": 0.0,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "erasing": 0.4,
    # Checkpointing
    "patience": 15, "save": True, "save_period": 5, "exist_ok": True,
    "project": "/app/models/yolo", "name": "xview_v2",
    "plots": True, "verbose": True,
}

print(f"Ultralytics: {__import__('ultralytics').__version__}")
try:
    cfg = get_cfg(DEFAULT_CFG, train_kwargs)
    print("✅ All kwargs VALID")
    print(f"  amp={cfg.amp}  half={cfg.half}  cls_pw={cfg.cls_pw}  cls={cfg.cls}")
    print(f"  copy_paste={cfg.copy_paste}  close_mosaic={cfg.close_mosaic}")
    print(f"  mixup={cfg.mixup}  erasing={cfg.erasing}")
except Exception as e:
    print(f"❌ INVALID kwargs: {e}")
    raise
