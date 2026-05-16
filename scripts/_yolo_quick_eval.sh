#!/usr/bin/env bash
# Build a temp mini-dataset of 200 annotated train images and run YOLO val on it
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"

docker run --rm \
  --name skylogic-yolo-eval \
  -v "$PROJ/data":/app/data \
  -v "$PROJ/models":/app/models \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  --memory=4g \
  skylogic-api \
  python - <<'PYEOF'
import yaml, subprocess, sys, os, shutil, tempfile
from pathlib import Path

root = Path("/app")
yaml_path = root / "data/yolo/skylogic.yaml"
ckpt     = root / "models/yolo/latest_best.pt"

# Pick first 200 annotated train images (images with non-empty label files)
label_dir = root / "data/yolo/labels/train"
img_dir   = root / "data/yolo/images/train"

annotated = []
for lf in sorted(label_dir.glob("*.txt")):
    if lf.stat().st_size > 0:
        img = img_dir / (lf.stem + ".png")
        if img.exists():
            annotated.append((img, lf))
    if len(annotated) >= 200:
        break

print(f"Selected {len(annotated)} annotated images for YOLO eval")

# Build tmp dataset
tmpdir = Path(tempfile.mkdtemp())
(tmpdir / "images/val").mkdir(parents=True)
(tmpdir / "labels/val").mkdir(parents=True)

for img, lbl in annotated:
    os.symlink(img, tmpdir / "images/val" / img.name)
    os.symlink(lbl, tmpdir / "labels/val" / lbl.name)

with open(yaml_path) as f:
    cfg = yaml.safe_load(f)

eval_cfg = {
    "path": str(tmpdir),
    "train": "images/val",
    "val":   "images/val",
    "nc": cfg["nc"],
    "names": cfg["names"],
}
eval_yaml = tmpdir / "eval.yaml"
eval_yaml.write_text(yaml.safe_dump(eval_cfg))

print(f"Running YOLO val on {len(annotated)} images (these are TRAINING images — metrics show in-sample fit)")
print("NOTE: The saved checkpoint is a 1-epoch smoke run — low mAP is expected.")
print()

from ultralytics import YOLO
import numpy as np

model = YOLO(str(ckpt))
metrics = model.val(
    data=str(eval_yaml),
    imgsz=512,
    batch=8,
    device="cpu",
    workers=0,
    verbose=False,
    plots=False,
)

map50   = float(metrics.box.map50) * 100
map5095 = float(metrics.box.map)   * 100
mp      = float(metrics.box.mp)    * 100
mr      = float(metrics.box.mr)    * 100

print()
print("=" * 65)
print("  YOLO DETECTOR — QUICK EVAL (200 annotated train images)")
print("=" * 65)
print(f"  Checkpoint   : latest_best.pt  (1-epoch smoke run)")
print(f"  Eval images  : 200  (annotated train-split images used as GT)")
print(f"  Classes      : 62")
print()
print(f"  mAP@0.5       : {map50:.2f}%")
print(f"  mAP@0.5:0.95  : {map5095:.2f}%")
print(f"  Mean Precision: {mp:.2f}%")
print(f"  Mean Recall   : {mr:.2f}%")
print()

# Top 10 classes by AP50
names = cfg["names"]
try:
    ap50s = np.array(metrics.box.ap50)
    if ap50s is not None and len(ap50s) > 0:
        top_idx = np.argsort(ap50s)[::-1][:10]
        print(f"  {'Class':<35} {'AP@0.5':>8}")
        print(f"  {'-'*35} {'-'*8}")
        for i in top_idx:
            cname = names[i] if i < len(names) else f"class_{i}"
            print(f"  {cname:<35} {float(ap50s[i])*100:>7.1f}%")
except Exception as e:
    print(f"  (per-class breakdown failed: {e})")

print("=" * 65)
shutil.rmtree(tmpdir, ignore_errors=True)
PYEOF
