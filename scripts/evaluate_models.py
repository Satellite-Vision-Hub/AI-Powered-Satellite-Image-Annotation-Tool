"""
SkyLogic MAS — Model Evaluation Script
Evaluates trained segmentor and YOLO detector checkpoints.

Usage (inside container):
  python scripts/evaluate_models.py --project-root /app --seg-samples 500 --yolo-samples 200
"""
from __future__ import annotations
import argparse
import ast
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evaluate")

SEG_CLASS_NAMES = {
    0: "background",
    1: "building",
    2: "damaged_building",
    3: "vehicle",
    4: "road",
    5: "vegetation",
    6: "water_flood",
    7: "debris_rubble",
    8: "construction",
    9: "container",
}

# Same mapping as train_segmentor.py
XVIEW_TO_SEG = {
    73: 1, 71: 1, 72: 1, 74: 1, 75: 1, 76: 1, 84: 1, 85: 1, 77: 1, 86: 1,
    79: 8, 78: 8, 87: 7,
    18: 3, 19: 3, 20: 3, 21: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3,
    34: 3, 35: 3, 36: 3, 37: 3, 38: 3, 53: 3, 54: 3, 55: 3, 56: 3, 57: 3,
    59: 3, 60: 3, 61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3,
    11: 3, 12: 3, 13: 3, 15: 3, 17: 3,
    40: 3, 41: 3, 42: 3, 44: 3, 45: 3, 47: 3, 49: 3, 50: 3, 51: 3, 52: 3,
    83: 3, 89: 9,
}


def parse_bbox(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return None
    return None


class WeakSegDataset(Dataset):
    """Identical to training dataset — same preprocessing pipeline."""

    def __init__(self, patches_dir, ann_by_patch, patch_files, split, img_size, num_classes=10):
        self.patches_dir = patches_dir
        self.ann_by_patch = ann_by_patch
        self.files = patch_files
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def _build_mask(self, fname, src_w, src_h):
        m = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        anns = self.ann_by_patch.get(fname, [])
        sx = self.img_size / float(src_w)
        sy = self.img_size / float(src_h)
        for cls_idx, (x1, y1, x2, y2) in anns:
            X1 = int(max(0, min(self.img_size - 1, round(x1 * sx))))
            Y1 = int(max(0, min(self.img_size - 1, round(y1 * sy))))
            X2 = int(max(0, min(self.img_size, round(x2 * sx))))
            Y2 = int(max(0, min(self.img_size, round(y2 * sy))))
            if X2 > X1 and Y2 > Y1:
                m[Y1:Y2, X1:X2] = cls_idx
        return m

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = self.patches_dir / self.split / fname
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        mask = self._build_mask(fname, w, h)
        return {"image": self.tf(img), "mask": torch.from_numpy(mask)}


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTOR EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_segmentor(args, root: Path, ann_by_patch: dict, train_files: list[str]):
    import segmentation_models_pytorch as smp

    ckpt_path = root / args.seg_checkpoint
    if not ckpt_path.exists():
        log.error(f"Segmentor checkpoint not found: {ckpt_path}")
        return

    log.info(f"Loading segmentor checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        num_classes = int(ckpt.get("num_classes", 10))
        saved_epoch = ckpt.get("epoch", "?")
        saved_loss = ckpt.get("loss", "?")
        state_dict = ckpt["model_state"]
    else:
        num_classes = 10
        saved_epoch = "?"
        saved_loss = "?"
        state_dict = ckpt

    log.info(f"  Checkpoint: epoch={saved_epoch}  train_loss={saved_loss}  classes={num_classes}")

    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    # Use last N annotated train patches as evaluation set (held-out by position)
    # Training used shuffle=True so all patches were seen; this evaluates on the same
    # distribution. We report this as "train-set evaluation" honestly.
    eval_files = train_files[-args.seg_samples:] if len(train_files) > args.seg_samples else train_files
    log.info(f"  Evaluating on {len(eval_files)} patches (last {args.seg_samples} from annotated train set)")
    log.info(f"  NOTE: No held-out val labels available (xView ships no val-set GT masks).")
    log.info(f"        These metrics reflect in-distribution performance on seen training data.")

    ds = WeakSegDataset(root / "data" / "patches", ann_by_patch, eval_files, "train",
                        args.imgsz, num_classes)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0

    # Accumulators for pixel metrics
    num_classes_eval = num_classes
    confusion = np.zeros((num_classes_eval, num_classes_eval), dtype=np.int64)

    log.info("  Running inference…")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch["image"]
            masks = batch["mask"].long()

            logits = model(images)
            # Resize logits to mask size if needed
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = nn.functional.interpolate(logits, size=masks.shape[-2:],
                                                   mode="bilinear", align_corners=False)

            loss = criterion(logits, masks)
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1)  # (B, H, W)

            # Update confusion matrix
            for pred, gt in zip(preds.numpy().flatten(), masks.numpy().flatten()):
                confusion[gt, pred] += 1

            if (i + 1) % 20 == 0:
                log.info(f"    batch {i+1}/{len(loader)}  running_loss={total_loss/n_batches:.4f}")

    avg_loss = total_loss / max(n_batches, 1)

    # ── Metrics ──────────────────────────────────────────────────────────────
    # Pixel accuracy: fraction of correctly classified pixels
    total_pixels = confusion.sum()
    correct_pixels = np.diag(confusion).sum()
    pixel_acc = 100.0 * correct_pixels / max(total_pixels, 1)

    # Per-class: precision, recall, IoU
    per_class = {}
    iou_list = []
    for c in range(num_classes_eval):
        tp = confusion[c, c]
        fn = confusion[c, :].sum() - tp          # true class c predicted as other
        fp = confusion[:, c].sum() - tp          # other class predicted as c
        support = confusion[c, :].sum()          # true pixels of class c
        if support == 0:
            per_class[c] = {"precision": 0.0, "recall": 0.0, "iou": 0.0, "support": 0}
            continue
        precision = 100.0 * tp / max(tp + fp, 1)
        recall    = 100.0 * tp / max(tp + fn, 1)
        iou       = 100.0 * tp / max(tp + fp + fn, 1)
        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "support": int(support),
        }
        iou_list.append(iou)

    # Mean IoU over classes that actually appear in GT
    present_classes = [c for c in range(num_classes_eval) if per_class[c]["support"] > 0]
    mean_iou = np.mean([per_class[c]["iou"] for c in present_classes]) if present_classes else 0.0

    # ── Report ───────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  SEGMENTOR (U-Net ResNet18) EVALUATION")
    print("=" * 65)
    print(f"  Checkpoint   : {ckpt_path.name}  (epoch {saved_epoch})")
    print(f"  Eval samples : {len(eval_files)} patches  ({total_pixels:,} pixels)")
    print(f"  Eval set     : last {len(eval_files)} annotated train patches")
    print()
    print(f"  Validation Loss   : {avg_loss:.4f}")
    print(f"  Pixel Accuracy    : {pixel_acc:.2f}%")
    print(f"  Mean IoU (mIoU)   : {mean_iou:.2f}%  (over {len(present_classes)} classes with GT)")
    print()
    print(f"  {'Class':<22} {'Precision':>10} {'Recall':>10} {'IoU':>10} {'GT pixels':>12}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for c in range(num_classes_eval):
        m = per_class[c]
        mark = " ← present" if m["support"] > 0 else ""
        print(f"  {SEG_CLASS_NAMES[c]:<22} {m['precision']:>9.1f}% {m['recall']:>9.1f}% "
              f"{m['iou']:>9.1f}% {m['support']:>12,}{mark}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# YOLO EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_yolo(args, root: Path):
    yolo_ckpt = root / args.yolo_checkpoint
    if not yolo_ckpt.exists():
        log.warning(f"YOLO checkpoint not found: {yolo_ckpt}. Skipping YOLO eval.")
        return

    yaml_path = root / "data" / "yolo" / "skylogic.yaml"
    if not yaml_path.exists():
        log.warning(f"YOLO dataset YAML not found: {yaml_path}. Skipping YOLO eval.")
        return

    log.info(f"Loading YOLO model: {yolo_ckpt}")
    from ultralytics import YOLO
    model = YOLO(str(yolo_ckpt))

    log.info("  NOTE: xView val set has no ground-truth labels → mAP will be 0.")
    log.info(f"  Evaluating on TRAIN split (first {args.yolo_samples} images) for meaningful metrics.")

    # Build a temp yaml pointing to train images only with correct labels
    import yaml, tempfile, shutil
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Create a mini eval yaml pointing at train subset
    train_img_dir = root / "data" / "yolo" / "images" / "train"
    train_imgs = sorted(train_img_dir.glob("*.png"))[:args.yolo_samples]
    if not train_imgs:
        log.warning("No train images found. Skipping YOLO eval.")
        return

    # Write temp image list
    tmpdir = Path(tempfile.mkdtemp())
    img_list = tmpdir / "eval_images.txt"
    img_list.write_text("\n".join(str(p) for p in train_imgs))

    eval_yaml = tmpdir / "eval.yaml"
    eval_cfg = {
        "path": str(root / "data" / "yolo"),
        "train": "images/train",
        "val":   "images/train",   # use train labels for eval
        "nc": cfg["nc"],
        "names": cfg["names"],
    }
    eval_yaml.write_text(yaml.safe_dump(eval_cfg))

    try:
        log.info(f"  Running validation on first {args.yolo_samples} train images…")
        metrics = model.val(
            data=str(eval_yaml),
            imgsz=args.imgsz,
            batch=4,
            device="cpu",
            workers=0,
            verbose=False,
            plots=False,
            max_det=300,
        )

        map50    = float(metrics.box.map50)   * 100
        map5095  = float(metrics.box.map)     * 100
        mp       = float(metrics.box.mp)      * 100   # mean precision
        mr       = float(metrics.box.mr)      * 100   # mean recall

        print()
        print("=" * 65)
        print("  YOLO DETECTOR EVALUATION")
        print("=" * 65)
        print(f"  Checkpoint   : {yolo_ckpt.name}")
        print(f"  Eval images  : {args.yolo_samples} train-split images  (train labels as GT)")
        print(f"  NOTE: These are TRAINING images — metrics reflect fitting, not generalisation.")
        print()
        print(f"  mAP@50        : {map50:.2f}%")
        print(f"  mAP@50-95     : {map5095:.2f}%")
        print(f"  Mean Precision: {mp:.2f}%")
        print(f"  Mean Recall   : {mr:.2f}%")

        # Per-class breakdown (top-10 by AP50)
        names = cfg["names"]
        try:
            ap50s = metrics.box.ap50
            if ap50s is not None and len(ap50s) > 0:
                top_k = min(10, len(ap50s))
                top_idx = np.argsort(ap50s)[::-1][:top_k]
                print()
                print(f"  {'Class':<35} {'AP@50':>8}")
                print(f"  {'-'*35} {'-'*8}")
                for i in top_idx:
                    cname = names[i] if i < len(names) else f"class_{i}"
                    print(f"  {cname:<35} {float(ap50s[i])*100:>7.1f}%")
        except Exception:
            pass

        print("=" * 65)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seg-checkpoint", default="models/segformer/latest_best.pth")
    parser.add_argument("--yolo-checkpoint", default="models/yolo/latest_best.pt")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--seg-samples", type=int, default=500,
                        help="Number of annotated train patches to evaluate segmentor on")
    parser.add_argument("--yolo-samples", type=int, default=200,
                        help="Number of train images to evaluate YOLO on")
    parser.add_argument("--skip-seg", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    log.info(f"Project root: {root}")

    # Load metadata (shared by both evals)
    log.info("Loading metadata…")
    patches  = pd.read_csv(root / "data" / "metadata" / "patches_metadata.csv")
    anns_df  = pd.read_csv(root / "data" / "metadata" / "annotations_metadata.csv")

    anns_df["seg_idx"] = anns_df["class_id"].map(XVIEW_TO_SEG)
    anns_df = anns_df.dropna(subset=["seg_idx"]).copy()
    anns_df["seg_idx"] = anns_df["seg_idx"].astype(int)

    ann_by_patch: dict = {}
    for row in anns_df.itertuples(index=False):
        bb = parse_bbox(row.bbox)
        if bb is None or len(bb) != 4:
            continue
        ann_by_patch.setdefault(row.patch_filename, []).append((int(row.seg_idx), bb))

    train_files = patches[patches["split"] == "train"]["filename"].tolist()
    train_files = [f for f in train_files if f in ann_by_patch]
    log.info(f"  {len(train_files)} annotated train patches available")

    if not args.skip_seg:
        evaluate_segmentor(args, root, ann_by_patch, train_files)

    if not args.skip_yolo:
        evaluate_yolo(args, root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
