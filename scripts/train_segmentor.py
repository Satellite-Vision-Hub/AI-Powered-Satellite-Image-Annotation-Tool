"""
SkyLogic MAS — Train Agent B (SegFormer / U-Net) using bbox-derived weak masks.

LIMITATION: The xView dataset shipped with this project provides bounding-box
annotations only. There are NO ground-truth segmentation masks. Training a
segmentor with hard CE loss against bounding-box "pseudo-masks" (every pixel
inside the bbox is treated as positive) is a well-known weak-supervision
approximation; results will be coarse and over-segment objects. This is the
honest workaround. To train a real segmentor, you would need polygon labels
(e.g. xView2 disaster masks or COCO-style polygons).

The script maps xView class IDs into the 10 SkyLogic disaster classes
(SegmentorAgent.SEG_CLASSES) using a heuristic mapping below.

Smoke run example:
    python scripts/train_segmentor.py --epochs 1 --limit-train 100 --imgsz 256

Full run example (resumable via --resume):
    python scripts/train_segmentor.py --epochs 20 --limit-train 0 --imgsz 512 \
        --batch 2 --save-dir models/segformer
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_segmentor")

# Heuristic xView class_id → SegmentorAgent SEG_CLASSES (10-class disaster taxonomy)
# 0 background, 1 building, 2 damaged_building, 3 vehicle, 4 road,
# 5 vegetation, 6 water_flood, 7 debris_rubble, 8 construction, 9 container
XVIEW_TO_SEG = {
    73: 1,  # Building
    71: 1, 72: 1, 74: 1, 75: 1, 76: 1, 84: 1, 85: 1,  # building-like / facility
    77: 1, 86: 1,  # Facility, Storage Tank
    79: 8,  # Construction Site
    78: 8,  # Damaged Building (xView)
    87: 7,  # Pylon → debris-ish (best-effort)
    18: 3, 19: 3, 20: 3, 21: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3,
    34: 3, 35: 3, 36: 3, 37: 3, 38: 3, 53: 3, 54: 3, 55: 3, 56: 3, 57: 3,
    59: 3, 60: 3, 61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3,  # all vehicles
    11: 3, 12: 3, 13: 3, 15: 3, 17: 3,  # aircraft / passenger vehicle (treat as vehicle)
    40: 3, 41: 3, 42: 3, 44: 3, 45: 3, 47: 3, 49: 3, 50: 3, 51: 3, 52: 3,  # boats/ships
    83: 3,  # Vehicle Lot
    89: 9,  # Shipping container lot
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
    """Returns (image_tensor, mask_tensor) where mask is rasterized bboxes."""

    def __init__(
        self,
        patches_dir: Path,
        ann_by_patch: dict[str, list[tuple[int, list[float]]]],
        patch_files: list[str],
        split: str,
        img_size: int,
        num_classes: int = 10,
    ):
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

    def _build_mask(self, fname: str, src_w: int, src_h: int) -> np.ndarray:
        """Rasterize bboxes into a class-index mask at img_size."""
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
                # Don't overwrite a more specific class with background later
                m[Y1:Y2, X1:X2] = cls_idx
        return m

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = self.patches_dir / self.split / fname
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        mask = self._build_mask(fname, w, h)
        return {"image": self.tf(img), "mask": torch.from_numpy(mask)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--limit-train", type=int, default=100,
                        help="Cap train patches (0 = all annotated)")
    parser.add_argument("--save-dir", default="models/segformer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    meta_dir = root / "data" / "metadata"
    patches_dir = root / "data" / "patches"
    save_dir = root / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading metadata…")
    patches = pd.read_csv(meta_dir / "patches_metadata.csv")
    anns = pd.read_csv(meta_dir / "annotations_metadata.csv")

    # Map xView IDs → seg taxonomy; drop unmapped
    anns["seg_idx"] = anns["class_id"].map(XVIEW_TO_SEG)
    anns = anns.dropna(subset=["seg_idx"]).copy()
    anns["seg_idx"] = anns["seg_idx"].astype(int)
    log.info(f"  {len(anns)} annotations after seg-class mapping")

    ann_by_patch: dict[str, list[tuple[int, list[float]]]] = {}
    for row in anns.itertuples(index=False):
        bb = parse_bbox(row.bbox)
        if bb is None or len(bb) != 4:
            continue
        ann_by_patch.setdefault(row.patch_filename, []).append((int(row.seg_idx), bb))

    train_files = patches[patches["split"] == "train"]["filename"].tolist()
    train_files = [f for f in train_files if f in ann_by_patch]  # only annotated
    if args.limit_train > 0:
        train_files = train_files[: args.limit_train]
    log.info(f"  Training on {len(train_files)} patches")

    ds = WeakSegDataset(patches_dir, ann_by_patch, train_files, "train",
                        args.imgsz, args.num_classes)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=False)

    # Build U-Net (SegFormer is overkill on CPU for a smoke run)
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=args.num_classes,
    ).to(args.device)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        log.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            start_epoch = ckpt.get("epoch", 0)
        else:
            model.load_state_dict(ckpt)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(start_epoch, start_epoch + args.epochs):
        model.train()
        running = 0.0
        n = 0
        for batch in loader:
            images = batch["image"].to(args.device)
            masks = batch["mask"].to(args.device).long()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n += 1
            if n % 10 == 0:
                log.info(f"  epoch {ep+1}  batch {n}/{len(loader)}  loss={running/n:.4f}")
        avg = running / max(n, 1)
        log.info(f"Epoch {ep+1}/{start_epoch + args.epochs} done — avg loss {avg:.4f}")

        # Always save a resumable checkpoint
        ckpt_path = save_dir / f"epoch_{ep+1}.pth"
        torch.save({
            "model_state": model.state_dict(),
            "epoch": ep + 1,
            "loss": avg,
            "num_classes": args.num_classes,
        }, ckpt_path)
        log.info(f"  saved {ckpt_path}")

        # Update latest
        latest = save_dir / "latest_best.pth"
        torch.save({
            "model_state": model.state_dict(),
            "epoch": ep + 1,
            "loss": avg,
            "num_classes": args.num_classes,
        }, latest)
        log.info(f"  saved {latest}")

    log.info("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
