"""
SkyLogic MAS — Train the YOLO detector (Agent A) on the prepared dataset.

Prerequisite: run scripts/build_yolo_dataset.py first.

Examples:
  # Smoke run — 1 epoch, tiny subset
  python scripts/train_detector.py --epochs 1 --imgsz 416 --batch 4 --name smoke

  # Stable full run (crash-safe CPU config + class-imbalance fixes)
  python scripts/train_detector.py \\
      --epochs 50 --imgsz 416 --batch 4 --amp-off \\
      --cls-pw 2.0 --fl-gamma 1.5 --annotated-only \\
      --name xview_v2

  # Resume an interrupted run
  python scripts/train_detector.py --resume models/yolo/xview_v2/weights/last.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_detector")


def build_annotated_only_yaml(data_yaml: Path, project_root: Path) -> Path:
    """
    Create a YAML pointing train at only annotated patches (non-empty label files).
    Returned yaml lives alongside the original in data/yolo/.
    """
    import yaml

    label_dir = project_root / "data" / "yolo" / "labels" / "train"
    img_dir   = project_root / "data" / "yolo" / "images" / "train"

    annotated_imgs = []
    for lf in sorted(label_dir.glob("*.txt")):
        if lf.stat().st_size > 0:
            img = img_dir / (lf.stem + ".png")
            if img.exists():
                annotated_imgs.append(str(img))

    log.info(f"  Annotated-only train set: {len(annotated_imgs)} images "
             f"(vs all {len(list(img_dir.glob('*.png')))} incl. backgrounds)")

    txt_path = project_root / "data" / "yolo" / "annotated_train.txt"
    txt_path.write_text("\n".join(annotated_imgs) + "\n")

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    cfg["train"] = str(txt_path)   # absolute path list accepted by Ultralytics

    out_yaml = project_root / "data" / "yolo" / "skylogic_annotated.yaml"
    out_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    log.info(f"  Wrote annotated-only YAML: {out_yaml}")
    return out_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    # ── Dataset & model ──────────────────────────────────────────────────────
    parser.add_argument("--data", default="data/yolo/skylogic.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Base checkpoint (yolov8n.pt for CPU)")
    parser.add_argument("--annotated-only", action="store_true",
                        help="Rebuild train list to only include annotated patches")
    # ── Training hypers ──────────────────────────────────────────────────────
    parser.add_argument("--epochs",   type=int,   default=30)
    parser.add_argument("--imgsz",    type=int,   default=416)
    parser.add_argument("--batch",    type=int,   default=4)
    parser.add_argument("--lr0",      type=float, default=0.01)
    parser.add_argument("--lrf",      type=float, default=0.01,
                        help="Final LR as fraction of lr0")
    parser.add_argument("--patience", type=int,   default=15,
                        help="Early-stop patience (epochs without improvement)")
    # ── Crash-fix flags ───────────────────────────────────────────────────────
    parser.add_argument("--amp-off",  action="store_true",
                        help="Disable AMP (fixes SIGSEGV on CPU/WSL2)")
    parser.add_argument("--workers",  type=int,   default=0,
                        help="DataLoader workers (0 = single-threaded, safest on WSL2)")
    parser.add_argument("--cache",    default=False,
                        help="Cache dataset: 'ram', 'disk', or False")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of dataset to use (0.1 = 10%%) — for fast iteration")
    parser.add_argument("--no-val",   action="store_true",
                        help="Skip validation phase each epoch (debug/stability)")
    # ── Class-imbalance flags ─────────────────────────────────────────────────
    parser.add_argument("--cls-pw",   type=float, default=1.0,
                        help="BCE positive-class weight for classification head. "
                             ">1 increases penalty for missing any class. (valid in 8.4.x)")
    parser.add_argument("--cls-gain", type=float, default=0.5,
                        help="Classification loss coefficient. Default 0.5; raise to "
                             "1.5 to triple the gradient signal on class predictions.")
    # NOTE: fl_gamma removed from Ultralytics 8.4.42 overrideable params.
    # Class imbalance is handled via cls_pw + cls_gain + copy_paste + annotated-only data.
    # ── Augmentation tuning (lower = less memory pressure on WSL2 CPU) ────────
    parser.add_argument("--mosaic",     type=float, default=1.0,
                        help="Mosaic prob; lower (or 0) reduces memory load")
    parser.add_argument("--mixup",      type=float, default=0.05,
                        help="MixUp prob; 0 disables")
    parser.add_argument("--copy-paste", type=float, default=0.1,
                        help="Copy-Paste prob; 0 disables")
    parser.add_argument("--simple-aug", action="store_true",
                        help="Disable heavy aug (mosaic+mixup+copy_paste) — WSL2 stability mode")
    # ── Output / checkpointing ────────────────────────────────────────────────
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--project",     default="models/yolo")
    parser.add_argument("--name",        default="xview_v2")
    parser.add_argument("--save-period", type=int, default=5,
                        help="Save checkpoint every N epochs (in addition to best/last)")
    parser.add_argument("--plots",       action="store_true", default=True,
                        help="Save loss-curve PNG plots after training")
    # ── Resume ───────────────────────────────────────────────────────────────
    parser.add_argument("--resume", default=None,
                        help="Path to last.pt to resume a crashed/interrupted run")
    # ── Project root (for annotated-only yaml builder) ────────────────────────
    parser.add_argument("--project-root", default=".",
                        help="Root of the SkyLogic project (default: cwd)")
    args = parser.parse_args()

    from ultralytics import YOLO

    # ── Resume branch ────────────────────────────────────────────────────────
    if args.resume:
        log.info("Resuming from %s", args.resume)
        model = YOLO(args.resume)
        results = model.train(resume=True)
        log.info("Resume complete.")
        return 0

    # ── Validate dataset ─────────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        log.error("Dataset YAML not found: %s — run build_yolo_dataset.py first.", data_path)
        return 2

    # ── Optionally rebuild to annotated-only list ────────────────────────────
    if args.annotated_only:
        root = Path(args.project_root).resolve()
        data_path = build_annotated_only_yaml(data_path, root)

    # ── Compose train kwargs ─────────────────────────────────────────────────
    model = YOLO(args.model)

    train_kwargs = {
        # Core
        "data":    str(data_path),
        "epochs":  args.epochs,
        "imgsz":   args.imgsz,
        "batch":   args.batch,
        "device":  args.device,

        # Crash fixes
        "amp":     not args.amp_off,   # False when --amp-off passed
        "half":    False,              # Never use FP16 on CPU
        "workers": args.workers,
        "cache":   args.cache,         # 'ram' avoids per-batch I/O on WSL2
        "fraction": args.fraction,     # use subset of data for fast iteration
        "val":     not args.no_val,    # skip val phase if requested

        # LR schedule
        "lr0":  args.lr0,
        "lrf":  args.lrf,
        "optimizer": "SGD",
        "momentum":  0.937,
        "weight_decay": 5e-4,
        "warmup_epochs": 3.0,

        # ── Class imbalance (Ultralytics 8.4.x valid params) ─────────────────
        # fl_gamma was removed from overrideable kwargs in 8.4.x.
        # We compensate with three levers instead:
        #   1. cls_pw  — amplifies BCE positive-class loss (penalises FN more)
        #   2. cls     — triples the classification loss coefficient vs default
        #   3. copy_paste + close_mosaic=0 — keeps rare objects in every batch
        "cls_pw":    args.cls_pw,       # 2.0 → penalise missing rare classes harder
        "cls":       args.cls_gain,     # 1.5 → 3× default classification gradient
        "box":       7.5,               # keep default regression weight
        "dfl":       1.5,               # keep default DFL weight

        # ── Augmentation — maximise rare-class exposure ───────────────────
        # WSL2 CPU mode is fragile under heavy aug; --simple-aug forces all to 0
        "mosaic":       0.0 if args.simple_aug else args.mosaic,
        "close_mosaic": 0,     # 0 = keep mosaic for all epochs (default=10 turns it off)
        "mixup":        0.0 if args.simple_aug else args.mixup,
        "copy_paste":   0.0 if args.simple_aug else args.copy_paste,
        "degrees":      5.0,   # rotation ±5°
        "translate":    0.1,
        "scale":        0.5,
        "fliplr":       0.5,
        "flipud":       0.0,
        "hsv_h":        0.015,
        "hsv_s":        0.7,
        "hsv_v":        0.4,
        "erasing":      0.4,   # random erasing (prevents overfit on common objects)

        # Early stopping & checkpoints
        "patience":    args.patience,
        "save":        True,
        "save_period": args.save_period,  # checkpoint every 5 epochs
        "exist_ok":    True,

        # Output
        "project": args.project,
        "name":    args.name,
        "plots":   args.plots,
        "verbose": True,
    }

    log.info("=" * 60)
    log.info("YOLO training configuration")
    log.info("  AMP        : %s", "ON" if train_kwargs["amp"] else "OFF (crash-safe)")
    log.info("  batch/imgsz: %d / %d", args.batch, args.imgsz)
    log.info("  cls_pw     : %.1f  (>1 = amplify positive-class BCE penalty)", args.cls_pw)
    log.info("  cls_gain   : %.1f  (classification loss coefficient, default=0.5)", args.cls_gain)
    log.info("  copy_paste : %.2f (rare-class exposure via object pasting)", train_kwargs["copy_paste"])
    log.info("  close_mosaic: %d  (0=keep mosaic all epochs)", train_kwargs["close_mosaic"])
    log.info("  annotated  : %s", "only" if args.annotated_only else "all (incl. backgrounds)")
    log.info("  epochs     : %d  patience=%d", args.epochs, args.patience)
    log.info("=" * 60)

    results = model.train(**train_kwargs)
    log.info("Training complete.")

    # ── Wire latest_best.pt ──────────────────────────────────────────────────
    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        log.info("Best weights: %s  (%.1f MB)", best.resolve(), best.stat().st_size / 1e6)
        latest_link = Path(args.project) / "latest_best.pt"
        try:
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            try:
                latest_link.symlink_to(best.resolve())
            except (OSError, NotImplementedError):
                import shutil
                shutil.copy2(best, latest_link)
            log.info("Updated %s", latest_link)
        except Exception as e:
            log.warning("Could not update latest_best.pt: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
