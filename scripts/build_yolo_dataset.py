"""
SkyLogic MAS — Build YOLO-format dataset from annotations CSV.

Reads:
  data/metadata/annotations_metadata.csv  (per-bbox rows in pixel coords)
  data/metadata/patches_metadata.csv      (per-patch rows with split + size)
  data/patches/{train,val}/*.png

Writes:
  data/yolo/images/{train,val}/*.png   (symlinks/copies)
  data/yolo/labels/{train,val}/*.txt   (YOLO normalized: cls cx cy w h)
  data/yolo/skylogic.yaml              (Ultralytics dataset config)
  data/yolo/class_map.json             (sparse xView ID -> contiguous 0..N-1)

Patches with no annotations are skipped from the labels but still copied
as images (Ultralytics treats empty-label images as background examples).

The script is idempotent and skips re-creating existing files.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("build_yolo_dataset")


def parse_bbox(value) -> list[float] | None:
    """Parse a bbox cell which may be a JSON-like string '[x1, y1, x2, y2]'."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root (default: cwd)")
    parser.add_argument("--limit-train", type=int, default=0,
                        help="If >0, only use N train patches (smoke runs)")
    parser.add_argument("--limit-val", type=int, default=0,
                        help="If >0, only use N val patches (smoke runs)")
    parser.add_argument("--use-symlinks", action="store_true",
                        help="Symlink images instead of copying (faster, Linux-only)")
    parser.add_argument("--min-classes", type=int, default=1,
                        help="Drop classes with fewer than N annotations")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    meta_dir = root / "data" / "metadata"
    patches_dir = root / "data" / "patches"
    out_dir = root / "data" / "yolo"

    annotations_csv = meta_dir / "annotations_metadata.csv"
    patches_csv = meta_dir / "patches_metadata.csv"

    if not annotations_csv.exists() or not patches_csv.exists():
        log.error("Metadata CSVs missing. Run scripts/ingest.py first.")
        return 2

    log.info(f"Loading {patches_csv.name}…")
    patches_df = pd.read_csv(patches_csv)
    log.info(f"  {len(patches_df)} patches")

    log.info(f"Loading {annotations_csv.name}…")
    ann_df = pd.read_csv(annotations_csv)
    log.info(f"  {len(ann_df)} annotations")

    # --- Drop rare classes if requested ---
    class_counts = ann_df["class_id"].value_counts()
    keep_ids = set(class_counts[class_counts >= args.min_classes].index.tolist())
    ann_df = ann_df[ann_df["class_id"].isin(keep_ids)].copy()
    log.info(f"  After min_classes={args.min_classes}: {len(ann_df)} annotations / {len(keep_ids)} classes")

    # --- Build contiguous class map ---
    sparse_ids = sorted(keep_ids)
    class_id_to_idx = {sid: idx for idx, sid in enumerate(sparse_ids)}
    # name lookup
    id_to_name: dict[int, str] = (
        ann_df.groupby("class_id")["class_name"]
        .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else f"class_{s.name}")
        .to_dict()
    )
    names = {idx: id_to_name.get(sid, f"class_{sid}") for sid, idx in class_id_to_idx.items()}

    # --- Optionally subsample for smoke runs ---
    train_patches = patches_df[patches_df["split"] == "train"]["filename"].tolist()
    val_patches = patches_df[patches_df["split"] == "val"]["filename"].tolist()

    # Prefer patches that actually have annotations (more useful for smoke training)
    annotated = set(ann_df["patch_filename"].unique())
    if args.limit_train > 0:
        annotated_train = [p for p in train_patches if p in annotated]
        train_patches = annotated_train[: args.limit_train]
        log.info(f"  Subsampling train → {len(train_patches)} patches (annotated only)")
    if args.limit_val > 0:
        # Validation patches in this dataset have NO annotations (val/ has no GT)
        # so just take the first N regardless.
        val_patches = val_patches[: args.limit_val]
        log.info(f"  Subsampling val   → {len(val_patches)} patches")

    train_set = set(train_patches)
    val_set = set(val_patches)

    # --- Layout ---
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # --- Group annotations by patch ---
    log.info("Indexing annotations by patch…")
    ann_by_patch: dict[str, list[tuple[int, list[float]]]] = {}
    for row in ann_df.itertuples(index=False):
        bbox = parse_bbox(row.bbox)
        if bbox is None or len(bbox) != 4:
            continue
        cls_idx = class_id_to_idx[row.class_id]
        ann_by_patch.setdefault(row.patch_filename, []).append((cls_idx, bbox))

    # --- Write out ---
    written_images = 0
    written_labels = 0
    skipped = 0

    patches_meta = patches_df.set_index("filename")[["split", "actual_width", "actual_height"]]

    def link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        if args.use_symlinks:
            try:
                dst.symlink_to(src)
                return
            except OSError:
                pass
        shutil.copy2(src, dst)

    for fname in (train_patches + val_patches):
        if fname in train_set:
            split = "train"
        elif fname in val_set:
            split = "val"
        else:
            continue

        src_img = patches_dir / split / fname
        if not src_img.exists():
            skipped += 1
            continue

        dst_img = out_dir / "images" / split / fname
        link_or_copy(src_img, dst_img)
        written_images += 1

        # Write labels
        label_path = out_dir / "labels" / split / (Path(fname).stem + ".txt")
        if fname in ann_by_patch:
            try:
                w = float(patches_meta.loc[fname, "actual_width"])
                h = float(patches_meta.loc[fname, "actual_height"])
            except KeyError:
                w = h = 512.0
            lines = []
            for cls_idx, (x1, y1, x2, y2) in ann_by_patch[fname]:
                x1, x2 = sorted([max(0.0, x1), min(w, x2)])
                y1, y2 = sorted([max(0.0, y1), min(h, y2)])
                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                nbw = bw / w
                nbh = bh / h
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nbw:.6f} {nbh:.6f}")
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
            if lines:
                written_labels += 1
        else:
            # background image — empty label file
            label_path.write_text("")

    log.info(f"  Images written: {written_images}")
    log.info(f"  Labels written: {written_labels}")
    log.info(f"  Patches skipped (image missing): {skipped}")

    # --- Dataset YAML ---
    yaml_data = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val" if val_patches else "images/train",  # fall back if no val
        "nc": len(names),
        "names": [names[i] for i in range(len(names))],
    }
    yaml_path = out_dir / "skylogic.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False))
    log.info(f"Wrote {yaml_path}")

    map_path = out_dir / "class_map.json"
    map_path.write_text(json.dumps(
        {"sparse_to_contiguous": class_id_to_idx, "names": names},
        indent=2,
    ))
    log.info(f"Wrote {map_path}")

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
