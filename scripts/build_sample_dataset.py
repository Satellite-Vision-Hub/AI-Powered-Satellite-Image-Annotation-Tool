#!/usr/bin/env python3
"""
build_sample_dataset.py - Create sample_dataset_9k/, a COPY-ONLY ~9k sample.

SAFETY GUARANTEES
  * Never moves or deletes anything from the original dataset.
  * Pure shutil.copy2 - the original data/ tree is read-only here.
  * Idempotent - re-running skips images already copied (size-matched),
    so it is safe to resume an interrupted copy without recopying.

WHAT IT BUILDS
  sample_dataset_9k/
    images/{train,val,test}/*.png   copied 512x512 patches (annotated only)
    labels/{train,val,test}/*.txt   matching YOLO label files
    data.yaml                       YOLOv12 dataset config (62 classes)
    class_map.json                  sparse xView id -> contiguous id (copied)
    seg_class_map.json              contiguous id -> 10-class disaster seg index
    dataset_stats.json              per-split counts + class histogram
    SAMPLE_INFO.txt                 human-readable summary

The sample = ALL annotated patches (every patch that has >=1 object). This
maximises class diversity for a ~9k-image experimental subset; only empty
background patches are excluded.

Usage:
    python scripts/build_sample_dataset.py
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

# ----------------------------------------------------------------------
# Paths - resolved relative to the project root (parent of scripts/)
# ----------------------------------------------------------------------
PROJ        = Path(__file__).resolve().parent.parent
SRC_IMAGES  = PROJ / "data" / "patches" / "train"          # real PNG patches
SRC_LABELS  = PROJ / "data" / "yolo" / "labels" / "train"  # YOLO label .txt
ANNOTATED   = PROJ / "data" / "yolo" / "annotated_train.txt"
CLASS_MAP   = PROJ / "data" / "yolo" / "class_map.json"
SKY_YAML    = PROJ / "data" / "yolo" / "skylogic.yaml"
OUT         = PROJ / "sample_dataset_9k"

# 80/10/10 train/val/test - original val/ has only empty labels, so we build
# a *real* val + test split out of the annotated images.
SPLITS = (("train", 0.80), ("val", 0.10), ("test", 0.10))
SEED   = 42

# xView SPARSE class id -> 10-class disaster segmentation index.
# 0 background, 1 building, 2 damaged_building, 3 vehicle, 4 road,
# 5 vegetation, 6 water_flood, 7 debris_rubble, 8 construction, 9 container
XVIEW_TO_SEG = {
    73: 1, 71: 1, 72: 1, 74: 1, 75: 1, 76: 1, 84: 1, 85: 1,
    77: 1, 86: 1,
    79: 8, 78: 8, 87: 7,
    18: 3, 19: 3, 20: 3, 21: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3,
    34: 3, 35: 3, 36: 3, 37: 3, 38: 3, 53: 3, 54: 3, 55: 3, 56: 3, 57: 3,
    59: 3, 60: 3, 61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3,
    11: 3, 12: 3, 13: 3, 15: 3, 17: 3,
    40: 3, 41: 3, 42: 3, 44: 3, 45: 3, 47: 3, 49: 3, 50: 3, 51: 3, 52: 3,
    83: 3, 89: 9,
}
SEG_CLASS_NAMES = [
    "background", "building", "damaged_building", "vehicle", "road",
    "vegetation", "water_flood", "debris_rubble", "construction", "container",
]


def parse_yaml_names(yaml_path: Path) -> list[str]:
    """Minimal parser for the `names:` list in skylogic.yaml (no PyYAML dep)."""
    names: list[str] = []
    in_names = False
    for raw in yaml_path.read_text(encoding="utf-8").splitlines():
        if raw.strip().startswith("names:"):
            in_names = True
            continue
        if in_names:
            s = raw.strip()
            if s.startswith("- "):
                names.append(s[2:].strip())
            elif s and not s.startswith("#"):
                break  # left the names block
    return names


def main() -> int:
    print("=" * 64)
    print("build_sample_dataset.py - copy-only ~9k sample builder")
    print("=" * 64)

    # ---- Validate sources ------------------------------------------------
    for p in (SRC_IMAGES, SRC_LABELS, ANNOTATED, CLASS_MAP, SKY_YAML):
        if not p.exists():
            print(f"ERROR: required source missing: {p}")
            return 2

    names = parse_yaml_names(SKY_YAML)
    if len(names) != 62:
        print(f"WARNING: expected 62 class names, parsed {len(names)}")
    class_map = json.loads(CLASS_MAP.read_text(encoding="utf-8"))
    sparse_to_contig = {int(k): int(v) for k, v in
                        class_map.get("sparse_to_contiguous", {}).items()}

    # contiguous id -> seg index (derive from sparse map + XVIEW_TO_SEG)
    contig_to_seg = {}
    for sparse_id, contig_id in sparse_to_contig.items():
        contig_to_seg[contig_id] = XVIEW_TO_SEG.get(sparse_id, 0)

    # ---- Load annotated image stems -------------------------------------
    stems = []
    for line in ANNOTATED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            stems.append(Path(line).stem)
    stems = sorted(set(stems))
    print(f"Annotated images listed: {len(stems)}")

    # ---- Deterministic shuffle + split ----------------------------------
    random.seed(SEED)
    random.shuffle(stems)
    n = len(stems)
    n_train = int(n * SPLITS[0][1])
    n_val = int(n * SPLITS[1][1])
    split_items = {
        "train": stems[:n_train],
        "val":   stems[n_train:n_train + n_val],
        "test":  stems[n_train + n_val:],
    }
    for s, items in split_items.items():
        print(f"  {s:5s}: {len(items)} images")

    # ---- Create output dirs ---------------------------------------------
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ---- Copy (idempotent) ----------------------------------------------
    copied = skipped = missing = 0
    per_split_counts: dict[str, int] = {}
    class_hist: Counter = Counter()
    seg_hist: Counter = Counter()

    for split, items in split_items.items():
        ok = 0
        for stem in items:
            src_img = SRC_IMAGES / f"{stem}.png"
            src_lbl = SRC_LABELS / f"{stem}.txt"
            if not src_img.exists() or not src_lbl.exists():
                missing += 1
                continue
            dst_img = OUT / "images" / split / f"{stem}.png"
            dst_lbl = OUT / "labels" / split / f"{stem}.txt"

            # idempotent image copy - skip if already present and same size
            if dst_img.exists() and dst_img.stat().st_size == src_img.stat().st_size:
                skipped += 1
            else:
                shutil.copy2(src_img, dst_img)
                copied += 1
            shutil.copy2(src_lbl, dst_lbl)  # labels are tiny, always refresh

            # tally class histogram from the label file
            for ln in src_lbl.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                cid = int(float(ln.split()[0]))
                class_hist[cid] += 1
                seg_hist[contig_to_seg.get(cid, 0)] += 1
            ok += 1
        per_split_counts[split] = ok
        print(f"  {split:5s}: {ok} image/label pairs ready")

    print(f"\nCopy summary: copied={copied}  skipped(existing)={skipped}  "
          f"missing={missing}")

    # ---- Write data.yaml (YOLOv12 dataset config) -----------------------
    # `path` is a placeholder; the Colab notebook patches it at runtime.
    names_block = "\n".join(f"  {i}: {nm}" for i, nm in enumerate(names))
    data_yaml = (
        "# YOLOv12 dataset config - sample_dataset_9k\n"
        "# NOTE: the Colab notebook rewrites `path:` to the runtime location.\n"
        "path: ./sample_dataset_9k\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(names)}\n"
        f"names:\n{names_block}\n"
    )
    (OUT / "data.yaml").write_text(data_yaml, encoding="utf-8")

    # ---- Copy class_map.json + write seg_class_map.json -----------------
    shutil.copy2(CLASS_MAP, OUT / "class_map.json")
    seg_map = {
        "seg_class_names": SEG_CLASS_NAMES,
        "num_seg_classes": len(SEG_CLASS_NAMES),
        "contiguous_to_seg": {str(k): v for k, v in sorted(contig_to_seg.items())},
        "note": "Maps YOLO contiguous class id (0-61) -> 10-class disaster "
                "segmentation index. Used to rasterize bbox pseudo-masks for "
                "SegFormer weak-supervision training.",
    }
    (OUT / "seg_class_map.json").write_text(
        json.dumps(seg_map, indent=2), encoding="utf-8")

    # ---- Write dataset_stats.json ---------------------------------------
    total_imgs = sum(per_split_counts.values())
    stats = {
        "total_images": total_imgs,
        "splits": per_split_counts,
        "split_ratios": {s: r for s, r in SPLITS},
        "num_yolo_classes": len(names),
        "num_seg_classes": len(SEG_CLASS_NAMES),
        "total_annotations": sum(class_hist.values()),
        "yolo_class_histogram": {
            (names[i] if i < len(names) else f"class_{i}"): class_hist.get(i, 0)
            for i in range(len(names))
        },
        "seg_class_histogram": {
            SEG_CLASS_NAMES[i]: seg_hist.get(i, 0)
            for i in range(len(SEG_CLASS_NAMES))
        },
        "classes_present": sum(1 for i in range(len(names))
                               if class_hist.get(i, 0) > 0),
        "seed": SEED,
        "source": "data/patches/train (copy-only, original untouched)",
    }
    (OUT / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    # ---- Human-readable summary -----------------------------------------
    rare = sorted((class_hist.get(i, 0),
                   names[i] if i < len(names) else f"class_{i}")
                  for i in range(len(names)))[:8]
    info = [
        "SAMPLE DATASET 9K - build summary",
        "=" * 40,
        f"Total images : {total_imgs}",
        f"  train      : {per_split_counts.get('train', 0)}",
        f"  val        : {per_split_counts.get('val', 0)}",
        f"  test       : {per_split_counts.get('test', 0)}",
        f"YOLO classes : {len(names)} ({stats['classes_present']} present in sample)",
        f"Annotations  : {stats['total_annotations']}",
        f"Seg classes  : {len(SEG_CLASS_NAMES)}",
        "",
        "8 rarest classes (count, name):",
        *[f"  {c:5d}  {nm}" for c, nm in rare],
        "",
        "Original dataset was NOT modified - copy-only build.",
    ]
    (OUT / "SAMPLE_INFO.txt").write_text("\n".join(info), encoding="utf-8")

    print("\n" + "\n".join(info))
    print(f"\nOutput: {OUT}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
