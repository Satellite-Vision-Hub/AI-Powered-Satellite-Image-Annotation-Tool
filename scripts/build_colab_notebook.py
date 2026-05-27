#!/usr/bin/env python3
"""
build_colab_notebook.py - Generate SkyLogic_Sample9k_Colab.ipynb.

Produces a fully self-contained Colab notebook that:
  1.  Mounts Drive
  2.  Reads the ORIGINAL full dataset (zip or folder) from Drive
  3.  Builds a copy-only ~9k sample inside Colab/Drive (original untouched)
  4.  Verifies every label (format, class range, bbox normalisation)
  5.  Trains YOLOv12 (resume-aware) + SegFormer (resume-aware)
  6.  Saves everything to Drive and prints the final paths

Re-run this generator any time to refresh the notebook.

Usage:
    python scripts/build_colab_notebook.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "SkyLogic_Sample9k_Colab.ipynb"

CELLS: list[dict] = []


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text: str) -> None:
    CELLS.append({"cell_type": "code", "metadata": {},
                  "execution_count": None, "outputs": [], "source": text})


# ======================================================================
# Title
# ======================================================================
md("""# SkyLogic - Sample-9k Colab Pipeline (YOLOv12 + SegFormer)

**Self-contained for Google Colab T4 GPU.**

This notebook reads the **original/full** SkyLogic dataset (zip or folder)
from your Google Drive, builds a **copy-only ~9,000-image sample** while
preserving class diversity, verifies every label, then trains YOLOv12
detection + SegFormer segmentation. The original Drive dataset is **never
modified**.

**To run:**
1. `Runtime -> Change runtime type -> T4 GPU`.
2. Edit `DRIVE_DATASET_SOURCE` in the **Config** cell (Section 2) to point at
   your uploaded dataset (zip or folder).
3. `Runtime -> Run all`.

If Colab disconnects, just **Run all again** - both YOLOv12 and SegFormer
auto-resume from their last checkpoint on Drive.""")

# ======================================================================
# 1 - Environment setup
# ======================================================================
md("## 1 - Environment setup")

code("""# 1.1 - GPU + Torch sanity check
import subprocess
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
import torch
assert torch.cuda.is_available(), \\
    'No GPU! Runtime -> Change runtime type -> T4 GPU, then Run all.'
print('Torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))""")

# ======================================================================
# 2 - Drive mount and paths
# ======================================================================
md("## 2 - Drive mount and paths")

code("""# 2.1 - Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')""")

code("""# 2.2 - CONFIG: edit the Drive locations + sampling/training knobs below.
import os, logging, sys
from pathlib import Path

# ---- Drive inputs / outputs (EDIT THESE) ----------------------------
DRIVE_DATASET_SOURCE = '/content/drive/MyDrive/data'   # SkyLogic data folder on Drive (zip or folder)
DRIVE_PROJECT_DIR    = '/content/drive/MyDrive/SkyLogic_sample9k_experiment'

# ---- Sampling --------------------------------------------------------
SAMPLE_SIZE     = 9000          # approximate target image count
FORCE_RESAMPLE  = False         # set True to rebuild the Drive sample from scratch
SAMPLE_SEED     = 42
SPLIT_RATIOS    = (0.80, 0.10, 0.10)   # train, val, test (used if source has no usable splits)

# ---- Verification gate ----------------------------------------------
ALLOW_INVALID_LABELS = False    # set True to proceed despite verification errors

# ---- YOLOv12 detection -----------------------------------------------
YOLO_MODEL    = 'yolo12s.pt'    # lightweight; 'yolo12n.pt' = even faster on T4
YOLO_EPOCHS   = 50
YOLO_IMGSZ    = 512
YOLO_BATCH    = 16
YOLO_PATIENCE = 15
YOLO_RUN_NAME = 'sample9k'

# ---- SegFormer segmentation -----------------------------------------
SEG_BACKBONE      = 'nvidia/mit-b0'
SEG_EPOCHS        = 20
SEG_IMGSZ         = 512
SEG_BATCH         = 8
SEG_LR            = 6e-5
SEG_VAL_EVERY     = 2            # validate every N epochs
NUM_SEG_CLASSES   = 10

# ---- Derived paths --------------------------------------------------
DRIVE_SAMPLE_DIR = f'{DRIVE_PROJECT_DIR}/sample_dataset_9k'   # canonical sample
LOCAL_SOURCE_DIR = '/content/_source'                          # source extract
LOCAL_SAMPLE_DIR = '/content/sample_dataset_9k'                # fast SSD mirror
YOLO_PROJECT_DIR = f'{DRIVE_PROJECT_DIR}/yolov12'
SEG_DIR          = f'{DRIVE_PROJECT_DIR}/segformer'
RESULTS_DIR      = f'{DRIVE_PROJECT_DIR}/results'
LOG_FILE         = f'{DRIVE_PROJECT_DIR}/run.log'

for d in (DRIVE_PROJECT_DIR, DRIVE_SAMPLE_DIR, YOLO_PROJECT_DIR, SEG_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# Logging: tee everything to a Drive file so it survives disconnects.
for h in list(logging.root.handlers):
    logging.root.removeHandler(h)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger('skylogic')
log.info('=== SkyLogic Sample-9k run started ===')
log.info(f'DRIVE_DATASET_SOURCE = {DRIVE_DATASET_SOURCE}')
log.info(f'DRIVE_PROJECT_DIR    = {DRIVE_PROJECT_DIR}')
log.info(f'SAMPLE_SIZE = {SAMPLE_SIZE}  FORCE_RESAMPLE = {FORCE_RESAMPLE}')""")

# ======================================================================
# 3 - Dataset extraction/sampling
# ======================================================================
md("""## 3 - Dataset extraction/sampling
- Source `.zip` is extracted to `/content/_source` once (cached).
- Source folder is read directly (Drive FUSE) - only **sampled** files are copied.
- The sample is built on Drive (persistent) **and** mirrored to local SSD (fast).
- If a valid sample already exists on Drive and `FORCE_RESAMPLE = False`, it is
  reused without recopying.""")

code("""# 3.1 - Stage source: extract zip to local SSD, or locate the folder.
import shutil, zipfile, time

def stage_source():
    src = DRIVE_DATASET_SOURCE
    if not os.path.exists(src):
        raise FileNotFoundError(f'DRIVE_DATASET_SOURCE not found: {src}')
    if os.path.isdir(src):
        log.info(f'Source is a folder (read directly from Drive): {src}')
        return Path(src)
    if not src.lower().endswith('.zip'):
        raise ValueError(f'Source must be a folder or .zip: {src}')
    target = Path(LOCAL_SOURCE_DIR)
    marker = target / '.extracted'
    if marker.exists() and not FORCE_RESAMPLE:
        log.info(f'Source zip already extracted at {target} - reusing.')
        return target
    if target.exists() and FORCE_RESAMPLE:
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    log.info(f'Extracting {src} -> {target} ...')
    t0 = time.time()
    with zipfile.ZipFile(src) as zf:
        zf.extractall(target)
    marker.touch()
    log.info(f'Extracted in {time.time()-t0:.0f}s')
    return target

SOURCE_ROOT = stage_source()
log.info(f'SOURCE_ROOT = {SOURCE_ROOT}')""")

code("""# 3.2 - Auto-detect YOLO layout in the source + (if needed) synthesise YOLO
#       labels from metadata CSVs. Also locates class names + sparse->contig map.
import yaml, ast

def detect_layout(root):
    \"\"\"Return {split: (image_dir, label_dir)} by probing common patterns.

    Handles, in order:
      1. Pre-built YOLO sample:      images/{split}/      + labels/{split}/
      2. Repacked sample folder:     sample_dataset_9k/images/{split}/ + .../labels/{split}/
      3. Standard variant:           {split}/images/      + {split}/labels/
      4. SkyLogic project repo root: data/patches/{split} + data/yolo/labels/{split}
      5. data/yolo images variant:   data/yolo/images/{split} + data/yolo/labels/{split}
      6. SkyLogic data/ as root:     patches/{split}      + yolo/labels/{split}
    \"\"\"
    splits = {}
    for split in ('train', 'val', 'test'):
        candidates = [
            (root / 'images' / split,                           root / 'labels' / split),
            (root / 'sample_dataset_9k' / 'images' / split,     root / 'sample_dataset_9k' / 'labels' / split),
            (root / split / 'images',                           root / split / 'labels'),
            (root / 'data' / 'patches' / split,                 root / 'data' / 'yolo' / 'labels' / split),
            (root / 'data' / 'yolo' / 'images' / split,         root / 'data' / 'yolo' / 'labels' / split),
            (root / 'patches' / split,                          root / 'yolo' / 'labels' / split),
        ]
        for img_dir, lbl_dir in candidates:
            if img_dir.is_dir() and lbl_dir.is_dir():
                if next(img_dir.glob('*.png'), None) or next(img_dir.glob('*.jpg'), None):
                    splits[split] = (img_dir, lbl_dir); break
    if not splits:
        for img_dir, lbl_dir in [
            (root / 'images', root / 'labels'),
            (root,            root),
        ]:
            if img_dir.is_dir() and lbl_dir.is_dir() and (
                next(img_dir.glob('*.png'), None) or next(img_dir.glob('*.jpg'), None)):
                splits['train'] = (img_dir, lbl_dir); break
    return splits

def find_names(root):
    candidates = [
        root / 'data.yaml', root / 'skylogic.yaml',
        root / 'data' / 'yolo' / 'skylogic.yaml',
        root / 'data' / 'yolo' / 'data.yaml',
        root / 'sample_dataset_9k' / 'data.yaml',
    ]
    for c in candidates:
        if c.exists():
            try:
                cfg = yaml.safe_load(c.read_text())
                n = cfg.get('names')
                if isinstance(n, dict): return [n[i] for i in sorted(n.keys())]
                if isinstance(n, list): return n
            except Exception as e:
                log.warning(f'  could not parse {c}: {e}')
    return None

def find_class_map(root):
    for c in [root / 'class_map.json',
              root / 'data' / 'yolo' / 'class_map.json',
              root / 'sample_dataset_9k' / 'class_map.json']:
        if c.exists():
            try: return json.loads(c.read_text())
            except Exception: pass
    return None

def synthesize_yolo_labels_from_metadata(root):
    \"\"\"Build YOLO labels from data/metadata/{annotations,patches}_metadata.csv.

    Looks for the CSVs in two locations:
      - root/metadata/...      (when SOURCE_ROOT points at the data/ folder)
      - root/data/metadata/... (when SOURCE_ROOT points at the project repo root)

    Returns (synth_dir, sparse_to_contig, names_list, patches_root) or
    (None, None, None, None) if metadata files are missing.

    Output structure: /content/_synth_labels/{train,val,test}/*.txt
    The patches images are referenced in-place; only labels are written.
    \"\"\"
    import pandas as pd
    candidates = [
        (root / 'metadata',         root / 'patches'),
        (root / 'data' / 'metadata',root / 'data' / 'patches'),
    ]
    meta_dir = patches_root = None
    for m, p in candidates:
        if (m / 'annotations_metadata.csv').exists() \\
                and (m / 'patches_metadata.csv').exists() \\
                and p.is_dir():
            meta_dir = m; patches_root = p; break
    if meta_dir is None:
        return None, None, None, None

    log.info(f'No precomputed YOLO labels - synthesising from {meta_dir}/...')
    out_dir = Path('/content/_synth_labels')
    marker  = out_dir / '.synthesised'

    log.info('  reading annotations_metadata.csv ...')
    ann = pd.read_csv(meta_dir / 'annotations_metadata.csv')
    log.info('  reading patches_metadata.csv ...')
    p   = pd.read_csv(meta_dir / 'patches_metadata.csv')

    # Sparse class id -> contiguous + class names list (preserve sparse order).
    id_name = ann[['class_id','class_name']].drop_duplicates('class_id') \\
                  .sort_values('class_id')
    sparse_ids = [int(c) for c in id_name['class_id']]
    sparse_to_contig = {sid: i for i, sid in enumerate(sparse_ids)}
    names_list = [str(n) for n in id_name['class_name'].tolist()]
    log.info(f'  {len(names_list)} classes derived from CSV: '
             f'sparse {min(sparse_ids)}..{max(sparse_ids)} -> contiguous 0..{len(names_list)-1}')

    if marker.exists() and not FORCE_RESAMPLE:
        log.info(f'  labels already synthesised at {out_dir} - reusing')
        return out_dir, sparse_to_contig, names_list, patches_root

    log.info(f'  writing synthesised labels under {out_dir} (one-time cost)')
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    for split in ('train', 'val', 'test'):
        (out_dir / split).mkdir()

    # Map patch_filename -> (W, H, split)
    p_info = {}
    for row in p.itertuples(index=False):
        try:
            p_info[str(row.filename)] = (int(row.width), int(row.height), str(row.split))
        except Exception:
            continue

    written = {'train': 0, 'val': 0, 'test': 0}
    no_meta = 0
    n_grp = ann['patch_filename'].nunique()
    log.info(f'  iterating {n_grp} unique patches with annotations ...')
    for i, (pf, group) in enumerate(ann.groupby('patch_filename'), 1):
        if pf not in p_info:
            no_meta += 1; continue
        W, H, split = p_info[pf]
        if split not in ('train', 'val', 'test'):
            split = 'train'
        lines = []
        for _, r in group.iterrows():
            try:
                bb = ast.literal_eval(str(r['bbox']))
                if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                    continue
                x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
            except Exception:
                continue
            contig = sparse_to_contig.get(int(r['class_id']))
            if contig is None: continue
            cx = ((x1 + x2) / 2.0) / W
            cy = ((y1 + y2) / 2.0) / H
            bw = abs(x2 - x1) / W
            bh = abs(y2 - y1) / H
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                    and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                continue
            lines.append(f'{contig} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
        if lines:
            stem = Path(pf).stem
            (out_dir / split / f'{stem}.txt').write_text('\\n'.join(lines) + '\\n')
            written[split] += 1
        if i % 5000 == 0:
            log.info(f'    {i}/{n_grp} patches processed')
    marker.touch()
    log.info(f'  synthesised labels per split: {written}  '
             f'(patches with no size-metadata: {no_meta})')
    return out_dir, sparse_to_contig, names_list, patches_root

# ---- Run detection ---------------------------------------------------
source_splits = detect_layout(SOURCE_ROOT)
names         = find_names(SOURCE_ROOT)
class_map     = find_class_map(SOURCE_ROOT)

# Fallback: if labels weren't found anywhere but metadata CSVs are present,
# synthesise YOLO labels from them.
if not source_splits:
    synth_dir, synth_s2c, synth_names, patches_root = \\
        synthesize_yolo_labels_from_metadata(SOURCE_ROOT)
    if synth_dir is not None:
        for split in ('train', 'val', 'test'):
            img_dir = patches_root / split
            lbl_dir = synth_dir / split
            if img_dir.is_dir() and lbl_dir.is_dir() \\
                    and next(img_dir.glob('*.png'), None):
                source_splits[split] = (img_dir, lbl_dir)
        if names is None:
            names = synth_names
        if class_map is None and synth_s2c:
            class_map = {'sparse_to_contiguous': {str(k): v for k, v in synth_s2c.items()}}
        log.info(f'After synthesis - splits: {list(source_splits)}')

log.info(f'Detected splits: {list(source_splits)}')
for s, (i, l) in source_splits.items():
    log.info(f'  {s}: images={i}  labels={l}')
log.info(f'Class names file : '
         f'{"found ("+str(len(names))+" classes)" if names else "not found - will infer"}')
log.info(f'class_map.json   : {"found" if class_map else "not found"}')

assert source_splits, ('No image/label layout detected in source - '
                       'and no metadata CSVs found to synthesise labels from. '
                       'Check DRIVE_DATASET_SOURCE.')""")

# Debug cell - prints everything we know about source + detected layout.
code("""# 3.3 - DEBUG: source state + detected layout (before sampling)
print('=' * 78)
print(' DEBUG: dataset source + detected layout')
print('=' * 78)
print(f'DRIVE_DATASET_SOURCE   = {DRIVE_DATASET_SOURCE}')
print(f'exists                 = {os.path.exists(DRIVE_DATASET_SOURCE)}')
print(f'SOURCE_ROOT (post-stg) = {SOURCE_ROOT}')
print(f'SOURCE_ROOT exists     = {Path(SOURCE_ROOT).exists()}')
print()
print('Top-level entries inside SOURCE_ROOT:')
try:
    for item in sorted(Path(SOURCE_ROOT).iterdir()):
        kind = 'D' if item.is_dir() else 'F'
        sz = ''
        if item.is_file():
            try: sz = f'  ({item.stat().st_size:,} bytes)'
            except Exception: pass
        print(f'  [{kind}] {item.name}{sz}')
except Exception as e:
    print(f'  (could not list: {e})')

print()
print(f'Detected splits: {list(source_splits)}')
total_imgs = total_lbls = 0
for split, (img_dir, lbl_dir) in source_splits.items():
    try:
        n_img = sum(1 for p in img_dir.iterdir() if p.is_file())
    except Exception: n_img = -1
    try:
        n_lbl = sum(1 for _ in lbl_dir.glob('*.txt'))
    except Exception: n_lbl = -1
    total_imgs += max(0, n_img); total_lbls += max(0, n_lbl)
    print(f'  {split}:')
    print(f'    images_dir = {img_dir}  ({n_img} files)')
    print(f'    labels_dir = {lbl_dir}  ({n_lbl} files)')
print(f'TOTAL images detected: {total_imgs}')
print(f'TOTAL labels detected: {total_lbls}')
print(f'Class names           : {len(names) if names else 0}')
print(f'class_map.json present: {bool(class_map)}')
print('=' * 78)
log.info(f'DEBUG layout summary - splits={list(source_splits)} '
         f'imgs={total_imgs} lbls={total_lbls} '
         f'names={len(names) if names else 0}')""")

code("""# 3.3 - Collect annotated stems per split + decide sampling mode.
import random

def collect_annotated(splits):
    out = {}
    for split, (img_dir, lbl_dir) in splits.items():
        anns = []
        for lp in lbl_dir.glob('*.txt'):
            if lp.stat().st_size == 0:
                continue
            stem = lp.stem
            ext = None
            for cand in ('.png', '.jpg', '.jpeg'):
                if (img_dir / f'{stem}{cand}').exists():
                    ext = cand; break
            if ext is not None:
                anns.append((stem, ext))
        out[split] = sorted(anns)
    return out

source_ann = collect_annotated(source_splits)
for s, v in source_ann.items():
    log.info(f'  annotated in {s}: {len(v)}')

# Use existing train/val/test splits if all three have a meaningful amount
# of annotated data; otherwise re-split everything 80/10/10.
present_with_anns = [s for s in ('train','val','test') if len(source_ann.get(s, [])) > 50]
total_ann = sum(len(v) for v in source_ann.values())
assert total_ann > 0, 'No annotated images found - cannot sample.'

PRESERVE_SPLITS = (len(present_with_anns) >= 2)
log.info(f'Total annotated images: {total_ann}')
log.info(f'Sampling mode: {"preserve source splits" if PRESERVE_SPLITS else "fresh 80/10/10 split"}')""")

code("""# 3.4 - Build per-split sample lists (class-stratified when sub-sampling).
from collections import defaultdict

def label_classes(lbl_path):
    cls = set()
    for ln in lbl_path.read_text().splitlines():
        p = ln.split()
        if not p: continue
        try: cls.add(int(float(p[0])))
        except ValueError: pass
    return cls

def stratified_pick(pool, lbl_dir_fn, target_n, seed):
    \"\"\"Round-robin pick over rare-first class buckets until we hit target_n.\"\"\"
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for item in pool:
        for c in label_classes(lbl_dir_fn(item)):
            buckets[c].append(item)
    for k in buckets:
        rng.shuffle(buckets[k])
    rare_first = sorted(buckets.keys(), key=lambda c: len(buckets[c]))
    selected = []
    seen = set()
    cursor = {c: 0 for c in rare_first}
    safety = 0
    while len(selected) < target_n and safety < 2 * len(pool) + 10:
        safety += 1
        progressed = False
        for c in rare_first:
            while cursor[c] < len(buckets[c]):
                cand = buckets[c][cursor[c]]; cursor[c] += 1
                key = (cand[0], cand[1])
                if key not in seen:
                    seen.add(key); selected.append(cand); progressed = True; break
            if len(selected) >= target_n: break
        if not progressed: break
    # If still short, fall back to random fill from remaining
    if len(selected) < target_n:
        leftovers = [it for it in pool if (it[0], it[1]) not in seen]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:target_n - len(selected)])
    return selected[:target_n]

def build_sample_lists():
    rng = random.Random(SAMPLE_SEED)
    if PRESERVE_SPLITS:
        # Keep splits, sub-sample each proportionally if needed.
        result = {'train': [], 'val': [], 'test': []}
        total = sum(len(source_ann.get(s, [])) for s in present_with_anns)
        for split in ('train', 'val', 'test'):
            v = source_ann.get(split, [])
            if not v:
                continue
            quota = max(1, int(SAMPLE_SIZE * len(v) / total))
            pool = [(split, stem, ext) for (stem, ext) in v]
            if len(pool) <= quota:
                result[split] = pool
            else:
                result[split] = stratified_pick(
                    pool, lambda it: source_splits[it[0]][1] / f'{it[1]}.txt',
                    quota, SAMPLE_SEED + ord(split[0]))
        return result
    # Single-split / re-split path: pool everything, stratified pick to SAMPLE_SIZE,
    # then 80/10/10.
    pool = []
    for split, v in source_ann.items():
        for stem, ext in v:
            pool.append((split, stem, ext))
    if len(pool) <= SAMPLE_SIZE:
        chosen = list(pool)
    else:
        chosen = stratified_pick(
            pool, lambda it: source_splits[it[0]][1] / f'{it[1]}.txt',
            SAMPLE_SIZE, SAMPLE_SEED)
    rng.shuffle(chosen)
    n = len(chosen)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val   = int(n * SPLIT_RATIOS[1])
    return {'train': chosen[:n_train],
            'val':   chosen[n_train:n_train + n_val],
            'test':  chosen[n_train + n_val:]}

sample_lists = build_sample_lists()
for s in ('train', 'val', 'test'):
    log.info(f'  sample {s}: {len(sample_lists[s])} images')
total_sample = sum(len(v) for v in sample_lists.values())
log.info(f'Total sampled: {total_sample} (target ~{SAMPLE_SIZE})')""")

code("""# 3.5 - Check Drive for an existing valid sample (reuse if present).
def existing_sample_valid(drive_dir, sample_lists):
    base = Path(drive_dir)
    if FORCE_RESAMPLE or not base.exists():
        return False
    if not (base / 'data.yaml').exists():
        return False
    for split, items in sample_lists.items():
        if not items: continue
        img_dir = base / 'images' / split
        lbl_dir = base / 'labels' / split
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            return False
        # Spot-check 50 random expected files
        rng = random.Random(99)
        check = rng.sample(items, min(50, len(items)))
        for src_split, stem, ext in check:
            if not (img_dir / f'{stem}{ext}').exists(): return False
            if not (lbl_dir / f'{stem}.txt').exists(): return False
    return True

SAMPLE_REUSED = existing_sample_valid(DRIVE_SAMPLE_DIR, sample_lists)
log.info(f'Existing Drive sample valid: {SAMPLE_REUSED}'
         + ('  -> skipping copy' if SAMPLE_REUSED else ''))""")

code("""# 3.6 - Copy the sampled images + labels to Drive (idempotent, skip-if-exists).
def copy_sample_to_drive(target_root, sample_lists, source_splits, log_every=500):
    target_root = Path(target_root)
    copied = skipped = failed = 0
    for split, items in sample_lists.items():
        if not items: continue
        img_out = target_root / 'images' / split
        lbl_out = target_root / 'labels' / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for i, (src_split, stem, ext) in enumerate(items, 1):
            src_img = source_splits[src_split][0] / f'{stem}{ext}'
            src_lbl = source_splits[src_split][1] / f'{stem}.txt'
            dst_img = img_out / f'{stem}{ext}'
            dst_lbl = lbl_out / f'{stem}.txt'
            try:
                if (not dst_img.exists()
                        or dst_img.stat().st_size != src_img.stat().st_size):
                    shutil.copy2(src_img, dst_img); copied += 1
                else:
                    skipped += 1
                shutil.copy2(src_lbl, dst_lbl)
            except Exception as e:
                failed += 1
                if failed <= 3:
                    log.warning(f'  copy failed {stem}: {e}')
            if i % log_every == 0:
                log.info(f'    {split}: {i}/{len(items)}  copied={copied} reused={skipped}')
        log.info(f'  {split} done: {len(items)} pairs ready (copied={copied} reused={skipped})')
    log.info(f'Copy summary: copied={copied} reused={skipped} failed={failed}')

if SAMPLE_REUSED:
    log.info('Skipping copy - Drive sample already complete.')
else:
    log.info(f'Building sample at {DRIVE_SAMPLE_DIR}  '
             '(Drive FUSE writes are slow; this may take a few minutes)...')
    copy_sample_to_drive(DRIVE_SAMPLE_DIR, sample_lists, source_splits)""")

code("""# 3.7 - Write/refresh data.yaml + class_map + seg_class_map + stats on Drive.
# Names: parsed if available, else inferred from labels.
if names is None:
    log.warning('Class names not found - inferring from label contents.')
    max_cls = -1
    for split, items in sample_lists.items():
        for src_split, stem, _ext in items:
            lp = source_splits[src_split][1] / f'{stem}.txt'
            try:
                for ln in lp.read_text().splitlines():
                    p = ln.split()
                    if p:
                        try: max_cls = max(max_cls, int(float(p[0])))
                        except ValueError: pass
            except Exception:
                pass
    nc = max_cls + 1
    names = [f'class_{i}' for i in range(max(nc, 1))]
NUM_CLASSES = len(names)
log.info(f'NUM_CLASSES = {NUM_CLASSES}')

# data.yaml on Drive - the local mirror gets its own copy with path updated.
names_block = '\\n'.join(f'  {i}: {n}' for i, n in enumerate(names))
data_yaml_text = (
    '# Auto-generated by SkyLogic Sample-9k Colab notebook.\\n'
    '# `path:` is rewritten to the local mirror at training time for speed.\\n'
    f'path: {DRIVE_SAMPLE_DIR}\\n'
    'train: images/train\\n'
    'val: images/val\\n'
    'test: images/test\\n'
    f'nc: {NUM_CLASSES}\\n'
    'names:\\n' + names_block + '\\n'
)
Path(DRIVE_SAMPLE_DIR, 'data.yaml').write_text(data_yaml_text)

if class_map is not None:
    Path(DRIVE_SAMPLE_DIR, 'class_map.json').write_text(json.dumps(class_map, indent=2))

# 10-class disaster mapping for SegFormer pseudo-masks.
SEG_CLASS_NAMES = ['background','building','damaged_building','vehicle','road',
                   'vegetation','water_flood','debris_rubble','construction','container']
XVIEW_SPARSE_TO_SEG = {
    73:1, 71:1,72:1,74:1,75:1,76:1,84:1,85:1, 77:1,86:1,
    79:8, 78:8, 87:7,
    18:3,19:3,20:3,21:3,23:3,24:3,25:3,26:3,27:3,28:3,
    34:3,35:3,36:3,37:3,38:3,53:3,54:3,55:3,56:3,57:3,
    59:3,60:3,61:3,62:3,63:3,64:3,65:3,66:3,
    11:3,12:3,13:3,15:3,17:3,
    40:3,41:3,42:3,44:3,45:3,47:3,49:3,50:3,51:3,52:3,
    83:3, 89:9,
}
contig_to_seg = {}
if class_map and 'sparse_to_contiguous' in class_map:
    s2c = {int(k): int(v) for k, v in class_map['sparse_to_contiguous'].items()}
    for sparse_id, contig_id in s2c.items():
        contig_to_seg[contig_id] = XVIEW_SPARSE_TO_SEG.get(sparse_id, 0)
    for c in range(NUM_CLASSES):
        contig_to_seg.setdefault(c, 0)
else:
    # name-based heuristic (works for generic xView-like names)
    BUILD = {'building','damaged building','facility','shed','hut','tent','hangar'}
    VEHIC = {'car','truck','bus','vehicle','plane','aircraft','boat','ship','yacht',
             'ferry','train','locomotive','passenger','cargo','tugboat','barge',
             'motorboat','sailboat','helicopter','railway','locomotive'}
    CONSTR= {'construction','crane','bulldozer','excavator','cement','grader',
             'dump','loader','scraper','tower crane','reach stacker'}
    CONT  = {'container','shipping'}
    for i, nm in enumerate(names):
        n = nm.lower()
        if   any(w in n for w in CONT):   contig_to_seg[i] = 9
        elif any(w in n for w in CONSTR): contig_to_seg[i] = 8
        elif any(w in n for w in BUILD):  contig_to_seg[i] = 1
        elif any(w in n for w in VEHIC):  contig_to_seg[i] = 3
        else:                              contig_to_seg[i] = 0
seg_map_obj = {
    'seg_class_names': SEG_CLASS_NAMES,
    'num_seg_classes': len(SEG_CLASS_NAMES),
    'contiguous_to_seg': {str(k): v for k, v in sorted(contig_to_seg.items())},
}
Path(DRIVE_SAMPLE_DIR, 'seg_class_map.json').write_text(json.dumps(seg_map_obj, indent=2))

# Stats snapshot
stats = {
    'total_images': total_sample,
    'splits': {s: len(v) for s, v in sample_lists.items()},
    'split_ratios': list(SPLIT_RATIOS),
    'sampling_mode': 'preserve' if PRESERVE_SPLITS else 'fresh 80/10/10',
    'sample_size_config': SAMPLE_SIZE,
    'num_classes': NUM_CLASSES,
    'seed': SAMPLE_SEED,
    'reused_existing_drive_sample': SAMPLE_REUSED,
}
Path(DRIVE_SAMPLE_DIR, 'dataset_stats.json').write_text(json.dumps(stats, indent=2))
log.info(f'Wrote data.yaml + seg_class_map.json + dataset_stats.json to {DRIVE_SAMPLE_DIR}')""")

code("""# 3.8 - Mirror the Drive sample to /content for fast training reads.
def mirror_to_local(drive_root, local_root, log_every=1000):
    drive_root = Path(drive_root); local_root = Path(local_root)
    # Quick-skip if local already matches Drive counts (and no force).
    if local_root.exists() and (local_root / 'data.yaml').exists() and not FORCE_RESAMPLE:
        ok = True
        for split in ('train','val','test'):
            d = drive_root / 'images' / split
            l = local_root / 'images' / split
            if d.exists() and not l.exists(): ok = False; break
            if d.exists() and l.exists():
                if len(list(d.iterdir())) != len(list(l.iterdir())):
                    ok = False; break
        if ok:
            log.info(f'Local mirror already valid at {local_root} - skipping.')
            return
    if local_root.exists() and FORCE_RESAMPLE:
        shutil.rmtree(local_root, ignore_errors=True)
    local_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in drive_root.rglob('*'):
        if src.is_dir(): continue
        rel = src.relative_to(drive_root)
        dst = local_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        shutil.copy2(src, dst); n += 1
        if n % log_every == 0:
            log.info(f'    mirrored {n} files ...')
    log.info(f'Local mirror complete: {n} new files copied to {local_root}')

mirror_to_local(DRIVE_SAMPLE_DIR, LOCAL_SAMPLE_DIR)

# Rewrite the LOCAL data.yaml `path:` so YOLOv12 reads from /content (fast).
local_yaml = Path(LOCAL_SAMPLE_DIR) / 'data.yaml'
cfg = yaml.safe_load(local_yaml.read_text())
cfg['path'] = LOCAL_SAMPLE_DIR
local_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
DATA_YAML = str(local_yaml)
log.info(f'Local data.yaml ready at {DATA_YAML}  (path: {LOCAL_SAMPLE_DIR})')""")

# ======================================================================
# 4 - Dataset verification
# ======================================================================
md("""## 4 - Dataset verification
Hard checks before any training. Each label is parsed line-by-line:

- image <-> label match (both directions)
- 5 whitespace-separated fields per line: `class x_center y_center width height`
- class id is an integer in `[0, NUM_CLASSES)`
- bbox coordinates are numeric and in `[0, 1]`
- empty labels are counted (warning, not error)

If any **hard** error is found, the cell raises `VerificationError` and stops
the notebook. To proceed anyway, set `ALLOW_INVALID_LABELS = True` in the
**Config** cell.""")

code("""# 4.1 - Strong dataset verification
class VerificationError(RuntimeError): pass

def verify_dataset(local_root, num_classes):
    base = Path(local_root)
    per_split = {}
    issue_samples = {'unmatched_images':[], 'unmatched_labels':[],
                     'bad_format':[], 'bad_class_id':[], 'bad_bbox':[]}
    class_counts = {}
    for split in ('train', 'val', 'test'):
        img_dir = base / 'images' / split
        lbl_dir = base / 'labels' / split
        if not img_dir.exists() and not lbl_dir.exists():
            per_split[split] = {'images':0,'labels':0}; continue
        img_map = {p.stem: p.suffix for p in img_dir.glob('*')
                   if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg')}
        lbl_set = {p.stem for p in lbl_dir.glob('*.txt')}
        only_img = set(img_map) - lbl_set
        only_lbl = lbl_set - set(img_map)
        for s in sorted(only_img)[:10]: issue_samples['unmatched_images'].append(f'{split}/{s}')
        for s in sorted(only_lbl)[:10]: issue_samples['unmatched_labels'].append(f'{split}/{s}')
        empty = bad_fmt = bad_cls = bad_box = 0
        for stem in img_map:
            lp = lbl_dir / f'{stem}.txt'
            if not lp.exists(): continue
            content = lp.read_text().strip()
            if not content:
                empty += 1; continue
            for li, ln in enumerate(content.splitlines(), 1):
                p = ln.split()
                if len(p) != 5:
                    bad_fmt += 1
                    if len(issue_samples['bad_format']) < 10:
                        issue_samples['bad_format'].append(
                            f'{split}/{stem}.txt:{li} ({len(p)} fields)')
                    continue
                try:
                    cf = float(p[0])
                    if cf != int(cf): raise ValueError('not int')
                    c = int(cf)
                    if c < 0 or c >= num_classes:
                        raise ValueError(f'id {c} out of [0,{num_classes-1}]')
                    class_counts[c] = class_counts.get(c, 0) + 1
                except ValueError as e:
                    bad_cls += 1
                    if len(issue_samples['bad_class_id']) < 10:
                        issue_samples['bad_class_id'].append(
                            f'{split}/{stem}.txt:{li} ({e})')
                    continue
                try:
                    coords = [float(x) for x in p[1:]]
                except ValueError:
                    bad_box += 1
                    if len(issue_samples['bad_bbox']) < 10:
                        issue_samples['bad_bbox'].append(
                            f'{split}/{stem}.txt:{li} (non-numeric bbox)')
                    continue
                if any(not (0.0 <= x <= 1.0) for x in coords):
                    bad_box += 1
                    if len(issue_samples['bad_bbox']) < 10:
                        issue_samples['bad_bbox'].append(
                            f'{split}/{stem}.txt:{li} (coords {coords} not in [0,1])')
        per_split[split] = {
            'images':       len(img_map),
            'labels':       len(lbl_set),
            'only_image':   len(only_img),
            'only_label':   len(only_lbl),
            'empty_labels': empty,
            'bad_format':   bad_fmt,
            'bad_class_id': bad_cls,
            'bad_bbox':     bad_box,
        }
    return per_split, class_counts, issue_samples

per_split, class_counts, issue_samples = verify_dataset(LOCAL_SAMPLE_DIR, NUM_CLASSES)

log.info('=== DATASET VERIFICATION REPORT ===')
for s, st in per_split.items():
    log.info(f'  {s:5s} | images={st["images"]:5d}  labels={st["labels"]:5d}'
             f'  only_img={st.get("only_image",0)}  only_lbl={st.get("only_label",0)}'
             f'  empty={st.get("empty_labels",0)}'
             f'  bad_fmt={st.get("bad_format",0)}'
             f'  bad_cls={st.get("bad_class_id",0)}'
             f'  bad_bbox={st.get("bad_bbox",0)}')
log.info(f'  Total annotations counted: {sum(class_counts.values())}')
log.info(f'  Classes present in sample: {sum(1 for n in class_counts.values() if n>0)}/{NUM_CLASSES}')

if class_counts:
    sorted_cls = sorted(class_counts.items(), key=lambda kv: kv[1])
    log.info('  --- 5 rarest classes ---')
    for c, n in sorted_cls[:5]:
        nm = names[c] if c < len(names) else f'class_{c}'
        log.info(f'    {nm:30s} {n}')
    log.info('  --- 10 most common classes ---')
    for c, n in sorted_cls[-10:][::-1]:
        nm = names[c] if c < len(names) else f'class_{c}'
        log.info(f'    {nm:30s} {n}')

hard_errors = sum(st.get('only_image',0) + st.get('only_label',0)
                  + st.get('bad_format',0) + st.get('bad_class_id',0)
                  + st.get('bad_bbox',0) for st in per_split.values())

# Save report to Drive (always)
Path(RESULTS_DIR, 'dataset_verification.json').write_text(json.dumps({
    'per_split': per_split, 'class_counts': class_counts,
    'issue_samples': {k: v for k, v in issue_samples.items() if v},
    'num_classes': NUM_CLASSES,
}, indent=2))

if hard_errors > 0:
    log.warning(f'!!! Verification found {hard_errors} hard errors. Examples:')
    for k, vs in issue_samples.items():
        if vs:
            log.warning(f'  {k}:')
            for v in vs[:5]:
                log.warning(f'    {v}')
    if not ALLOW_INVALID_LABELS:
        raise VerificationError(
            f'Dataset has {hard_errors} hard errors (see above). Fix them, '
            'or set ALLOW_INVALID_LABELS = True in the Config cell to proceed.')
    log.warning('ALLOW_INVALID_LABELS = True -> proceeding despite errors.')
else:
    log.info('Dataset verification PASSED.')""")

# ======================================================================
# 5 - YOLOv12 training
# ======================================================================
md("## 5 - YOLOv12 training")

code("""# 5.1 - Install YOLOv12 + SegFormer dependencies
!pip install -q -U ultralytics
!pip install -q -U transformers
import ultralytics, transformers
ultralytics.checks()
log.info(f'ultralytics {ultralytics.__version__} | transformers {transformers.__version__}')""")

code("""# 5.2 - YOLOv12 training (resume-aware: checkpoints live on Drive)
from ultralytics import YOLO

run_dir     = Path(YOLO_PROJECT_DIR) / YOLO_RUN_NAME
last_ckpt   = run_dir / 'weights' / 'last.pt'
best_ckpt   = run_dir / 'weights' / 'best.pt'
results_csv = run_dir / 'results.csv'

done_epochs = 0
if results_csv.exists():
    done_epochs = max(0, sum(1 for _ in open(results_csv)) - 1)

if done_epochs >= YOLO_EPOCHS and best_ckpt.exists():
    log.info(f'YOLOv12 already trained {done_epochs} epochs - skipping training.')
    yolo_model = YOLO(str(best_ckpt))
elif last_ckpt.exists():
    log.info(f'Resuming YOLOv12 from {last_ckpt} (~{done_epochs} epochs done)')
    yolo_model = YOLO(str(last_ckpt))
    yolo_model.train(resume=True)
else:
    log.info(f'Fresh YOLOv12 training: {YOLO_MODEL}')
    yolo_model = YOLO(YOLO_MODEL)
    yolo_model.train(
        data=DATA_YAML, epochs=YOLO_EPOCHS, imgsz=YOLO_IMGSZ, batch=YOLO_BATCH,
        patience=YOLO_PATIENCE, device=0,
        project=YOLO_PROJECT_DIR, name=YOLO_RUN_NAME, exist_ok=True,
        save_period=5,         # frequent epoch checkpoints (last.pt every epoch)
        workers=2, cache='disk', amp=True,
        mosaic=0.5, mixup=0.05, close_mosaic=10,
        optimizer='auto', lr0=0.01, lrf=0.01, seed=SAMPLE_SEED,
        plots=True, verbose=True)
log.info('YOLOv12 training stage complete.')""")

# ======================================================================
# 6 - YOLOv12 validation/inference
# ======================================================================
md("## 6 - YOLOv12 validation/inference")

code("""# 6.1 - YOLOv12 validation on the val split
yolo_model = YOLO(str(best_ckpt))
val_metrics = yolo_model.val(data=DATA_YAML, imgsz=YOLO_IMGSZ, batch=YOLO_BATCH,
                             device=0, split='val', verbose=False)
yolo_scores = {
    'mAP_0.5':      float(val_metrics.box.map50),
    'mAP_0.5:0.95': float(val_metrics.box.map),
    'precision':    float(val_metrics.box.mp),
    'recall':       float(val_metrics.box.mr),
}
for k, v in yolo_scores.items():
    log.info(f'  YOLOv12 {k:14s}: {v:.4f}')""")

code("""# 6.2 - YOLOv12 inference visualisation on random test images
import random, matplotlib.pyplot as plt
test_imgs = sorted((Path(LOCAL_SAMPLE_DIR) / 'images' / 'test').glob('*.png')) \\
            + sorted((Path(LOCAL_SAMPLE_DIR) / 'images' / 'test').glob('*.jpg'))
if test_imgs:
    random.seed(0)
    sample = random.sample(test_imgs, min(6, len(test_imgs)))
    preds = yolo_model.predict(sample, imgsz=YOLO_IMGSZ, conf=0.25, device=0,
                                verbose=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, pred in zip(axes.flat, preds):
        ax.imshow(pred.plot()[..., ::-1])
        ax.set_title(Path(pred.path).name, fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    viz = f'{RESULTS_DIR}/yolov12_predictions.png'
    plt.savefig(viz, dpi=110, bbox_inches='tight'); plt.show()
    log.info(f'Saved {viz}')
else:
    log.warning('No test images available for inference visualisation.')""")

code("""# 6.3 - YOLOv12 metrics export (CSV + JSON)
yolo_metrics = {'model': YOLO_MODEL, 'epochs_config': YOLO_EPOCHS,
                'imgsz': YOLO_IMGSZ, 'batch': YOLO_BATCH,
                'epochs_done': done_epochs, **yolo_scores}
with open(f'{RESULTS_DIR}/yolov12_metrics.json', 'w') as f:
    json.dump(yolo_metrics, f, indent=2)
if results_csv.exists():
    shutil.copy2(results_csv, f'{RESULTS_DIR}/yolov12_results.csv')
log.info('YOLOv12 metrics exported:\\n' + json.dumps(yolo_metrics, indent=2))""")

# ======================================================================
# 7 - SegFormer training
# ======================================================================
md("## 7 - SegFormer training")

code("""# 7.1 - SegFormer dataset: bbox -> weak pseudo-mask (10 disaster classes)
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

_seg = json.loads((Path(LOCAL_SAMPLE_DIR) / 'seg_class_map.json').read_text())
CONTIG_TO_SEG = {int(k): int(v) for k, v in _seg['contiguous_to_seg'].items()}
SEG_NAMES = _seg['seg_class_names']

class WeakSegDataset(Dataset):
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD  = np.array([0.229, 0.224, 0.225], np.float32)
    def __init__(self, split, img_size):
        base = Path(LOCAL_SAMPLE_DIR)
        self.imgs = sorted((base/'images'/split).glob('*.png')) \\
                  + sorted((base/'images'/split).glob('*.jpg'))
        self.lbl_dir = base / 'labels' / split
        self.S = img_size
    def __len__(self): return len(self.imgs)
    def _mask(self, stem):
        m = np.zeros((self.S, self.S), np.int64)
        lp = self.lbl_dir / f'{stem}.txt'
        if not lp.exists(): return m
        boxes = []
        for ln in lp.read_text().splitlines():
            p = ln.split()
            if len(p) != 5: continue
            try:
                cid = int(float(p[0])); cx, cy, w, h = map(float, p[1:])
            except ValueError:
                continue
            seg = CONTIG_TO_SEG.get(cid, 0)
            if seg > 0:
                boxes.append((w*h, seg, cx, cy, w, h))
        for _, seg, cx, cy, w, h in sorted(boxes, reverse=True):
            x1 = max(0, int((cx - w/2) * self.S))
            y1 = max(0, int((cy - h/2) * self.S))
            x2 = min(self.S, int((cx + w/2) * self.S))
            y2 = min(self.S, int((cy + h/2) * self.S))
            m[y1:y2, x1:x2] = seg
        return m
    def __getitem__(self, i):
        p = self.imgs[i]
        img = Image.open(p).convert('RGB').resize((self.S, self.S))
        arr = (np.asarray(img, np.float32)/255.0 - self.MEAN) / self.STD
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        y = torch.from_numpy(self._mask(p.stem))
        return x, y

seg_train = WeakSegDataset('train', SEG_IMGSZ)
seg_val   = WeakSegDataset('val',   SEG_IMGSZ)
log.info(f'SegFormer dataset: {len(seg_train)} train / {len(seg_val)} val')""")

code("""# 7.2 - SegFormer training (resume-aware, checkpoints every epoch to Drive)
import time
from transformers import SegformerForSemanticSegmentation

SEG_CKPT = f'{SEG_DIR}/segformer_last.pt'
SEG_BEST = f'{SEG_DIR}/segformer_best.pt'

train_loader = DataLoader(seg_train, batch_size=SEG_BATCH, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(seg_val, batch_size=SEG_BATCH, shuffle=False,
                          num_workers=2, pin_memory=True)

seg_model = SegformerForSemanticSegmentation.from_pretrained(
    SEG_BACKBONE, num_labels=NUM_SEG_CLASSES,
    ignore_mismatched_sizes=True).to('cuda')
opt = torch.optim.AdamW(seg_model.parameters(), lr=SEG_LR, weight_decay=1e-4)
scaler = torch.amp.GradScaler('cuda')

start_epoch, best_miou = 0, 0.0
if os.path.exists(SEG_CKPT):
    ck = torch.load(SEG_CKPT, map_location='cuda')
    seg_model.load_state_dict(ck['model'])
    opt.load_state_dict(ck['opt'])
    start_epoch = ck['epoch'] + 1
    best_miou = ck.get('best_miou', 0.0)
    log.info(f'Resuming SegFormer @ epoch {start_epoch}  best_mIoU={best_miou:.4f}')

def seg_evaluate():
    seg_model.eval()
    cm = np.zeros((NUM_SEG_CLASSES, NUM_SEG_CLASSES), np.int64)
    with torch.no_grad():
        for x, y in val_loader:
            logits = seg_model(pixel_values=x.to('cuda')).logits
            logits = torch.nn.functional.interpolate(
                logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
            pred = logits.argmax(1).cpu().numpy()
            tgt = y.numpy()
            for p_, t_ in zip(pred, tgt):
                k = (t_ >= 0) & (t_ < NUM_SEG_CLASSES)
                cm += np.bincount(NUM_SEG_CLASSES * t_[k] + p_[k],
                                  minlength=NUM_SEG_CLASSES**2
                                  ).reshape(NUM_SEG_CLASSES, NUM_SEG_CLASSES)
    acc = float(np.diag(cm).sum() / max(cm.sum(), 1))
    denom = cm.sum(1) + cm.sum(0) - np.diag(cm)
    iou = np.diag(cm) / np.maximum(denom, 1)
    present = cm.sum(1) > 0
    miou = float(iou[present].mean() if present.any() else 0.0)
    return acc, miou

if start_epoch >= SEG_EPOCHS:
    log.info(f'SegFormer already trained {start_epoch} epochs - skipping training.')
else:
    for epoch in range(start_epoch, SEG_EPOCHS):
        seg_model.train()
        t0, tot = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to('cuda'), y.to('cuda')
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                out = seg_model(pixel_values=x, labels=y)
            scaler.scale(out.loss).backward()
            scaler.step(opt); scaler.update()
            tot += out.loss.item()
        msg = (f'epoch {epoch+1}/{SEG_EPOCHS}  loss={tot/len(train_loader):.4f}'
               f'  {time.time()-t0:.0f}s')
        torch.save({'epoch': epoch, 'model': seg_model.state_dict(),
                    'opt': opt.state_dict(), 'best_miou': best_miou}, SEG_CKPT)
        if (epoch + 1) % SEG_VAL_EVERY == 0 or (epoch + 1) == SEG_EPOCHS:
            acc, miou = seg_evaluate()
            msg += f'  pixAcc={acc:.4f}  mIoU={miou:.4f}'
            if miou > best_miou:
                best_miou = miou
                torch.save({'epoch': epoch, 'model': seg_model.state_dict(),
                            'best_miou': best_miou}, SEG_BEST)
                msg += '  <-- best'
        log.info(msg)
    log.info(f'SegFormer training done. Best mIoU: {best_miou:.4f}')""")

# ======================================================================
# 8 - Evaluation / metrics export
# ======================================================================
md("## 8 - Evaluation/metrics export")

code("""# 8.1 - SegFormer evaluation + visualisation
import matplotlib.pyplot as plt
if os.path.exists(SEG_BEST):
    seg_model.load_state_dict(torch.load(SEG_BEST, map_location='cuda')['model'])
seg_acc, seg_miou = seg_evaluate()
log.info(f'SegFormer  pixel-accuracy={seg_acc:.4f}  mIoU={seg_miou:.4f}')

seg_model.eval()
n_show = min(3, len(seg_val))
if n_show:
    fig, axes = plt.subplots(n_show, 3, figsize=(13, 4*n_show))
    if n_show == 1:
        axes = np.array([axes])
    step = max(1, len(seg_val) // (n_show + 1))
    for row in range(n_show):
        x, y = seg_val[row * step]
        with torch.no_grad():
            lo = seg_model(pixel_values=x.unsqueeze(0).to('cuda')).logits
            lo = torch.nn.functional.interpolate(lo, size=y.shape[-2:],
                                                 mode='bilinear', align_corners=False)
            pr = lo.argmax(1)[0].cpu().numpy()
        img = (x.permute(1, 2, 0).numpy() * WeakSegDataset.STD
               + WeakSegDataset.MEAN).clip(0, 1)
        axes[row,0].imshow(img);                                axes[row,0].set_title('image')
        axes[row,1].imshow(y.numpy(), vmin=0, vmax=9, cmap='tab10'); axes[row,1].set_title('pseudo-mask (GT)')
        axes[row,2].imshow(pr,         vmin=0, vmax=9, cmap='tab10'); axes[row,2].set_title('SegFormer prediction')
        for c in range(3): axes[row,c].axis('off')
    plt.tight_layout()
    seg_viz = f'{RESULTS_DIR}/segformer_predictions.png'
    plt.savefig(seg_viz, dpi=110, bbox_inches='tight'); plt.show()
    log.info(f'Saved {seg_viz}')""")

code("""# 8.2 - Consolidated metrics JSON (one file per model + an overall summary)
seg_metrics = {
    'backbone': SEG_BACKBONE, 'epochs_config': SEG_EPOCHS,
    'imgsz': SEG_IMGSZ, 'batch': SEG_BATCH,
    'pixel_accuracy': float(seg_acc),
    'mIoU': float(seg_miou),
    'best_mIoU': float(best_miou),
}
with open(f'{RESULTS_DIR}/segformer_metrics.json', 'w') as f:
    json.dump(seg_metrics, f, indent=2)

experiment_summary = {
    'yolov12':   yolo_metrics,
    'segformer': seg_metrics,
    'dataset':   stats,
    'verification': per_split,
}
with open(f'{RESULTS_DIR}/experiment_summary.json', 'w') as f:
    json.dump(experiment_summary, f, indent=2)
log.info('=== EXPERIMENT SUMMARY ===\\n' + json.dumps(experiment_summary, indent=2))""")

# ======================================================================
# 9 - Checkpoints/results saved to Drive
# ======================================================================
md("## 9 - Checkpoints/results saved to Drive")

code("""# 9.1 - Print the exact Drive locations of every produced artifact.
print('=' * 78)
print(' SkyLogic Sample-9k - Drive artifacts')
print('=' * 78)
print()
print('Sampled dataset:')
print(f'  {DRIVE_SAMPLE_DIR}/')
print(f'    images/{{train,val,test}}/')
print(f'    labels/{{train,val,test}}/')
print(f'    data.yaml, class_map.json, seg_class_map.json, dataset_stats.json')
print()
print('YOLOv12 checkpoints:')
print(f'  {YOLO_PROJECT_DIR}/{YOLO_RUN_NAME}/weights/best.pt')
print(f'  {YOLO_PROJECT_DIR}/{YOLO_RUN_NAME}/weights/last.pt')
print(f'  (+ epoch{{N}}.pt every {{save_period=5}} epochs)')
print()
print('YOLOv12 metrics:')
print(f'  {RESULTS_DIR}/yolov12_metrics.json')
print(f'  {RESULTS_DIR}/yolov12_results.csv   (per-epoch losses + mAP)')
print(f'  {RESULTS_DIR}/yolov12_predictions.png')
print()
print('SegFormer checkpoints:')
print(f'  {SEG_DIR}/segformer_last.pt   (resumable, saved every epoch)')
print(f'  {SEG_DIR}/segformer_best.pt   (best mIoU)')
print()
print('SegFormer metrics:')
print(f'  {RESULTS_DIR}/segformer_metrics.json')
print(f'  {RESULTS_DIR}/segformer_predictions.png')
print()
print('Other:')
print(f'  {RESULTS_DIR}/experiment_summary.json')
print(f'  {RESULTS_DIR}/dataset_verification.json')
print(f'  {LOG_FILE}')
print('=' * 78)
log.info('Run finished.')""")


# ======================================================================
# Assemble + write the notebook
# ======================================================================
def main() -> int:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
    n_md = sum(1 for c in CELLS if c["cell_type"] == "markdown")
    print(f"Wrote {OUT.name}: {len(CELLS)} cells ({n_code} code, {n_md} markdown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
