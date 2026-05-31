#!/usr/bin/env python3
"""
build_colab_notebook.py - Generate SkyLogic_Sample9k_Colab.ipynb.

Produces a fully self-contained Colab notebook that:
  1.  Mounts Drive and reads the ORIGINAL full dataset (read-only) from Drive
  2.  Synthesises/validates YOLO labels and analyses the full class distribution
  3.  MERGES the 62 fine-grained xView classes into broader, balanced semantic
      groups and DROPS unknown/ambiguous classes (e.g. class_75, class_82)
  4.  Builds a CLASS-BALANCED ~9,000-image dataset (under-samples majority
      groups, drops too-rare groups) in a SEPARATE experiment folder on Drive
  5.  Prints a FINAL TRAINING DATASET SUMMARY, then trains YOLOv12 (resume-aware)
  6.  Reports REAL validation metrics only (never hardcodes accuracy)

The original Drive dataset is treated as strictly read-only - every output is
written into /content/drive/MyDrive/SkyLogic_balanced_yolov12_experiment.

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
md("""# SkyLogic - Balanced YOLOv12 Pipeline (class merging + balancing)

**Self-contained for Google Colab T4 GPU.**

This notebook reads the **original/full** SkyLogic dataset (folder or zip) from
your Google Drive, then:

1. Synthesises/validates YOLO labels and analyses the **full class distribution**.
2. **Merges** the 62 fine-grained xView classes into broader, balanced semantic
   groups (aircraft, small vehicle, truck/bus, rail, maritime, construction,
   building/facility, storage tank, shipping container, infrastructure) and
   **drops unknown/ambiguous classes** (e.g. `class_75`, `class_82`).
3. Builds a **class-balanced ~9,000-image dataset** (under-samples the majority
   groups, drops too-rare groups, remaps every label to the new group ids).
4. Prints a **FINAL TRAINING DATASET SUMMARY**, then trains **YOLOv12** and
   reports **real validation metrics only**.

> **Safety:** the original Drive dataset is treated as **strictly read-only**.
> Everything is written to a separate experiment folder on Drive.

**To run:**
1. `Runtime -> Change runtime type -> T4 GPU`.
2. Edit `DRIVE_DATASET_SOURCE` in the **Config** cell (Section 2) if your data
   is not at `/content/drive/MyDrive/data`.
3. `Runtime -> Run all`.

If Colab disconnects, just **Run all again** - the balanced dataset is reused
and YOLOv12 auto-resumes from its last checkpoint on Drive.""")

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
# 2 - Drive mount and config
# ======================================================================
md("## 2 - Drive mount and config")

code("""# 2.1 - Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')""")

code("""# 2.2 - CONFIG: edit the Drive locations + balancing/training knobs below.
import os, logging, sys
from pathlib import Path

# ---- Drive input (READ-ONLY) + outputs (EDIT IF NEEDED) -------------
DRIVE_DATASET_SOURCE = '/content/drive/MyDrive/data'   # original SkyLogic data (folder or .zip)
DRIVE_PROJECT_DIR    = '/content/drive/MyDrive/SkyLogic_balanced_yolov12_experiment'
BALANCED_DIR         = f'{DRIVE_PROJECT_DIR}/balanced_dataset_9k'   # all outputs land here

# ---- Class merging + balancing --------------------------------------
TARGET_TOTAL    = 9000          # approximate balanced image target
MIN_GROUP_IMAGES = 50           # drop a merged group rarer than this many images
FORCE_RESAMPLE  = False         # True rebuilds the balanced dataset from scratch
SAMPLE_SEED     = 42
SPLIT_RATIOS    = (0.80, 0.10, 0.10)   # train/val/test when re-splitting
# Balancing is image-level + rare-class-first with a per-group cap of
# ceil(TARGET_TOTAL / num_groups). Detection data is multi-label, so a few
# ubiquitous classes (buildings, cars) co-occur in most scenes and cannot be
# made truly rare without discarding most images.
#   True  -> after capping, top up to ~TARGET_TOTAL (keeps the most data; the
#            cap still under-samples the majority groups first).
#   False -> stop at the caps (a smaller, more strictly balanced dataset).
BALANCE_FILL_TO_TARGET = True

# ---- Verification gate ----------------------------------------------
ALLOW_INVALID_LABELS = False    # True proceeds even if verification finds errors

# ---- YOLOv12 detection (lightweight, tuned for T4) -------------------
YOLO_MODEL    = 'yolo12s.pt'    # 'yolo12n.pt' = even faster/smaller on T4
YOLO_EPOCHS   = 60              # early stopping (patience) usually stops sooner
YOLO_IMGSZ    = 512            # patches are 512x512; 640 is an option (slower)
YOLO_BATCH    = 16
YOLO_PATIENCE = 15
YOLO_RUN_NAME = 'balanced9k'

# ---- Derived paths --------------------------------------------------
LOCAL_SOURCE_DIR    = '/content/_source'              # zip extract cache
LOCAL_BALANCED_DIR  = '/content/balanced_dataset_9k'  # fast SSD mirror for training
YOLO_PROJECT_DIR    = f'{DRIVE_PROJECT_DIR}/yolov12'
RESULTS_DIR         = f'{DRIVE_PROJECT_DIR}/results'
LOG_FILE            = f'{DRIVE_PROJECT_DIR}/run.log'

for d in (DRIVE_PROJECT_DIR, BALANCED_DIR, YOLO_PROJECT_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# Logging: tee everything to a Drive file so it survives disconnects.
for h in list(logging.root.handlers):
    logging.root.removeHandler(h)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger('skylogic')
log.info('=== SkyLogic balanced-YOLOv12 run started ===')
log.info(f'DRIVE_DATASET_SOURCE = {DRIVE_DATASET_SOURCE}  (READ-ONLY)')
log.info(f'BALANCED_DIR         = {BALANCED_DIR}')
log.info(f'TARGET_TOTAL = {TARGET_TOTAL}  MIN_GROUP_IMAGES = {MIN_GROUP_IMAGES}  '
         f'FORCE_RESAMPLE = {FORCE_RESAMPLE}')""")

# ======================================================================
# 3 - Stage source + detect/synthesise YOLO labels
# ======================================================================
md("""## 3 - Stage source + detect/synthesise YOLO labels
- A source `.zip` is extracted once to `/content/_source` (cached); a source
  folder is read **in place** from Drive (never modified).
- The YOLO layout is auto-detected. If the dataset ships only metadata CSVs
  (xView style), YOLO labels are **synthesised** from them under
  `/content/_synth_labels` (the original Drive files are left untouched).""")

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

code("""# 3.2 - Auto-detect YOLO layout + (if needed) synthesise labels from metadata.
import yaml, ast

def detect_layout(root):
    \"\"\"Return {split: (image_dir, label_dir)} by probing common patterns.\"\"\"
    splits = {}
    for split in ('train', 'val', 'test'):
        candidates = [
            (root / 'images' / split,                       root / 'labels' / split),
            (root / 'sample_dataset_9k' / 'images' / split, root / 'sample_dataset_9k' / 'labels' / split),
            (root / split / 'images',                       root / split / 'labels'),
            (root / 'data' / 'patches' / split,             root / 'data' / 'yolo' / 'labels' / split),
            (root / 'data' / 'yolo' / 'images' / split,     root / 'data' / 'yolo' / 'labels' / split),
            (root / 'patches' / split,                      root / 'yolo' / 'labels' / split),
        ]
        for img_dir, lbl_dir in candidates:
            if img_dir.is_dir() and lbl_dir.is_dir():
                if next(img_dir.glob('*.png'), None) or next(img_dir.glob('*.jpg'), None):
                    splits[split] = (img_dir, lbl_dir); break
    if not splits:
        for img_dir, lbl_dir in [(root / 'images', root / 'labels'), (root, root)]:
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
    \"\"\"Build YOLO labels from data/metadata/{annotations,patches}_metadata.csv.\"\"\"
    import pandas as pd
    candidates = [
        (root / 'metadata',          root / 'patches'),
        (root / 'data' / 'metadata', root / 'data' / 'patches'),
    ]
    meta_dir = patches_root = None
    for m, p in candidates:
        if (m / 'annotations_metadata.csv').exists() \\
                and (m / 'patches_metadata.csv').exists() and p.is_dir():
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

    id_name = ann[['class_id','class_name']].drop_duplicates('class_id') \\
                  .sort_values('class_id')
    sparse_ids = [int(c) for c in id_name['class_id']]
    sparse_to_contig = {sid: i for i, sid in enumerate(sparse_ids)}
    names_list = [str(n) for n in id_name['class_name'].tolist()]
    log.info(f'  {len(names_list)} classes derived from CSV: '
             f'sparse {min(sparse_ids)}..{max(sparse_ids)} -> 0..{len(names_list)-1}')

    if marker.exists() and not FORCE_RESAMPLE:
        log.info(f'  labels already synthesised at {out_dir} - reusing')
        return out_dir, sparse_to_contig, names_list, patches_root

    log.info(f'  writing synthesised labels under {out_dir} (one-time cost)')
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    for split in ('train', 'val', 'test'):
        (out_dir / split).mkdir()

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
    log.info(f'  synthesised labels per split: {written}  (no size-metadata: {no_meta})')
    return out_dir, sparse_to_contig, names_list, patches_root

# ---- Run detection ---------------------------------------------------
source_splits = detect_layout(SOURCE_ROOT)
names         = find_names(SOURCE_ROOT)
class_map     = find_class_map(SOURCE_ROOT)

if not source_splits:
    synth_dir, synth_s2c, synth_names, patches_root = \\
        synthesize_yolo_labels_from_metadata(SOURCE_ROOT)
    if synth_dir is not None:
        for split in ('train', 'val', 'test'):
            img_dir = patches_root / split
            lbl_dir = synth_dir / split
            if img_dir.is_dir() and lbl_dir.is_dir() and next(img_dir.glob('*.png'), None):
                source_splits[split] = (img_dir, lbl_dir)
        if names is None:
            names = synth_names
        if class_map is None and synth_s2c:
            class_map = {'sparse_to_contiguous': {str(k): v for k, v in synth_s2c.items()}}
        log.info(f'After synthesis - splits: {list(source_splits)}')

log.info(f'Detected splits: {list(source_splits)}')
for s, (i, l) in source_splits.items():
    log.info(f'  {s}: images={i}  labels={l}')
log.info(f'Class names: {"found ("+str(len(names))+")" if names else "not found - will infer"}')

assert source_splits, ('No image/label layout detected in source - and no '
                       'metadata CSVs found to synthesise labels from. '
                       'Check DRIVE_DATASET_SOURCE.')""")

code("""# 3.3 - DEBUG report: source state + detected layout (before any analysis).
print('=' * 78)
print(' DEBUG: dataset source + detected layout')
print('=' * 78)
print(f'DRIVE_DATASET_SOURCE   = {DRIVE_DATASET_SOURCE}')
print(f'exists                 = {os.path.exists(DRIVE_DATASET_SOURCE)}')
print(f'SOURCE_ROOT (staged)   = {SOURCE_ROOT}')
print()
print('Top-level entries inside SOURCE_ROOT:')
try:
    for item in sorted(Path(SOURCE_ROOT).iterdir()):
        kind = 'D' if item.is_dir() else 'F'
        print(f'  [{kind}] {item.name}')
except Exception as e:
    print(f'  (could not list: {e})')
print()
print(f'Detected splits: {list(source_splits)}')
dbg_imgs = dbg_lbls = 0
for split, (img_dir, lbl_dir) in source_splits.items():
    try: n_img = sum(1 for p in img_dir.iterdir() if p.is_file())
    except Exception: n_img = -1
    try: n_lbl = sum(1 for _ in lbl_dir.glob('*.txt'))
    except Exception: n_lbl = -1
    dbg_imgs += max(0, n_img); dbg_lbls += max(0, n_lbl)
    print(f'  {split}:  images_dir={img_dir} ({n_img})  labels_dir={lbl_dir} ({n_lbl})')
print(f'TOTAL images detected : {dbg_imgs}')
print(f'TOTAL labels detected : {dbg_lbls}')
print(f'Raw class count       : {len(names) if names else 0}')
print('=' * 78)
log.info(f'DEBUG layout - splits={list(source_splits)} imgs={dbg_imgs} '
         f'lbls={dbg_lbls} raw_classes={len(names) if names else 0}')""")

# ======================================================================
# 4 - Full class analysis (validate every label)
# ======================================================================
md("""## 4 - Full class analysis (validate every label)
Every label file is parsed line-by-line. A line is **valid** only if it has 5
whitespace-separated fields `class x_center y_center width height`, an integer
class id in range, and normalised coordinates in `[0, 1]` with positive size.
Invalid lines are **dropped and counted**; images left with no valid label are
excluded. The full **original** class distribution is saved to Drive.""")

code("""# 4.1 - Parse + validate all labels; build the ORIGINAL class distribution.
import re, math, csv
from collections import defaultdict, Counter

if not names:
    log.warning('No class names found - inferring class count from labels.')
    mx = -1
    for split, (img_dir, lbl_dir) in source_splits.items():
        for lp in lbl_dir.glob('*.txt'):
            for ln in lp.read_text().splitlines():
                ps = ln.split()
                if ps:
                    try: mx = max(mx, int(float(ps[0])))
                    except ValueError: pass
    names = [f'class_{i}' for i in range(max(mx + 1, 1))]
NUM_RAW_CLASSES = len(names)

def _find_ext(img_dir, stem):
    for cand in ('.png', '.jpg', '.jpeg'):
        if (img_dir / (stem + cand)).exists():
            return cand
    return None

def parse_yolo_line(ln, num_classes):
    p = ln.split()
    if len(p) != 5:
        return None, 'bad_format'
    try:
        cf = float(p[0])
        if cf != int(cf):
            return None, 'bad_class_id'
        cid = int(cf)
        coords = [float(x) for x in p[1:]]
    except ValueError:
        return None, 'bad_number'
    if cid < 0 or cid >= num_classes:
        return None, 'bad_class_id'
    cx, cy, w, h = coords
    if any(not (0.0 <= v <= 1.0) for v in coords) or w <= 0 or h <= 0:
        return None, 'bad_bbox'
    return (cid, cx, cy, w, h), None

item_lines      = {}        # (split, stem, ext) -> [(cid, cx, cy, w, h), ...]
orig_img_count  = Counter() # cid -> images containing it
orig_lbl_count  = Counter() # cid -> total boxes
invalid_counter = Counter() # reason -> count
n_items = n_empty = n_no_image = 0

for split, (img_dir, lbl_dir) in source_splits.items():
    for lp in lbl_dir.glob('*.txt'):
        stem = lp.stem
        ext  = _find_ext(img_dir, stem)
        if ext is None:
            n_no_image += 1; continue
        valid, present = [], set()
        for ln in lp.read_text().splitlines():
            if not ln.strip():
                continue
            parsed, reason = parse_yolo_line(ln, NUM_RAW_CLASSES)
            if parsed is None:
                invalid_counter[reason] += 1; continue
            valid.append(parsed); present.add(parsed[0])
            orig_lbl_count[parsed[0]] += 1
        if not valid:
            n_empty += 1; continue
        item_lines[(split, stem, ext)] = valid
        for c in present:
            orig_img_count[c] += 1
        n_items += 1

log.info(f'Analysed {n_items} annotated images  '
         f'(empty/all-invalid: {n_empty}, label-without-image: {n_no_image})')
log.info(f'Invalid label lines dropped: {dict(invalid_counter)}  '
         f'total={sum(invalid_counter.values())}')

original_distribution = {}
for c in range(NUM_RAW_CLASSES):
    nm = names[c] if c < len(names) else f'class_{c}'
    original_distribution[nm] = {'class_id': c,
                                 'images': int(orig_img_count.get(c, 0)),
                                 'labels': int(orig_lbl_count.get(c, 0))}
Path(RESULTS_DIR, 'original_class_distribution.json').write_text(
    json.dumps(original_distribution, indent=2))
log.info(f'Wrote original_class_distribution.json ({NUM_RAW_CLASSES} raw classes)')

# Quick on-screen peek at the most/least common raw classes.
ranked = sorted(range(NUM_RAW_CLASSES), key=lambda c: orig_img_count.get(c, 0))
print('Rarest 8 raw classes :',
      [(names[c], orig_img_count.get(c, 0)) for c in ranked[:8]])
print('Top 8 raw classes    :',
      [(names[c], orig_img_count.get(c, 0)) for c in ranked[-8:][::-1]])""")

# ======================================================================
# 5 - Merge classes into balanced semantic groups
# ======================================================================
md("""## 5 - Merge classes into balanced semantic groups
The 62 fine-grained xView classes are merged into broader groups. Any class not
covered by the mapping - including the placeholder/unknown classes `class_75`
and `class_82` - is **dropped** (never trained on). Groups that end up rarer
than `MIN_GROUP_IMAGES` are also dropped so the remaining classes can be
balanced. The mapping and both distributions are saved to Drive.""")

code("""# 5.1 - Explicit class-group mapping (edit here to regroup).
# Keys are the final merged group names; values are the raw xView class names
# that fold into them. Names are matched case-insensitively and '/'-insensitively
# so minor spelling/spacing differences still resolve.
CLASS_GROUPS = {
    'Aircraft': [
        'Fixed-wing Aircraft', 'Small Aircraft', 'Cargo Plane', 'Helicopter'],
    'Small Vehicle': [
        'Passenger Vehicle', 'Small Car', 'Pickup Truck'],
    'Truck / Bus': [
        'Bus', 'Utility Truck', 'Truck', 'Cargo Truck', 'Truck w/Box',
        'Truck Tractor', 'Trailer', 'Truck w/Flatbed', 'Truck w/Liquid',
        'Crane Truck', 'Dump Truck', 'Haul Truck'],
    'Rail Vehicle': [
        'Railway Vehicle', 'Passenger Car', 'Cargo Car', 'Flat Car',
        'Tank car', 'Locomotive'],
    'Maritime Vessel': [
        'Maritime Vessel', 'Motorboat', 'Sailboat', 'Tugboat', 'Barge',
        'Fishing Vessel', 'Ferry', 'Yacht', 'Container Ship', 'Oil Tanker'],
    'Construction Equipment': [
        'Engineering Vehicle', 'Tower crane', 'Container Crane', 'Reach Stacker',
        'Straddle Carrier', 'Mobile Crane', 'Scraper/Tractor',
        'Front loader/Bulldozer', 'Excavator', 'Cement Mixer', 'Ground Grader'],
    'Building / Facility': [
        'Hut/Tent', 'Shed', 'Building', 'Aircraft Hangar', 'Damaged Building',
        'Facility', 'Construction Site', 'Vehicle Lot', 'Helipad'],
    'Storage Tank': [
        'Storage Tank'],
    'Shipping Container': [
        'Shipping container lot', 'Shipping Container'],
    'Infrastructure': [
        'Pylon', 'Tower'],
}

def _norm(s):
    return ' '.join(str(s).lower().replace('/', ' ').split())

name_to_group = {}
for g, members in CLASS_GROUPS.items():
    for m in members:
        name_to_group[_norm(m)] = g

UNKNOWN_RE = re.compile(r'^class_\\d+$', re.IGNORECASE)

contig_to_group_name = {}
unknown_classes = []
for c in range(NUM_RAW_CLASSES):
    nm = (names[c] if c < len(names) else f'class_{c}').strip()
    if UNKNOWN_RE.match(nm):
        contig_to_group_name[c] = None; unknown_classes.append(nm); continue
    g = name_to_group.get(_norm(nm))
    contig_to_group_name[c] = g
    if g is None:
        unknown_classes.append(nm)

# Merged distribution BEFORE balancing.
group_img_count = Counter()
group_lbl_count = Counter()
for key, lines in item_lines.items():
    gpresent = set()
    for (cid, *_rest) in lines:
        g = contig_to_group_name.get(cid)
        if g is not None:
            gpresent.add(g); group_lbl_count[g] += 1
    for g in gpresent:
        group_img_count[g] += 1

# Keep groups with enough images (preserve CLASS_GROUPS order); drop the rest.
kept_group_names   = [g for g in CLASS_GROUPS if group_img_count.get(g, 0) >= MIN_GROUP_IMAGES]
dropped_rare_groups = {g: int(group_img_count.get(g, 0)) for g in CLASS_GROUPS
                       if 0 < group_img_count.get(g, 0) < MIN_GROUP_IMAGES}
group_to_final_id  = {g: i for i, g in enumerate(kept_group_names)}
contig_to_final_gid = {c: group_to_final_id.get(g)
                       for c, g in contig_to_group_name.items()}
dropped_rare_classes = [names[c] for c in range(NUM_RAW_CLASSES)
                        if contig_to_group_name.get(c) in dropped_rare_groups]

log.info('=== CLASS MERGE MAP ===')
for g in CLASS_GROUPS:
    fid = group_to_final_id.get(g, '-')
    status = ('KEEP' if g in group_to_final_id
              else ('DROP(rare)' if g in dropped_rare_groups else 'DROP(absent)'))
    log.info(f'  [{str(fid):>2}] {g:24s} imgs={group_img_count.get(g,0):5d} '
             f'lbls={group_lbl_count.get(g,0):6d}  {status}')
log.info(f'Unknown/ambiguous classes dropped: {unknown_classes}')
log.info(f'Kept groups ({len(kept_group_names)}): {kept_group_names}')

class_group_mapping = {
    'class_groups': CLASS_GROUPS,
    'kept_groups': kept_group_names,
    'group_to_final_id': group_to_final_id,
    'dropped_unknown_classes': unknown_classes,
    'dropped_rare_groups': dropped_rare_groups,
    'dropped_rare_classes': dropped_rare_classes,
    'min_group_images': MIN_GROUP_IMAGES,
    'raw_class_names': list(names),
}
Path(RESULTS_DIR, 'class_group_mapping.json').write_text(
    json.dumps(class_group_mapping, indent=2))
log.info('Wrote class_group_mapping.json')

assert kept_group_names, ('No class groups survived the MIN_GROUP_IMAGES filter - '
                          'lower MIN_GROUP_IMAGES in the Config cell.')""")

# ======================================================================
# 6 - Build the class-balanced ~9k dataset
# ======================================================================
md("""## 6 - Build the class-balanced ~9k dataset
Every image is remapped to the final group ids (labels for dropped classes are
discarded; images left with no label are excluded). Images are then selected
**rare-group-first** with a per-group cap so no single group dominates, then
topped up toward `TARGET_TOTAL`. All labels are rewritten with the new group ids
into a **separate** Drive folder - the original dataset is never touched.

> **Honest note on balance.** Object detection is multi-label: one image can
> contain several groups, and a few classes (buildings, small vehicles) appear
> in most satellite scenes. So perfect per-class balance is impossible without
> throwing away most images. This pipeline does **image-level** balancing -
> rare-class images are all kept and majority-only images are under-sampled by
> the per-group cap - and reports the **real** resulting distribution and
> imbalance ratio rather than pretending the classes are perfectly even.""")

code("""# 6.1 - Remap to final group ids; drop images with no kept label.
item_groups        = {}   # key -> set(final gid)
item_primary       = {}   # key -> rarest present gid
item_source_split  = {}   # key -> source split name
group_freq         = Counter()
n_drop_unmapped    = 0
for key, lines in item_lines.items():
    gids = set()
    for (cid, *_rest) in lines:
        fg = contig_to_final_gid.get(cid)
        if fg is not None:
            gids.add(fg)
    if not gids:
        n_drop_unmapped += 1; continue
    item_groups[key] = gids
    item_source_split[key] = key[0]
    for g in gids:
        group_freq[g] += 1
for key, gids in item_groups.items():
    item_primary[key] = min(gids, key=lambda g: group_freq[g])
log.info(f'Images with >=1 kept label: {len(item_groups)}  '
         f'(dropped, no kept label: {n_drop_unmapped})')""")

code("""# 6.2 - Class-balanced selection (rare-first; per-group cap under-samples majority).
# Each image is bucketed by its RAREST present group, so rare groups are never
# capped away. Every bucket is capped at ceil(TARGET_TOTAL / num_groups); if
# BALANCE_FILL_TO_TARGET we then top up toward TARGET_TOTAL from the groups that
# still have spare images (otherwise we stop at the caps for a stricter balance).
import random
def build_balanced_selection(item_primary, n_groups, target_total, fill, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for key, primary in item_primary.items():
        buckets[primary].append(key)
    for g in buckets:
        rng.shuffle(buckets[g])
    cap = max(1, math.ceil(target_total / max(1, n_groups)))
    selected, cursor = [], {}
    for g, items in buckets.items():
        n = min(len(items), cap)
        selected.extend(items[:n]); cursor[g] = n
    if fill:
        progressing = True
        while len(selected) < target_total and progressing:
            progressing = False
            for g, items in buckets.items():
                if cursor[g] < len(items):
                    selected.append(items[cursor[g]]); cursor[g] += 1; progressing = True
                    if len(selected) >= target_total:
                        break
    return selected[:target_total], cap

K = len(kept_group_names)
selected, per_group_cap = build_balanced_selection(
    item_primary, K, TARGET_TOTAL, BALANCE_FILL_TO_TARGET, SAMPLE_SEED)
selected_set = set(selected)
n_drop_not_selected = len(item_groups) - len(selected)
log.info(f'Balanced selection: {len(selected)} images  '
         f'(target ~{TARGET_TOTAL}, {K} groups, cap/grp {per_group_cap}, '
         f'fill_to_target={BALANCE_FILL_TO_TARGET}, not selected: {n_drop_not_selected})')""")

code("""# 6.3 - Split selected images (preserve source splits if usable, else 80/10/10).
present_splits  = [s for s in ('train','val','test')
                   if sum(1 for k in selected if k[0] == s) > 50]
PRESERVE_SPLITS = len(present_splits) >= 2

def assign_splits(selected, preserve, ratios, seed):
    out = {'train': [], 'val': [], 'test': []}
    if preserve:
        for key in selected:
            s = key[0] if key[0] in out else 'train'
            out[s].append(key)
        return out
    rng = random.Random(seed + 1)
    by_primary = defaultdict(list)
    for key in selected:
        by_primary[item_primary[key]].append(key)
    for g, items in by_primary.items():
        rng.shuffle(items)
        n = len(items); ntr = int(n * ratios[0]); nval = int(n * ratios[1])
        out['train'] += items[:ntr]
        out['val']   += items[ntr:ntr + nval]
        out['test']  += items[ntr + nval:]
    return out

split_assignment = assign_splits(selected, PRESERVE_SPLITS, SPLIT_RATIOS, SAMPLE_SEED)
for s in ('train','val','test'):
    log.info(f'  split {s}: {len(split_assignment[s])} images')
log.info(f'Split mode: {"preserve source" if PRESERVE_SPLITS else "fresh stratified 80/10/10"}')

# Final balanced per-group distribution (images + labels).
balanced_img_count = Counter()
balanced_lbl_count = Counter()
for key in selected:
    gids = set()
    for (cid, *_rest) in item_lines[key]:
        fg = contig_to_final_gid.get(cid)
        if fg is not None:
            balanced_lbl_count[fg] += 1; gids.add(fg)
    for g in gids:
        balanced_img_count[g] += 1""")

code("""# 6.4 - Write the balanced dataset to Drive (idempotent; remapped labels).
def mirror_to_local(drive_root, local_root, log_every=1000):
    drive_root = Path(drive_root); local_root = Path(local_root)
    if local_root.exists() and (local_root/'data.yaml').exists() and not FORCE_RESAMPLE:
        ok = True
        for split in ('train','val','test'):
            d = drive_root/'images'/split; l = local_root/'images'/split
            if d.exists() and not l.exists(): ok = False; break
            if d.exists() and l.exists() and len(list(d.iterdir())) != len(list(l.iterdir())):
                ok = False; break
        if ok:
            log.info(f'Local mirror already valid at {local_root} - skipping.'); return
    if local_root.exists() and FORCE_RESAMPLE:
        shutil.rmtree(local_root, ignore_errors=True)
    local_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in drive_root.rglob('*'):
        if src.is_dir(): continue
        dst = local_root / src.relative_to(drive_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == src.stat().st_size: continue
        shutil.copy2(src, dst); n += 1
        if n % log_every == 0: log.info(f'    mirrored {n} files ...')
    log.info(f'Local mirror complete: {n} new files -> {local_root}')

def balanced_exists_valid(drive_dir, split_assignment):
    base = Path(drive_dir)
    if FORCE_RESAMPLE or not base.exists() or not (base/'data.yaml').exists():
        return False
    rng = random.Random(99)
    for split, items in split_assignment.items():
        if not items: continue
        img_dir = base/'images'/split; lbl_dir = base/'labels'/split
        if not img_dir.is_dir() or not lbl_dir.is_dir(): return False
        for (s, stem, ext) in rng.sample(items, min(40, len(items))):
            if not (img_dir/f'{stem}{ext}').exists(): return False
            if not (lbl_dir/f'{stem}.txt').exists():  return False
    return True

def write_remapped_label(dst_lbl, key):
    out = []
    for (cid, cx, cy, w, h) in item_lines[key]:
        fg = contig_to_final_gid.get(cid)
        if fg is None: continue
        out.append(f'{fg} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    dst_lbl.write_text('\\n'.join(out) + '\\n')

def build_balanced_dataset(drive_dir, split_assignment, source_splits, log_every=500):
    base = Path(drive_dir); copied = skipped = failed = 0
    for split, items in split_assignment.items():
        img_out = base/'images'/split; lbl_out = base/'labels'/split
        img_out.mkdir(parents=True, exist_ok=True); lbl_out.mkdir(parents=True, exist_ok=True)
        for i, key in enumerate(items, 1):
            s, stem, ext = key
            src_img = source_splits[s][0] / f'{stem}{ext}'
            dst_img = img_out / f'{stem}{ext}'; dst_lbl = lbl_out / f'{stem}.txt'
            try:
                if not dst_img.exists() or dst_img.stat().st_size != src_img.stat().st_size:
                    shutil.copy2(src_img, dst_img); copied += 1
                else:
                    skipped += 1
                write_remapped_label(dst_lbl, key)
            except Exception as e:
                failed += 1
                if failed <= 3: log.warning(f'  copy failed {stem}: {e}')
            if i % log_every == 0:
                log.info(f'    {split}: {i}/{len(items)} copied={copied} reused={skipped}')
        log.info(f'  {split} done: {len(items)} pairs (copied={copied} reused={skipped})')
    log.info(f'Copy summary: copied={copied} reused={skipped} failed={failed}')

BALANCED_REUSED = balanced_exists_valid(BALANCED_DIR, split_assignment)
if BALANCED_REUSED:
    log.info('Existing balanced dataset on Drive is valid - skipping copy.')
else:
    log.info(f'Writing balanced dataset to {BALANCED_DIR} '
             '(Drive FUSE writes are slow; this may take a few minutes)...')
    build_balanced_dataset(BALANCED_DIR, split_assignment, source_splits)""")

code("""# 6.5 - data.yaml + class_group_mapping.json + balance report (JSON + CSV).
NUM_CLASSES        = K
group_names_final  = list(kept_group_names)   # list index == final group id

names_block = '\\n'.join(f'  {i}: {n}' for i, n in enumerate(group_names_final))
data_yaml_text = (
    '# Auto-generated balanced dataset (YOLOv12) - merged + class-balanced.\\n'
    f'path: {BALANCED_DIR}\\n'
    'train: images/train\\n'
    'val: images/val\\n'
    'test: images/test\\n'
    f'nc: {NUM_CLASSES}\\n'
    'names:\\n' + names_block + '\\n'
)
Path(BALANCED_DIR, 'data.yaml').write_text(data_yaml_text)
Path(BALANCED_DIR, 'class_group_mapping.json').write_text(
    json.dumps(class_group_mapping, indent=2))

balance_report = {
    'target_total': TARGET_TOTAL,
    'min_group_images': MIN_GROUP_IMAGES,
    'seed': SAMPLE_SEED,
    'split_mode': 'preserve' if PRESERVE_SPLITS else 'fresh_stratified_80_10_10',
    'raw_num_classes': NUM_RAW_CLASSES,
    'final_num_classes': NUM_CLASSES,
    'final_group_names': group_names_final,
    'splits': {s: len(v) for s, v in split_assignment.items()},
    'final_total_images': len(selected),
    'original_distribution': original_distribution,
    'class_group_mapping': {g: CLASS_GROUPS[g] for g in CLASS_GROUPS},
    'merged_distribution_before_balance': {
        g: {'images': int(group_img_count.get(g, 0)),
            'labels': int(group_lbl_count.get(g, 0))} for g in CLASS_GROUPS},
    'balanced_distribution': {
        group_names_final[g]: {'images': int(balanced_img_count.get(g, 0)),
                               'labels': int(balanced_lbl_count.get(g, 0))}
        for g in range(NUM_CLASSES)},
    'per_group_cap': int(per_group_cap),
    'balance_fill_to_target': bool(BALANCE_FILL_TO_TARGET),
    'imbalance_ratio_images': round(
        max(balanced_img_count.values()) / max(1, min(balanced_img_count.get(g, 0)
            for g in range(NUM_CLASSES))), 2) if balanced_img_count else None,
    'dropped_unknown_classes': unknown_classes,
    'dropped_rare_groups': dropped_rare_groups,
    'dropped_rare_classes': dropped_rare_classes,
    'dropped_invalid_lines': int(sum(invalid_counter.values())),
    'dropped_invalid_breakdown': dict(invalid_counter),
    'dropped_images_no_kept_label': int(n_drop_unmapped),
    'dropped_images_not_selected': int(n_drop_not_selected),
    'images_without_matching_file': int(n_no_image),
    'empty_or_all_invalid_images': int(n_empty),
}
Path(BALANCED_DIR, 'dataset_balance_report.json').write_text(json.dumps(balance_report, indent=2))
Path(RESULTS_DIR, 'dataset_balance_report.json').write_text(json.dumps(balance_report, indent=2))

csv_path = Path(BALANCED_DIR, 'dataset_balance_report.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['original_id','original_name','group_name','final_group_id',
                'images_with_class','labels_of_class','status'])
    for c in range(NUM_RAW_CLASSES):
        nm = names[c] if c < len(names) else f'class_{c}'
        g  = contig_to_group_name.get(c)
        fid = group_to_final_id.get(g, '')
        status = ('dropped_unknown' if g is None
                  else 'dropped_rare_group' if g in dropped_rare_groups else 'kept')
        w.writerow([c, nm, g if g else '', fid,
                    int(orig_img_count.get(c, 0)), int(orig_lbl_count.get(c, 0)), status])
shutil.copy2(csv_path, Path(RESULTS_DIR, 'dataset_balance_report.csv'))
log.info(f'Wrote data.yaml (nc={NUM_CLASSES}) + balance report JSON/CSV to {BALANCED_DIR}')""")

code("""# 6.6 - Mirror balanced dataset to local SSD; rewrite data.yaml path for speed.
mirror_to_local(BALANCED_DIR, LOCAL_BALANCED_DIR)
local_yaml = Path(LOCAL_BALANCED_DIR) / 'data.yaml'
cfg = yaml.safe_load(local_yaml.read_text())
cfg['path'] = LOCAL_BALANCED_DIR
local_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
DATA_YAML = str(local_yaml)
names = group_names_final            # downstream verify/plots use final group names
log.info(f'Local data.yaml ready: {DATA_YAML}  (nc={NUM_CLASSES}, path={LOCAL_BALANCED_DIR})')""")

# ======================================================================
# 7 - Verify the balanced dataset
# ======================================================================
md("""## 7 - Verify the balanced dataset
A final hard check on the **balanced** dataset: image<->label matching, 5-field
lines, class ids in `[0, NUM_CLASSES)`, and bbox coords in `[0, 1]`. The labels
were generated by this notebook, so this should pass cleanly; it is a safety
net. Set `ALLOW_INVALID_LABELS = True` to proceed despite errors.""")

code("""# 7.1 - Strong verification of the balanced dataset.
class VerificationError(RuntimeError): pass

def verify_dataset(local_root, num_classes):
    base = Path(local_root); per_split = {}; class_counts = {}
    issues = {'unmatched_images':[], 'unmatched_labels':[],
              'bad_format':[], 'bad_class_id':[], 'bad_bbox':[]}
    for split in ('train', 'val', 'test'):
        img_dir = base/'images'/split; lbl_dir = base/'labels'/split
        if not img_dir.exists() and not lbl_dir.exists():
            per_split[split] = {'images':0,'labels':0}; continue
        img_map = {p.stem for p in img_dir.glob('*')
                   if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg')}
        lbl_set = {p.stem for p in lbl_dir.glob('*.txt')}
        only_img, only_lbl = img_map - lbl_set, lbl_set - img_map
        for s in sorted(only_img)[:10]: issues['unmatched_images'].append(f'{split}/{s}')
        for s in sorted(only_lbl)[:10]: issues['unmatched_labels'].append(f'{split}/{s}')
        empty = bad_fmt = bad_cls = bad_box = 0
        for stem in img_map:
            lp = lbl_dir / f'{stem}.txt'
            if not lp.exists(): continue
            content = lp.read_text().strip()
            if not content: empty += 1; continue
            for li, ln in enumerate(content.splitlines(), 1):
                p = ln.split()
                if len(p) != 5:
                    bad_fmt += 1
                    if len(issues['bad_format']) < 10:
                        issues['bad_format'].append(f'{split}/{stem}.txt:{li}')
                    continue
                try:
                    cf = float(p[0])
                    if cf != int(cf): raise ValueError('not int')
                    c = int(cf)
                    if c < 0 or c >= num_classes: raise ValueError(f'id {c} out of range')
                    class_counts[c] = class_counts.get(c, 0) + 1
                except ValueError:
                    bad_cls += 1
                    if len(issues['bad_class_id']) < 10:
                        issues['bad_class_id'].append(f'{split}/{stem}.txt:{li}')
                    continue
                try:
                    coords = [float(x) for x in p[1:]]
                except ValueError:
                    bad_box += 1; continue
                if any(not (0.0 <= x <= 1.0) for x in coords):
                    bad_box += 1
                    if len(issues['bad_bbox']) < 10:
                        issues['bad_bbox'].append(f'{split}/{stem}.txt:{li}')
        per_split[split] = {'images':len(img_map),'labels':len(lbl_set),
                            'only_image':len(only_img),'only_label':len(only_lbl),
                            'empty_labels':empty,'bad_format':bad_fmt,
                            'bad_class_id':bad_cls,'bad_bbox':bad_box}
    return per_split, class_counts, issues

per_split, class_counts, issue_samples = verify_dataset(LOCAL_BALANCED_DIR, NUM_CLASSES)
log.info('=== BALANCED DATASET VERIFICATION ===')
for s, st in per_split.items():
    log.info(f'  {s:5s} | images={st["images"]:5d} labels={st["labels"]:5d} '
             f'only_img={st.get("only_image",0)} only_lbl={st.get("only_label",0)} '
             f'empty={st.get("empty_labels",0)} bad_fmt={st.get("bad_format",0)} '
             f'bad_cls={st.get("bad_class_id",0)} bad_bbox={st.get("bad_bbox",0)}')
log.info(f'  Total annotations: {sum(class_counts.values())}  '
         f'classes present: {sum(1 for n in class_counts.values() if n>0)}/{NUM_CLASSES}')

hard_errors = sum(st.get('only_image',0)+st.get('only_label',0)+st.get('bad_format',0)
                  +st.get('bad_class_id',0)+st.get('bad_bbox',0) for st in per_split.values())
Path(RESULTS_DIR, 'dataset_verification.json').write_text(json.dumps({
    'per_split': per_split, 'class_counts': class_counts,
    'issue_samples': {k: v for k, v in issue_samples.items() if v},
    'num_classes': NUM_CLASSES}, indent=2))

if hard_errors > 0:
    log.warning(f'!!! Verification found {hard_errors} hard errors. Examples:')
    for k, vs in issue_samples.items():
        for v in vs[:5]: log.warning(f'  {k}: {v}')
    if not ALLOW_INVALID_LABELS:
        raise VerificationError(f'{hard_errors} hard errors - fix them or set '
                                'ALLOW_INVALID_LABELS = True in the Config cell.')
    log.warning('ALLOW_INVALID_LABELS = True -> proceeding despite errors.')
else:
    log.info('Balanced dataset verification PASSED.')""")

# ======================================================================
# 8 - FINAL TRAINING DATASET SUMMARY
# ======================================================================
md("""## 8 - FINAL TRAINING DATASET SUMMARY
Printed **before any training** so you can confirm exactly what YOLOv12 will
learn from: the real image counts, the final merged classes, and everything
that was dropped along the way.""")

code("""# 8.1 - FINAL TRAINING DATASET SUMMARY
print('=' * 78)
print(' FINAL TRAINING DATASET SUMMARY')
print('=' * 78)
print(f'Output dataset path : {BALANCED_DIR}')
print(f'Local training copy : {LOCAL_BALANCED_DIR}')
print(f'data.yaml           : {DATA_YAML}')
print()
print(f'Total sampled images: {len(selected)}   (target ~{TARGET_TOTAL})')
print(f'  train : {len(split_assignment["train"])}')
print(f'  val   : {len(split_assignment["val"])}')
print(f'  test  : {len(split_assignment["test"])}')
print()
print(f'Final classes (merged groups): {NUM_CLASSES}   '
      f'(per-group cap {per_group_cap}, fill_to_target={BALANCE_FILL_TO_TARGET})')
print(f'  {"id":>3}  {"group":24s} {"images":>8s} {"labels":>9s}')
for g in range(NUM_CLASSES):
    print(f'  {g:>3}  {group_names_final[g]:24s} '
          f'{balanced_img_count.get(g,0):8d} {balanced_lbl_count.get(g,0):9d}')
_iv = [balanced_img_count.get(g, 0) for g in range(NUM_CLASSES)]
print(f'  image imbalance ratio (max/min): '
      f'{max(_iv)/max(1,min(_iv)):.1f}x  (multi-label; not perfectly balanceable)')
print()
print(f'Dropped unknown/ambiguous classes : {unknown_classes}')
print(f'Dropped rare-group classes        : {dropped_rare_classes}')
print(f'Dropped rare groups               : {dropped_rare_groups}')
print(f'Dropped invalid label lines       : {sum(invalid_counter.values())} '
      f'{dict(invalid_counter)}')
print(f'Dropped images (no kept label)    : {n_drop_unmapped}')
print(f'Dropped images (balance/not sel.) : {n_drop_not_selected}')
print('=' * 78)
log.info('FINAL TRAINING DATASET SUMMARY printed - starting YOLOv12 next.')""")

# ======================================================================
# 9 - YOLOv12 training
# ======================================================================
md("""## 9 - YOLOv12 training
Resume-aware: checkpoints are written straight to Drive (`last.pt` every epoch,
`epoch{N}.pt` every `save_period`). On a re-run after a disconnect, training
resumes automatically from `last.pt`.

> **On accuracy:** this notebook reports **only the real validation metrics**
> measured below. It does **not** hardcode or guarantee any number. Whether
> YOLOv12 reaches high mAP depends on dataset/label quality, how visually
> separable the merged groups are, images-per-class after balancing, the model
> size (`yolo12n/s/m`), image resolution, and how long you train.""")

code("""# 9.1 - Install YOLOv12 (Ultralytics).
!pip install -q -U ultralytics
import ultralytics
ultralytics.checks()
log.info(f'ultralytics {ultralytics.__version__}')""")

code("""# 9.2 - YOLOv12 training (resume-aware; checkpoints live on Drive).
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
    log.info(f'Fresh YOLOv12 training: {YOLO_MODEL}  (nc={NUM_CLASSES})')
    yolo_model = YOLO(YOLO_MODEL)
    yolo_model.train(
        data=DATA_YAML, epochs=YOLO_EPOCHS, imgsz=YOLO_IMGSZ, batch=YOLO_BATCH,
        patience=YOLO_PATIENCE, device=0,
        project=YOLO_PROJECT_DIR, name=YOLO_RUN_NAME, exist_ok=True,
        save_period=5, workers=2, cache='disk', amp=True,
        mosaic=0.5, mixup=0.05, close_mosaic=10,
        optimizer='auto', lr0=0.01, lrf=0.01, seed=SAMPLE_SEED,
        plots=True, verbose=True)
log.info('YOLOv12 training stage complete.')""")

# ======================================================================
# 10 - Validation, inference, metrics (real numbers only)
# ======================================================================
md("## 10 - Validation, inference, metrics (real numbers only)")

code("""# 10.1 - Validate the best checkpoint on the val split (REAL metrics).
yolo_model  = YOLO(str(best_ckpt))
val_metrics = yolo_model.val(data=DATA_YAML, imgsz=YOLO_IMGSZ, batch=YOLO_BATCH,
                             device=0, split='val', verbose=False)
yolo_scores = {
    'mAP_0.5':      float(val_metrics.box.map50),
    'mAP_0.5:0.95': float(val_metrics.box.map),
    'precision':    float(val_metrics.box.mp),
    'recall':       float(val_metrics.box.mr),
}
print('--- YOLOv12 validation metrics (measured, not hardcoded) ---')
for k, v in yolo_scores.items():
    print(f'  {k:14s}: {v:.4f}')
    log.info(f'  YOLOv12 {k:14s}: {v:.4f}')""")

code("""# 10.2 - Inference visualisation on random test images.
import matplotlib.pyplot as plt
test_imgs = sorted((Path(LOCAL_BALANCED_DIR)/'images'/'test').glob('*.png')) \\
          + sorted((Path(LOCAL_BALANCED_DIR)/'images'/'test').glob('*.jpg'))
if test_imgs:
    random.seed(0)
    sample = random.sample(test_imgs, min(6, len(test_imgs)))
    preds = yolo_model.predict(sample, imgsz=YOLO_IMGSZ, conf=0.25, device=0, verbose=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, pred in zip(axes.flat, preds):
        ax.imshow(pred.plot()[..., ::-1]); ax.set_title(Path(pred.path).name, fontsize=8)
        ax.axis('off')
    for ax in axes.flat[len(preds):]: ax.axis('off')
    plt.tight_layout()
    viz = f'{RESULTS_DIR}/yolov12_predictions.png'
    plt.savefig(viz, dpi=110, bbox_inches='tight'); plt.show()
    log.info(f'Saved {viz}')
else:
    log.warning('No test images available for inference visualisation.')""")

code("""# 10.3 - Export YOLOv12 metrics + per-class table (JSON + CSV).
yolo_metrics = {'model': YOLO_MODEL, 'epochs_config': YOLO_EPOCHS,
                'imgsz': YOLO_IMGSZ, 'batch': YOLO_BATCH,
                'epochs_done': done_epochs, 'num_classes': NUM_CLASSES,
                'class_names': group_names_final, **yolo_scores}

# Per-class AP (real values from the validator), when available.
try:
    per_class_ap = {group_names_final[int(ci)]: float(ap)
                    for ci, ap in zip(val_metrics.box.ap_class_index,
                                      val_metrics.box.ap50)}
    yolo_metrics['per_class_mAP50'] = per_class_ap
    print('--- per-class mAP@0.5 ---')
    for nm, ap in per_class_ap.items():
        print(f'  {nm:24s}: {ap:.4f}')
except Exception as e:
    log.warning(f'per-class AP unavailable: {e}')

with open(f'{RESULTS_DIR}/yolov12_metrics.json', 'w') as f:
    json.dump(yolo_metrics, f, indent=2)
if results_csv.exists():
    shutil.copy2(results_csv, f'{RESULTS_DIR}/yolov12_results.csv')

experiment_summary = {
    'yolov12': yolo_metrics,
    'dataset_balance': balance_report,
    'verification': per_split,
}
with open(f'{RESULTS_DIR}/experiment_summary.json', 'w') as f:
    json.dump(experiment_summary, f, indent=2)
log.info('YOLOv12 metrics exported:\\n' + json.dumps(yolo_metrics, indent=2))""")

# ======================================================================
# 11 - Drive artifacts
# ======================================================================
md("## 11 - Drive artifacts")

code("""# 11.1 - Print the exact Drive locations of every produced artifact.
print('=' * 78)
print(' SkyLogic balanced-YOLOv12 - Drive artifacts')
print('=' * 78)
print()
print('Balanced dataset (separate from the read-only original):')
print(f'  {BALANCED_DIR}/')
print(f'    images/{{train,val,test}}/   labels/{{train,val,test}}/')
print(f'    data.yaml')
print(f'    class_group_mapping.json')
print(f'    dataset_balance_report.json')
print(f'    dataset_balance_report.csv')
print()
print('YOLOv12 checkpoints:')
print(f'  {YOLO_PROJECT_DIR}/{YOLO_RUN_NAME}/weights/best.pt')
print(f'  {YOLO_PROJECT_DIR}/{YOLO_RUN_NAME}/weights/last.pt  (+ epoch{{N}}.pt)')
print()
print('Reports + metrics:')
print(f'  {RESULTS_DIR}/original_class_distribution.json')
print(f'  {RESULTS_DIR}/class_group_mapping.json')
print(f'  {RESULTS_DIR}/dataset_balance_report.json / .csv')
print(f'  {RESULTS_DIR}/dataset_verification.json')
print(f'  {RESULTS_DIR}/yolov12_metrics.json')
print(f'  {RESULTS_DIR}/yolov12_results.csv')
print(f'  {RESULTS_DIR}/yolov12_predictions.png')
print(f'  {RESULTS_DIR}/experiment_summary.json')
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
