# SkyLogic — Sample-9k Experimental Dataset & Colab Pipeline

A lightweight, **copy-based** ~9,000-image sample of the xView-derived SkyLogic
dataset, plus a self-contained Google Colab notebook for fast YOLOv12 +
SegFormer experimentation on a **T4 GPU**.

> The original `data/` tree was **never modified** — this folder was built with
> file copies only (`scripts/build_sample_dataset.py`).

---

## 1. What's in this folder

```
sample_dataset_9k/
├── images/
│   ├── train/   7,268 PNG patches (512×512)
│   ├── val/       908 PNG patches
│   └── test/      909 PNG patches
├── labels/
│   ├── train/   7,268 YOLO label files
│   ├── val/       908 YOLO label files
│   └── test/      909 YOLO label files
├── data.yaml            YOLOv12 dataset config (62 classes)
├── class_map.json       sparse xView id → contiguous id
├── seg_class_map.json   contiguous id → 10-class disaster seg index
├── dataset_stats.json   per-split counts + class histogram
├── SAMPLE_INFO.txt      human-readable build summary
└── README.md            this file
```

**9,085 images total** — every annotated patch in the source `train` split
(empty-background patches excluded), so all 62 classes are represented.
Split 80 / 10 / 10 with a fixed seed (42) for reproducibility.

---

## 2. Quick start on Google Colab

The notebook is **self-contained** and supports two upload modes:

| You upload to Drive | Notebook does |
|--------------------|---------------|
| The **original/full** dataset (zip or folder) | Samples ~9,000 images in Colab, copies them to Drive once for reuse |
| A pre-built `sample_dataset_9k.zip` | Detects the existing sample layout and uses it directly |

### Step 1 — Upload to Google Drive
Upload either:
- the **original/full** dataset as a `.zip` (recommended — one big file uploads
  far more reliably than thousands of loose files), **or**
- the prebuilt `sample_dataset_9k.zip` from this folder.

### Step 2 — Open the notebook
Open `SkyLogic_Sample9k_Colab.ipynb` in Google Colab
(`File → Upload notebook`, or open it from Drive).

### Step 3 — Select the GPU
`Runtime → Change runtime type → T4 GPU → Save`.

### Step 4 — Edit two variables in the Config cell, then Run all
In the notebook's **Config** cell (Section 2):
```python
DRIVE_DATASET_SOURCE = '/content/drive/MyDrive/your_upload.zip'   # or folder
DRIVE_PROJECT_DIR    = '/content/drive/MyDrive/SkyLogic_sample9k_experiment'
```

Then `Runtime → Run all`. The notebook will:
1. Mount Drive
2. Stage your source (extract zip / use folder directly)
3. Auto-detect the YOLO layout
4. Build a class-stratified ~9,000-image sample (or reuse the one already on
   Drive — controlled by `FORCE_RESAMPLE`)
5. Run hard label-format verification (stops on errors unless
   `ALLOW_INVALID_LABELS = True`)
6. Train YOLOv12 and SegFormer with checkpoints on Drive
7. Print every output path at the end

---

## 3. What the notebook does

| Stage | Detail |
|-------|--------|
| Environment | GPU check, mount Drive, automatic logging to `run.log` |
| Dependencies | Installs latest `ultralytics` (YOLOv12) + `transformers` (SegFormer) |
| Dataset | Stages zip→local SSD (cached), verifies integrity |
| **YOLOv12** | Train → validate → inference visualisation → metrics export |
| **SegFormer** | Weak pseudo-mask dataset → train → evaluate → visualise |
| Results | All metrics + plots saved to Drive |

### Models
- **Detection:** `yolo12s.pt` (lightweight YOLOv12; switch to `yolo12n.pt` in the
  Config cell for even faster runs).
- **Segmentation:** SegFormer `nvidia/mit-b0` (smallest backbone), trained with
  weak supervision — bounding boxes are rasterised into 10-class disaster masks.

### Lightweight settings (tuned for ~9k images on T4)
| | YOLOv12 | SegFormer |
|--|---------|-----------|
| epochs | 50 (patience 15) | 20 |
| image size | 512 | 512 |
| batch | 16 | 8 |
| augmentation | light (mosaic 0.5, mixup 0.05) | — |

---

## 4. Resume after a Colab disconnect

Colab sessions drop. This pipeline is built for it:

- **Sampling** — if `DRIVE_PROJECT_DIR/sample_dataset_9k/` already contains a
  valid sample, it is reused (set `FORCE_RESAMPLE = True` to rebuild).
- **YOLOv12** — checkpoints (`last.pt`, every-5-epoch snapshots) are written
  straight to Drive. On re-run, training resumes from `last.pt` automatically.
- **SegFormer** — a full checkpoint (`segformer_last.pt`) is saved to Drive
  **every epoch**. On re-run, it resumes from the saved epoch.

If Colab disconnects, just reconnect and **`Runtime → Run all`** again. Nothing
restarts from scratch; completed stages are skipped.

---

## 5. Outputs (written to Drive)

```
MyDrive/SkyLogic_Sample9k/
├── yolov12/sample9k/        training run — weights/, results.csv, plots
├── segformer/               segformer_last.pt, segformer_best.pt
├── results/                 *_metrics.json, *_predictions.png, experiment_summary.json
└── run.log                  full timestamped log
```

---

## 6. Regenerating this sample

From the project root:

```bash
python scripts/build_sample_dataset.py     # rebuild sample_dataset_9k/ (copy-only)
python scripts/zip_sample_dataset.py       # repackage sample_dataset_9k.zip
python scripts/build_colab_notebook.py     # regenerate the Colab notebook
```

All three are idempotent and never touch the original `data/` tree.
