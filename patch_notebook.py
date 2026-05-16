"""
Patch SkyLogic_Colab_GPU.ipynb with 3 resilience improvements:
  1. Cell 8 (c8-copy)      : parallel Drive->/content copy (10x faster)
  2. Cell 11 (c11-yolo-train): YOLO --resume from last.pt
  3. Cell 16 (c16-sam)     : cache SAM checkpoint on Drive (avoid 2.5 GB redownload)

Idempotent — running twice produces the same result.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "SkyLogic_Colab_GPU.ipynb"

# ── New cell source bodies ─────────────────────────────────────────────────

CELL_8_NEW = """\
# ── CELL 8: Copy Patches to Local SSD (parallel, resumable) ──────────────────
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm.auto import tqdm

# Parallel copy: ~10x faster than sequential shutil.copy2 on Drive FUSE.
# Skips files that already exist locally → safe to rerun mid-copy.
def _copy_one(args):
    src, dst = args
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst.stat().st_size

def copy_patches_parallel(src_dir, dst_dir, desc, workers=32):
    files = sorted(Path(src_dir).glob('*.png'))
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    tasks = [(f, Path(dst_dir) / f.name) for f in files]
    # Quick skip if everything already copied
    pending = [(s, d) for s, d in tasks if not d.exists()]
    if not pending:
        print(f'{desc}: all {len(files):,} files already cached locally')
        return len(files)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_copy_one, t) for t in pending]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=desc, unit='img'):
            pass
    return len(files)

n_train_loc = copy_patches_parallel(DRIVE_TRAIN_PAT, LOCAL_TRAIN_PAT, 'Train', workers=32)
n_val_loc   = copy_patches_parallel(DRIVE_VAL_PAT,   LOCAL_VAL_PAT,   'Val',   workers=32)
print(f'\\nLocal train: {n_train_loc:,} | val: {n_val_loc:,}')
"""

CELL_11_NEW = """\
# ── CELL 11: YOLOv10 — Train (resume + checkpoint-aware) ─────────────────────
import time, json
from pathlib import Path

MODEL_RESULTS = {}
yolo_status, yolo_metrics, yolo_error = 'not_started', {}, None
yolo_best_weights = None

# Look for best.pt (training already complete) or last.pt (interrupted training)
best_pts = list(Path(YOLO_SAVE_DIR).rglob('best.pt'))
last_pts = list(Path(YOLO_SAVE_DIR).rglob('last.pt'))

if best_pts and not FORCE_RETRAIN_YOLO:
    yolo_best_weights = str(best_pts[-1])
    yolo_status = 'resumed'
    print(f'Existing YOLO best.pt found, skipping training: {yolo_best_weights}')
elif last_pts and not FORCE_RETRAIN_YOLO:
    # Interrupted training → resume from last.pt with Ultralytics native resume
    resume_ckpt = str(last_pts[-1])
    print(f'Found interrupted YOLO training at: {resume_ckpt}')
    print('Resuming from last.pt (Ultralytics native resume=True)...')
    try:
        from ultralytics import YOLO
        m = YOLO(resume_ckpt)
        t0 = time.time()
        m.train(resume=True)
        train_time_min = (time.time() - t0) / 60
        yolo_metrics['train_time_min'] = round(train_time_min, 2)
        yolo_metrics['resumed_from'] = resume_ckpt
        best_pts = list(Path(YOLO_SAVE_DIR).rglob('best.pt'))
        yolo_best_weights = str(best_pts[-1]) if best_pts else resume_ckpt
        yolo_status = 'completed'
        print(f'YOLO resume completed in {train_time_min:.1f} min -> {yolo_best_weights}')
    except Exception as e:
        import traceback
        yolo_error = traceback.format_exc()
        yolo_status = 'failed'
        with open(f'{LOGS_DIR}/yolo_error.txt', 'w') as f: f.write(yolo_error)
        print(f'YOLO RESUME FAILED (continuing):\\n{yolo_error[-500:]}')
else:
    try:
        from skylogic.agents.detector import DetectorAgent
        detector = DetectorAgent(model_path=None, device=DEVICE, num_classes=60)
        t0 = time.time()
        detector.train(data_yaml=DATA_YAML_PATH,
                       epochs=EPOCHS_YOLO, batch_size=BATCH_YOLO,
                       img_size=512, save_dir=YOLO_SAVE_DIR)
        train_time_min = (time.time() - t0) / 60
        yolo_metrics['train_time_min'] = round(train_time_min, 2)
        best_pts = list(Path(YOLO_SAVE_DIR).rglob('best.pt'))
        yolo_best_weights = str(best_pts[-1]) if best_pts else None
        yolo_status = 'completed'
        print(f'YOLO trained in {train_time_min:.1f} min -> {yolo_best_weights}')
    except Exception as e:
        import traceback
        yolo_error = traceback.format_exc()
        yolo_status = 'failed'
        with open(f'{LOGS_DIR}/yolo_error.txt', 'w') as f: f.write(yolo_error)
        print(f'YOLO FAILED (continuing):\\n{yolo_error[-500:]}')
"""

CELL_16_NEW = """\
# ── CELL 16: SAM — Download + Inference (Drive-cached) ───────────────────────
import os, time, json, shutil, numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

sam_status, sam_metrics, sam_error = 'not_started', {}, None

# Use ViT-B if VRAM tight, ViT-H otherwise
SAM_MODEL_TYPE = 'vit_h' if VRAM_GB >= 14 else 'vit_b'
SAM_URL = {
    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
}[SAM_MODEL_TYPE]

# Drive-side cache: saves ~5 min per cold start on rerun
SAM_DRIVE_CKPT = f'{SAM_SAVE_DIR}/sam_{SAM_MODEL_TYPE}.pth'
SAM_LOCAL_CKPT = f'/content/models/sam_{SAM_MODEL_TYPE}.pth'
os.makedirs('/content/models', exist_ok=True)

if Path(SAM_LOCAL_CKPT).exists():
    print(f'SAM checkpoint already on /content: {SAM_LOCAL_CKPT}')
elif Path(SAM_DRIVE_CKPT).exists():
    print(f'Copying SAM checkpoint from Drive cache: {SAM_DRIVE_CKPT}')
    shutil.copy2(SAM_DRIVE_CKPT, SAM_LOCAL_CKPT)
else:
    print(f'Downloading SAM {SAM_MODEL_TYPE} from facebook (no Drive cache)...')
    !wget -q --show-progress '{SAM_URL}' -O '{SAM_LOCAL_CKPT}'
    # Cache to Drive for next session
    print(f'Caching SAM checkpoint to Drive: {SAM_DRIVE_CKPT}')
    try:
        shutil.copy2(SAM_LOCAL_CKPT, SAM_DRIVE_CKPT)
    except Exception as e:
        print(f'(Drive cache failed, will redownload next time: {e})')

SAM_CKPT = SAM_LOCAL_CKPT
print(f'SAM checkpoint: {SAM_CKPT} ({Path(SAM_CKPT).stat().st_size/1e6:.1f} MB)')

try:
    from skylogic.agents.sam_agent import SAMAgent
    sam_agent = SAMAgent(checkpoint_path=SAM_CKPT, model_type=SAM_MODEL_TYPE, device=DEVICE)
    if sam_agent.predictor is None: raise RuntimeError('SAM predictor failed to load')

    imgs = sorted(Path(LOCAL_VAL_PAT).glob('*.png'))[:6] or sorted(Path(LOCAL_TRAIN_PAT).glob('*.png'))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    total_ms, ok_count = 0.0, 0
    for i, ip in enumerate(imgs[:6]):
        t0 = time.time()
        if not sam_agent.set_image(str(ip)):
            axes[i].set_title(f'{ip.name[:15]}\\n(skip)', fontsize=7); axes[i].axis('off'); continue
        r = sam_agent.predict_click([[256, 256]], [1], multimask_output=True)
        ms = (time.time()-t0)*1000; total_ms += ms; ok_count += 1
        img = np.array(Image.open(ip).convert('RGB').resize((512, 512)))
        axes[i].imshow(img)
        if r.get('best_mask') is not None:
            m = r['best_mask']; m = np.array(m) if isinstance(m, list) else m
            ov = np.zeros((*m.shape, 4), dtype=np.uint8); ov[m > 0] = [0, 255, 0, 120]
            axes[i].imshow(ov); axes[i].plot(256, 256, 'r*', markersize=10)
            axes[i].set_title(f'{ip.name[:15]}\\nscore={r["best_score"]:.3f} | {ms:.0f}ms', fontsize=7)
        else:
            axes[i].set_title(f'{ip.name[:15]}\\nno mask', fontsize=7)
        axes[i].axis('off')
    plt.suptitle(f'SAM {SAM_MODEL_TYPE.upper()} click predictions', fontsize=12); plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/sam_predictions.png', dpi=150, bbox_inches='tight'); plt.show()

    sam_metrics = {'model_type': SAM_MODEL_TYPE, 'samples_run': ok_count,
                    'inference_ms_avg': round(total_ms / max(1, ok_count), 2),
                    'drive_cached': Path(SAM_DRIVE_CKPT).exists(),
                    'note': 'Inference only (no training)'}
    with open(f'{LOGS_DIR}/sam_metrics.json', 'w') as f: json.dump(sam_metrics, f, indent=2)
    sam_status = 'completed'
    print(f'SAM avg inference: {sam_metrics["inference_ms_avg"]:.1f} ms/image')
except Exception as e:
    import traceback
    sam_error = traceback.format_exc()
    sam_status = 'failed'
    with open(f'{LOGS_DIR}/sam_error.txt', 'w') as f: f.write(sam_error)
    print(f'SAM FAILED (continuing):\\n{sam_error[-500:]}')

MODEL_RESULTS['SAM'] = {'status': sam_status, 'metrics': sam_metrics, 'error': sam_error}
"""

PATCHES = {
    'c8-copy':       CELL_8_NEW,
    'c11-yolo-train': CELL_11_NEW,
    'c16-sam':       CELL_16_NEW,
}


def main():
    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
    applied = []
    skipped = []
    for cell in nb['cells']:
        cid = cell.get('id')
        if cid in PATCHES:
            new_src = PATCHES[cid]
            old_src = ''.join(cell['source'])
            if old_src.strip() == new_src.strip():
                skipped.append(cid)
                continue
            # Split new source back into list of lines (each ending with \n except last)
            lines = new_src.splitlines(keepends=True)
            cell['source'] = lines
            applied.append(cid)
    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + '\n',
        encoding='utf-8'
    )
    print(f'Applied : {applied}')
    print(f'Skipped : {skipped}  (already up to date)')
    print(f'Notebook: {NB_PATH} ({NB_PATH.stat().st_size/1024:.1f} KB)')


if __name__ == '__main__':
    main()
