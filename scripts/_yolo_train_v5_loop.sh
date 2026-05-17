#!/usr/bin/env bash
# YOLO v5: continue from v4 best.pt with stronger settings
#   - 200 epochs, imgsz=512 (native), mosaic=0.5, mixup=0.10, copy_paste=0.15
#   - Same stable infra: ext4 data, OMP/MKL=4, batch=4, workers=0, auto-resume
set -u
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"
EXT4="${HOME}/skylogic_data"
RUN_DIR="${PROJ}/models/yolo/xview_v5"
LAST="${RUN_DIR}/weights/last.pt"
V4_BEST="${PROJ}/models/yolo/xview_v4/weights/best.pt"
LOG="${PROJ}/runs/logs/yolo_v5_loop.log"

mkdir -p "${PROJ}/runs/logs"
echo "==== YOLO v5 auto-resume loop started at $(date) ====" | tee -a "$LOG"
echo "  init=v4/best.pt  fraction=1.0  epochs=200  imgsz=512" | tee -a "$LOG"
echo "  mosaic=0.5  mixup=0.10  copy_paste=0.15  batch=4  workers=0" | tee -a "$LOG"

MAX_ITERS=200   # extra safety; we expect mostly 1-2 iterations with current stability
ITER=0
PREV_EPOCH=-1
TARGET_EPOCHS=200

while [ $ITER -lt $MAX_ITERS ]; do
  ITER=$((ITER + 1))
  echo "" | tee -a "$LOG"
  echo "---- Iteration $ITER at $(date +%H:%M:%S) ----" | tee -a "$LOG"

  # Stop any running container (safety)
  docker rm -f skylogic-trainer-yolo 2>/dev/null || true

  # Find the latest TRULY RESUMABLE checkpoint (epoch_N.pt with optimizer state).
  # IMPORTANT: best.pt and last.pt have epoch=-1 (Ultralytics "stripped" format) and
  # are NOT resumable — using them causes Ultralytics to silently fall back to
  # defaults (coco8.yaml, batch=16, mosaic=1.0) and create garbage train-N/ runs.
  LATEST_N=$(ls "${RUN_DIR}/weights/epoch"*.pt 2>/dev/null \
             | sed 's|.*/epoch||; s|\.pt$||' \
             | sort -n | tail -1)

  if [ -n "$LATEST_N" ]; then
    LATEST_CKPT="${RUN_DIR}/weights/epoch${LATEST_N}.pt"
    echo "  Resume from epoch${LATEST_N}.pt ($(stat -c %y "$LATEST_CKPT" | cut -d. -f1))" | tee -a "$LOG"
    EXTRA_ARGS=( --resume "/app/models/yolo/xview_v5/weights/epoch${LATEST_N}.pt" )
  else
    echo "  Fresh start — init from v4/best.pt + native 512 + stronger aug" | tee -a "$LOG"
    EXTRA_ARGS=( \
      --project-root /app \
      --data /app/data/yolo/skylogic_split.yaml \
      --model /app/models/yolo/xview_v4/weights/best.pt \
      --epochs 200 \
      --imgsz 512 --batch 4 --amp-off --workers 0 \
      --fraction 1.0 \
      --cls-pw 1.0 --cls-gain 2.0 \
      --mosaic 0.5 --mixup 0.10 --copy-paste 0.15 \
      --lr0 0.01 --lrf 0.01 \
      --patience 15 --save-period 1 \
      --project /app/models/yolo --name xview_v5 \
    )
  fi

  docker run -d \
    --name skylogic-trainer-yolo \
    --restart no \
    -v "${EXT4}":/app/data \
    -v "${PROJ}/models":/app/models \
    -v "${PROJ}/runs":/app/runs \
    -v "${PROJ}/scripts":/app/scripts:ro \
    -w /app \
    -e PYTHONUNBUFFERED=1 \
    -e OMP_NUM_THREADS=4 \
    -e MKL_NUM_THREADS=4 \
    skylogic-api \
    python -u scripts/train_detector.py "${EXTRA_ARGS[@]}" >/dev/null

  echo "  Waiting for container to exit..." | tee -a "$LOG"
  while true; do
    STATUS=$(docker inspect skylogic-trainer-yolo --format '{{.State.Status}}' 2>/dev/null)
    [ "$STATUS" = "exited" ] && break
    [ -z "$STATUS" ]          && break
    sleep 20
  done

  EXITCODE=$(docker inspect skylogic-trainer-yolo --format '{{.State.ExitCode}}' 2>/dev/null)
  echo "  Exited (code=$EXITCODE)" | tee -a "$LOG"

  if [ -f "${RUN_DIR}/results.csv" ]; then
    EPOCH=$(tail -1 "${RUN_DIR}/results.csv" | cut -d, -f1)
    echo "  results.csv epoch reached: $EPOCH" | tee -a "$LOG"
    if [ -n "$EPOCH" ] && [ "$EPOCH" != "epoch" ]; then
      PREV_EPOCH=$EPOCH
      if [ "$EPOCH" -ge "$TARGET_EPOCHS" ]; then
        echo "  ✅ Reached target $TARGET_EPOCHS epochs!" | tee -a "$LOG"
        break
      fi
    fi
  fi

  sleep 5
done

echo "" | tee -a "$LOG"
echo "==== Loop ended at $(date) (iter=$ITER, last_epoch=$PREV_EPOCH) ====" | tee -a "$LOG"
