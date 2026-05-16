#!/usr/bin/env bash
# YOLO v4: full data + 100 epochs + light mosaic, keeping stable WSL2 infra
# Auto-resume: each crash → restart from last.pt. Stops at epoch 100 or MAX_ITERS.
set -u
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"
EXT4="${HOME}/skylogic_data"
RUN_DIR="${PROJ}/models/yolo/xview_v4"
LAST="${RUN_DIR}/weights/last.pt"
LOG="${PROJ}/runs/logs/yolo_v4_loop.log"

mkdir -p "${PROJ}/runs/logs"
echo "==== YOLO v4 auto-resume loop started at $(date) ====" | tee -a "$LOG"
echo "  fraction=1.0  epochs=100  mosaic=0.3 (light)  batch=4  workers=0" | tee -a "$LOG"

MAX_ITERS=150   # 100 epochs × ~12min ÷ ~10min per WSL2 cycle = ~120 iter worst-case
ITER=0
PREV_EPOCH=-1
TARGET_EPOCHS=100

while [ $ITER -lt $MAX_ITERS ]; do
  ITER=$((ITER + 1))
  echo "" | tee -a "$LOG"
  echo "---- Iteration $ITER at $(date +%H:%M:%S) ----" | tee -a "$LOG"

  # Stop any running container
  docker rm -f skylogic-trainer-yolo 2>/dev/null || true

  # Decide: fresh start or resume
  if [ ! -f "$LAST" ]; then
    echo "  Fresh start (no last.pt) — full data + light aug" | tee -a "$LOG"
    EXTRA_ARGS=( \
      --project-root /app \
      --data /app/data/yolo/skylogic_split.yaml \
      --model /app/models/yolov8n.pt \
      --epochs 100 \
      --imgsz 416 --batch 4 --amp-off --workers 0 \
      --fraction 1.0 \
      --cls-pw 1.0 --cls-gain 2.0 \
      --mosaic 0.3 --mixup 0.05 --copy-paste 0.1 \
      --lr0 0.01 --lrf 0.01 \
      --patience 15 --save-period 1 \
      --project /app/models/yolo --name xview_v4 \
    )
  else
    echo "  Resume from last.pt ($(stat -c %y "$LAST" | cut -d. -f1))" | tee -a "$LOG"
    EXTRA_ARGS=( --resume /app/models/yolo/xview_v4/weights/last.pt )
  fi

  # Launch
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

  # Wait until container exits
  echo "  Waiting for container to exit..." | tee -a "$LOG"
  while true; do
    STATUS=$(docker inspect skylogic-trainer-yolo --format '{{.State.Status}}' 2>/dev/null)
    [ "$STATUS" = "exited" ] && break
    [ -z "$STATUS" ]          && break
    sleep 20
  done

  EXITCODE=$(docker inspect skylogic-trainer-yolo --format '{{.State.ExitCode}}' 2>/dev/null)
  echo "  Exited (code=$EXITCODE)" | tee -a "$LOG"

  # Pull current epoch count from results.csv
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
