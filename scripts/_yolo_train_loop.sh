#!/usr/bin/env bash
# Auto-resume YOLO training loop: WSL2 silently kills training around the 5-minute mark,
# but we save last.pt every epoch. Each crash → restart from last.pt.
# Stops automatically when training reports "complete" or max iterations reached.
set -u
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"
EXT4="${HOME}/skylogic_data"
RUN_DIR="${PROJ}/models/yolo/xview_v3"
LAST="${RUN_DIR}/weights/last.pt"
LOG="${PROJ}/runs/logs/yolo_loop.log"

mkdir -p "${PROJ}/runs/logs"
echo "==== Auto-resume training loop started at $(date) ====" | tee -a "$LOG"

MAX_ITERS=60   # safety cap (50 epochs target, but allow some retries)
ITER=0
PREV_EPOCH=-1

while [ $ITER -lt $MAX_ITERS ]; do
  ITER=$((ITER + 1))
  echo "" | tee -a "$LOG"
  echo "---- Iteration $ITER (cumulative restart) at $(date +%H:%M:%S) ----" | tee -a "$LOG"

  # Stop any running container
  docker rm -f skylogic-trainer-yolo 2>/dev/null || true

  # Decide: fresh start or resume
  if [ ! -f "$LAST" ]; then
    echo "  Fresh start (no last.pt)" | tee -a "$LOG"
    EXTRA_ARGS=( \
      --project-root /app \
      --data /app/data/yolo/skylogic_split.yaml \
      --model /app/models/yolov8n.pt \
      --epochs 50 \
      --imgsz 416 --batch 4 --amp-off --workers 0 \
      --fraction 0.2 \
      --cls-pw 1.0 --cls-gain 2.0 --simple-aug \
      --lr0 0.01 --lrf 0.01 \
      --patience 15 --save-period 1 \
      --project /app/models/yolo --name xview_v3 \
    )
  else
    echo "  Resume from last.pt ($(stat -c %y "$LAST" | cut -d. -f1))" | tee -a "$LOG"
    EXTRA_ARGS=( --resume /app/models/yolo/xview_v3/weights/last.pt )
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

  # Wait until it exits
  echo "  Waiting for container to exit..." | tee -a "$LOG"
  while true; do
    STATUS=$(docker inspect skylogic-trainer-yolo --format '{{.State.Status}}' 2>/dev/null)
    [ "$STATUS" = "exited" ] && break
    [ -z "$STATUS" ]          && break
    sleep 15
  done

  EXITCODE=$(docker inspect skylogic-trainer-yolo --format '{{.State.ExitCode}}' 2>/dev/null)
  echo "  Exited (code=$EXITCODE)" | tee -a "$LOG"

  # Pull current epoch count from results.csv
  if [ -f "${RUN_DIR}/results.csv" ]; then
    EPOCH=$(tail -1 "${RUN_DIR}/results.csv" | cut -d, -f1)
    echo "  results.csv epoch reached: $EPOCH" | tee -a "$LOG"
    if [ -n "$EPOCH" ] && [ "$EPOCH" != "epoch" ]; then
      # Check if we made progress
      if [ "$EPOCH" -le "$PREV_EPOCH" ]; then
        echo "  No progress this iteration (still at epoch $EPOCH), but continuing..." | tee -a "$LOG"
      fi
      PREV_EPOCH=$EPOCH

      # Stop if we reached 50 epochs
      if [ "$EPOCH" -ge 50 ]; then
        echo "  ✅ Reached target 50 epochs!" | tee -a "$LOG"
        break
      fi
    fi
  fi

  # Brief pause before next attempt
  sleep 5
done

echo "" | tee -a "$LOG"
echo "==== Loop ended at $(date) (iterations: $ITER, last epoch: $PREV_EPOCH) ====" | tee -a "$LOG"
