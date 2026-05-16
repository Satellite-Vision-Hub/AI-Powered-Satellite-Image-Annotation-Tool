#!/usr/bin/env bash
# YOLO training only — runs inside skylogic-api image.
set -u
mkdir -p /app/runs/logs
LOG="/app/runs/logs/train_detector_$(date +%Y%m%d_%H%M%S).log"
SUMMARY="/app/runs/logs/SUMMARY.txt"

echo "===== YOLO TRAINING START $(date) =====" | tee -a "$SUMMARY"
echo "Log: $LOG" | tee -a "$SUMMARY"

python -u scripts/train_detector.py \
  --data /app/data/yolo/skylogic.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 512 \
  --batch 8 \
  --device cpu \
  --project /app/models/yolo \
  --name xview_full_run \
  --workers 2 \
  --patience 10 \
  --lr0 0.01 \
  > "$LOG" 2>&1
RC=$?
echo "===== YOLO TRAINING END $(date) exit=$RC =====" | tee -a "$SUMMARY"
exit $RC
