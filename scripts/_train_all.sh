#!/usr/bin/env bash
# Runs YOLO full training, then segmentor full training, into mounted volumes.
# Designed to run inside the skylogic-api image with /app as workdir.
set -u
mkdir -p /app/runs/logs
LOG_YOLO="/app/runs/logs/train_detector_$(date +%Y%m%d_%H%M%S).log"
LOG_SEG="/app/runs/logs/train_segmentor_$(date +%Y%m%d_%H%M%S).log"
SUMMARY="/app/runs/logs/SUMMARY.txt"

echo "===== YOLO TRAINING START $(date) =====" | tee -a "$SUMMARY"
echo "Log file: $LOG_YOLO" | tee -a "$SUMMARY"
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
  > "$LOG_YOLO" 2>&1
YOLO_EXIT=$?
echo "===== YOLO TRAINING END $(date) exit=$YOLO_EXIT =====" | tee -a "$SUMMARY"

echo "===== SEGMENTOR TRAINING START $(date) =====" | tee -a "$SUMMARY"
echo "Log file: $LOG_SEG" | tee -a "$SUMMARY"
python -u scripts/train_segmentor.py \
  --project-root /app \
  --epochs 20 \
  --imgsz 512 \
  --batch 2 \
  --limit-train 0 \
  --save-dir /app/models/segformer \
  > "$LOG_SEG" 2>&1
SEG_EXIT=$?
echo "===== SEGMENTOR TRAINING END $(date) exit=$SEG_EXIT =====" | tee -a "$SUMMARY"

echo "ALL DONE. yolo_exit=$YOLO_EXIT seg_exit=$SEG_EXIT" | tee -a "$SUMMARY"
exit $((YOLO_EXIT + SEG_EXIT))
