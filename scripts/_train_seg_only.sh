#!/usr/bin/env bash
# Segmentor training only — runs inside skylogic-api image.
set -u
mkdir -p /app/runs/logs
LOG="/app/runs/logs/train_segmentor_$(date +%Y%m%d_%H%M%S).log"
SUMMARY="/app/runs/logs/SUMMARY.txt"

echo "===== SEGMENTOR TRAINING START $(date) =====" | tee -a "$SUMMARY"
echo "Log: $LOG" | tee -a "$SUMMARY"

python -u scripts/train_segmentor.py \
  --project-root /app \
  --epochs 20 \
  --imgsz 512 \
  --batch 2 \
  --limit-train 0 \
  --save-dir /app/models/segformer \
  > "$LOG" 2>&1
RC=$?
echo "===== SEGMENTOR TRAINING END $(date) exit=$RC =====" | tee -a "$SUMMARY"
exit $RC
