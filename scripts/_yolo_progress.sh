#!/usr/bin/env bash
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"
LOG=$(ls -t "$PROJ/runs/logs/train_detector_"*.log 2>/dev/null | head -1)
echo "log = $LOG"
echo "log size t0 = $(wc -c < "$LOG") bytes"
sleep 30
echo "log size t1 = $(wc -c < "$LOG") bytes  (after 30s)"

# Extract just the LAST progress segment after the final carriage return
# Read raw bytes, swap \r with \n, then take last few "lines"
echo
echo "==== Last 5 progress segments (split on CR) ===="
tr '\r' '\n' < "$LOG" | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^\s+[0-9]+/50 " | tail -5 | cut -c 1-180

echo
echo "==== Run dir contents ===="
ls -la "$PROJ/models/yolo/xview_full_run/" 2>/dev/null
echo
echo "==== Weights subdir ===="
ls -la "$PROJ/models/yolo/xview_full_run/weights/" 2>/dev/null || echo "(weights dir not created yet — saved after first epoch)"
