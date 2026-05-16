#!/usr/bin/env bash
# Show only the LAST 25 lines of the YOLO training log + container metrics.
LOG=$(ls -t "/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool/runs/logs/train_detector_"*.log 2>/dev/null | head -1)
echo "LOG=$LOG"
echo "size=$(wc -c < "$LOG") bytes  lines=$(wc -l < "$LOG")"
echo
echo "==== Tail (last 30 lines, with long lines truncated) ===="
tail -30 "$LOG" | cut -c 1-300
echo
echo "==== Container ===="
docker ps --filter name=skylogic-trainer --format '{{.Status}}'
docker stats --no-stream skylogic-trainer --format 'CPU={{.CPUPerc}} MEM={{.MemUsage}}'
