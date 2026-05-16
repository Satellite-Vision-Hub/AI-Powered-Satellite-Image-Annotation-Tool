#!/usr/bin/env bash
LOG="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool/runs/logs/train_detector_20260428_152007.log"

echo "==== Log size & last modified ===="
ls -la "$LOG"

echo
echo "==== Significant log lines (Epoch / Scanning / Error / Train info) ===="
grep -E "^\s*Epoch|train: New cache|val: New cache|Scanning|Image sizes|Starting training|optimizer|Plotting|Logging|Traceback|Error|ERROR|Killed|OOM|^\s*[0-9]+/[0-9]+\s" "$LOG" 2>/dev/null | tail -25

echo
echo "==== Last 8 lines of raw log ===="
tail -8 "$LOG"

echo
echo "==== Container resources ===="
docker stats --no-stream skylogic-trainer --format 'CPU={{.CPUPerc}} MEM={{.MemUsage}} BLOCK={{.BlockIO}}'
docker ps --filter name=skylogic-trainer --format '{{.Status}}'
