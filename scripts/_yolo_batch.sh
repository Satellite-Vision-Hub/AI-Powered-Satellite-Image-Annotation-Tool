#!/usr/bin/env bash
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"
LOG=$(ls -t "$PROJ/runs/logs/train_detector_"*.log 2>/dev/null | head -1)

# Extract numbers like " 234/2157 " from anywhere in the file (raw byte search)
echo "==== Latest 10 batch markers ===="
grep -aoE '[0-9]{1,4}/2157' "$LOG" | tail -10

echo
echo "==== Latest 4 timing markers (Xs/it or time-remaining) ===="
grep -aoE '[0-9]+\.[0-9]+s/it[^ ]*' "$LOG" | tail -4
grep -aoE '[0-9]+:[0-9]+:[0-9]+<[0-9:]+' "$LOG" | tail -4

echo
echo "==== Latest box_loss seen ===="
grep -aoE 'box_loss[^|]*' "$LOG" | tail -1 | head -c 60
echo

echo
echo "==== Last 200 bytes of log file (raw) ===="
tail -c 200 "$LOG" | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*m//g' | tail -3 | cut -c 1-200
