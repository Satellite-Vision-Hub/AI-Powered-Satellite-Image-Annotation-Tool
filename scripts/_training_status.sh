#!/usr/bin/env bash
# Read-only training status check
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"

echo "======================================================"
echo "1. CONTAINER STATUS"
echo "======================================================"
docker ps -a --filter name=skylogic --format "table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.RunningFor}}"

echo
echo "======================================================"
echo "2. TRAINER RESOURCE USAGE"
echo "======================================================"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.BlockIO}}" 2>/dev/null | grep -E "NAME|trainer" || echo "(no trainer running)"

echo
echo "======================================================"
echo "3. YOLO TRAINING LOG — key lines only"
echo "======================================================"
LOG=$(ls -t "$PROJ/runs/logs/train_detector_"*.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
  echo "File: $LOG  ($(wc -l < "$LOG") lines, $(du -sh "$LOG" | cut -f1))"
  echo
  echo "--- First significant lines ---"
  grep -a -m1 "Starting training for" "$LOG" | cut -c1-120
  echo
  echo "--- Latest epoch progress ---"
  grep -a -E "^\s+[0-9]+/[0-9]+\s" "$LOG" | tail -5 | cut -c1-200
  echo
  echo "--- Last 10 raw lines (truncated at 200 chars) ---"
  tail -10 "$LOG" | cut -c1-200
  echo
  echo "--- Any errors ---"
  grep -a -iE "error|traceback|killed|oom|exception|out of memory|no space" "$LOG" | grep -v "WARNING" | tail -10 | cut -c1-200
else
  echo "No YOLO log found"
fi

echo
echo "======================================================"
echo "4. SEGMENTOR TRAINING LOG — key lines only"
echo "======================================================"
SLOG=$(ls -t "$PROJ/runs/logs/train_segmentor_"*.log 2>/dev/null | head -1)
if [ -n "$SLOG" ]; then
  echo "File: $SLOG  ($(wc -l < "$SLOG") lines, $(du -sh "$SLOG" | cut -f1))"
  grep -a -E "Epoch|epoch|loss|error|Error|Killed|OOM" "$SLOG" | tail -10 | cut -c1-180
else
  echo "No segmentor log found yet (runs after YOLO completes)"
fi

echo
echo "======================================================"
echo "5. DISK SPACE"
echo "======================================================"
echo "--- WSL root filesystem ---"
df -h / 2>/dev/null | tail -1
echo
echo "--- Windows D: drive (where project lives) ---"
df -h "/mnt/d" 2>/dev/null | tail -1
echo
echo "--- Top consumers in project/models ---"
du -sh "$PROJ/models/"* 2>/dev/null | sort -rh | head -10
echo
echo "--- Top consumers in project/runs ---"
du -sh "$PROJ/runs/"* 2>/dev/null | sort -rh | head -5

echo
echo "======================================================"
echo "6. CHECKPOINT FILES"
echo "======================================================"
echo "--- YOLO checkpoints ---"
find "$PROJ/models/yolo" -name "*.pt" -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}'
echo
echo "--- Segmentor checkpoints ---"
find "$PROJ/models/segformer" -name "*.pth" -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}'

echo
echo "======================================================"
echo "7. TRAINING SUMMARY FILE"
echo "======================================================"
cat "$PROJ/runs/logs/SUMMARY.txt" 2>/dev/null || echo "(no summary yet)"
