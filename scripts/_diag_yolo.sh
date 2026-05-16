#!/usr/bin/env bash
echo "==== YOLO docker logs (last 30 lines, ANSI stripped, truncated) ===="
docker logs --tail 30 skylogic-trainer-yolo 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | cut -c 1-260

echo
echo "==== Processes in YOLO container ===="
docker top skylogic-trainer-yolo 2>&1 | head -20

echo
echo "==== Resource state ===="
docker stats --no-stream skylogic-trainer-yolo --format 'CPU={{.CPUPerc}} MEM={{.MemUsage}} BLOCK={{.BlockIO}} PIDS={{.PIDs}}'

echo
echo "==== Tail of yolo log ===="
LOG=$(ls -t "/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool/runs/logs/train_detector_"*.log 2>/dev/null | head -1)
tail -5 "$LOG" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | cut -c 1-260
