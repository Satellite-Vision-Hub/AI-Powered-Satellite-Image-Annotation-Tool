#!/usr/bin/env bash
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"
LOG_Y=$(ls -t "$PROJ/runs/logs/train_detector_"*.log 2>/dev/null | head -1)
LOG_S=$(ls -t "$PROJ/runs/logs/train_segmentor_"*.log 2>/dev/null | head -1)

echo "===================================================="
echo "  YOLO log: $(basename "$LOG_Y") ($(wc -c < "$LOG_Y") B)"
echo "===================================================="
tail -8 "$LOG_Y" 2>&1 | cut -c 1-260
echo
echo "===================================================="
echo "  Segmentor log: $(basename "$LOG_S") ($(wc -c < "$LOG_S") B)"
echo "===================================================="
if [ -f "$LOG_S" ]; then
  tail -15 "$LOG_S" 2>&1 | cut -c 1-260
else
  echo "(no segmentor log yet)"
fi
echo
echo "===================================================="
echo "  Container resources"
echo "===================================================="
docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' skylogic-trainer-yolo skylogic-trainer-seg 2>&1
echo
echo "===================================================="
echo "  Container statuses"
echo "===================================================="
docker ps -a --filter name=skylogic-trainer --format 'table {{.Names}}\t{{.Status}}'
