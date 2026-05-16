#!/usr/bin/env bash
# Quick "is training still healthy?" status — run anytime from WSL.
# Usage:  bash scripts/watch_training.sh
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"
cd "$PROJ" 2>/dev/null || true

BLUE='\033[1;36m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; NC='\033[0m'

echo -e "${BLUE}=== Containers ===${NC}"
docker ps -a --filter name=skylogic-trainer --format 'table {{.Names}}\t{{.Status}}'

echo
echo -e "${BLUE}=== Resources ===${NC}"
docker stats --no-stream \
  --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' \
  skylogic-trainer-yolo skylogic-trainer-seg 2>/dev/null

echo
echo -e "${BLUE}=== YOLO progress (last 6 lines) ===${NC}"
LOG_Y=$(ls -t "$PROJ/runs/logs/train_detector_"*.log 2>/dev/null | head -1)
if [ -n "$LOG_Y" ]; then
  echo "log: $(basename "$LOG_Y")"
  tail -6 "$LOG_Y" | sed 's/\x1b\[[0-9;]*m//g' | cut -c 1-200
fi

echo
echo -e "${BLUE}=== Segmentor progress (last 6 lines) ===${NC}"
LOG_S=$(ls -t "$PROJ/runs/logs/train_segmentor_"*.log 2>/dev/null | head -1)
if [ -n "$LOG_S" ]; then
  echo "log: $(basename "$LOG_S")"
  tail -6 "$LOG_S" | sed 's/\x1b\[[0-9;]*m//g' | cut -c 1-200
fi

echo
echo -e "${BLUE}=== Saved checkpoints ===${NC}"
echo -e "${YELLOW}YOLO:${NC}"
ls -lh "$PROJ/models/yolo/" 2>/dev/null | grep -E "\.(pt|pth)$"
ls -lh "$PROJ/models/yolo/xview_full_run/weights/" 2>/dev/null | grep -E "\.(pt|pth)$"
echo -e "${YELLOW}Segmentor:${NC}"
ls -lh "$PROJ/models/segformer/" 2>/dev/null | grep -E "\.(pt|pth)$"

echo
echo -e "${BLUE}=== Host RAM ===${NC}"
free -m | head -2
