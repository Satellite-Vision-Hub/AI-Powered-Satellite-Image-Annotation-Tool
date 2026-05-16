#!/usr/bin/env bash
# Launch YOLO v2 training: crash-safe + class-imbalance fixes
set -u
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"
mkdir -p "$PROJ/runs/logs"

echo "==== 1. Kill any existing YOLO trainer ===="
docker rm -f skylogic-trainer-yolo 2>/dev/null || true

echo
echo "==== 2. Memory before launch ===="
free -m | head -2

echo
echo "==== 3. Launching YOLO v2 training container ===="
docker run -d \
  --name skylogic-trainer-yolo \
  --restart no \
  -v "$PROJ/data":/app/data \
  -v "$PROJ/models":/app/models \
  -v "$PROJ/runs":/app/runs \
  -v "$PROJ/scripts":/app/scripts:ro \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  --memory=8g \
  --memory-swap=10g \
  skylogic-api \
  python -u scripts/train_detector.py \
    --project-root /app \
    --data /app/data/yolo/skylogic.yaml \
    --model yolov8n.pt \
    --epochs 50 \
    --imgsz 416 \
    --batch 4 \
    --amp-off \
    --workers 0 \
    --cls-pw 2.0 \
    --cls-gain 1.5 \
    --annotated-only \
    --lr0 0.01 \
    --lrf 0.01 \
    --patience 15 \
    --save-period 5 \
    --project /app/models/yolo \
    --name xview_v2

CSTATUS=$?
if [ $CSTATUS -ne 0 ]; then
  echo "ERROR: docker run failed (exit $CSTATUS)"
  exit 1
fi

echo
echo "==== 4. Waiting 30s for startup ===="
sleep 30

echo
echo "==== 5. Container state ===="
docker ps --filter name=skylogic-trainer-yolo --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
docker stats --no-stream skylogic-trainer-yolo --format "CPU={{.CPUPerc}} MEM={{.MemUsage}}"

echo
echo "==== 6. Initial stdout (entrypoint + training config) ===="
docker logs skylogic-trainer-yolo 2>&1 | grep -v "^\[K" | head -35

echo
echo "==== 7. Waiting another 90s for dataset scanning to complete ===="
sleep 90

echo
echo "==== 8. Training log — first epoch beginning? ===="
docker logs skylogic-trainer-yolo 2>&1 | grep -a -E "Epoch|GPU_mem|AMP|annotated|fl_gamma|cls_pw|SIGSEGV|Traceback|Error|Killed|Starting training" | tail -15
docker stats --no-stream skylogic-trainer-yolo --format "CPU={{.CPUPerc}} MEM={{.MemUsage}}"

echo
echo "==== DONE — container running in background ===="
echo "Monitor with:"
echo "  docker logs -f skylogic-trainer-yolo 2>&1 | grep -a -E 'Epoch|GPU_mem|Error'"
echo "  docker stats skylogic-trainer-yolo"
