#!/usr/bin/env bash
set -e
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"

echo "==== 1. Ensure runs/ dir exists on host (for log files) ===="
mkdir -p "$PROJ/runs/logs"

echo "==== 2. Stop API service to free SAM memory ===="
cd "$PROJ" && docker compose -p skylogic stop api
sleep 2

echo "==== 3. Memory after stop ===="
free -m | head -2

echo "==== 4. Remove any stale trainer container ===="
docker rm -f skylogic-trainer 2>/dev/null || true

echo "==== 5. Launch detached training container (YOLO 50 ep -> Seg 20 ep) ===="
docker run -d \
  --name skylogic-trainer \
  --restart no \
  -v "$PROJ/data":/app/data \
  -v "$PROJ/models":/app/models \
  -v "$PROJ/runs":/app/runs \
  -v "$PROJ/scripts":/app/scripts:ro \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  --memory=10g \
  skylogic-api \
  bash /app/scripts/_train_all.sh

echo "==== 6. Initial container state ===="
sleep 3
docker ps --filter name=skylogic-trainer --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"

echo "==== 7. Initial log preview (15s wait then tail) ===="
sleep 15
docker logs skylogic-trainer 2>&1 | tail -20
echo
echo "==== 8. Disk: training log files written so far ===="
ls -la "$PROJ/runs/logs/" 2>&1 || true
