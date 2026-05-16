#!/usr/bin/env bash
set -u
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"

echo "==== 1. Stop sequential trainer (caches saved → restart will be instant) ===="
docker rm -f skylogic-trainer 2>/dev/null || true
docker rm -f skylogic-trainer-yolo 2>/dev/null || true
docker rm -f skylogic-trainer-seg 2>/dev/null || true
sleep 2

echo "==== 2. Verify dataset caches survived ===="
ls -lh "$PROJ/data/yolo/labels/"*.cache 2>&1 | head -5

echo "==== 3. Stop API (free SAM RAM if it was somehow running) ===="
cd "$PROJ" && docker compose -p skylogic stop api 2>&1 | tail -1

echo "==== 4. Memory snapshot before launching parallel trainers ===="
free -m | head -2

echo
echo "==== 5. Launch YOLO trainer (8 cores, 4 GB cap) ===="
docker run -d \
  --name skylogic-trainer-yolo \
  --restart no \
  --cpus=2 \
  --memory=4g \
  -v "$PROJ/data":/app/data \
  -v "$PROJ/models":/app/models \
  -v "$PROJ/runs":/app/runs \
  -v "$PROJ/scripts":/app/scripts:ro \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  --no-healthcheck \
  --entrypoint bash \
  skylogic-api \
  /app/scripts/_train_yolo_only.sh

echo
echo "==== 6. Launch Segmentor trainer (8 cores, 4 GB cap) ===="
docker run -d \
  --name skylogic-trainer-seg \
  --restart no \
  --cpus=2 \
  --memory=4g \
  -v "$PROJ/data":/app/data \
  -v "$PROJ/models":/app/models \
  -v "$PROJ/runs":/app/runs \
  -v "$PROJ/scripts":/app/scripts:ro \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  --no-healthcheck \
  --entrypoint bash \
  skylogic-api \
  /app/scripts/_train_seg_only.sh

echo
echo "==== 7. Container state ===="
sleep 4
docker ps --filter name=skylogic-trainer --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"

echo
echo "==== 8. Memory after both launched ===="
free -m | head -2
