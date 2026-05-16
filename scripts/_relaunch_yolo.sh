#!/usr/bin/env bash
set -u
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"

echo "==== Killing stalled YOLO trainer ===="
docker rm -f skylogic-trainer-yolo 2>/dev/null || true

echo "==== Relaunching YOLO with explicit OMP_NUM_THREADS=2, no cpu cap ===="
docker run -d \
  --name skylogic-trainer-yolo \
  --restart no \
  --memory=4g \
  -v "$PROJ/data":/app/data \
  -v "$PROJ/models":/app/models \
  -v "$PROJ/runs":/app/runs \
  -v "$PROJ/scripts":/app/scripts:ro \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 \
  -e OPENBLAS_NUM_THREADS=2 \
  --no-healthcheck \
  --entrypoint bash \
  skylogic-api \
  /app/scripts/_train_yolo_only.sh

sleep 4
echo
echo "==== Container state ===="
docker ps --filter name=skylogic-trainer --format 'table {{.Names}}\t{{.Status}}'
