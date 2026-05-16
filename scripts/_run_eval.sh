#!/usr/bin/env bash
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"

echo "Cleaning up any previous eval container..."
docker rm -f skylogic-eval 2>/dev/null || true

echo "Launching evaluation container..."
docker run --rm \
  --name skylogic-eval \
  -v "$PROJ/data":/app/data \
  -v "$PROJ/models":/app/models \
  -v "$PROJ/scripts":/app/scripts \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  --memory=6g \
  skylogic-api \
  python scripts/evaluate_models.py \
    --project-root /app \
    --seg-samples 500 \
    --yolo-samples 200 \
    --imgsz 512
