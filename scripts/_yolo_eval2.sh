#!/usr/bin/env bash
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"
OUT="$PROJ/runs/logs/yolo_eval_$(date +%Y%m%d_%H%M%S).log"

docker run --rm \
  --name skylogic-yolo-eval \
  -v "$PROJ/data":/app/data \
  -v "$PROJ/models":/app/models \
  -v "$PROJ/runs":/app/runs \
  -v "$PROJ/scripts":/app/scripts \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  --memory=4g \
  skylogic-api \
  python scripts/evaluate_models.py \
    --project-root /app \
    --skip-seg \
    --yolo-samples 200 \
    --imgsz 512 > "$OUT" 2>&1

echo "Exit=$?"
echo "Output file: $OUT"
cat "$OUT" | grep -v "^\[K" | grep -v "^$" | tail -50
