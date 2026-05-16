#!/usr/bin/env bash
set -e
echo "==== Wiping prior smoke-size YOLO dataset ===="
docker exec skylogic-api bash -c 'rm -rf /app/data/yolo/images /app/data/yolo/labels /app/data/yolo/skylogic.yaml /app/data/yolo/class_map.json'

echo "==== Building full YOLO dataset (all train + val patches) ===="
# --use-symlinks avoids duplicating ~17k images (saves disk + IO)
docker exec -w /app skylogic-api python scripts/build_yolo_dataset.py \
  --project-root /app \
  --limit-train 0 \
  --limit-val 0 \
  --use-symlinks 2>&1 | tail -40

echo
echo "==== Resulting dataset counts ===="
docker exec skylogic-api bash -c 'echo "train images: $(ls /app/data/yolo/images/train/ | wc -l)"; echo "val images:   $(ls /app/data/yolo/images/val/ | wc -l)"; echo "train labels: $(ls /app/data/yolo/labels/train/ | wc -l)"; echo "val labels:   $(ls /app/data/yolo/labels/val/ | wc -l)"'

echo
echo "==== skylogic.yaml ===="
docker exec skylogic-api cat /app/data/yolo/skylogic.yaml | head -10
