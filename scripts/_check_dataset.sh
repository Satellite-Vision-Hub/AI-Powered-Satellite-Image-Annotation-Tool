#!/usr/bin/env bash
echo "==== Train images ===="
docker exec skylogic-api bash -c 'ls /app/data/yolo/images/train/ | wc -l'
echo "==== Val images ===="
docker exec skylogic-api bash -c 'ls /app/data/yolo/images/val/ | wc -l'
echo "==== Train labels ===="
docker exec skylogic-api bash -c 'ls /app/data/yolo/labels/train/ | wc -l'
echo "==== Val labels ===="
docker exec skylogic-api bash -c 'ls /app/data/yolo/labels/val/ | wc -l'
echo "==== Patches dir totals (used by segmentor) ===="
docker exec skylogic-api bash -c 'ls /app/data/patches/train/ | wc -l'
docker exec skylogic-api bash -c 'ls /app/data/patches/val/ 2>&1 | head -1'
echo "==== Annotations CSV (used by segmentor) ===="
docker exec skylogic-api bash -c 'ls -la /app/data/metadata/'
