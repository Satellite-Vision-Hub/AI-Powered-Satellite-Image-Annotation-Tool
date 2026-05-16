#!/usr/bin/env bash
echo "=== Container state ==="
docker ps -a --filter name=skylogic-trainer-yolo --format "table {{.Names}}\t{{.Status}}"
docker stats --no-stream skylogic-trainer-yolo --format "CPU={{.CPUPerc}} MEM={{.MemUsage}}" 2>/dev/null || true

echo
echo "=== Stdout (filtered, last 30 lines) ==="
docker logs skylogic-trainer-yolo 2>&1 | grep -av "^\[K" | grep -av "^$" | tail -30 | cut -c1-200

echo
echo "=== Key signals ==="
docker logs skylogic-trainer-yolo 2>&1 | grep -a -iE "annotated|cls_pw|fl_gamma|AMP|GPU_mem|Starting training for|Epoch|Error|Traceback|Killed|exit|Scanning" | grep -av "^\[K" | tail -20 | cut -c1-200
