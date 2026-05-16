#!/usr/bin/env bash
set -u
BASE="${BASE:-http://localhost:8000}"

echo "==== Pick a real patch ===="
read PID IMG_URL <<< $(curl -fsS "$BASE/api/patches/?split=train&limit=1" | python3 -c "import sys,json; r=json.load(sys.stdin)[0]; print(r['id'], r['image_url'])")
# Strip leading "/data/" so the route can prepend settings.data_dir (=/app/data)
IMG_PATH="${IMG_URL#/data/}"
echo "patch_id=$PID"
echo "image_url=$IMG_URL"
echo "image_path=$IMG_PATH"

echo
echo "==== docker stats BEFORE ===="
docker stats --no-stream --format "{{.Name}}  CPU={{.CPUPerc}}  MEM={{.MemUsage}}" skylogic-api

echo
echo "==== POST /api/sam/click (this took the OOM hit at 8 GB) ===="
START=$(date +%s)
RESP=$(curl -s -w "\n__HTTP=%{http_code}__\n" -X POST "$BASE/api/sam/click" \
  -H "Content-Type: application/json" \
  -d "{\"patch_id\": $PID, \"image_path\": \"$IMG_PATH\", \"clicks\": [[256, 256]], \"labels\": [1], \"auto_annotate\": false}")
END=$(date +%s)
echo "elapsed: $((END-START))s"
echo "$RESP" | head -c 1200; echo

echo
echo "==== docker stats AFTER ===="
docker stats --no-stream --format "{{.Name}}  CPU={{.CPUPerc}}  MEM={{.MemUsage}}" skylogic-api

echo
echo "==== Qdrant point count after click ===="
curl -fsS "$BASE/api/sam/status" | python3 -m json.tool
