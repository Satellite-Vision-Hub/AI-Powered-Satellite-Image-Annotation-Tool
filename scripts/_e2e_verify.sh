#!/usr/bin/env bash
# End-to-end smoke verification for the SkyLogic stack.
set -u
BASE="${BASE:-http://localhost:8000}"
pass=0; fail=0
check() { if [ "$1" -eq 0 ]; then echo "  PASS  $2"; pass=$((pass+1)); else echo "  FAIL  $2"; fail=$((fail+1)); fi; }

echo "=== 1. /health ==="
H=$(curl -fsS "$BASE/health")
echo "$H"
echo "$H" | grep -q '"status":"healthy"'; check $? "/health returns healthy"

echo
echo "=== 2. /api/patches/count ==="
PC=$(curl -fsS "$BASE/api/patches/count")
echo "$PC"
echo "$PC" | grep -qE '"count": *[0-9]+'; check $? "patches count present"

echo
echo "=== 3. /api/patches/?limit=2 ==="
P=$(curl -fsS "$BASE/api/patches/?limit=2")
echo "$P" | head -c 400; echo
echo "$P" | grep -q '"image_url"'; check $? "patch listing returns image_url"

echo
echo "=== 4. /auth/login ==="
LOGIN=$(curl -fsS -X POST -d 'username=SkyLogic&password=skylogic2026' "$BASE/auth/login")
echo "$LOGIN" | head -c 250; echo
echo "$LOGIN" | grep -q '"access_token"'; check $? "login returns token"
TOKEN=$(echo "$LOGIN" | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))")

echo
echo "=== 5. /auth/me ==="
ME=$(curl -fsS "$BASE/auth/me" -H "Authorization: Bearer $TOKEN")
echo "$ME"
echo "$ME" | grep -q '"username":"SkyLogic"'; check $? "/auth/me identifies seeded user"

echo
echo "=== 6. /api/sam/status ==="
S=$(curl -fsS "$BASE/api/sam/status")
echo "$S"
echo "$S" | grep -q '"sam_loaded": *true'; check $? "SAM ViT-H loaded"
echo "$S" | grep -qE '"status": *"connected"'; check $? "Qdrant connected"

echo
echo "=== 7. /api/predictions/status ==="
PS=$(curl -fsS "$BASE/api/predictions/status")
echo "$PS"
echo "$PS" | grep -qE '"total_patches": *[0-9]+'; check $? "predictions/status has totals"

echo
echo "=== 8. Frontend HTML at / ==="
HTTP=$(curl -s -o /tmp/_index.html -w "%{http_code}" "$BASE/")
SIZE=$(wc -c < /tmp/_index.html)
echo "HTTP=$HTTP  size=$SIZE"
[ "$HTTP" = "200" ]; check $? "frontend index served"
grep -q "SkyLogic" /tmp/_index.html; check $? "index contains 'SkyLogic'"

echo
echo "=== 9. Single-patch prediction (real inference through trained YOLO + U-Net) ==="
# Pick first train patch from the DB
PID=$(curl -fsS "$BASE/api/patches/?split=train&limit=1" | python3 -c "import sys,json; rows=json.load(sys.stdin); print(rows[0]['id'] if rows else 0)")
echo "patch_id=$PID"
if [ "$PID" -gt 0 ]; then
  PRED=$(curl -fsS -X POST -H 'Content-Type: application/json' \
    -d "{\"patch_id\": $PID, \"use_ensemble\": true}" \
    "$BASE/api/predictions/single")
  echo "$PRED" | head -c 800; echo
  echo "$PRED" | grep -q '"saved"'; check $? "prediction completed and saved"
else
  echo "  SKIP  no patches in DB"
fi

echo
echo "============================================================"
echo "  PASSED: $pass    FAILED: $fail"
echo "============================================================"
[ "$fail" -eq 0 ]
