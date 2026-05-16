#!/usr/bin/env bash
set -u
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"

echo "==== 1. Bring stack down ===="
cd "$PROJ" && docker compose -p skylogic down

echo
echo "==== 2. Wipe stub /tmp/skylogic via privileged container ===="
docker run --rm -v /tmp:/host_tmp alpine sh -c 'rm -rf /host_tmp/skylogic && echo wiped'
ls -la /tmp/ | grep skylogic || echo "(/tmp/skylogic gone)"

echo
echo "==== 3. Create real symlink ===="
ln -sfn "$PROJ" /tmp/skylogic
ls -la /tmp/skylogic

echo
echo "==== 4. Bring stack back up from /tmp/skylogic ===="
cd /tmp/skylogic && docker compose -p skylogic up -d

echo
echo "==== 5. Wait for healthy (poll up to 3 minutes) ===="
for i in $(seq 1 36); do
  status=$(docker compose -p skylogic ps --format '{{.Status}}' | tr '\n' '|')
  echo "  [$i] $status"
  if echo "$status" | grep -qv 'healthy'; then
    if echo "$status" | grep -q 'unhealthy\|Exit\|Restarting'; then
      echo "  !! container in bad state"
    fi
  fi
  unhealthy=$(docker compose -p skylogic ps --format '{{.Status}}' | grep -vc 'healthy')
  if [ "$unhealthy" -eq 0 ]; then
    echo "  ALL HEALTHY"
    break
  fi
  sleep 5
done

echo
echo "==== 6. Final compose ps ===="
docker compose -p skylogic ps

echo
echo "==== 7. Verify mounts now contain data ===="
docker exec skylogic-api ls /app/models/sam/ 2>&1 | head -5
docker exec skylogic-api ls /app/data/patches/train/ 2>&1 | head -5

echo
echo "==== 8. Run e2e verifier ===="
bash scripts/_e2e_verify.sh
