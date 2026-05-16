#!/usr/bin/env bash
echo "==== Python process: CPU/state via /proc ===="
PID=$(docker exec skylogic-trainer-yolo bash -c 'ls /proc | grep -E "^[0-9]+$" | head -10')
echo "pids in container: $PID"
docker exec skylogic-trainer-yolo bash -c 'for p in $(pgrep -f train_detector); do echo "--- PID $p ---"; cat /proc/$p/status 2>/dev/null | grep -E "State|Threads|VmRSS"; cat /proc/$p/wchan 2>/dev/null; echo; done'

echo
echo "==== py-spy/strace on python ===="
docker exec skylogic-trainer-yolo bash -c 'pgrep -f train_detector' || true
echo "(no py-spy — using stack peek)"
docker exec skylogic-trainer-yolo bash -c 'PID=$(pgrep -f train_detector | head -1); cat /proc/$PID/stack 2>/dev/null || echo "no stack readable"'

echo
echo "==== Filesystem check: yolov8n.pt cache ===="
docker exec skylogic-trainer-yolo find / -name "yolov8n.pt" -size +1M 2>/dev/null | head -5

echo
echo "==== HF/network: any active connections? ===="
docker exec skylogic-trainer-yolo bash -c 'ss -tnp 2>/dev/null | head -20 || cat /proc/net/tcp | head -20'

echo
echo "==== Recent file activity in workdir ===="
docker exec skylogic-trainer-yolo bash -c 'find /app -type f -newer /app/scripts -mmin -3 2>/dev/null | head -20'
