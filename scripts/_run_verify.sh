#!/usr/bin/env bash
set -u
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"
cd "$PROJ" || { echo "FAILED to cd into project"; exit 1; }
echo "==== PWD ===="
pwd
echo
echo "==== docker compose ps ===="
docker compose -p skylogic ps
echo
echo "==== Running e2e verifier ===="
bash scripts/_e2e_verify.sh
