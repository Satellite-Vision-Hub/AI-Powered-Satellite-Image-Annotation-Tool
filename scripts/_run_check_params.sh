#!/usr/bin/env bash
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"
docker run --rm \
  --entrypoint python \
  -v "$PROJ/scripts":/app/scripts \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  skylogic-api \
  scripts/_check_valid_params.py 2>/dev/null
