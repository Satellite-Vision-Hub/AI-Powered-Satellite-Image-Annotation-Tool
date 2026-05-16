#!/usr/bin/env bash
PROJ="/mnt/d/EGSA/last semester/Graduation Project/graduation project data/graduation project data/AI-Powered-Satellite-Image-Annotation-Tool"
docker run --rm \
  --entrypoint python \
  -v "$PROJ/data":/app/data \
  -v "$PROJ/models":/app/models \
  -v "$PROJ/scripts":/app/scripts \
  -w /app \
  -e PYTHONUNBUFFERED=1 \
  skylogic-api \
  scripts/_test_kwargs.py 2>&1
