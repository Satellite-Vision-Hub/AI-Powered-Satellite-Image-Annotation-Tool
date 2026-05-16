# ============================================================
# SkyLogic MAS — FastAPI Dockerfile (CPU-Only)
# Layered pip install so torch/AI deps cache separately from
# small/transient deps. Reduces re-build cost massively.
# ============================================================

FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libgeos-dev libproj-dev libspatialindex-dev \
    libgl1 libglib2.0-0 \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

# --- Heavy AI deps in their own layer (cache key) ---
# CPU-only torch from PyTorch's index
RUN pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2.0" "torchvision>=0.17.0"

# --- AI ecosystem (won't change often) ---
RUN pip install \
        "ultralytics>=8.3.0,<8.5" \
        "transformers>=4.47.0" \
        "segmentation-models-pytorch>=0.3.4" \
        "ensemble-boxes>=1.0.9" \
        "qdrant-client>=1.12.0" \
        "git+https://github.com/facebookresearch/segment-anything.git"

# --- App-level deps from requirements.txt (small + flexible) ---
COPY requirements.txt .
RUN pip install -r requirements.txt

# --- Application code ---
COPY skylogic/ ./skylogic/
COPY scripts/ ./scripts/
COPY frontend/ ./frontend/
COPY .env.example ./.env.example
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /app/data /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "skylogic.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
