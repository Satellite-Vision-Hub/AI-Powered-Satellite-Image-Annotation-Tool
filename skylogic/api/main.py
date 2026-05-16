"""
SkyLogic MAS — FastAPI Application Entry
Main application with CORS, startup events, and route registration.
Runs standalone with SQLite — no Docker needed.
"""

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from skylogic.config import settings
from skylogic.database import init_db, SessionLocal
from skylogic.api.auth import seed_default_user

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("=" * 50)
    logger.info("SkyLogic MAS — Starting Up")
    logger.info(f"Database: {settings.database_url}")
    logger.info(f"Device: {settings.torch_device}")
    logger.info("=" * 50)

    # Initialize database tables
    init_db()
    logger.info("Database tables created")

    # Seed default user
    db = SessionLocal()
    try:
        seed_default_user(db)
    finally:
        db.close()

    # Ensure data directories exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.patches_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    yield

    logger.info("SkyLogic MAS — Shutting Down")


app = FastAPI(
    title="SkyLogic MAS",
    description=(
        "Multi-Agent Satellite Disaster Intelligence System. "
        "AI-powered annotation with YOLOv10, SegFormer, and SAM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static data directory for serving patch images
data_dir = str(settings.data_dir)
os.makedirs(data_dir, exist_ok=True)
app.mount("/data", StaticFiles(directory=data_dir), name="data")

# Mount frontend static files
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="frontend")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))


# --- Health Check ---
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "service": "SkyLogic MAS",
        "version": "1.0.0",
        "device": settings.torch_device,
        "database": "sqlite" if settings.is_sqlite else "postgresql",
    }


# --- Register Routers ---
from skylogic.api.routes import patches, annotations, predictions, sam  # noqa: E402

app.include_router(patches.router)
app.include_router(annotations.router)
app.include_router(predictions.router)
app.include_router(sam.router)

from skylogic.api.auth import router as auth_router  # noqa: E402
app.include_router(auth_router)
