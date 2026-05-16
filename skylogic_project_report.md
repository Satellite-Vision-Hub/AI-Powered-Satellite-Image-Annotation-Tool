# SkyLogic MAS — AI-Powered Satellite Image Annotation Tool
## Comprehensive Project Report

**Author**: Antigravity AI Assistant  
**Date**: April 8, 2026  
**Project Location**: `d:\graduation project data\AI-Powered-Satellite-Image-Annotation-Tool`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Development Timeline](#3-development-timeline)
4. [Complete File Inventory](#4-complete-file-inventory)
5. [Detailed File Descriptions](#5-detailed-file-descriptions)
6. [Design Decisions & Rationale](#6-design-decisions--rationale)
7. [What's Missing — Steps Required to Run](#7-whats-missing--steps-required-to-run)
8. [Startup Instructions](#8-startup-instructions)

---

## 1. Project Overview

### What Was Built

**SkyLogic MAS** (Multi-Agent System) is a full-stack AI-powered platform for annotating satellite imagery using the **xView** overhead imagery dataset. The system uses three concurrent AI agents that collaborate to automatically detect and segment objects in satellite images, enabling both automated and human-in-the-loop annotation workflows.

### Core Objectives

| Objective | Description |
|---|---|
| **Automated Annotation** | Use AI models to auto-detect and segment objects in satellite imagery (60+ xView classes) |
| **Multi-Agent Architecture** | Three independent AI agents (Detector, Segmentor, Interactive SAM) that coordinate through an orchestrator |
| **Ensemble Fusion** | Combine predictions from multiple agents via Weighted Boxes Fusion (WBF) for higher accuracy |
| **Human-in-the-Loop** | Interactive annotation UI with click-to-segment, manual bounding boxes, and verification workflows |
| **Similarity Search** | SAM embeddings stored in Qdrant vector database for "click one, annotate all similar" capability |
| **xView Dataset Support** | Full pipeline for ingesting, tiling, and labeling the xView overhead imagery dataset (60+ object classes) |

### Technology Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | FastAPI (Python 3.11) |
| **Database** | SQLite (default) / PostGIS 16 (production) |
| **Vector Store** | Qdrant v1.7.4 (SAM embedding similarity search) |
| **Object Detection** | YOLOv10 (via Ultralytics) |
| **Semantic Segmentation** | SegFormer (HuggingFace) / U-Net (fallback) |
| **Interactive Segmentation** | Segment Anything Model (SAM ViT-H) |
| **Ensemble** | Weighted Boxes Fusion (WBF) |
| **Authentication** | JWT (python-jose + bcrypt) |
| **Containerization** | Docker Compose |
| **Frontend** | Vanilla HTML/CSS/JavaScript (no build step) |
| **Geo-processing** | Rasterio, GDAL, Shapely, GeoPandas |

---

## 2. System Architecture

The system follows a multi-agent architecture with five distinct layers:

```
┌─────────────────────────────────────────────┐
│                  FRONTEND                    │
│   SkyLogic Dashboard (HTML/CSS/JS)          │
│   - Login, Dashboard, Annotator, AI Agents  │
│   - Canvas-based annotation with tools      │
└─────────────────┬──────────────────────────┘
                  │ HTTP/REST
┌─────────────────▼──────────────────────────┐
│              FastAPI BACKEND (:8000)         │
│   Routes: /health, /auth, /api/patches,     │
│           /api/annotations, /api/predictions │
│           /api/sam                           │
└──┬──────────┬──────────┬───────────────────┘
   │          │          │
   ▼          ▼          ▼
┌──────┐ ┌──────┐ ┌───────────┐
│Agent │ │Agent │ │  Agent C  │
│  A   │ │  B   │ │   SAM     │
│YOLO  │ │SegFor│ │  ViT-H    │
│v10   │ │mer   │ │           │
└──┬───┘ └──┬───┘ └─────┬─────┘
   │        │           │
   ▼        ▼           ▼
┌────────────────┐ ┌──────────┐
│ WBF Ensemble   │ │  Qdrant  │
│ Fusion         │ │  Vector  │
│                │ │  Store   │
└───────┬────────┘ └──────────┘
        │
        ▼
┌────────────────┐
│   PostGIS /    │
│   SQLite       │
│   Database     │
└────────────────┘
```

### Agent Roles

| Agent | Model | Role |
|---|---|---|
| **Agent A: Detector** | YOLOv10 (Ultralytics) | Fast multi-class object detection across 60+ xView classes (vehicles, buildings, aircraft, maritime vessels, etc.) |
| **Agent B: Segmentor** | SegFormer / U-Net | Pixel-level semantic segmentation into 10 disaster-related classes (buildings, damage, floods, debris, water, vegetation, etc.) |
| **Agent C: SAM** | Segment Anything ViT-H | Interactive click-based segmentation — user clicks a point, SAM generates a precise mask. Embedding stored in Qdrant for cross-image similarity search |

### Data Flow

1. **Ingestion**: Raw xView ZIP files → Extraction → 512×512 Tiling → Label Mapping → CSV metadata
2. **Database Load**: CSV metadata → SQLAlchemy ORM → SQLite/PostGIS tables
3. **Prediction**: Patch image → Agent A + Agent B → WBF Fusion → Master Predictions → Database
4. **SAM Interactive**: User click → SAM mask → Feature embedding → Qdrant → Similarity search → Auto-annotations

---

## 3. Development Timeline

### Conversation 1: SkyLogic MAS Disaster Intelligence (March 27–28, 2026)

**Objective**: Initial architecture planning for a disaster detection system targeting Japan using USGS satellite imagery.

**What was done**:
- Designed the original project architecture under a separate `skylogic-mas` directory
- Created an implementation plan with Docker + FastAPI + PostGIS
- Planned USGS STAC API integration with `pystac-client`
- Proposed a Stacking Ensemble approach with SegFormer, U-Net, and EfficientNet
- Created scaffolding files for the disaster intelligence system
- Analyzed Stacking vs. Weighted Averaging for ensemble design

**Outcome**: The plan was fully designed and scaffolding code created. This was later pivoted to focus on xView dataset annotation instead.

### Conversation 2: Deleting Project Files (March 29, 2026)

**Objective**: Clean slate reset — delete all files from the `skylogic-mas` directory.

**What was done**: All files in the original `skylogic-mas` project were removed.

### Conversation 3: Full SkyLogic MAS Implementation (prior to current session)

**Objective**: Build the complete AI-Powered Satellite Image Annotation Tool under the `AI-Powered-Satellite-Image-Annotation-Tool` directory, pivoted to the xView dataset.

**What was done**: All 30+ files were created, implementing the full system described in this report. The system was redesigned to:
- Use xView (overhead imagery) instead of USGS disaster imagery
- Focus on annotation workflows instead of just disaster detection
- Add an interactive frontend with canvas-based annotation tools
- Add Weighted Boxes Fusion instead of Stacking
- Add SAM ViT-H with Qdrant vector similarity search

---

## 4. Complete File Inventory

### Root-Level Files (4 files)

| File | Size | Purpose |
|---|---|---|
| `docker-compose.yml` | 2,826 B | Docker stack: PostGIS + Qdrant + API |
| `Dockerfile` | 1,029 B | Python 3.11-slim with GDAL + geo tools |
| `.env.example` | 736 B | Environment variable template |
| `requirements.txt` | 1,472 B | All Python dependencies (CPU-only) |

### Backend Core — `skylogic/` (3 files)

| File | Size | Purpose |
|---|---|---|
| `__init__.py` | 119 B | Package init with version string |
| `config.py` | 2,237 B | Pydantic Settings — all configuration |
| `database.py` | 1,152 B | SQLAlchemy engine, session, init_db() |

### ORM Models — `skylogic/models/` (4 files)

| File | Size | Purpose |
|---|---|---|
| `__init__.py` | 43 B | Package init |
| `user.py` | 846 B | User model (JWT auth) |
| `patch.py` | 1,508 B | Patch model (512×512 tiles) |
| `annotation.py` | 1,885 B | Annotation model (human + AI sources) |

### API Layer — `skylogic/api/` (7 files)

| File | Size | Purpose |
|---|---|---|
| `__init__.py` | 36 B | Package init |
| `main.py` | 3,159 B | FastAPI app with CORS, lifespan, routes |
| `auth.py` | 5,012 B | JWT authentication + user management |
| `routes/__init__.py` | 43 B | Package init |
| `routes/patches.py` | 3,802 B | Patches CRUD + CSV bulk loader |
| `routes/annotations.py` | 5,468 B | Annotations CRUD + bulk operations |
| `routes/predictions.py` | 6,599 B | AI prediction endpoints (single + batch) |
| `routes/sam.py` | 6,388 B | SAM click→mask→Qdrant endpoints |

### AI Agents — `skylogic/agents/` (5 files)

| File | Size | Purpose |
|---|---|---|
| `__init__.py` | 42 B | Package init |
| `detector.py` | 6,580 B | Agent A: YOLOv10 train + predict |
| `segmentor.py` | 11,066 B | Agent B: SegFormer + U-Net fallback |
| `sam_agent.py` | 7,174 B | Agent C: SAM ViT-H click segmentation |
| `orchestrator.py` | 5,046 B | Multi-agent coordination |

### Ensemble — `skylogic/ensemble/` (2 files)

| File | Size | Purpose |
|---|---|---|
| `__init__.py` | 41 B | Package init |
| `wbf_fusion.py` | 7,450 B | WBF merge + NMS fallback |

### Data Ingestion — `skylogic/ingestion/` (4 files)

| File | Size | Purpose |
|---|---|---|
| `__init__.py` | 47 B | Package init |
| `extractor.py` | 2,381 B | ZIP extraction with progress bars |
| `tiler.py` | 8,244 B | GeoTIFF → 512×512 patches |
| `label_mapper.py` | 7,415 B | xView GeoJSON → patch-level labels |

### Vector Store — `skylogic/vector_store/` (2 files)

| File | Size | Purpose |
|---|---|---|
| `__init__.py` | 45 B | Package init |
| `qdrant_client.py` | 5,889 B | Qdrant store/search for SAM embeddings |

### Scripts — `scripts/` (1 file)

| File | Size | Purpose |
|---|---|---|
| `ingest.py` | 4,643 B | Full ingestion pipeline: Extract → Tile → Map → CSV |

### Frontend — `frontend/` (3 files)

| File | Size | Purpose |
|---|---|---|
| `index.html` | 24,717 B | Multi-view dashboard UI |
| `styles.css` | 29,680 B | Premium dark theme (600+ lines) |
| `app.js` | 27,088 B | Client-side logic: auth, canvas, API calls |

### Dataset Files (present in workspace)

| File | Size | Purpose |
|---|---|---|
| `train_images.zip` | ~14.3 GB | xView training satellite images |
| `train_labels.zip` | ~46.7 MB | xView training GeoJSON labels |
| `val_images.zip` | ~4.9 GB | xView validation images |

**Total: ~35 source code files + 3 dataset archives**

---

## 5. Detailed File Descriptions

### 5.1 Environment & Infrastructure

#### `docker-compose.yml`
**Why it was created**: The system needs three services running together — a spatial database (PostGIS) for storing annotations with geo-coordinates, a vector database (Qdrant) for SAM embedding similarity search, and a FastAPI backend. Docker Compose orchestrates all three.

**What it does**:
- Defines a `postgis` service using the `postgis/postgis:16-3.4` image on port 5432 with health checks
- Defines a `qdrant` service (v1.7.4) on ports 6333/6334 for REST and gRPC
- Defines an `api` service that builds from the Dockerfile, depends on both databases being healthy
- All services share a `skylogic-net` bridge network
- Persistent volumes for both databases

#### `Dockerfile`
**Why**: The FastAPI backend requires GDAL/GEOS/PROJ system libraries for geo-processing (rasterio, geoalchemy2). A custom Docker image ensures these are consistently installed.

**What it does**: Based on `python:3.11-slim-bookworm`, installs GDAL system deps, copies requirements and code, runs uvicorn with 2 workers.

#### `.env.example`
**Why**: Centralizes all configurable variables (database credentials, ports, JWT secrets, PyTorch device, batch sizes) in one place.

**What it does**: Template file with default values. Must be copied to `.env` before running.

#### `requirements.txt`
**Why**: Pins all 30+ Python dependencies with version constraints. Uses CPU-only PyTorch index URL for the user's AMD Ryzen 5 4500U system (no CUDA GPU).

**Key dependencies**: FastAPI, SQLAlchemy, GeoAlchemy2, PyTorch (CPU), Ultralytics (YOLO), HuggingFace Transformers, Segment Anything Model, ensemble-boxes (WBF), Qdrant client, Rasterio, OpenCV.

---

### 5.2 Configuration & Database

#### `skylogic/config.py`
**Why**: Pydantic Settings provides type-safe configuration that auto-loads from `.env` files. All paths, ports, credentials, and ML hyperparameters are centralized here.

**What it does**: Defines a `Settings` class with fields for database URL, Qdrant connection, JWT config, PyTorch device, tile size, xView class count, and seed user credentials. Provides computed properties for data directories.

#### `skylogic/database.py`
**Why**: SQLAlchemy ORM needs a central engine and session factory. The system must work with both SQLite (for local dev without Docker) and PostgreSQL/PostGIS (production).

**What it does**: Creates the engine (with SQLite's `check_same_thread=False` or PostgreSQL connection pooling), defines `SessionLocal`, provides `get_db()` FastAPI dependency, and `init_db()` to create all tables on startup.

---

### 5.3 ORM Models

#### `skylogic/models/user.py`
**Why**: JWT authentication requires a user table to store credentials.

**What it does**: Defines `User` model with `id`, `username`, `hashed_password`, `full_name`, `is_active`, `is_admin`, timestamps. Used by the auth system for login/register.

#### `skylogic/models/patch.py`
**Why**: Each satellite image is cut into 512×512 tiles (patches). The database needs to track every patch with its geo-referencing data and prediction status.

**What it does**: Defines `Patch` model with `filename`, `source_image`, `split` (train/val), `affine_transform` (JSON), `crs`, `width`, `height`, `x_offset`, `y_offset`, `is_predicted`, `prediction_count`.

#### `skylogic/models/annotation.py`
**Why**: Both human-drawn and AI-generated annotations need to be stored identically, with tracking for source (human/yolo/segformer/sam/ensemble), confidence, and verification status.

**What it does**: Defines `Annotation` model with `patch_id` (FK), `class_id`, `class_name`, `bbox` (JSON), `mask_path`, `confidence`, `source`, `geo_bbox_wkt`, `qdrant_point_id`, `created_by` (FK to users), `is_verified`.

---

### 5.4 AI Agents

#### `skylogic/agents/detector.py` — Agent A
**Why**: Object detection is the primary annotation method — quickly identifying and localizing all 60+ xView object classes (vehicles, aircraft, buildings, ships, construction equipment, etc.).

**What it does**:
- Wraps YOLOv10 (via Ultralytics) with `predict()` and `train()` methods
- Falls back to YOLOv8n pretrained weights if no custom model exists
- `predict()` returns list of `{bbox, class_id, class_name, confidence, source}` dicts
- `train()` fine-tunes on xView patches with early stopping, patience=10
- `generate_data_yaml()` creates YOLO-format dataset configuration
- CPU-only with configurable confidence/IoU thresholds

#### `skylogic/agents/segmentor.py` — Agent B
**Why**: Semantic segmentation provides pixel-level classification that detection cannot — identifying flood zones, debris fields, damaged vs. undamaged buildings as continuous regions.

**What it does**:
- Loads SegFormer from HuggingFace Transformers (MiT-B0 backbone)
- Falls back to U-Net (ResNet18 encoder from `segmentation-models-pytorch`) if Transformers unavailable
- `predict()` returns `{mask, probabilities, class_areas}` — a 2D class map and per-class pixel counts
- `mask_to_bboxes()` converts segmentation masks to bounding boxes (for WBF fusion with Agent A)
- `train()` supports fine-tuning with AdamW optimizer
- 10 segmentation classes: background, building, damaged_building, vehicle, road, vegetation, water_flood, debris_rubble, construction, container

#### `skylogic/agents/sam_agent.py` — Agent C
**Why**: Interactive annotation with SAM enables "click one, annotate many" — the user clicks on an object, SAM generates a precise boundary mask, and the embedding is used to find similar objects across the entire dataset.

**What it does**:
- Loads SAM ViT-H model from a local checkpoint file
- `set_image()` pre-computes the image embedding (expensive, cached)
- `predict_click()` generates up to 3 masks from user-provided click coordinates
- `extract_embedding()` performs masked average pooling on SAM encoder features to produce a 256-d L2-normalized embedding vector
- `mask_to_bbox()` and `mask_to_polygon()` convert binary masks to annotation coordinates
- Requires downloading the ~2.5 GB ViT-H checkpoint separately

#### `skylogic/agents/orchestrator.py`
**Why**: A central coordinator runs Agent A and Agent B together on each image, collects results, and feeds them to the ensemble fusion.

**What it does**:
- Initializes all three agents on construction
- `run_all_agents()` runs detection then segmentation sequentially (to avoid OOM on CPU), converts segmentation masks to bounding boxes
- `run_batch_prediction()` iterates over image lists with optional progress callbacks
- Logs timing for each agent execution

---

### 5.5 Ensemble Fusion

#### `skylogic/ensemble/wbf_fusion.py`
**Why**: WBF combines overlapping predictions from both agents by weighted averaging of bounding box coordinates based on confidence, producing more accurate "Master Predictions" than either agent alone.

**What it does**:
- Normalizes bounding boxes to [0,1] range for WBF processing
- Runs `weighted_boxes_fusion()` from the `ensemble-boxes` library
- YOLOv10 predictions weighted at 2.0, SegFormer at 1.0 (detector is more spatially precise)
- Falls back to simple NMS (Non-Maximum Suppression) if `ensemble-boxes` library is not installed
- Implements custom IoU computation for NMS fallback
- Returns list of `{bbox, class_id, confidence, source: "ensemble"}` dicts

---

### 5.6 Data Ingestion Pipeline

#### `skylogic/ingestion/extractor.py`
**Why**: The xView dataset comes as large ZIP archives (~20 GB total). Extraction needs progress bars and skip-if-already-done logic.

**What it does**: Extracts `train_images.zip`, `train_labels.zip`, and `val_images.zip` into `data/raw/` with `tqdm` progress bars. Skips already-extracted datasets.

#### `skylogic/ingestion/tiler.py`
**Why**: Satellite images in xView are very large (multi-thousand pixel). They must be cut into 512×512 patches for AI model input. Geo-referencing must be preserved for each patch.

**What it does**:
- Opens images as GeoTIFF via `rasterio` (preserving CRS and affine transforms)
- Falls back to standard PIL opening for non-GeoTIFF images
- Cuts with configurable overlap (default 64px) using sliding window
- Pads edge patches to maintain 512×512 dimensions
- Handles uint8 and uint16 image data
- Returns metadata dicts with filename, source_image, split, offsets, affine transform, CRS

#### `skylogic/ingestion/label_mapper.py`
**Why**: xView labels are in GeoJSON format with bounding boxes in source image coordinates. These must be mapped to patch-relative coordinates for each 512×512 tile.

**What it does**:
- Parses xView GeoJSON features (Polygon geometry or `bounds_imcoords` properties)
- Contains the full 60+ class mapping (class ID → class name) for all xView categories
- `map_labels_to_patches()` maps each label to overlapping patches, converting to patch-relative [x_min, y_min, x_max, y_max] coordinates
- Filters out labels that are clipped too small (<4px in either dimension)

#### `scripts/ingest.py`
**Why**: A single orchestration script that runs the full 4-step pipeline: Extract → Tile → Map Labels → Save CSV.

**What it does**: Calls extractor → tiler → label_mapper in sequence, saves results as CSV files in `data/metadata/`. This CSV is later loaded into the database via the API endpoint.

---

### 5.7 Vector Store

#### `skylogic/vector_store/qdrant_client.py`
**Why**: SAM generates 256-dimensional feature embeddings for each segmented object. Storing these in a vector database enables cosine-similarity search to find visually similar objects across the entire dataset.

**What it does**:
- Connects to Qdrant on startup, creates `sam_embeddings` collection if missing
- `store_embedding()` upserts a point with vector + metadata payload
- `search_similar()` performs cosine similarity search with configurable top_k and score threshold
- Gracefully handles Qdrant being unavailable (vector search simply disabled)

---

### 5.8 API Layer

#### `skylogic/api/main.py`
**Why**: The central FastAPI application connects all modules — creates database tables on startup, seeds the default user, mounts the frontend, and registers all API routers.

**What it does**:
- Lifespan handler: init_db(), seed_default_user(), create directories
- CORS middleware (allow all origins for local dev)
- Mounts `/data` for serving patch images
- Mounts `/static` for frontend files
- Registers 5 routers: patches, annotations, predictions, sam, auth
- Health check endpoint at `/health`

#### `skylogic/api/auth.py`
**Why**: Authentication prevents unauthorized access and tracks who created which annotations.

**What it does**:
- JWT token creation/verification (HS256 algorithm, configurable expiry)
- bcrypt password hashing via `passlib`
- `POST /auth/register` — new user signup
- `POST /auth/login` — returns JWT token (OAuth2 password flow)
- `GET /auth/me` — current user info
- Seeds default admin user (`SkyLogic` / `skylogic2026`) on startup

#### `skylogic/api/routes/patches.py`
**Why**: CRUD operations for patch management and a bulk CSV loader to populate the database from ingestion output.

**What it does**:
- `GET /api/patches/` — list with filtering by split and prediction status
- `GET /api/patches/count` — total count
- `GET /api/patches/{id}` — single patch
- `POST /api/patches/load-from-csv` — bulk loads patches_metadata.csv into the database

#### `skylogic/api/routes/annotations.py`
**Why**: Full CRUD for managing annotation data (both human and AI), including bulk creation for batch predictions.

**What it does**:
- `GET /api/annotations/` — list with filtering by patch_id, source, class_id
- `POST /api/annotations/` — create single annotation
- `POST /api/annotations/bulk` — bulk create (for AI batch predictions)
- `PUT /api/annotations/{id}` — edit (class, bbox, confidence, verification)
- `DELETE /api/annotations/{id}` — delete single
- `DELETE /api/annotations/patch/{patch_id}` — delete all for a patch

#### `skylogic/api/routes/predictions.py`
**Why**: Triggers the AI agent pipeline to auto-annotate patches and provides progress monitoring.

**What it does**:
- `POST /api/predictions/single` — run all agents + WBF on one patch, save results
- `POST /api/predictions/batch` — background task for batch processing (configurable limit)
- `GET /api/predictions/status` — progress: total vs. predicted patches, AI annotation count

#### `skylogic/api/routes/sam.py`
**Why**: Exposes SAM interactive segmentation as API endpoints for the frontend canvas.

**What it does**:
- `POST /api/sam/click` — receives click coordinates, generates SAM mask, extracts embedding, stores in Qdrant, optionally auto-annotates similar objects
- `POST /api/sam/search-similar` — explicit similarity search by Qdrant point ID
- `GET /api/sam/status` — reports SAM model and Qdrant availability

---

### 5.9 Frontend

#### `frontend/index.html`
**Why**: A visual dashboard for interacting with the system — monitoring status, browsing patches, annotating, and running predictions.

**What it does**: 4-view single-page app:
- **Login Modal**: JWT authentication with pre-filled default credentials
- **Dashboard**: Stats cards (patches, predicted, annotations, progress), recent patches grid, system status panel
- **Annotator**: Canvas-based annotation with toolbar (Select, Bbox, SAM Click, Delete), patch browser sidebar, annotation list
- **AI Agents**: Visual cards for each agent with model details and status badges
- **Predictions**: Batch prediction launcher with progress bar and results table

#### `frontend/styles.css`
**Why**: Premium dark-themed UI design (600+ lines) for a professional-grade look.

**What it does**: Custom CSS properties for theming, glassmorphism cards, smooth animations/transitions, responsive grid layouts, custom badges, SVG icons, agent status indicators.

#### `frontend/app.js`
**Why**: Client-side logic for interacting with all API endpoints without a build step (no React/Vue needed).

**What it does**: JWT token management, API fetch wrappers, view navigation, canvas rendering for annotations with mouse events, system status polling, prediction progress tracking, toast notifications.

---

## 6. Design Decisions & Rationale

### 6.1 CPU-Only Operation
**Decision**: All PyTorch operations use `device="cpu"` with batch_size=2 and workers=2.  
**Why**: The target system is an AMD Ryzen 5 4500U laptop without a CUDA-capable GPU. CPU-only PyTorch wheels are specified in requirements.txt to avoid downloading large CUDA binaries.

### 6.2 SQLite Default with PostGIS Option
**Decision**: The system defaults to SQLite but can switch to PostGIS via environment variable.  
**Why**: SQLite runs immediately without Docker, making development easier. PostGIS is available for production deployment when spatial queries and geo-indexing are needed.

### 6.3 No Node.js / No Build Step
**Decision**: Frontend is vanilla HTML/CSS/JS served as static files.  
**Why**: Eliminates the need for Node.js, npm, webpack, or any frontend build tooling. The frontend is served directly by FastAPI's `StaticFiles` mount.

### 6.4 SegFormer with U-Net Fallback
**Decision**: Agent B tries to load SegFormer first, automatically falls back to U-Net if the HuggingFace `transformers` library fails.  
**Why**: SegFormer is a modern transformer-based segmentor with better accuracy, but U-Net (via `segmentation-models-pytorch`) is more lightweight and always available as a reliable fallback.

### 6.5 WBF with NMS Fallback
**Decision**: Ensemble uses WBF from `ensemble-boxes`, but falls back to custom NMS if the library is unavailable.  
**Why**: WBF produces better localization by weighted averaging of box coordinates, but the system should still work without the optional dependency.

### 6.6 Lazy Agent Initialization
**Decision**: AI models are loaded only when first called (deferred imports), not on app startup.  
**Why**: Loading three large PyTorch models simultaneously could exceed memory on a 8–16 GB RAM system. Lazy loading means only the needed model is loaded.

### 6.7 CSV Intermediary for Ingestion
**Decision**: The ingestion pipeline writes CSV files to `data/metadata/` before database population.  
**Why**: Decouples ingestion from the database. You can run ingestion without PostGIS or even without the API running, then load the CSV later.

### 6.8 Weighted Boxes Fusion (WBF) over Stacking
**Decision**: WBF was chosen for the final implementation instead of the initially proposed Stacking ensemble.  
**Why**: WBF is simpler, does not require training a meta-learner, and works well for combining detection + segmentation outputs. The initial Stacking proposal was more appropriate for a fully integrated training loop, which was not the immediate priority.

---

## 7. What's Missing — Steps Required to Run

> [!CAUTION]
> The following items are required before the program can run. Without these, the system will not start or will function with limited capabilities.

### 7.1 CRITICAL: Missing Files That Must Be Created

| Item | Status | Action Needed |
|---|---|---|
| **`.env` file** | ❌ Missing | Copy `.env.example` to `.env` and update values |
| **`data/` directory** | ❌ Missing | Created automatically on first run |
| **`models/` directory** | ❌ Missing | Created automatically on first run |

### 7.2 CRITICAL: Dependencies Not Yet Installed

| Item | Status | Action Needed |
|---|---|---|
| **Python virtual environment** | ⚠️ `.venv` exists | Verify packages are installed: `pip install -r requirements.txt` |
| **PyTorch CPU** | ❌ Unknown | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| **Docker Desktop** | ❌ Unknown | Install Docker Desktop for Windows |
| **Docker Compose** | ❌ Unknown | Included with Docker Desktop |

### 7.3 CRITICAL: Services That Must Be Running

| Service | Port | How to Start |
|---|---|---|
| **PostGIS** | 5432 | `docker-compose up -d postgis` (or use SQLite default) |
| **Qdrant** | 6333 | `docker-compose up -d qdrant` (optional, SAM search won't work without it) |

### 7.4 IMPORTANT: Data Ingestion Must Be Run

| Step | Status | Command |
|---|---|---|
| Extract ZIP archives | ❌ Not done | `python scripts/ingest.py` (Step 1 of 4) |
| Tile images to 512×512 | ❌ Not done | Included in `ingest.py` (Step 2) |
| Map labels to patches | ❌ Not done | Included in `ingest.py` (Step 3) |
| Save CSV metadata | ❌ Not done | Included in `ingest.py` (Step 4) |
| Load CSV into database | ❌ Not done | `POST http://localhost:8000/api/patches/load-from-csv` |

> [!WARNING]
> The ingestion pipeline will take a **long time** (potentially hours) due to the ~20 GB dataset. The tiling step is particularly slow as it opens each large satellite image and cuts it into thousands of 512×512 patches.

### 7.5 IMPORTANT: Model Weights Not Included

| Model | Size | How to Obtain |
|---|---|---|
| **YOLOv8n pretrained** | ~6 MB | Auto-downloaded by Ultralytics on first use |
| **SegFormer (untrained)** | ~6 MB | Auto-initialized by HuggingFace Transformers |
| **SAM ViT-H checkpoint** | ~2.5 GB | Must be manually downloaded (see instructions below) |
| **Fine-tuned YOLO weights** | — | Must be trained with `detector.train()` on xView data |
| **Fine-tuned SegFormer weights** | — | Must be trained with `segmentor.train()` on xView data |

**SAM ViT-H download command**:
```bash
mkdir -p models/sam
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam/sam_vit_h_4b8939.pth
```

### 7.6 IMPORTANT: Model Training Required

The AI agents currently use pretrained weights that are **not fine-tuned on xView data**:

- **Agent A (YOLOv10)**: Uses YOLOv8n pretrained on COCO, not xView. Will detect general objects but not xView-specific classes. **Must be fine-tuned** on the tiled xView patches with YOLO-format labels.
- **Agent B (SegFormer)**: Uses randomly initialized weights (not pretrained on any satellite data). **Must be fine-tuned** on annotated segmentation masks before predictions are meaningful.
- **Agent C (SAM)**: Uses Facebook's pretrained ViT-H weights (works out-of-the-box for interactive segmentation). **No fine-tuning needed**, but requires downloading the 2.5 GB checkpoint.

### 7.7 NICE-TO-HAVE: Additional Items

| Item | Status | Notes |
|---|---|---|
| Database migrations (Alembic) | ❌ Not configured | `init_db()` creates tables via `CREATE TABLE IF NOT EXISTS`, but schema migrations need Alembic setup |
| YOLO training data YAML | ❌ Not generated | Must be generated via `detector.generate_data_yaml()` after ingestion |
| Segmentation training masks | ❌ Not created | xView labels are bounding boxes; segmentation masks would need to be generated or manually created |
| Unit tests | ❌ Not created | No test suite exists |
| Production HTTPS | ❌ Not configured | CORS allows all origins; no TLS/SSL setup |
| Logging to file | ❌ Not configured | Logs go to stdout only |
| `README.md` content | ⚠️ Minimal | Only contains the repository title, not setup instructions |

---

## 8. Startup Instructions

### Quick Start (SQLite Mode — No Docker Required)

```bash
# 1. Navigate to project
cd "d:\graduation project data\AI-Powered-Satellite-Image-Annotation-Tool"

# 2. Copy environment file
copy .env.example .env

# 3. Activate virtual environment
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run data ingestion (SLOW — 20 GB dataset)
python scripts\ingest.py

# 6. Start the API server
uvicorn skylogic.api.main:app --host 0.0.0.0 --port 8000 --reload

# 7. Load patches metadata into database
# Open another terminal and run:
curl -X POST http://localhost:8000/api/patches/load-from-csv

# 8. Open browser
# Navigate to http://localhost:8000
# Login: SkyLogic / skylogic2026
```

### Full Docker Mode (PostGIS + Qdrant)

```bash
# 1. Copy and edit environment file
copy .env.example .env

# 2. Start all services
docker-compose up -d

# 3. Run ingestion (from host, pointing to Docker DB)
python scripts\ingest.py

# 4. Load patches into PostGIS
curl -X POST http://localhost:8000/api/patches/load-from-csv

# 5. Open http://localhost:8000
```

---

*End of Report*
