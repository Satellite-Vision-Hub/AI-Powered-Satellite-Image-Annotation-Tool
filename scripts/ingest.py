"""
SkyLogic MAS — Data Ingestion Orchestration Script
Runs the full pipeline: Extract → Tile → Map Labels → Populate DB.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from skylogic.config import settings
from skylogic.ingestion.extractor import extract_all_datasets
from skylogic.ingestion.tiler import TilingEngine
from skylogic.ingestion.label_mapper import (
    load_xview_geojson,
    parse_feature_bbox,
    map_labels_to_patches,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("skylogic.ingest")


def run_ingestion():
    """Execute the full data ingestion pipeline."""

    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Extract ZIPs ──────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Extracting dataset archives")
    logger.info("=" * 60)
    extracted = extract_all_datasets(settings.project_root, data_dir)
    logger.info(f"Extracted datasets: {list(extracted.keys())}")

    # ── Step 2: Tile images ───────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Tiling satellite images into 512x512 patches")
    logger.info("=" * 60)

    import pandas as pd

    patches_dir = settings.patches_dir
    patches_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = data_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    existing_meta_csv = meta_dir / "patches_metadata.csv"
    all_patches_meta = []

    # ── Fast path: reuse existing patch metadata if available ──
    if existing_meta_csv.exists() and any(patches_dir.rglob("*.png")):
        logger.info("Existing patches detected — loading metadata (skip re-tiling)")
        df = pd.read_csv(existing_meta_csv)
        all_patches_meta = df.to_dict("records")
        logger.info(f"Loaded {len(all_patches_meta)} patch records from cache")
    else:
        # ── Full tiling ──
        tiler = TilingEngine(
            tile_size=settings.tile_size,
            overlap=settings.tile_overlap,
        )

        # Tile training images
        if "train_images" in extracted:
            train_patches = tiler.tile_directory(
                images_dir=extracted["train_images"],
                output_dir=patches_dir / "train",
                split="train",
            )
            all_patches_meta.extend(train_patches)
            logger.info(f"Train patches: {len(train_patches)}")

        # Tile validation images
        if "val_images" in extracted:
            val_patches = tiler.tile_directory(
                images_dir=extracted["val_images"],
                output_dir=patches_dir / "val",
                split="val",
            )
            all_patches_meta.extend(val_patches)
            logger.info(f"Val patches: {len(val_patches)}")

    # ── Step 3: Map labels to patches ─────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Mapping xView labels to patches")
    logger.info("=" * 60)

    annotations = []
    if "train_labels" in extracted:
        features = load_xview_geojson(extracted["train_labels"])
        parsed_labels = []
        for feat in features:
            parsed = parse_feature_bbox(feat)
            if parsed:
                parsed_labels.append(parsed)

        logger.info(f"Parsed {len(parsed_labels)} labels from GeoJSON")

        # Map to train patches only
        train_patches_meta = [p for p in all_patches_meta if p["split"] == "train"]
        annotations = map_labels_to_patches(
            parsed_labels, train_patches_meta, tile_size=settings.tile_size
        )

    # ── Step 4: Save metadata ─────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Saving metadata")
    logger.info("=" * 60)


    if all_patches_meta:
        patches_df = pd.DataFrame(all_patches_meta)
        patches_csv = meta_dir / "patches_metadata.csv"
        patches_df.to_csv(patches_csv, index=False)
        logger.info(f"Saved {len(patches_df)} patch records → {patches_csv}")

    # Save annotations metadata
    if annotations:
        annot_df = pd.DataFrame(annotations)
        annot_csv = meta_dir / "annotations_metadata.csv"
        annot_df.to_csv(annot_csv, index=False)
        logger.info(f"Saved {len(annot_df)} annotation records → {annot_csv}")

    # ── Summary ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"  Total patches:     {len(all_patches_meta)}")
    logger.info(f"  Total annotations: {len(annotations)}")
    logger.info(f"  Data directory:    {data_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_ingestion()
