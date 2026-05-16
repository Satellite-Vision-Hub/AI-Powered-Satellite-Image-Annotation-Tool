"""
SkyLogic MAS — xView Label Mapper
Maps xView GeoJSON labels to the corresponding 512x512 patches.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# xView class mapping — all 60+ classes
XVIEW_CLASSES = {
    11: "Fixed-wing Aircraft",
    12: "Small Aircraft",
    13: "Cargo Plane",
    15: "Helicopter",
    17: "Passenger Vehicle",
    18: "Small Car",
    19: "Bus",
    20: "Pickup Truck",
    21: "Utility Truck",
    23: "Truck",
    24: "Cargo Truck",
    25: "Truck w/Box",
    26: "Truck Tractor",
    27: "Trailer",
    28: "Truck w/Flatbed",
    29: "Truck w/Liquid",
    32: "Crane Truck",
    33: "Railway Vehicle",
    34: "Passenger Car",
    35: "Cargo Car",
    36: "Flat Car",
    37: "Tank car",
    38: "Locomotive",
    40: "Maritime Vessel",
    41: "Motorboat",
    42: "Sailboat",
    44: "Tugboat",
    45: "Barge",
    47: "Fishing Vessel",
    49: "Ferry",
    50: "Yacht",
    51: "Container Ship",
    52: "Oil Tanker",
    53: "Engineering Vehicle",
    54: "Tower crane",
    55: "Container Crane",
    56: "Reach Stacker",
    57: "Straddle Carrier",
    59: "Mobile Crane",
    60: "Dump Truck",
    61: "Haul Truck",
    62: "Scraper/Tractor",
    63: "Front loader/Bulldozer",
    64: "Excavator",
    65: "Cement Mixer",
    66: "Ground Grader",
    71: "Hut/Tent",
    72: "Shed",
    73: "Building",
    74: "Aircraft Hangar",
    76: "Damaged Building",
    77: "Facility",
    79: "Construction Site",
    83: "Vehicle Lot",
    84: "Helipad",
    86: "Storage Tank",
    89: "Shipping container lot",
    91: "Shipping Container",
    93: "Pylon",
    94: "Tower",
}

# Reverse map: class_id → class_name
CLASS_ID_TO_NAME = XVIEW_CLASSES
CLASS_NAME_TO_ID = {v: k for k, v in XVIEW_CLASSES.items()}


def load_xview_geojson(label_path: Path) -> list[dict]:
    """
    Load xView GeoJSON labels file.
    
    Args:
        label_path: Path to the xView GeoJSON labels file.
    
    Returns:
        List of feature dicts with bounding boxes and class IDs.
    """
    # Search for the GeoJSON file
    if label_path.is_dir():
        geojson_files = list(label_path.rglob("*.geojson"))
        if not geojson_files:
            # Also check for .json files
            geojson_files = list(label_path.rglob("*.json"))
        if not geojson_files:
            logger.warning(f"No GeoJSON/JSON label files found in {label_path}")
            return []
        label_path = geojson_files[0]
        logger.info(f"Found label file: {label_path}")

    with open(label_path, "r") as f:
        data = json.load(f)

    features = data.get("features", [])
    logger.info(f"Loaded {len(features)} label features from {label_path.name}")
    return features


def parse_feature_bbox(feature: dict) -> Optional[dict]:
    """
    Extract bounding box and class from a GeoJSON feature.
    
    xView stores pixel-space bounding boxes in the ``bounds_imcoords``
    property (format: "x_min,y_min,x_max,y_max").  The geometry field
    contains geo-coordinates (lon/lat) which are NOT suitable for
    matching against pixel-based tile offsets.
    
    Returns:
        Dict with keys: image_id, class_id, class_name, bbox [x_min, y_min, x_max, y_max]
        or None if unparseable.
    """
    try:
        props = feature.get("properties", {})

        # Get image filename
        image_id = props.get("image_id", "")
        if not image_id:
            return None

        # Get class
        class_id = int(props.get("type_id", 0))
        class_name = CLASS_ID_TO_NAME.get(class_id, f"class_{class_id}")

        # ── Use bounds_imcoords (pixel coordinates) ──────────────
        # This is the authoritative pixel-space bbox in xView.
        bounds_str = props.get("bounds_imcoords", "")
        if not bounds_str:
            return None

        bbox = [float(x) for x in bounds_str.split(",")]
        if len(bbox) != 4:
            return None

        return {
            "image_id": image_id,
            "class_id": class_id,
            "class_name": class_name,
            "bbox": bbox,  # [x_min, y_min, x_max, y_max] in pixel coords
        }
    except (KeyError, ValueError, TypeError) as e:
        logger.debug(f"Failed to parse feature: {e}")
        return None


def map_labels_to_patches(
    labels: list[dict],
    patches_meta: list[dict],
    tile_size: int = 512,
) -> list[dict]:
    """
    Map parsed xView labels to their corresponding patches.
    
    For each label, finds which patch(es) it falls into based on
    the patch's x_offset, y_offset within the source image.
    
    Args:
        labels: List of parsed label dicts (from parse_feature_bbox).
        patches_meta: List of patch metadata dicts (from TilingEngine).
        tile_size: Patch dimensions.
    
    Returns:
        List of annotation dicts with patch-relative bounding boxes.
    """
    # Group patches by source image
    patches_by_image = {}
    for pm in patches_meta:
        src = pm["source_image"]
        if src not in patches_by_image:
            patches_by_image[src] = []
        patches_by_image[src].append(pm)

    annotations = []

    for label in tqdm(labels, desc="Mapping labels to patches", unit="label"):
        if label is None:
            continue

        image_id = label["image_id"]
        bbox = label["bbox"]  # [x_min, y_min, x_max, y_max] in source image coords

        # Find patches from the same source image
        matching_patches = patches_by_image.get(image_id, [])

        for pm in matching_patches:
            x_off = pm["x_offset"]
            y_off = pm["y_offset"]

            # Check if bbox overlaps with this patch
            patch_x_min = x_off
            patch_y_min = y_off
            patch_x_max = x_off + tile_size
            patch_y_max = y_off + tile_size

            # Intersection check
            if (bbox[0] >= patch_x_max or bbox[2] <= patch_x_min or
                    bbox[1] >= patch_y_max or bbox[3] <= patch_y_min):
                continue

            # Convert to patch-relative coordinates
            rel_bbox = [
                max(0, bbox[0] - x_off),
                max(0, bbox[1] - y_off),
                min(tile_size, bbox[2] - x_off),
                min(tile_size, bbox[3] - y_off),
            ]

            # Skip if clipped bbox is too small
            rel_w = rel_bbox[2] - rel_bbox[0]
            rel_h = rel_bbox[3] - rel_bbox[1]
            if rel_w < 4 or rel_h < 4:
                continue

            annotations.append({
                "patch_filename": pm["filename"],
                "class_id": label["class_id"],
                "class_name": label["class_name"],
                "bbox": rel_bbox,
                "source": "xview_ground_truth",
                "confidence": 1.0,
            })

    logger.info(f"Mapped {len(annotations)} annotations to patches")
    return annotations
