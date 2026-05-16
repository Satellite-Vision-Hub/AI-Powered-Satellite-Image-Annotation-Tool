"""
SkyLogic MAS — ZIP Extraction Service
Extracts xView dataset archives into structured /data/raw directory.
"""

import zipfile
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_zip(zip_path: Path, output_dir: Path, description: str = "") -> Path:
    """
    Extract a ZIP archive to the specified output directory.
    
    Args:
        zip_path: Path to the ZIP file.
        output_dir: Destination directory for extraction.
        description: Label for the progress bar.
    
    Returns:
        Path to the output directory.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {zip_path.name} → {output_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc=description or zip_path.stem, unit="file"):
            zf.extract(member, output_dir)

    logger.info(f"Extraction complete: {len(members)} files → {output_dir}")
    return output_dir


def extract_all_datasets(project_root: Path, data_dir: Path) -> dict:
    """
    Extract all xView dataset ZIPs from the project root.
    
    Expected files:
        - train_images.zip
        - train_labels.zip
        - val_images.zip
    
    Args:
        project_root: Root directory containing the ZIP files.
        data_dir: Base data directory (data/raw/).
    
    Returns:
        Dict mapping dataset name to extraction path.
    """
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "train_images": project_root / "train_images.zip",
        "train_labels": project_root / "train_labels.zip",
        "val_images": project_root / "val_images.zip",
    }

    results = {}

    for name, zip_path in datasets.items():
        target_dir = raw_dir / name
        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info(f"Skipping {name}: already extracted at {target_dir}")
            results[name] = target_dir
            continue

        if not zip_path.exists():
            logger.warning(f"ZIP not found, skipping: {zip_path}")
            continue

        results[name] = extract_zip(zip_path, target_dir, description=name)

    return results
