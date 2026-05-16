"""
SkyLogic MAS — GeoTIFF Tiling Engine
Converts large satellite images into 512x512 patches with geo-referencing.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TilingEngine:
    """
    Cuts large GeoTIFF/TIFF images into fixed-size patches.
    Preserves affine transforms for geo-referencing.
    """

    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

    def tile_image(
        self,
        image_path: Path,
        output_dir: Path,
        split: str = "train",
    ) -> list[dict]:
        """
        Tile a single image into patches.
        
        Args:
            image_path: Path to the source image (TIFF/GeoTIFF or standard image).
            output_dir: Directory to save patches.
            split: Dataset split label ('train' or 'val').
        
        Returns:
            List of patch metadata dicts.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        patches_meta = []

        try:
            # Try opening as GeoTIFF with rasterio
            with rasterio.open(str(image_path)) as src:
                img_width = src.width
                img_height = src.height
                transform = src.transform
                crs = str(src.crs) if src.crs else None

                patch_idx = 0
                for y_off in range(0, img_height, self.stride):
                    for x_off in range(0, img_width, self.stride):
                        # Clamp window to image bounds
                        win_width = min(self.tile_size, img_width - x_off)
                        win_height = min(self.tile_size, img_height - y_off)

                        # Skip tiny edge patches
                        if win_width < self.tile_size // 2 or win_height < self.tile_size // 2:
                            continue

                        window = Window(x_off, y_off, win_width, win_height)
                        patch_data = src.read(window=window)

                        # Compute patch affine transform
                        patch_transform = rasterio.windows.transform(window, transform)

                        # Pad if needed to maintain consistent tile_size
                        if win_width < self.tile_size or win_height < self.tile_size:
                            padded = np.zeros(
                                (patch_data.shape[0], self.tile_size, self.tile_size),
                                dtype=patch_data.dtype,
                            )
                            padded[:, :win_height, :win_width] = patch_data
                            patch_data = padded

                        # Save patch as PNG
                        patch_name = f"{image_path.stem}_p{patch_idx:05d}.png"
                        patch_path = output_dir / patch_name

                        # Convert to HWC for PIL
                        if patch_data.shape[0] <= 4:
                            img_array = np.moveaxis(patch_data, 0, -1)
                        else:
                            img_array = patch_data[:3]
                            img_array = np.moveaxis(img_array, 0, -1)

                        # Handle different dtypes
                        if img_array.dtype == np.uint16:
                            img_array = (img_array / 256).astype(np.uint8)
                        elif img_array.dtype != np.uint8:
                            img_array = img_array.astype(np.uint8)

                        pil_img = Image.fromarray(img_array)
                        pil_img.save(str(patch_path))

                        meta = {
                            "filename": patch_name,
                            "source_image": image_path.name,
                            "split": split,
                            "x_offset": x_off,
                            "y_offset": y_off,
                            "width": self.tile_size,
                            "height": self.tile_size,
                            "actual_width": win_width,
                            "actual_height": win_height,
                            "affine_transform": list(patch_transform)[:6],
                            "crs": crs,
                        }
                        patches_meta.append(meta)
                        patch_idx += 1

        except rasterio.errors.RasterioIOError:
            # Fallback: open as standard image with PIL
            logger.info(f"Opening {image_path.name} as standard image (not GeoTIFF)")
            patches_meta = self._tile_standard_image(image_path, output_dir, split)

        return patches_meta

    def _tile_standard_image(
        self, image_path: Path, output_dir: Path, split: str
    ) -> list[dict]:
        """Tile a standard image (PNG/JPEG) without geo-referencing."""
        img = Image.open(image_path)
        img_array = np.array(img)

        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

        img_height, img_width = img_array.shape[:2]
        patches_meta = []
        patch_idx = 0

        for y_off in range(0, img_height, self.stride):
            for x_off in range(0, img_width, self.stride):
                win_width = min(self.tile_size, img_width - x_off)
                win_height = min(self.tile_size, img_height - y_off)

                if win_width < self.tile_size // 2 or win_height < self.tile_size // 2:
                    continue

                patch = img_array[y_off:y_off + win_height, x_off:x_off + win_width]

                # Pad if needed
                if win_width < self.tile_size or win_height < self.tile_size:
                    padded = np.zeros(
                        (self.tile_size, self.tile_size, patch.shape[2]),
                        dtype=patch.dtype,
                    )
                    padded[:win_height, :win_width] = patch
                    patch = padded

                patch_name = f"{image_path.stem}_p{patch_idx:05d}.png"
                patch_path = output_dir / patch_name
                Image.fromarray(patch).save(str(patch_path))

                meta = {
                    "filename": patch_name,
                    "source_image": image_path.name,
                    "split": split,
                    "x_offset": x_off,
                    "y_offset": y_off,
                    "width": self.tile_size,
                    "height": self.tile_size,
                    "actual_width": win_width,
                    "actual_height": win_height,
                    "affine_transform": None,
                    "crs": None,
                }
                patches_meta.append(meta)
                patch_idx += 1

        return patches_meta

    def tile_directory(
        self,
        images_dir: Path,
        output_dir: Path,
        split: str = "train",
        extensions: tuple = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
    ) -> list[dict]:
        """
        Tile all images in a directory.
        
        Args:
            images_dir: Directory containing source images.
            output_dir: Directory to save patches.
            split: Dataset split label.
            extensions: Accepted image file extensions.
        
        Returns:
            Combined list of all patch metadata.
        """
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return []

        image_files = sorted([
            f for f in images_dir.rglob("*")
            if f.suffix.lower() in extensions and f.is_file()
        ])

        logger.info(f"Found {len(image_files)} images in {images_dir}")

        all_patches = []
        for img_path in tqdm(image_files, desc=f"Tiling {split}", unit="image"):
            patches = self.tile_image(img_path, output_dir, split)
            all_patches.extend(patches)

        logger.info(f"Total patches generated ({split}): {len(all_patches)}")
        return all_patches
