"""
SkyLogic MAS — Agent C: Interactive Segment Anything Model (SAM)
Click-based annotation with ViT-H + Qdrant similarity search.
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SAMAgent:
    """
    Agent C — Interactive SAM for click-based segmentation.
    
    Workflow:
        1. User clicks (x, y) on the UI canvas
        2. SAM generates a precise mask for that click
        3. Feature embedding is extracted and stored in Qdrant
        4. Similarity search finds and auto-annotates similar objects globally
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "vit_h",
        device: str = "cpu",
    ):
        self.device = device
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.sam_model = None
        self.predictor = None
        self._current_image = None
        self._current_embedding = None

        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: Optional[str] = None):
        """Load SAM ViT-H model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor

            if checkpoint_path and Path(checkpoint_path).exists():
                logger.info(f"Loading SAM {self.model_type} from {checkpoint_path}")
                self.sam_model = sam_model_registry[self.model_type](
                    checkpoint=checkpoint_path
                )
                self.sam_model.to(self.device)
                self.predictor = SamPredictor(self.sam_model)
                logger.info(f"SAM {self.model_type} loaded on {self.device}")
            else:
                logger.warning(
                    f"SAM checkpoint not found at {checkpoint_path}. "
                    "Download ViT-H weights from: "
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                )
                self.sam_model = None
                self.predictor = None

        except ImportError as e:
            logger.error(f"segment_anything not installed: {e}")
            self.sam_model = None
            self.predictor = None

    def set_image(self, image: np.ndarray | str | Path) -> bool:
        """
        Set the current image for SAM predictor.
        Pre-computes the image embedding (expensive step).

        Args:
            image: Image as numpy array (HWC, uint8) or file path.

        Returns:
            True if image was set successfully.
        """
        if self.predictor is None:
            logger.warning("SAM not loaded. Cannot set image.")
            return False

        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
            image_array = np.array(pil_img)
        else:
            image_array = image

        self._current_image = image_array
        self.predictor.set_image(image_array)
        logger.info(f"SAM image set: {image_array.shape}")
        return True

    def predict_click(
        self,
        point_coords: list[list[int]],
        point_labels: list[int],
        multimask_output: bool = True,
    ) -> dict:
        """
        Generate mask from user click(s).

        Args:
            point_coords: List of [x, y] coordinates.
            point_labels: List of labels (1=foreground, 0=background).
            multimask_output: If True, returns 3 masks with different granularity.

        Returns:
            Dict with masks, scores, and logits.
        """
        if self.predictor is None:
            return {"masks": [], "scores": [], "error": "SAM not loaded"}

        if self._current_image is None:
            return {"masks": [], "scores": [], "error": "No image set"}

        coords = np.array(point_coords)
        labels = np.array(point_labels)

        masks, scores, logits = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=multimask_output,
        )

        # Select best mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = float(scores[best_idx])

        return {
            "masks": masks.tolist() if len(masks) <= 3 else [best_mask.tolist()],
            "best_mask": best_mask,
            "best_score": best_score,
            "scores": scores.tolist(),
            "all_masks": masks,
        }

    def extract_embedding(
        self,
        mask: np.ndarray,
        image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract a feature embedding for the masked region.
        Uses the SAM image encoder features.

        Args:
            mask: Binary mask (H, W) for the region of interest.
            image: Optional image override (uses current if None).

        Returns:
            Feature embedding vector (256-d).
        """
        if self.predictor is None or self._current_image is None:
            return np.zeros(256, dtype=np.float32)

        # Get image embedding from SAM encoder
        features = self.predictor.get_image_embedding()
        features_np = features.squeeze(0).cpu().numpy()  # (256, H', W')

        # Resize mask to feature map size
        feat_h, feat_w = features_np.shape[1], features_np.shape[2]
        from PIL import Image as PILImage
        mask_resized = np.array(
            PILImage.fromarray(mask.astype(np.uint8)).resize(
                (feat_w, feat_h), PILImage.NEAREST
            )
        )

        # Masked average pooling
        mask_bool = mask_resized > 0
        if mask_bool.sum() == 0:
            return np.mean(features_np, axis=(1, 2)).astype(np.float32)

        embedding = np.zeros(features_np.shape[0], dtype=np.float32)
        for c in range(features_np.shape[0]):
            embedding[c] = features_np[c][mask_bool].mean()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def mask_to_bbox(self, mask: np.ndarray) -> list[int]:
        """Convert a binary mask to [x_min, y_min, x_max, y_max] bounding box."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return [0, 0, 0, 0]
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    def mask_to_polygon(self, mask: np.ndarray) -> list:
        """Convert binary mask to polygon coordinates."""
        import cv2
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygons = []
        for contour in contours:
            if len(contour) >= 3:
                polygon = contour.squeeze().tolist()
                polygons.append(polygon)
        return polygons

    def generate_point_id(self) -> str:
        """Generate a unique Qdrant point ID."""
        return str(uuid.uuid4())
