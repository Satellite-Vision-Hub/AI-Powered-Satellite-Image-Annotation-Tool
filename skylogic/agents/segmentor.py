"""
SkyLogic MAS — Agent B: SegFormer Semantic Segmentor
Identifies disaster regions (floods, debris, damage) via pixel-level segmentation.
CPU-optimized with small batch sizes.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Segmentation class labels for disaster detection
SEG_CLASSES = {
    0: "background",
    1: "building",
    2: "damaged_building",
    3: "vehicle",
    4: "road",
    5: "vegetation",
    6: "water_flood",
    7: "debris_rubble",
    8: "construction",
    9: "container",
}


class SegmentorAgent:
    """
    Agent B — SegFormer-based semantic segmentation for disaster regions.
    Falls back to a lightweight U-Net if SegFormer is unavailable.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        num_classes: int = 10,
        img_size: int = 512,
    ):
        self.device = device
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.model_path = model_path
        # Tracks which API the loaded model expects ("segformer" -> pixel_values=, else direct call)
        self._model_kind: str = "unet"
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str] = None):
        """Load fine-tuned U-Net checkpoint, SegFormer scratch, or U-Net imagenet (in that order)."""
        if model_path and Path(model_path).exists():
            try:
                logger.info(f"Loading fine-tuned segmentor checkpoint: {model_path}")
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                # Support both raw state_dict and {'model_state': ..., 'num_classes': ...}
                if isinstance(ckpt, dict) and "model_state" in ckpt:
                    nc = int(ckpt.get("num_classes", self.num_classes))
                    self.num_classes = nc
                    import segmentation_models_pytorch as smp
                    self.model = smp.Unet(
                        encoder_name="resnet18",
                        encoder_weights=None,  # we're loading trained weights
                        in_channels=3,
                        classes=nc,
                    )
                    self.model.load_state_dict(ckpt["model_state"])
                    self.model.to(self.device)
                    self.model.eval()
                    self._model_kind = "unet"
                    logger.info(f"U-Net checkpoint loaded ({nc} classes)")
                    return
                if isinstance(ckpt, dict):
                    # Plain state_dict
                    import segmentation_models_pytorch as smp
                    self.model = smp.Unet(
                        encoder_name="resnet18", encoder_weights=None,
                        in_channels=3, classes=self.num_classes,
                    )
                    self.model.load_state_dict(ckpt)
                    self.model.to(self.device).eval()
                    return
                # Fully serialised module
                self.model = ckpt
                self.model.to(self.device).eval()
                return
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Checkpoint load failed ({e}); falling back to pretrained U-Net")

        # No checkpoint — try SegFormer first, then U-Net
        try:
            self._load_segformer()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"SegFormer load failed ({e}), falling back to U-Net")
            self._load_unet()

    def _load_segformer(self):
        """Load SegFormer from HuggingFace transformers."""
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerConfig

            config = SegformerConfig(
                num_labels=self.num_classes,
                num_channels=3,
                hidden_sizes=[32, 64, 160, 256],
                decoder_hidden_size=256,
            )
            self.model = SegformerForSemanticSegmentation(config)
            self.model.to(self.device)
            self.model.eval()
            self._model_kind = "segformer"
            logger.info(f"SegFormer loaded on {self.device}")
        except ImportError:
            logger.warning("transformers not available, falling back to U-Net")
            self._load_unet()

    def _load_unet(self):
        """Load lightweight U-Net from segmentation_models_pytorch."""
        try:
            import segmentation_models_pytorch as smp

            self.model = smp.Unet(
                encoder_name="resnet18",
                encoder_weights="imagenet",
                in_channels=3,
                classes=self.num_classes,
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"U-Net (ResNet18) loaded on {self.device}")
        except ImportError:
            logger.error("Neither transformers nor smp available. No segmentor loaded.")
            self.model = None

    def train(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 30,
        lr: float = 1e-4,
        save_dir: str = "./models/segformer",
    ) -> dict:
        """
        Train the segmentation model.

        Args:
            train_loader: PyTorch DataLoader for training.
            val_loader: Optional validation DataLoader.
            epochs: Number of training epochs.
            lr: Learning rate.
            save_dir: Directory to save model weights.

        Returns:
            Training summary dict.
        """
        if self.model is None:
            raise RuntimeError("Segmentor model not loaded.")

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

            for batch in train_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device).long()

                optimizer.zero_grad()

                # Handle SegFormer vs U-Net output format
                if self._model_kind == "segformer":
                    # SegFormer returns SegformerOutput
                    outputs = self.model(pixel_values=images)
                    logits = outputs.logits
                    # Upsample to match mask size
                    logits = nn.functional.interpolate(
                        logits, size=masks.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                else:
                    logits = self.model(images)

                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_train_loss = epoch_loss / max(batch_count, 1)
            history["train_loss"].append(avg_train_loss)

            # Validation
            avg_val_loss = 0.0
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_count = 0
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch["image"].to(self.device)
                        masks = batch["mask"].to(self.device).long()

                        if self._model_kind == "segformer":
                            outputs = self.model(pixel_values=images)
                            logits = outputs.logits
                            logits = nn.functional.interpolate(
                                logits, size=masks.shape[-2:],
                                mode="bilinear", align_corners=False,
                            )
                        else:
                            logits = self.model(images)

                        loss = criterion(logits, masks)
                        val_loss += loss.item()
                        val_count += 1

                avg_val_loss = val_loss / max(val_count, 1)
                history["val_loss"].append(avg_val_loss)

            logger.info(
                f"Epoch {epoch+1}/{epochs} — "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )

            # Save best model
            if avg_val_loss < best_loss or not val_loader:
                check_loss = avg_train_loss if not val_loader else avg_val_loss
                if check_loss < best_loss:
                    best_loss = check_loss
                    torch.save(self.model.state_dict(), save_path / "best.pth")
                    logger.info(f"  → Saved best model (loss={best_loss:.4f})")

        # Save final model
        torch.save(self.model.state_dict(), save_path / "final.pth")
        return {"status": "complete", "history": history, "best_loss": best_loss}

    def predict(self, image: np.ndarray | str | Path) -> dict:
        """
        Run segmentation on a single image.

        Args:
            image: Image as numpy array (HWC uint8) or file path.

        Returns:
            Dict with:
                - mask: (H, W) numpy array of class indices
                - probabilities: (num_classes, H, W) numpy array
                - class_areas: dict of class_name → pixel count
        """
        if self.model is None:
            return {"mask": None, "probabilities": None, "class_areas": {}}

        self.model.eval()

        # Load image
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self._model_kind == "segformer":
                outputs = self.model(pixel_values=input_tensor)
                logits = outputs.logits
                logits = nn.functional.interpolate(
                    logits, size=(self.img_size, self.img_size),
                    mode="bilinear", align_corners=False,
                )
            else:
                logits = self.model(input_tensor)

        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        mask = np.argmax(probs, axis=0)

        # Calculate class areas
        class_areas = {}
        for cls_id, cls_name in SEG_CLASSES.items():
            area = int(np.sum(mask == cls_id))
            if area > 0:
                class_areas[cls_name] = area

        return {
            "mask": mask,
            "probabilities": probs,
            "class_areas": class_areas,
            "source": "segformer",
        }

    def mask_to_bboxes(
        self,
        mask: np.ndarray,
        min_area: int = 100,
    ) -> list[dict]:
        """
        Convert a segmentation mask into bounding boxes for each connected region.
        Used for fusion with YOLOv10 detections.

        Args:
            mask: (H, W) array of class indices.
            min_area: Minimum pixel area for a region to be kept.

        Returns:
            List of bbox dicts: {bbox, class_id, class_name, confidence}
        """
        import cv2

        bboxes = []
        for cls_id in range(1, self.num_classes):
            # Binary mask for this class
            binary = (mask == cls_id).astype(np.uint8)
            if binary.sum() == 0:
                continue

            # Find connected components
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                cls_name = SEG_CLASSES.get(cls_id, f"class_{cls_id}")

                bboxes.append({
                    "bbox": [x, y, x + w, y + h],
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": 0.7,  # Default confidence for segmentation-derived boxes
                    "source": "segformer",
                })

        return bboxes
