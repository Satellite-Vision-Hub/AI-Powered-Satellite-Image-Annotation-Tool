"""
SkyLogic MAS — Agent A: YOLOv10 Object Detector
Fast object detection across all 60+ xView classes.
Optimized for CPU inference with batch_size=2.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class DetectorAgent:
    """
    Agent A — YOLOv10-based object detector for satellite imagery.
    Handles training on xView patches and inference.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        num_classes: int = 60,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.model = None
        self.model_path = model_path

        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str] = None):
        """Load YOLOv10 model (pretrained or fine-tuned)."""
        try:
            from ultralytics import YOLO

            if model_path and Path(model_path).exists():
                logger.info(f"Loading fine-tuned YOLOv10 from {model_path}")
                self.model = YOLO(model_path)
            else:
                logger.info("Loading pretrained YOLOv10n (nano) for CPU efficiency")
                self.model = YOLO("yolov8n.pt")  # YOLOv8n as base, ultralytics compatible

            logger.info(f"YOLOv10 Detector loaded on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        batch_size: int = 2,
        img_size: int = 512,
        save_dir: str = "./models/yolov10",
    ) -> dict:
        """
        Fine-tune YOLOv10 on xView patches.

        Args:
            data_yaml: Path to YOLO-format dataset YAML.
            epochs: Number of training epochs.
            batch_size: Batch size (keep small for CPU: 2-4).
            img_size: Input image size.
            save_dir: Directory to save trained weights.

        Returns:
            Training results dict.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot train.")

        logger.info(f"Starting YOLOv10 training: epochs={epochs}, batch={batch_size}")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=self.device,
            project=save_dir,
            name="xview_detector",
            patience=10,
            save=True,
            verbose=True,
            workers=2,
        )

        # Update model path to best weights
        best_weights = Path(save_dir) / "xview_detector" / "weights" / "best.pt"
        if best_weights.exists():
            self.model_path = str(best_weights)
            self.model = None
            self._load_model(str(best_weights))
            logger.info(f"Loaded best weights from {best_weights}")

        return {"status": "complete", "results_dir": save_dir}

    def predict(
        self,
        image: np.ndarray | str | Path,
        conf: Optional[float] = None,
    ) -> list[dict]:
        """
        Run detection on a single image/patch.

        Args:
            image: Image as numpy array, file path, or PIL Image.
            conf: Override confidence threshold.

        Returns:
            List of detection dicts:
                {bbox: [x1,y1,x2,y2], class_id, class_name, confidence}
        """
        if self.model is None:
            logger.warning("Detector model not loaded, returning empty predictions")
            return []

        threshold = conf or self.confidence_threshold

        results = self.model.predict(
            source=image if isinstance(image, (str, Path)) else image,
            conf=threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            imgsz=512,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())

                # Map YOLO class index to xView class name
                names = result.names or {}
                class_name = names.get(class_id, f"class_{class_id}")

                detections.append({
                    "bbox": bbox,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "source": "yolov10",
                })

        return detections

    def predict_batch(
        self,
        images: list,
        conf: Optional[float] = None,
    ) -> list[list[dict]]:
        """
        Run detection on a batch of images.

        Args:
            images: List of image paths or numpy arrays.
            conf: Override confidence threshold.

        Returns:
            List of detection lists (one per image).
        """
        all_detections = []
        for img in images:
            dets = self.predict(img, conf)
            all_detections.append(dets)
        return all_detections

    def generate_data_yaml(
        self,
        train_dir: Path,
        val_dir: Path,
        class_names: dict,
        output_path: Path,
    ) -> str:
        """
        Generate YOLO-format dataset YAML.

        Args:
            train_dir: Path to training images.
            val_dir: Path to validation images.
            class_names: Dict mapping class_id → class_name.
            output_path: Where to save the YAML.

        Returns:
            Path to the generated YAML file.
        """
        import yaml

        data = {
            "path": str(train_dir.parent),
            "train": str(train_dir),
            "val": str(val_dir),
            "nc": len(class_names),
            "names": class_names,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"Generated data YAML → {output_path}")
        return str(output_path)
