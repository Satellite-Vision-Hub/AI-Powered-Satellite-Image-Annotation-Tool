"""
SkyLogic MAS — Agent Orchestrator
Coordinates multi-agent execution and result aggregation.
"""

import logging
import time
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from skylogic.agents.detector import DetectorAgent
from skylogic.agents.segmentor import SegmentorAgent
from skylogic.agents.sam_agent import SAMAgent
from skylogic.config import settings

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Coordinates the three AI agents and manages result aggregation.
    
    Agents:
        A) DetectorAgent (YOLOv10) — object detection
        B) SegmentorAgent (SegFormer) — semantic segmentation
        C) SAMAgent (ViT-H) — interactive click-based masks
    """

    def __init__(
        self,
        detector_path: Optional[str] = None,
        segmentor_path: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = device

        logger.info("Initializing Agent Orchestrator...")

        # Initialize agents
        self.detector = DetectorAgent(
            model_path=detector_path,
            device=device,
            num_classes=settings.xview_num_classes,
        )

        self.segmentor = SegmentorAgent(
            model_path=segmentor_path,
            device=device,
        )

        self.sam = SAMAgent(
            checkpoint_path=sam_checkpoint,
            model_type="vit_h",
            device=device,
        )

        logger.info("All agents initialized")

    def run_detection(self, image_path: str | Path) -> list[dict]:
        """Run Agent A (YOLOv10) detection."""
        start = time.time()
        detections = self.detector.predict(str(image_path))
        elapsed = time.time() - start
        logger.info(f"Agent A: {len(detections)} detections in {elapsed:.2f}s")
        return detections

    def run_segmentation(self, image_path: str | Path) -> dict:
        """Run Agent B (SegFormer) segmentation."""
        start = time.time()
        result = self.segmentor.predict(str(image_path))
        elapsed = time.time() - start
        n_classes = len(result.get("class_areas", {}))
        logger.info(f"Agent B: {n_classes} classes in {elapsed:.2f}s")
        return result

    def run_sam_click(
        self,
        image_path: str | Path,
        click_x: int,
        click_y: int,
    ) -> dict:
        """Run Agent C (SAM) click prediction."""
        start = time.time()

        # Set image (pre-compute embedding)
        self.sam.set_image(str(image_path))

        # Predict mask from click
        result = self.sam.predict_click(
            point_coords=[[click_x, click_y]],
            point_labels=[1],
        )
        elapsed = time.time() - start
        logger.info(f"Agent C: SAM click ({click_x}, {click_y}) in {elapsed:.2f}s")
        return result

    def run_all_agents(
        self,
        image_path: str | Path,
    ) -> dict:
        """
        Run Agent A + Agent B concurrently on the same image.
        
        Returns dict with detection and segmentation results.
        """
        start = time.time()
        results = {
            "detections": [],
            "segmentation": {},
            "seg_bboxes": [],
        }

        # Run agents (sequentially on CPU to avoid memory issues)
        logger.info(f"Running all agents on: {Path(image_path).name}")

        # Agent A: Detection
        results["detections"] = self.run_detection(image_path)

        # Agent B: Segmentation
        seg_result = self.run_segmentation(image_path)
        results["segmentation"] = seg_result

        # Convert seg masks to bboxes for fusion
        if seg_result.get("mask") is not None:
            results["seg_bboxes"] = self.segmentor.mask_to_bboxes(
                seg_result["mask"]
            )

        elapsed = time.time() - start
        logger.info(
            f"All agents complete in {elapsed:.2f}s — "
            f"Detections: {len(results['detections'])}, "
            f"Seg bboxes: {len(results['seg_bboxes'])}"
        )

        return results

    def run_batch_prediction(
        self,
        image_paths: list[str | Path],
        callback=None,
    ) -> list[dict]:
        """
        Run detection + segmentation on a batch of images.
        
        Args:
            image_paths: List of image file paths.
            callback: Optional function called after each image with (index, total, results).
        
        Returns:
            List of result dicts per image.
        """
        all_results = []
        total = len(image_paths)

        for i, img_path in enumerate(image_paths):
            logger.info(f"Processing {i+1}/{total}: {Path(img_path).name}")
            result = self.run_all_agents(img_path)
            result["image_path"] = str(img_path)
            all_results.append(result)

            if callback:
                callback(i, total, result)

        return all_results
