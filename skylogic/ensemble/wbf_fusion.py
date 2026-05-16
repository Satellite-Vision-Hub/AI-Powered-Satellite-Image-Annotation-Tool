"""
SkyLogic MAS — Weighted Boxes Fusion (WBF) Logic
Merges predictions from YOLOv10 and SegFormer into unified "Master Predictions".
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class WBFFusion:
    """
    Ensemble runner that combines detection (Agent A) and segmentation (Agent B)
    predictions using Weighted Boxes Fusion.
    
    WBF weights boxes by confidence scores to produce a consensus output,
    reducing false positives and improving localization accuracy.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        skip_box_threshold: float = 0.01,
        weights: Optional[list[float]] = None,
    ):
        """
        Args:
            iou_threshold: IoU threshold for box matching.
            skip_box_threshold: Minimum confidence to keep a box.
            weights: Weights for each model [detector_weight, segmentor_weight].
        """
        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold
        self.weights = weights or [2.0, 1.0]  # YOLOv10 weighted higher

    def fuse(
        self,
        detections: list[dict],
        seg_bboxes: list[dict],
        image_width: int = 512,
        image_height: int = 512,
    ) -> list[dict]:
        """
        Fuse detections from Agent A + Agent B using WBF.
        
        Args:
            detections: YOLOv10 detections [{bbox, class_id, confidence, ...}]
            seg_bboxes: SegFormer-derived bboxes [{bbox, class_id, confidence, ...}]
            image_width: Image width for normalization.
            image_height: Image height for normalization.
        
        Returns:
            List of fused "Master Prediction" dicts.
        """
        if not detections and not seg_bboxes:
            return []

        try:
            from ensemble_boxes import weighted_boxes_fusion
        except ImportError:
            logger.warning("ensemble_boxes not installed. Falling back to simple merge.")
            return self._simple_merge(detections, seg_bboxes)

        # Prepare inputs for WBF
        # WBF expects: list of [boxes_per_model], [scores_per_model], [labels_per_model]
        # Boxes must be normalized to [0, 1]

        boxes_list = []
        scores_list = []
        labels_list = []

        # Model 1: YOLOv10 detections
        det_boxes = []
        det_scores = []
        det_labels = []
        for d in detections:
            bbox = d["bbox"]
            norm_box = [
                bbox[0] / image_width,
                bbox[1] / image_height,
                bbox[2] / image_width,
                bbox[3] / image_height,
            ]
            # Clip to [0, 1]
            norm_box = [max(0, min(1, v)) for v in norm_box]
            # Ensure x1 < x2 and y1 < y2
            if norm_box[0] >= norm_box[2] or norm_box[1] >= norm_box[3]:
                continue
            det_boxes.append(norm_box)
            det_scores.append(d.get("confidence", 0.5))
            det_labels.append(d.get("class_id", 0))

        boxes_list.append(det_boxes if det_boxes else [[0, 0, 0, 0]])
        scores_list.append(det_scores if det_scores else [0.0])
        labels_list.append(det_labels if det_labels else [0])

        # Model 2: SegFormer-derived bboxes
        seg_boxes_norm = []
        seg_scores = []
        seg_labels = []
        for s in seg_bboxes:
            bbox = s["bbox"]
            norm_box = [
                bbox[0] / image_width,
                bbox[1] / image_height,
                bbox[2] / image_width,
                bbox[3] / image_height,
            ]
            norm_box = [max(0, min(1, v)) for v in norm_box]
            if norm_box[0] >= norm_box[2] or norm_box[1] >= norm_box[3]:
                continue
            seg_boxes_norm.append(norm_box)
            seg_scores.append(s.get("confidence", 0.5))
            seg_labels.append(s.get("class_id", 0))

        boxes_list.append(seg_boxes_norm if seg_boxes_norm else [[0, 0, 0, 0]])
        scores_list.append(seg_scores if seg_scores else [0.0])
        labels_list.append(seg_labels if seg_labels else [0])

        # Run WBF
        try:
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=self.weights,
                iou_thr=self.iou_threshold,
                skip_box_thr=self.skip_box_threshold,
            )
        except Exception as e:
            logger.error(f"WBF fusion failed: {e}")
            return self._simple_merge(detections, seg_bboxes)

        # Convert back to pixel coordinates
        master_predictions = []
        for i in range(len(fused_boxes)):
            box = fused_boxes[i]
            pixel_box = [
                float(box[0] * image_width),
                float(box[1] * image_height),
                float(box[2] * image_width),
                float(box[3] * image_height),
            ]

            master_predictions.append({
                "bbox": pixel_box,
                "class_id": int(fused_labels[i]),
                "confidence": float(fused_scores[i]),
                "source": "ensemble",
            })

        # Filter out dummy boxes
        master_predictions = [
            p for p in master_predictions
            if p["confidence"] > self.skip_box_threshold
            and p["bbox"] != [0.0, 0.0, 0.0, 0.0]
        ]

        logger.info(
            f"WBF Fusion: {len(detections)} det + {len(seg_bboxes)} seg "
            f"→ {len(master_predictions)} master predictions"
        )

        return master_predictions

    def _simple_merge(
        self,
        detections: list[dict],
        seg_bboxes: list[dict],
    ) -> list[dict]:
        """Fallback: simple concatenation with NMS when WBF is unavailable."""
        merged = []
        for d in detections:
            d["source"] = "ensemble"
            merged.append(d)
        for s in seg_bboxes:
            s["source"] = "ensemble"
            merged.append(s)

        # Simple NMS by IoU
        if len(merged) > 1:
            merged = self._nms(merged, iou_threshold=self.iou_threshold)

        return merged

    def _nms(self, predictions: list[dict], iou_threshold: float = 0.5) -> list[dict]:
        """Simple non-maximum suppression."""
        if not predictions:
            return []

        # Sort by confidence
        preds = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
        keep = []

        while preds:
            best = preds.pop(0)
            keep.append(best)
            preds = [
                p for p in preds
                if self._compute_iou(best["bbox"], p["bbox"]) < iou_threshold
            ]

        return keep

    @staticmethod
    def _compute_iou(bbox1: list, bbox2: list) -> float:
        """Compute Intersection over Union between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
