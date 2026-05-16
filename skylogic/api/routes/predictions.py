"""
SkyLogic MAS — Predictions API Routes
Triggers AI agents for auto-prediction on patches.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from skylogic.database import get_db
from skylogic.models.patch import Patch
from skylogic.models.annotation import Annotation
from skylogic.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/predictions", tags=["Predictions"])

# Global orchestrator (lazy init — deferred import to avoid loading torch at startup)
_orchestrator = None
_wbf = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from skylogic.agents.orchestrator import AgentOrchestrator
        from pathlib import Path as _P
        det = _P(settings.yolo_weights)
        seg = _P(settings.segformer_weights)
        sam = _P(settings.sam_checkpoint)
        _orchestrator = AgentOrchestrator(
            detector_path=str(det) if det.exists() else None,
            segmentor_path=str(seg) if seg.exists() else None,
            sam_checkpoint=str(sam) if sam.exists() else None,
            device=settings.torch_device,
        )
    return _orchestrator


def get_wbf():
    global _wbf
    if _wbf is None:
        from skylogic.ensemble.wbf_fusion import WBFFusion
        _wbf = WBFFusion()
    return _wbf


class PredictionRequest(BaseModel):
    patch_id: int
    use_ensemble: bool = True


class BatchPredictionRequest(BaseModel):
    split: Optional[str] = "train"
    limit: Optional[int] = 50
    skip_predicted: bool = True
    use_ensemble: bool = True


@router.post("/single")
def predict_single_patch(
    req: PredictionRequest,
    db: Session = Depends(get_db),
):
    """Run AI prediction on a single patch."""
    patch = db.query(Patch).filter(Patch.id == req.patch_id).first()
    if not patch:
        raise HTTPException(status_code=404, detail="Patch not found")

    # Find patch image file
    image_path = settings.patches_dir / patch.split / patch.filename
    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Patch image not found: {image_path}",
        )

    orchestrator = get_orchestrator()
    wbf = get_wbf()

    # Run agents
    agent_results = orchestrator.run_all_agents(str(image_path))

    # Fuse results
    if req.use_ensemble:
        master_predictions = wbf.fuse(
            detections=agent_results["detections"],
            seg_bboxes=agent_results["seg_bboxes"],
        )
    else:
        master_predictions = agent_results["detections"]

    # Save to database
    saved_count = 0
    for pred in master_predictions:
        annotation = Annotation(
            patch_id=patch.id,
            class_id=pred.get("class_id", 0),
            class_name=pred.get("class_name"),
            bbox=pred.get("bbox"),
            confidence=pred.get("confidence", 0.5),
            source=pred.get("source", "ensemble"),
        )
        db.add(annotation)
        saved_count += 1

    # Update patch status
    patch.is_predicted = 1
    patch.prediction_count = saved_count
    db.commit()

    return {
        "patch_id": patch.id,
        "predictions": master_predictions,
        "saved": saved_count,
        "segmentation_classes": agent_results.get("segmentation", {}).get("class_areas", {}),
    }


def _run_batch_prediction(
    split: str,
    limit: int,
    skip_predicted: bool,
    use_ensemble: bool,
):
    """Background task for batch prediction."""
    from skylogic.database import SessionLocal

    db = SessionLocal()
    try:
        query = db.query(Patch).filter(Patch.split == split)
        if skip_predicted:
            query = query.filter(Patch.is_predicted == 0)

        patches = query.limit(limit).all()
        orchestrator = get_orchestrator()
        wbf = get_wbf()

        for i, patch in enumerate(patches):
            image_path = settings.patches_dir / patch.split / patch.filename
            if not image_path.exists():
                continue

            try:
                results = orchestrator.run_all_agents(str(image_path))

                if use_ensemble:
                    predictions = wbf.fuse(
                        detections=results["detections"],
                        seg_bboxes=results["seg_bboxes"],
                    )
                else:
                    predictions = results["detections"]

                for pred in predictions:
                    annotation = Annotation(
                        patch_id=patch.id,
                        class_id=pred.get("class_id", 0),
                        class_name=pred.get("class_name"),
                        bbox=pred.get("bbox"),
                        confidence=pred.get("confidence", 0.5),
                        source=pred.get("source", "ensemble"),
                    )
                    db.add(annotation)

                patch.is_predicted = 1
                patch.prediction_count = len(predictions)
                db.commit()

                logger.info(f"Batch [{i+1}/{len(patches)}] {patch.filename}: {len(predictions)} predictions")

            except Exception as e:
                logger.error(f"Batch prediction failed for {patch.filename}: {e}")
                db.rollback()

    finally:
        db.close()


@router.post("/batch")
def predict_batch(
    req: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Run AI predictions on multiple patches (background task).
    This is the "Full Dataset Auto-Prediction" endpoint.
    """
    query = db.query(Patch).filter(Patch.split == req.split)
    if req.skip_predicted:
        query = query.filter(Patch.is_predicted == 0)

    total_pending = query.count()

    background_tasks.add_task(
        _run_batch_prediction,
        split=req.split,
        limit=req.limit,
        skip_predicted=req.skip_predicted,
        use_ensemble=req.use_ensemble,
    )

    return {
        "status": "started",
        "total_pending": total_pending,
        "processing_limit": req.limit,
        "message": f"Batch prediction started for up to {req.limit} patches",
    }


@router.get("/status")
def prediction_status(db: Session = Depends(get_db)):
    """Get overall prediction progress."""
    total = db.query(Patch).count()
    predicted = db.query(Patch).filter(Patch.is_predicted == 1).count()
    total_annotations = db.query(Annotation).filter(
        Annotation.source.in_(["yolov10", "segformer", "ensemble"])
    ).count()

    return {
        "total_patches": total,
        "predicted_patches": predicted,
        "pending_patches": total - predicted,
        "progress_percent": round(predicted / max(total, 1) * 100, 1),
        "total_ai_annotations": total_annotations,
    }
