"""
SkyLogic MAS — SAM Interactive Annotation API Routes
Handles click-based segmentation and Qdrant similarity search.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from skylogic.database import get_db
from skylogic.models.annotation import Annotation
from skylogic.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sam", tags=["SAM Interactive"])

# Lazy-initialized globals (deferred imports to avoid loading torch at startup)
_sam_agent = None
_qdrant = None


def get_sam():
    global _sam_agent
    if _sam_agent is None:
        from skylogic.agents.sam_agent import SAMAgent
        from pathlib import Path as _P
        sam_path = _P(settings.sam_checkpoint)
        if not sam_path.exists():
            sam_path = settings.models_dir / "sam" / "sam_vit_h_4b8939.pth"
        _sam_agent = SAMAgent(
            checkpoint_path=str(sam_path) if sam_path.exists() else None,
            device=settings.torch_device,
        )
    return _sam_agent


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        from skylogic.vector_store.qdrant_client import QdrantStore
        _qdrant = QdrantStore()
    return _qdrant


class ClickRequest(BaseModel):
    patch_id: int
    image_path: str
    clicks: list[list[int]]        # [[x1,y1], [x2,y2], ...]
    labels: list[int]              # [1, 0, ...] (1=fg, 0=bg)
    auto_annotate: bool = True     # Trigger similarity search


class ClickResponse(BaseModel):
    patch_id: int
    bbox: list[int]
    score: float
    point_id: Optional[str] = None
    similar_count: int = 0


class SimilaritySearchRequest(BaseModel):
    point_id: str
    top_k: int = 20
    score_threshold: float = 0.75


@router.post("/click", response_model=ClickResponse)
def sam_click(
    req: ClickRequest,
    db: Session = Depends(get_db),
):
    """
    Process a user click for SAM segmentation.
    
    1. Set image and generate mask from click
    2. Extract embedding
    3. Store in Qdrant
    4. Optionally trigger similarity auto-annotation
    """
    import numpy as np
    from pathlib import Path

    sam = get_sam()
    qdrant = get_qdrant()

    # Verify image exists
    image_path = Path(req.image_path)
    if not image_path.exists():
        # Try relative to data dir
        image_path = settings.data_dir / req.image_path.lstrip("/")
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

    # Set image for SAM
    success = sam.set_image(str(image_path))
    if not success:
        raise HTTPException(
            status_code=503,
            detail="SAM model not available. Download ViT-H checkpoint.",
        )

    # Generate mask
    result = sam.predict_click(
        point_coords=req.clicks,
        point_labels=req.labels,
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    best_mask = result.get("best_mask")
    if best_mask is None:
        raise HTTPException(status_code=500, detail="No mask generated")

    best_mask = np.array(best_mask)
    bbox = sam.mask_to_bbox(best_mask)
    score = result.get("best_score", 0.0)

    # Extract embedding and store in Qdrant
    point_id = sam.generate_point_id()
    embedding = sam.extract_embedding(best_mask)

    metadata = {
        "patch_id": req.patch_id,
        "bbox": bbox,
        "score": score,
        "image_path": str(image_path),
    }
    qdrant.store_embedding(point_id, embedding, metadata)

    # Save annotation to DB
    annotation = Annotation(
        patch_id=req.patch_id,
        class_id=0,
        class_name="sam_segment",
        bbox=bbox,
        confidence=score,
        source="sam",
        qdrant_point_id=point_id,
    )
    db.add(annotation)
    db.commit()

    # Auto-annotate similar objects
    similar_count = 0
    if req.auto_annotate:
        similar = qdrant.search_similar(
            embedding, top_k=20, score_threshold=0.75
        )
        # Skip the query itself
        similar = [s for s in similar if s["point_id"] != point_id]
        similar_count = len(similar)

        for s in similar:
            payload = s.get("payload", {})
            auto_annot = Annotation(
                patch_id=payload.get("patch_id", req.patch_id),
                class_id=0,
                class_name="sam_auto_segment",
                bbox=payload.get("bbox"),
                confidence=s["score"],
                source="sam",
                qdrant_point_id=s["point_id"],
            )
            db.add(auto_annot)

        db.commit()

    return ClickResponse(
        patch_id=req.patch_id,
        bbox=bbox,
        score=score,
        point_id=point_id,
        similar_count=similar_count,
    )


@router.post("/search-similar")
def search_similar(req: SimilaritySearchRequest):
    """Search for visually similar objects by Qdrant point ID."""
    qdrant = get_qdrant()

    # Get embedding for the reference point
    if qdrant.client is None:
        raise HTTPException(status_code=503, detail="Qdrant not connected")

    try:
        point = qdrant.client.retrieve(
            collection_name=qdrant.collection_name,
            ids=[req.point_id],
            with_vectors=True,
        )
        if not point:
            raise HTTPException(status_code=404, detail="Point not found in Qdrant")

        import numpy as np
        query_vector = np.array(point[0].vector)
        results = qdrant.search_similar(
            query_vector,
            top_k=req.top_k,
            score_threshold=req.score_threshold,
        )

        return {"results": results, "count": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def sam_status():
    """Check SAM and Qdrant availability."""
    result = {
        "sam_loaded": False,
        "sam_model_type": "vit_h",
        "qdrant": {"status": "not_initialized"},
    }

    try:
        sam = get_sam()
        result["sam_loaded"] = sam.sam_model is not None
        result["sam_model_type"] = sam.model_type
    except Exception as e:
        result["sam_error"] = str(e)

    try:
        qdrant = get_qdrant()
        result["qdrant"] = qdrant.get_collection_info()
    except Exception as e:
        result["qdrant"] = {"status": "error", "error": str(e)}

    return result

