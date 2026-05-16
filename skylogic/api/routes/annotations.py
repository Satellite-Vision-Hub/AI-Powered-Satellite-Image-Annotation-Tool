"""
SkyLogic MAS — Annotations API Routes
CRUD for human and AI-generated annotations. All changes persisted to PostGIS.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from skylogic.database import get_db
from skylogic.models.annotation import Annotation
from skylogic.models.user import User
from skylogic.api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/annotations", tags=["Annotations"])


class AnnotationCreate(BaseModel):
    patch_id: int
    class_id: int
    class_name: Optional[str] = None
    bbox: Optional[list[float]] = None
    confidence: float = 1.0
    source: str = "human"


class AnnotationUpdate(BaseModel):
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    bbox: Optional[list[float]] = None
    confidence: Optional[float] = None
    is_verified: Optional[int] = None


class AnnotationResponse(BaseModel):
    id: int
    patch_id: int
    class_id: int
    class_name: Optional[str]
    bbox: Optional[list]
    confidence: float
    source: str
    is_verified: int
    created_by: Optional[int]

    class Config:
        from_attributes = True


@router.get("/", response_model=list[AnnotationResponse])
def list_annotations(
    patch_id: Optional[int] = None,
    source: Optional[str] = None,
    class_id: Optional[int] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """List annotations with filtering."""
    query = db.query(Annotation)

    if patch_id is not None:
        query = query.filter(Annotation.patch_id == patch_id)
    if source:
        query = query.filter(Annotation.source == source)
    if class_id is not None:
        query = query.filter(Annotation.class_id == class_id)

    return query.offset(skip).limit(limit).all()


@router.get("/count")
def count_annotations(
    patch_id: Optional[int] = None,
    source: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Count annotations with optional filtering."""
    query = db.query(Annotation)
    if patch_id is not None:
        query = query.filter(Annotation.patch_id == patch_id)
    if source:
        query = query.filter(Annotation.source == source)
    return {"count": query.count()}


@router.post("/", response_model=AnnotationResponse)
def create_annotation(
    data: AnnotationCreate,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new annotation (human or AI)."""
    annotation = Annotation(
        patch_id=data.patch_id,
        class_id=data.class_id,
        class_name=data.class_name,
        bbox=data.bbox,
        confidence=data.confidence,
        source=data.source,
        created_by=user.id if user else None,
    )
    db.add(annotation)
    db.commit()
    db.refresh(annotation)
    return annotation


@router.post("/bulk", response_model=dict)
def create_annotations_bulk(
    annotations: list[AnnotationCreate],
    db: Session = Depends(get_db),
):
    """Bulk create annotations (for AI predictions)."""
    created = 0
    for data in annotations:
        annotation = Annotation(
            patch_id=data.patch_id,
            class_id=data.class_id,
            class_name=data.class_name,
            bbox=data.bbox,
            confidence=data.confidence,
            source=data.source,
        )
        db.add(annotation)
        created += 1

    db.commit()
    return {"created": created}


@router.put("/{annotation_id}", response_model=AnnotationResponse)
def update_annotation(
    annotation_id: int,
    data: AnnotationUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update an annotation (manual override/edit)."""
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")

    if data.class_id is not None:
        annotation.class_id = data.class_id
    if data.class_name is not None:
        annotation.class_name = data.class_name
    if data.bbox is not None:
        annotation.bbox = data.bbox
    if data.confidence is not None:
        annotation.confidence = data.confidence
    if data.is_verified is not None:
        annotation.is_verified = data.is_verified

    db.commit()
    db.refresh(annotation)
    return annotation


@router.delete("/{annotation_id}")
def delete_annotation(
    annotation_id: int,
    db: Session = Depends(get_db),
):
    """Delete an annotation."""
    annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")

    db.delete(annotation)
    db.commit()
    return {"deleted": annotation_id}


@router.delete("/patch/{patch_id}")
def delete_patch_annotations(
    patch_id: int,
    source: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Delete all annotations for a patch (optionally filtered by source)."""
    query = db.query(Annotation).filter(Annotation.patch_id == patch_id)
    if source:
        query = query.filter(Annotation.source == source)

    count = query.count()
    query.delete()
    db.commit()
    return {"deleted": count, "patch_id": patch_id}
