"""
SkyLogic MAS — Patches API Routes
CRUD operations for satellite image patches.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from skylogic.database import get_db
from skylogic.models.patch import Patch
from skylogic.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/patches", tags=["Patches"])


class PatchResponse(BaseModel):
    id: int
    filename: str
    source_image: str
    split: str
    width: int
    height: int
    x_offset: int
    y_offset: int
    is_predicted: int
    prediction_count: int
    image_url: Optional[str] = None

    class Config:
        from_attributes = True


@router.get("/", response_model=list[PatchResponse])
def list_patches(
    split: Optional[str] = None,
    predicted: Optional[bool] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List patches with optional filtering."""
    query = db.query(Patch)

    if split:
        query = query.filter(Patch.split == split)
    if predicted is not None:
        query = query.filter(Patch.is_predicted == (1 if predicted else 0))

    patches = query.offset(skip).limit(limit).all()

    result = []
    for p in patches:
        resp = PatchResponse.model_validate(p)
        resp.image_url = f"/data/patches/{p.split}/{p.filename}"
        result.append(resp)

    return result


@router.get("/count")
def count_patches(
    split: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Get total patch count."""
    query = db.query(Patch)
    if split:
        query = query.filter(Patch.split == split)
    return {"count": query.count()}


@router.get("/{patch_id}", response_model=PatchResponse)
def get_patch(patch_id: int, db: Session = Depends(get_db)):
    """Get a single patch by ID."""
    patch = db.query(Patch).filter(Patch.id == patch_id).first()
    if not patch:
        raise HTTPException(status_code=404, detail="Patch not found")

    resp = PatchResponse.model_validate(patch)
    resp.image_url = f"/data/patches/{patch.split}/{patch.filename}"
    return resp


@router.post("/load-from-csv")
def load_patches_from_csv(db: Session = Depends(get_db)):
    """Load patches from the metadata CSV into the database."""
    import pandas as pd

    csv_path = settings.data_dir / "metadata" / "patches_metadata.csv"
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Patches CSV not found at {csv_path}. Run ingestion first.",
        )

    df = pd.read_csv(csv_path)
    loaded = 0

    for _, row in df.iterrows():
        existing = db.query(Patch).filter(Patch.filename == row["filename"]).first()
        if existing:
            continue

        patch = Patch(
            filename=row["filename"],
            source_image=row["source_image"],
            split=row.get("split", "train"),
            x_offset=int(row.get("x_offset", 0)),
            y_offset=int(row.get("y_offset", 0)),
            width=int(row.get("width", 512)),
            height=int(row.get("height", 512)),
            crs=row.get("crs"),
        )

        # Parse affine transform if present
        affine_str = row.get("affine_transform")
        if affine_str and str(affine_str) != "nan":
            try:
                import ast
                patch.affine_transform = ast.literal_eval(str(affine_str))
            except (ValueError, SyntaxError):
                pass

        db.add(patch)
        loaded += 1

    db.commit()
    logger.info(f"Loaded {loaded} patches from CSV")
    return {"loaded": loaded, "total_in_csv": len(df)}
