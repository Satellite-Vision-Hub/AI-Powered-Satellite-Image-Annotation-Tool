"""
SkyLogic MAS — Configuration Module
Centralized settings loaded from environment variables.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    # --- Project Paths ---
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    data_dir: Path = Field(default=Path("./data"))
    models_dir: Path = Field(default=Path("./models"))

    # --- Database (SQLite by default, PostGIS if available) ---
    database_url: str = Field(
        default="sqlite:///./skylogic.db"
    )

    # --- Qdrant ---
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="sam_embeddings")

    # --- JWT Auth ---
    jwt_secret_key: str = Field(default="change-me-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiry_minutes: int = Field(default=1440)

    # --- PyTorch ---
    torch_device: str = Field(default="cpu")
    batch_size: int = Field(default=2)
    num_workers: int = Field(default=2)

    # --- Tiling ---
    tile_size: int = Field(default=512)
    tile_overlap: int = Field(default=64)

    # --- xView ---
    xview_num_classes: int = Field(default=60)

    # --- Model checkpoint paths (auto-resolved if files exist) ---
    yolo_weights: str = Field(default="models/yolo/latest_best.pt")
    segformer_weights: str = Field(default="models/segformer/latest_best.pth")
    sam_checkpoint: str = Field(default="models/sam/sam_vit_h_4b8939.pth")

    # --- Seeded User ---
    seed_username: str = Field(default="SkyLogic")
    seed_password: str = Field(default="skylogic2026")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def patches_dir(self) -> Path:
        return self.data_dir / "patches"

    @property
    def train_images_dir(self) -> Path:
        return self.raw_data_dir / "train_images"

    @property
    def val_images_dir(self) -> Path:
        return self.raw_data_dir / "val_images"

    @property
    def train_labels_dir(self) -> Path:
        return self.raw_data_dir / "train_labels"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


settings = Settings()
