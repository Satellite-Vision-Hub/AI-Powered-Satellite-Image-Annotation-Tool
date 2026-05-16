"""
SkyLogic MAS — Qdrant Vector Store Client
Stores and searches SAM embeddings for similarity-based auto-annotation.
"""

import logging
from typing import Optional

import numpy as np

from skylogic.config import settings

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Qdrant client for storing and searching SAM feature embeddings.
    
    Enables "click one, annotate all similar" workflow:
        1. User clicks an object → SAM generates mask + embedding
        2. Embedding is stored in Qdrant with metadata
        3. Similarity search finds visually similar objects across all patches
        4. Auto-annotations are created for similar regions
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        vector_size: int = 256,
    ):
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection
        self.vector_size = vector_size
        self.client = None

        self._connect()

    def _connect(self):
        """Initialize Qdrant client and ensure collection exists."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self.client = QdrantClient(host=self.host, port=self.port)

            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection exists: {self.collection_name}")

        except Exception as e:
            logger.warning(f"Qdrant connection failed: {e}. Vector search disabled.")
            self.client = None

    def store_embedding(
        self,
        point_id: str,
        embedding: np.ndarray,
        metadata: dict,
    ) -> bool:
        """
        Store a SAM embedding with metadata.
        
        Args:
            point_id: Unique ID for the point.
            embedding: Feature vector (256-d).
            metadata: Payload dict (patch_id, class_name, bbox, etc.).
        
        Returns:
            True if stored successfully.
        """
        if self.client is None:
            logger.warning("Qdrant not connected. Skipping store.")
            return False

        try:
            from qdrant_client.models import PointStruct

            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=metadata,
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

            logger.info(f"Stored embedding {point_id} in Qdrant")
            return True

        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            return False

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        score_threshold: float = 0.75,
    ) -> list[dict]:
        """
        Search for visually similar objects using cosine similarity.
        
        Args:
            query_embedding: Query feature vector (256-d).
            top_k: Number of results to return.
            score_threshold: Minimum similarity score (0-1).
        
        Returns:
            List of dicts with {point_id, score, payload}.
        """
        if self.client is None:
            return []

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
            )

            similar = []
            for hit in results:
                similar.append({
                    "point_id": str(hit.id),
                    "score": float(hit.score),
                    "payload": hit.payload,
                })

            logger.info(
                f"Similarity search: {len(similar)} results "
                f"(threshold={score_threshold})"
            )
            return similar

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def delete_embedding(self, point_id: str) -> bool:
        """Delete a stored embedding by ID."""
        if self.client is None:
            return False

        try:
            from qdrant_client.models import PointIdsList

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[point_id]),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete embedding: {e}")
            return False

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        if self.client is None:
            return {"status": "disconnected"}

        try:
            info = self.client.get_collection(self.collection_name)
            # qdrant-client >= 1.10 removed the legacy `vectors_count` attribute
            # (it lives on info.config.params.vectors now). Be tolerant of both.
            vec_count = getattr(info, "vectors_count", None)
            if vec_count is None:
                vec_count = getattr(info, "points_count", 0)
            return {
                "status": "connected",
                "points_count": getattr(info, "points_count", 0),
                "vectors_count": vec_count,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
