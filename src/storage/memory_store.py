"""Storage implementation using MongoDB and Milvus."""

import os
from typing import Any, Dict, List, Optional

from ..models import ConflictRecord, MemCell, MemScene, UserProfile
from .milvus_client import MilvusStorageClient
from .mongo_client import MongoStorageClient


class MemoryStore:
    """
    Persistent storage using MongoDB (structured) and Milvus (vectors).

    Replaces the previous in-memory/JSON implementation.
    """

    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        embedding_dim: int = 1536,
        **kwargs,
    ):
        """
        Initialize the memory store with database clients.

        Args:
            mongo_uri: Connection string for MongoDB
            milvus_host: Host for Milvus
            milvus_port: Port for Milvus
            embedding_dim: Dimension of embeddings (must match embedding service)
            mongo_client: Optional injected Mongo client (for testing)
            milvus_client: Optional injected Milvus client (for testing)
        """
        # Initialize clients
        # We allow env vars to override defaults
        mongo_uri = os.getenv("MONGO_URI", mongo_uri)
        milvus_host = os.getenv("MILVUS_HOST", milvus_host)
        milvus_port = os.getenv("MILVUS_PORT", milvus_port)

        if "mongo_client" in kwargs:
            self.mongo = kwargs["mongo_client"]
        else:
            self.mongo = MongoStorageClient(uri=mongo_uri)

        if "milvus_client" in kwargs:
            self.milvus = kwargs["milvus_client"]
            self._milvus_available = True
        else:
            try:
                self.milvus = MilvusStorageClient(
                    host=milvus_host, port=milvus_port, dim=embedding_dim
                )
                self._milvus_available = True
            except Exception:
                self._milvus_available = False

    def add_memcell(self, memcell: MemCell) -> None:
        """Add a MemCell to storage (Mongo + Milvus)."""
        # 1. Save structured data to Mongo
        self.mongo.add_memcell(memcell)

        # 2. Save vector to Milvus if embedding exists
        if self._milvus_available and memcell.embedding:
            self.milvus.add_memcell_embedding(memcell, memcell.embedding)

    def get_memcell(self, event_id: str) -> Optional[MemCell]:
        """Get a MemCell by ID."""
        return self.mongo.get_memcell(event_id)

    def get_all_memcells(self) -> List[MemCell]:
        """Get all MemCells."""
        return self.mongo.get_all_memcells()

    def get_memcells_by_ids(self, event_ids: List[str]) -> List[MemCell]:
        """Get multiple MemCells by their IDs."""
        return self.mongo.get_memcells_by_ids(event_ids)

    def get_memcells_by_scene(self, scene_id: str) -> List[MemCell]:
        """Get all MemCells in a scene."""
        # First get the scene to get the IDs
        scene = self.mongo.get_memscene(scene_id)
        if not scene or not scene.memcell_ids:
            return []

        # Then fetch the cells
        return self.mongo.get_memcells_by_ids(scene.memcell_ids)

    def add_memscene(self, memscene: MemScene) -> None:
        """Add a MemScene to storage."""
        # 1. Save structured data to Mongo
        self.mongo.add_memscene(memscene)

        # 2. Save vector to Milvus if centroid exists
        if self._milvus_available and memscene.centroid:
            self.milvus.add_memscene_embedding(memscene, memscene.centroid)

    def get_memscene(self, scene_id: str) -> Optional[MemScene]:
        """Get a MemScene by ID."""
        return self.mongo.get_memscene(scene_id)

    def get_all_memscenes(self) -> List[MemScene]:
        """Get all MemScenes."""
        return self.mongo.get_all_memscenes()

    def add_conflict(self, conflict: ConflictRecord) -> None:
        """Add a ConflictRecord."""
        self.mongo.add_conflict(conflict)

    def get_conflict(self, conflict_id: str) -> Optional[ConflictRecord]:
        """Get a ConflictRecord by ID."""
        return self.mongo.get_conflict(conflict_id)

    def get_all_conflicts(self) -> List[ConflictRecord]:
        """Get all ConflictRecords."""
        return self.mongo.get_all_conflicts()

    def get_unresolved_conflicts(self) -> List[ConflictRecord]:
        """Get all unresolved conflicts."""
        return self.mongo.get_unresolved_conflicts()

    def get_user_profile(self, user_id: str = "default") -> Optional[UserProfile]:
        """Get a UserProfile by ID."""
        return self.mongo.get_user_profile(user_id)

    def save_user_profile(self, profile: UserProfile) -> None:
        """Save or update a UserProfile."""
        self.mongo.save_user_profile(profile)

    def get_or_create_profile(self, user_id: str = "default") -> UserProfile:
        """Get a UserProfile, creating one if it doesn't exist."""
        profile = self.get_user_profile(user_id)
        if not profile:
            profile = UserProfile(user_id=user_id)
            self.save_user_profile(profile)
        return profile

    # --- Vector Search Methods (New) ---

    def search_memcells(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[MemCell]:
        """Search MemCells by vector similarity."""
        if not self._milvus_available:
            return []

        hits = self.milvus.search_memcells(query_embedding, top_k=top_k)
        # Fetch the full objects from Mongo
        event_ids = [hit["event_id"] for hit in hits]
        # Maintain order based on hits
        memcells = self.mongo.get_memcells_by_ids(event_ids)

        # Sort memcells by the order of event_ids (which are sorted by similarity)
        memcell_map = {mc.event_id: mc for mc in memcells}
        ordered_memcells = []
        for eid in event_ids:
            if eid in memcell_map:
                ordered_memcells.append(memcell_map[eid])

        return ordered_memcells

    # --- Legacy / Compatibility ---

    def save_to_disk(self) -> None:
        """No-op: Persistence is handled by databases."""
        pass

    def load_from_disk(self) -> None:
        """No-op: Persistence is handled by databases."""
        pass

    def clear(self) -> None:
        """Clear all data from databases."""
        self.mongo.clear()
        if self._milvus_available:
            self.milvus.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        # We could query counts from Mongo
        return {
            "memcell_count": self.mongo.memcells.count_documents({}),
            "memscene_count": self.mongo.memscenes.count_documents({}),
            "conflict_count": self.mongo.conflicts.count_documents({}),
            "profile_count": self.mongo.profiles.count_documents({}),
            "backend": "mongodb+milvus",
        }
