"""MemScene: Thematic grouping of MemCells."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MemScene:
    """
    Thematic grouping of MemCells that enables coherent context retrieval.

    Based on the MemScene concept from the EverMemOS paper:
    - Groups related MemCells by theme (career, health, relationships, etc.)
    - Maintains a centroid embedding for similarity-based clustering
    - Supports scene-level summaries for user profile evolution
    """

    scene_id: str
    theme: str
    memcell_ids: List[str]
    centroid: Optional[List[float]] = None
    summary: str = ""
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    earliest_timestamp: Optional[datetime] = None
    latest_timestamp: Optional[datetime] = None

    def __post_init__(self):
        if not self.scene_id:
            self.scene_id = f"scene_{uuid.uuid4().hex[:8]}"

    @property
    def memcell_count(self) -> int:
        """Return the number of MemCells in this scene."""
        return len(self.memcell_ids)

    def add_memcell(self, memcell_id: str) -> None:
        """Add a MemCell to this scene."""
        if memcell_id not in self.memcell_ids:
            self.memcell_ids.append(memcell_id)
            self.last_updated = datetime.now(timezone.utc)

    def update_time_range(self, timestamp: datetime) -> None:
        """Update the time range based on a new timestamp."""
        if self.earliest_timestamp is None or timestamp < self.earliest_timestamp:
            self.earliest_timestamp = timestamp
        if self.latest_timestamp is None or timestamp > self.latest_timestamp:
            self.latest_timestamp = timestamp

    def remove_memcell(self, memcell_id: str) -> bool:
        """Remove a MemCell from this scene. Returns True if found and removed."""
        if memcell_id in self.memcell_ids:
            self.memcell_ids.remove(memcell_id)
            self.last_updated = datetime.now(timezone.utc)
            return True
        return False

    def update_centroid(self, new_embedding: List[float], weight: float = 1.0) -> None:
        """Update the centroid with a new embedding."""
        if self.centroid is None:
            self.centroid = new_embedding
        else:
            # Incremental centroid update
            existing_weight = len(self.memcell_ids) - weight
            if existing_weight > 0:
                centroid_array = np.array(self.centroid)
                new_array = np.array(new_embedding)
                result = ((centroid_array * existing_weight) + (new_array * weight)) / (
                    existing_weight + weight
                )
                self.centroid = result.tolist()
            else:
                self.centroid = new_embedding
        self.last_updated = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scene_id": self.scene_id,
            "theme": self.theme,
            "memcell_ids": self.memcell_ids,
            "centroid": self.centroid,
            "summary": self.summary,
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata,
            "earliest_timestamp": self.earliest_timestamp.isoformat()
            if self.earliest_timestamp
            else None,
            "latest_timestamp": self.latest_timestamp.isoformat()
            if self.latest_timestamp
            else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemScene":
        """Create from dictionary."""
        return cls(
            scene_id=data["scene_id"],
            theme=data["theme"],
            memcell_ids=data["memcell_ids"],
            centroid=data.get("centroid"),
            summary=data.get("summary", ""),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            metadata=data.get("metadata", {}),
            earliest_timestamp=datetime.fromisoformat(data["earliest_timestamp"])
            if data.get("earliest_timestamp")
            else None,
            latest_timestamp=datetime.fromisoformat(data["latest_timestamp"])
            if data.get("latest_timestamp")
            else None,
        )

    @classmethod
    def create(
        cls,
        theme: str,
        initial_memcell_id: str,
        initial_embedding: List[float],
        summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MemScene":
        """Factory method to create a new MemScene with initial MemCell."""
        return cls(
            scene_id=f"scene_{uuid.uuid4().hex[:8]}",
            theme=theme,
            memcell_ids=[initial_memcell_id],
            centroid=initial_embedding,
            summary=summary,
            metadata=metadata or {},
        )
