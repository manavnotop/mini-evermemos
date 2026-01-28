"""MemCell: Atomic memory unit for the EverMemOS-inspired memory system."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ForesightItem:
    """Time-bounded information with validity intervals."""

    description: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    confidence: float = 1.0  # 0.0 to 1.0

    def is_valid_at(self, query_time: datetime) -> bool:
        """Check if this foresight is valid at the given query time."""
        if self.start_time and query_time < self.start_time:
            return False
        if self.end_time and query_time > self.end_time:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "description": self.description,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForesightItem":
        """Create from dictionary."""
        return cls(
            description=data["description"],
            start_time=datetime.fromisoformat(data["start_time"])
            if data.get("start_time")
            else None,
            end_time=datetime.fromisoformat(data["end_time"])
            if data.get("end_time")
            else None,
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class MemCell:
    """
    Atomic memory unit that captures episodic traces, atomic facts, and time-bounded foresight.

    Based on the MemCell concept from the EverMemOS paper:
    - Episode: Third-person narrative summary
    - Atomic Facts: Discrete, verifiable statements
    - Foresight: Time-bounded information with validity intervals
    - Metadata: Contextual grounding including timestamps and source
    """

    event_id: str
    episode: str
    atomic_facts: List[str]
    foresight: List[ForesightItem]
    timestamp: datetime
    source_messages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    scene_id: Optional[str] = None

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "episode": self.episode,
            "atomic_facts": self.atomic_facts,
            "foresight": [f.to_dict() for f in self.foresight],
            "timestamp": self.timestamp.isoformat(),
            "source_messages": self.source_messages,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "scene_id": self.scene_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemCell":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            episode=data["episode"],
            atomic_facts=data["atomic_facts"],
            foresight=[ForesightItem.from_dict(f) for f in data["foresight"]],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_messages=data["source_messages"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            scene_id=data.get("scene_id"),
        )

    @classmethod
    def create(
        cls,
        episode: str,
        atomic_facts: List[str],
        foresight: List[ForesightItem],
        source_messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "MemCell":
        """Factory method to create a MemCell."""
        return cls(
            event_id=str(uuid.uuid4()),
            episode=episode,
            atomic_facts=atomic_facts,
            foresight=foresight or [],
            timestamp=timestamp or datetime.now(timezone.utc),
            source_messages=source_messages,
            metadata=metadata or {},
        )
