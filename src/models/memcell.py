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
class AtomicFact:
    """An atomic fact with confidence score."""

    text: str
    confidence: float = 1.0  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicFact":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class MemCell:
    """
    Atomic memory unit that captures episodic traces, atomic facts, and time-bounded foresight.

    Based on the MemCell concept from the EverMemOS paper:
    - Episode: Third-person narrative summary
    - Atomic Facts: Discrete, verifiable statements with confidence scores
    - Foresight: Time-bounded information with validity intervals
    - Metadata: Contextual grounding including timestamps and source
    """

    event_id: str
    episode: str
    atomic_facts: List[AtomicFact]
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
            "atomic_facts": [f.to_dict() for f in self.atomic_facts],
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
            atomic_facts=[AtomicFact.from_dict(f) for f in data["atomic_facts"]],
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
        atomic_facts: List,
        foresight: List[ForesightItem],
        source_messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "MemCell":
        """Factory method to create a MemCell.

        Args:
            atomic_facts: List of dicts with 'text' and 'confidence' keys,
                         or List[str] for backward compatibility
        """
        # Handle both dicts (new format) and strings (old format)
        facts = []
        for f in atomic_facts:
            if isinstance(f, dict):
                facts.append(
                    AtomicFact(
                        text=f.get("text", f.get("fact", "")),
                        confidence=f.get("confidence", 1.0),
                    )
                )
            elif isinstance(f, str):
                # Backward compatibility: plain strings get confidence 1.0
                facts.append(AtomicFact(text=f, confidence=1.0))
            elif isinstance(f, AtomicFact):
                facts.append(f)

        return cls(
            event_id=str(uuid.uuid4()),
            episode=episode,
            atomic_facts=facts,
            foresight=foresight or [],
            timestamp=timestamp or datetime.now(timezone.utc),
            source_messages=source_messages,
            metadata=metadata or {},
        )
