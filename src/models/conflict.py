"""Conflict tracking for the memory system."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

ConflictResolution = Literal["recency", "keep_both", "user_choice", "manual"]


@dataclass
class ConflictRecord:
    """
    Record of a detected conflict between memory entries.

    Tracks what was contradicted, when it was detected, and how it was resolved.
    """

    conflict_id: str
    memcell_id: str  # The new MemCell that caused the conflict
    scene_id: str  # The MemScene where conflict was detected
    old_fact: str
    new_fact: str
    detected_at: datetime
    resolution: Optional[ConflictResolution] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.conflict_id:
            self.conflict_id = str(uuid.uuid4())

    @property
    def is_resolved(self) -> bool:
        """Check if this conflict has been resolved."""
        return self.resolution is not None and self.resolved_at is not None

    def resolve(self, resolution: ConflictResolution, notes: str = "") -> None:
        """Mark this conflict as resolved."""
        self.resolution = resolution
        self.resolved_at = datetime.now(timezone.utc)
        self.resolution_notes = notes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conflict_id": self.conflict_id,
            "memcell_id": self.memcell_id,
            "scene_id": self.scene_id,
            "old_fact": self.old_fact,
            "new_fact": self.new_fact,
            "detected_at": self.detected_at.isoformat(),
            "resolution": self.resolution,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConflictRecord":
        """Create from dictionary."""
        return cls(
            conflict_id=data["conflict_id"],
            memcell_id=data["memcell_id"],
            scene_id=data["scene_id"],
            old_fact=data["old_fact"],
            new_fact=data["new_fact"],
            detected_at=datetime.fromisoformat(data["detected_at"]),
            resolution=data.get("resolution"),
            resolved_at=datetime.fromisoformat(data["resolved_at"])
            if data.get("resolved_at")
            else None,
            resolution_notes=data.get("resolution_notes", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def create(
        cls,
        memcell_id: str,
        scene_id: str,
        old_fact: str,
        new_fact: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ConflictRecord":
        """Factory method to create a ConflictRecord."""
        return cls(
            conflict_id=str(uuid.uuid4()),
            memcell_id=memcell_id,
            scene_id=scene_id,
            old_fact=old_fact,
            new_fact=new_fact,
            detected_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )


@dataclass
class UserProfile:
    """
    Compact user profile extracted from consolidated MemScenes.

    Based on the User Profile concept from the EverMemOS paper:
    - Explicit facts: Verifiable attributes and time-varying measurements
    - Implicit traits: Preferences and habits inferred over time
    """

    user_id: str
    explicit_facts: Dict[str, Any] = field(default_factory=dict)
    implicit_traits: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.user_id:
            self.user_id = "default"

    def update_explicit_fact(
        self, key: str, value: Any, timestamp: Optional[datetime] = None
    ) -> None:
        """Update an explicit fact with a new value."""
        self.explicit_facts[key] = {
            "value": value,
            "updated_at": (timestamp or datetime.now(timezone.utc)).isoformat(),
        }
        self.last_updated = datetime.now(timezone.utc)

    def add_implicit_trait(self, trait: str) -> None:
        """Add an implicit trait if not already present."""
        if trait not in self.implicit_traits:
            self.implicit_traits.append(trait)
            self.last_updated = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "explicit_facts": self.explicit_facts,
            "implicit_traits": self.implicit_traits,
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            explicit_facts=data.get("explicit_facts", {}),
            implicit_traits=data.get("implicit_traits", []),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            metadata=data.get("metadata", {}),
        )
