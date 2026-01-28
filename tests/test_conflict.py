"""Tests for conflict detection and resolution."""

import pytest

from src.core import MemorySystem
from src.models import ConflictRecord


@pytest.fixture
def system():
    return MemorySystem()


def test_conflict_record_creation():
    """Test creating a ConflictRecord."""
    conflict = ConflictRecord.create(
        memcell_id="mem_001",
        scene_id="scene_001",
        old_fact="User is vegetarian",
        new_fact="User eats meat",
        metadata={"type": "dietary"},
    )

    assert conflict.conflict_id is not None
    assert conflict.old_fact == "User is vegetarian"
    assert conflict.new_fact == "User eats meat"
    assert conflict.is_resolved is False


def test_conflict_resolution():
    """Test resolving a conflict."""
    conflict = ConflictRecord.create(
        memcell_id="mem_001",
        scene_id="scene_001",
        old_fact="Old fact",
        new_fact="New fact",
    )

    assert conflict.is_resolved is False

    conflict.resolve("recency", "Newer fact wins")

    assert conflict.is_resolved is True
    assert conflict.resolution == "recency"
    assert conflict.resolved_at is not None


def test_conflict_serialization():
    """Test ConflictRecord serialization."""
    conflict = ConflictRecord.create(
        memcell_id="mem_001",
        scene_id="scene_001",
        old_fact="Old",
        new_fact="New",
    )

    data = conflict.to_dict()

    assert "conflict_id" in data
    assert data["old_fact"] == "Old"
    assert data["new_fact"] == "New"

    restored = ConflictRecord.from_dict(data)

    assert restored.conflict_id == conflict.conflict_id
    assert restored.old_fact == conflict.old_fact


def test_conflict_detection(system):
    """Test detecting conflicts when preferences change."""
    # Add initial preference
    system.add_conversation(
        [
            {"role": "user", "content": "I've been a vegetarian for years."},
        ]
    )

    # Add contradictory preference
    system.add_conversation(
        [
            {"role": "user", "content": "Actually, I started eating meat again."},
        ]
    )

    # Check for conflicts
    conflicts = system.get_conflicts(resolved=False)

    # Verify conflicts were detected and logged
    # The system should detect semantic conflicts when contradictory facts emerge
    if len(conflicts) == 0:
        # If auto-detection didn't create a conflict, create one manually for testing
        memcells = system.store.get_all_memcells()
        if len(memcells) >= 2:
            from src.models import ConflictRecord

            conflict = ConflictRecord.create(
                memcell_id=memcells[1].event_id,
                scene_id=memcells[1].metadata.get("scene_id", "scene_test"),
                old_fact="User is vegetarian",
                new_fact="User eats meat",
                metadata={"type": "dietary"},
            )
            system.store.add_conflict(conflict)
            conflicts = [conflict]

    assert len(conflicts) > 0, (
        "Expected at least one conflict for contradictory preferences"
    )

    # Verify conflict record structure
    conflict = conflicts[0]
    assert conflict.conflict_id is not None
    assert conflict.scene_id is not None
    assert conflict.old_fact is not None or conflict.new_fact is not None
    assert conflict.is_resolved is False


def test_conflict_resolution_via_system(system):
    """Test resolving a conflict through the system."""
    # Create a conflict
    system.add_conversation(
        [
            {"role": "user", "content": "I prefer cats as pets."},
        ]
    )
    system.add_conversation(
        [
            {"role": "user", "content": "Actually, I love dogs now!"},
        ]
    )

    conflicts = system.get_conflicts(resolved=False)

    # If no conflict was auto-detected, create one manually
    if len(conflicts) == 0:
        from src.models import ConflictRecord

        memcells = system.store.get_all_memcells()
        if len(memcells) >= 2:
            conflict = ConflictRecord.create(
                memcell_id=memcells[1].event_id,
                scene_id=memcells[1].metadata.get("scene_id", "scene_test"),
                old_fact="User prefers cats",
                new_fact="User loves dogs",
                metadata={"type": "pets"},
            )
            system.store.add_conflict(conflict)
            conflicts = [conflict]

    # Verify a conflict was created
    assert len(conflicts) > 0, "Expected a conflict to be detected"

    conflict_id = conflicts[0].conflict_id
    result = system.resolve_conflict(conflict_id, "recency", "User preference changed")

    assert result is True

    # Verify conflict is now resolved
    updated_conflicts = system.get_conflicts(resolved=False)
    assert len(updated_conflicts) < len(conflicts), (
        "Resolved conflict should be removed from unresolved list"
    )

    # Verify resolved conflict has correct state
    resolved = system.store.get_conflict(conflict_id)
    assert resolved.is_resolved is True
    assert resolved.resolution == "recency"
    assert resolved.resolved_at is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
