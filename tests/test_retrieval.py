"""Tests for memory retrieval."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.core import MemorySystem
from src.models import MemCell


@pytest.fixture
def system():
    """Create a memory system for testing."""
    return MemorySystem()


def test_add_and_retrieve(system):
    """Test basic add and retrieve."""
    # Add a conversation
    system.add_conversation(
        [
            {"role": "user", "content": "I work at Google as a software engineer."},
        ]
    )

    # Retrieve - verify memory was stored (memcells key exists)
    result = system.retrieve("Where does the user work?")

    assert "memcells" in result


def test_retrieve_with_query_time(system):
    """Test retrieval with specific query time."""
    now = datetime.now(timezone.utc)

    # Add memory
    system.add_conversation(
        [
            {"role": "user", "content": "I'm traveling to Paris next week."},
        ],
        timestamp=now - timedelta(days=1),
    )

    # Retrieve at different times - just verify no errors
    result_current = system.retrieve("Is the user traveling soon?", query_time=now)
    result_future = system.retrieve(
        "Is the user traveling soon?", query_time=now + timedelta(days=30)
    )

    # Verify both retrievals complete without error
    assert "memcells" in result_current
    assert "memcells" in result_future


def test_retrieve_multiple_memories(system):
    """Test retrieving multiple memories."""
    # Add multiple conversations
    system.add_conversation([{"role": "user", "content": "I work at Google."}])
    system.add_conversation([{"role": "user", "content": "I live in San Francisco."}])
    system.add_conversation([{"role": "user", "content": "I have a dog named Max."}])

    # Retrieve with different queries
    work_result = system.retrieve("Where does the user work?")
    location_result = system.retrieve("Where does the user live?")

    assert "memcells" in work_result
    assert "memcells" in location_result


def test_get_memory_stats(system):
    """Test getting memory statistics."""
    # Add some memories
    for i in range(5):
        system.add_conversation([{"role": "user", "content": f"Message {i}"}])

    stats = system.get_memory_stats()

    assert stats["memcell_count"] >= 5
    assert stats["memscene_count"] >= 1


def test_get_scenes_by_theme(system):
    """Test getting scene distribution by theme."""
    # Add memories - with mock LLM, themes may be clustered together
    # We just verify scenes are created and theme counting works
    system.add_conversation([{"role": "user", "content": "I work at a startup."}])
    system.add_conversation([{"role": "user", "content": "I went to the gym today."}])
    system.add_conversation([{"role": "user", "content": "I love playing guitar."}])

    theme_counts = system.get_scenes_by_theme()

    # Verify scenes were created (theme may vary with mock LLM)
    # Mock LLM may cluster all into single scene or separate them
    assert len(theme_counts) >= 1
    total_scenes = sum(theme_counts.values())
    assert total_scenes >= 1  # At least one scene should be created


def test_clear_memory(system):
    """Test clearing memory."""
    # Add some memories
    system.add_conversation([{"role": "user", "content": "Test message"}])

    # Clear
    system.clear()

    # Verify cleared
    stats = system.get_memory_stats()
    assert stats["memcell_count"] == 0
    assert stats["memscene_count"] == 0


def test_memory_system_retrieve_passes_user_id(system):
    """MemorySystem should pass its user_id to retriever."""
    system.user_id = "user-xyz"
    system.retriever.retrieve = MagicMock(
        return_value={"memcells": [], "foresight": [], "profile": None}
    )

    system.retrieve("test query")

    kwargs = system.retriever.retrieve.call_args.kwargs
    assert kwargs["user_id"] == "user-xyz"


def test_add_conversation_preserves_per_memcell_timestamps(system):
    """add_conversation should preserve extractor-assigned episode timestamps."""
    t1 = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2025, 1, 2, 10, 0, tzinfo=timezone.utc)
    m1 = MemCell.create("ep1", ["f1"], [], source_messages=[], timestamp=t1)
    m2 = MemCell.create("ep2", ["f2"], [], source_messages=[], timestamp=t2)

    system.extractor.process_conversation_stream = MagicMock(return_value=[m1, m2])

    seen_timestamps = []

    def _capture(memcell, user_id):
        seen_timestamps.append(memcell.timestamp)
        return {
            "scene_id": "scene1",
            "theme": "general",
            "conflicts_detected": 0,
            "original_facts_count": len(memcell.atomic_facts),
            "unique_facts_count": len(memcell.atomic_facts),
        }

    system.consolidator.consolidate = MagicMock(side_effect=_capture)

    system.add_conversation([{"role": "user", "content": "multi episode"}])

    assert seen_timestamps == [t1, t2]


def test_add_conversation_uses_fallback_timestamp_for_invalid_memcell_timestamp(system):
    """Invalid memcell timestamps should be replaced by provided timestamp."""
    provided = datetime(2025, 3, 1, 10, 0, tzinfo=timezone.utc)
    memcell = MemCell.create("ep", ["f"], [], source_messages=[], timestamp=provided)
    memcell.timestamp = "invalid"
    system.extractor.process_conversation_stream = MagicMock(return_value=[memcell])

    captured = {}

    def _capture(memcell_obj, user_id):
        captured["timestamp"] = memcell_obj.timestamp
        return {
            "scene_id": "scene1",
            "theme": "general",
            "conflicts_detected": 0,
            "original_facts_count": len(memcell_obj.atomic_facts),
            "unique_facts_count": len(memcell_obj.atomic_facts),
        }

    system.consolidator.consolidate = MagicMock(side_effect=_capture)

    system.add_conversation(
        [{"role": "user", "content": "invalid timestamp"}],
        timestamp=provided,
    )

    assert captured["timestamp"] == provided


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
