"""Tests for memory retrieval."""

from datetime import datetime, timedelta, timezone

import pytest

from src.core import MemorySystem


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
    # Add memories - with mock LLM, themes may be "general"
    # We just verify scenes are created
    system.add_conversation([{"role": "user", "content": "I work at a startup."}])
    system.add_conversation([{"role": "user", "content": "I went to the gym today."}])
    system.add_conversation([{"role": "user", "content": "I love playing guitar."}])

    theme_counts = system.get_scenes_by_theme()

    # Verify scenes were created (theme may vary with mock LLM)
    assert len(theme_counts) >= 1
    total_scenes = sum(theme_counts.values())
    assert total_scenes >= 3


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
