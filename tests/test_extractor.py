"""Tests for MemCell extraction."""

from datetime import datetime, timedelta, timezone

import pytest

from src.core import MemCellExtractor
from src.models import ForesightItem, MemCell
from src.utils import MockEmbeddings, MockProvider


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider for testing."""
    return MockProvider()


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    return MockEmbeddings(dim=384)


@pytest.fixture
def extractor(mock_llm, mock_embeddings):
    """Create a MemCell extractor for testing."""
    return MemCellExtractor(
        llm_provider=mock_llm,
        embedding_service=mock_embeddings,
    )


def test_memcell_creation(extractor):
    """Test creating a MemCell from messages."""
    messages = [
        {"role": "user", "content": "I work at Google."},
        {"role": "assistant", "content": "What do you do there?"},
        {"role": "user", "content": "I'm a software engineer."},
    ]

    memcell = extractor.create_memcell(messages)

    assert memcell is not None
    assert isinstance(memcell, MemCell)
    assert memcell.event_id is not None
    assert len(memcell.source_messages) == 3


def test_memcell_factory_method():
    """Test MemCell.create factory method."""
    memcell = MemCell.create(
        episode="User discussed their job.",
        atomic_facts=["User works at Google", "User is a software engineer"],
        foresight=[],
        source_messages=[{"role": "user", "content": "I work at Google."}],
    )

    assert memcell.event_id is not None
    assert memcell.episode == "User discussed their job."
    assert len(memcell.atomic_facts) == 2


def test_memcell_serialization():
    """Test MemCell to/from dict conversion."""
    memcell = MemCell.create(
        episode="Test episode",
        atomic_facts=["Fact 1", "Fact 2"],
        foresight=[
            ForesightItem(
                description="Test foresight",
                end_time=datetime.now(timezone.utc),
            )
        ],
        source_messages=[],
    )

    # Serialize
    data = memcell.to_dict()

    assert "event_id" in data
    assert data["episode"] == "Test episode"
    assert len(data["atomic_facts"]) == 2
    assert len(data["foresight"]) == 1

    # Deserialize
    restored = MemCell.from_dict(data)

    assert restored.event_id == memcell.event_id
    assert restored.episode == memcell.episode
    assert len(restored.atomic_facts) == 2


def test_foresight_validation():
    """Test ForesightItem validity checking."""
    now = datetime.now(timezone.utc)

    # Valid foresight
    valid = ForesightItem(
        description="Test",
        start_time=now - timedelta(days=1),
        end_time=now + timedelta(days=1),
    )
    assert valid.is_valid_at(now) is True

    # Expired foresight
    expired = ForesightItem(
        description="Test",
        start_time=now - timedelta(days=10),
        end_time=now - timedelta(days=5),
    )
    assert expired.is_valid_at(now) is False

    # Future foresight
    future = ForesightItem(
        description="Test",
        start_time=now + timedelta(days=5),
        end_time=now + timedelta(days=10),
    )
    assert future.is_valid_at(now) is False


def test_conversation_stream_processing(extractor):
    """Test processing a stream of messages."""
    messages = [
        {"role": "user", "content": "Message 1"},
        {"role": "user", "content": "Message 2"},
        {"role": "user", "content": "Message 3"},
    ]

    memcells = extractor.process_conversation_stream(messages, flush=True)

    # Should create at least one MemCell
    assert len(memcells) >= 1
    assert all(isinstance(m, MemCell) for m in memcells)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
