"""Tests for MemScene consolidation."""

import pytest

from src.core import MemSceneConsolidator
from src.models import MemCell, MemScene
from src.storage import MemoryStore, SearchIndex
from src.utils import MockEmbeddings, MockProvider


@pytest.fixture
def mock_llm():
    return MockProvider()


@pytest.fixture
def mock_embeddings():
    return MockEmbeddings(dim=384)


@pytest.fixture
def store():
    from tests.mock_db import MockMilvusStorageClient, MockMongoStorageClient

    return MemoryStore(
        mongo_client=MockMongoStorageClient(), milvus_client=MockMilvusStorageClient()
    )


@pytest.fixture
def index(mock_embeddings):
    return SearchIndex(embedding_service=mock_embeddings)


@pytest.fixture
def consolidator(mock_llm, mock_embeddings, store, index):
    return MemSceneConsolidator(
        llm_provider=mock_llm,
        embedding_service=mock_embeddings,
        memory_store=store,
        search_index=index,
    )


def test_memscene_creation():
    """Test creating a MemScene."""
    scene = MemScene.create(
        theme="career",
        initial_memcell_id="mem_001",
        initial_embedding=[0.1, 0.2, 0.3],
        summary="User discussed their job.",
    )

    assert scene.scene_id is not None
    assert scene.theme == "career"
    assert "mem_001" in scene.memcell_ids
    assert scene.centroid == [0.1, 0.2, 0.3]


def test_memscene_add_memcell():
    """Test adding MemCells to a scene."""
    scene = MemScene.create(
        theme="health",
        initial_memcell_id="mem_001",
        initial_embedding=[0.1, 0.2, 0.3],
    )

    scene.add_memcell("mem_002")
    scene.add_memcell("mem_003")

    assert len(scene.memcell_ids) == 3
    assert "mem_002" in scene.memcell_ids
    assert "mem_003" in scene.memcell_ids


def test_memscene_remove_memcell():
    """Test removing MemCells from a scene."""
    scene = MemScene.create(
        theme="test",
        initial_memcell_id="mem_001",
        initial_embedding=[0.1, 0.2, 0.3],
    )

    scene.add_memcell("mem_002")
    assert scene.remove_memcell("mem_002") is True
    assert "mem_002" not in scene.memcell_ids

    # Removing non-existent should return False
    assert scene.remove_memcell("nonexistent") is False


def test_memscene_serialization():
    """Test MemScene to/from dict conversion."""
    scene = MemScene.create(
        theme="test",
        initial_memcell_id="mem_001",
        initial_embedding=[0.1, 0.2, 0.3],
        summary="Test summary",
    )

    data = scene.to_dict()

    assert "scene_id" in data
    assert data["theme"] == "test"
    assert len(data["memcell_ids"]) == 1
    assert data["summary"] == "Test summary"

    restored = MemScene.from_dict(data)

    assert restored.scene_id == scene.scene_id
    assert restored.theme == scene.theme
    assert len(restored.memcell_ids) == 1


def test_cluster_memcell(consolidator):
    """Test clustering a MemCell into a scene."""
    memcell = MemCell.create(
        episode="I work at Google as a software engineer.",
        atomic_facts=["User works at Google", "User is a software engineer"],
        foresight=[],
        source_messages=[{"role": "user", "content": "I work at Google."}],
    )

    scene = consolidator.cluster_memcell(memcell)

    assert scene is not None
    assert isinstance(scene, MemScene)
    assert memcell.event_id in scene.memcell_ids


def test_consolidate_memcell(consolidator, store):
    """Test full consolidation process."""
    memcell = MemCell.create(
        episode="I went to the gym today.",
        atomic_facts=["User went to the gym"],
        foresight=[],
        source_messages=[{"role": "user", "content": "I went to the gym today."}],
    )

    result = consolidator.consolidate(memcell)

    assert result["scene_id"] is not None
    assert result["theme"] is not None
    assert result["conflicts_detected"] == 0

    # Check storage
    assert store.get_memcell(memcell.event_id) is not None
    assert store.get_memscene(result["scene_id"]) is not None


def test_scene_counting(consolidator):
    """Test counting scenes by theme."""
    # Add various MemCells
    for i in range(3):
        memcell = MemCell.create(
            episode=f"Career discussion {i}",
            atomic_facts=[f"Career fact {i}"],
            foresight=[],
            source_messages=[],
        )
        consolidator.cluster_memcell(memcell)

    for i in range(2):
        memcell = MemCell.create(
            episode=f"Health discussion {i}",
            atomic_facts=[f"Health fact {i}"],
            foresight=[],
            source_messages=[],
        )
        consolidator.cluster_memcell(memcell)

    stats = consolidator.get_consolidation_stats()

    # With semantic clustering, similar episodes cluster together
    # All career discussions cluster into 1 scene, health into 1 scene
    assert stats["scene_count"] == 2
    assert stats["scenes_by_theme"]["career"] == 1
    assert stats["scenes_by_theme"]["health"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
