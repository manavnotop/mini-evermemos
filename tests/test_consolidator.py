"""Tests for MemScene consolidation."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.core import MemSceneConsolidator
from src.core.constants import SIMILARITY_THRESHOLDS
from src.models import AtomicFact, ConflictRecord, MemCell, MemScene, UserProfile
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
    stored = store.get_memcell(memcell.event_id)
    assert stored is not None
    assert stored.scene_id == result["scene_id"]
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


def test_detect_conflicts_excludes_current_memcell(consolidator, store):
    """Conflict detection should skip facts from the incoming memcell itself."""
    memcell = MemCell.create(
        episode="User likes tea.",
        atomic_facts=[AtomicFact(text="User likes tea")],
        foresight=[],
        source_messages=[],
    )
    store.add_memcell(memcell)
    scene = MemScene.create("hobbies", memcell.event_id, [0.1, 0.2, 0.3])
    store.add_memscene(scene)

    llm = MagicMock()
    llm.complete_json.return_value = {"conflicts": []}
    consolidator.llm = llm

    conflicts = consolidator.detect_conflicts(memcell, scene)

    assert conflicts == []
    llm.complete_json.assert_not_called()


def test_detect_conflicts_uses_string_fact_keys_no_unhashable_error(consolidator, store):
    """AtomicFact objects should be handled safely without dict-key hash errors."""
    old_memcell = MemCell.create(
        episode="User is vegetarian.",
        atomic_facts=[AtomicFact(text="User is vegetarian")],
        foresight=[],
        source_messages=[],
    )
    new_memcell = MemCell.create(
        episode="User eats meat now.",
        atomic_facts=[AtomicFact(text="User eats meat now")],
        foresight=[],
        source_messages=[],
    )
    store.add_memcell(old_memcell)
    store.add_memcell(new_memcell)
    scene = MemScene.create("health", old_memcell.event_id, [0.1, 0.2, 0.3])
    scene.add_memcell(new_memcell.event_id)
    store.add_memscene(scene)

    llm = MagicMock()
    llm.complete_json.return_value = {"conflicts": []}
    consolidator.llm = llm

    conflicts = consolidator.detect_conflicts(new_memcell, scene)

    assert conflicts == []
    llm.complete_json.assert_called_once()


def test_detect_conflicts_related_scene_threshold_uses_constant(consolidator, store):
    """Cross-scene lookup should use the configured related-scene threshold."""
    old_memcell = MemCell.create(
        episode="User has a dog.",
        atomic_facts=[AtomicFact(text="User has a dog")],
        foresight=[],
        source_messages=[],
    )
    new_memcell = MemCell.create(
        episode="User has a cat.",
        atomic_facts=[AtomicFact(text="User has a cat")],
        foresight=[],
        source_messages=[],
    )
    store.add_memcell(old_memcell)
    store.add_memcell(new_memcell)
    scene = MemScene.create("relationships", old_memcell.event_id, [0.1, 0.2, 0.3])
    scene.add_memcell(new_memcell.event_id)
    store.add_memscene(scene)

    consolidator._find_related_scenes = MagicMock(return_value=[])
    llm = MagicMock()
    llm.complete_json.return_value = {"conflicts": []}
    consolidator.llm = llm

    consolidator.detect_conflicts(new_memcell, scene, check_cross_scene=True)

    consolidator._find_related_scenes.assert_called_once_with(
        scene, similarity_threshold=SIMILARITY_THRESHOLDS["related_scenes"]
    )


def test_global_dedup_uses_fact_embeddings_not_episode_embedding(store, index):
    """Dedup should compare fact embeddings, not existing MemCell episode embeddings."""

    class CustomEmbeddings:
        def embed(self, text):
            return [1.0, 0.0] if "episode" in text.lower() else [0.0, 1.0]

        def embed_batch(self, texts):
            vectors = []
            for text in texts:
                if "existing fact" in text.lower():
                    vectors.append([0.0, 1.0])
                elif "new fact" in text.lower():
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.5, 0.5])
            return vectors

        def similarity(self, a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    consolidator = MemSceneConsolidator(MockProvider(), CustomEmbeddings(), store, index)

    existing = MemCell.create(
        episode="Existing episode context.",
        atomic_facts=[AtomicFact(text="Existing fact")],
        foresight=[],
        source_messages=[],
    )
    existing.embedding = [1.0, 0.0]
    store.add_memcell(existing)
    scene = MemScene.create("general", existing.event_id, [1.0, 0.0])
    store.add_memscene(scene)

    new_facts = [AtomicFact(text="New fact")]
    unique_facts, _, unique_count = consolidator.deduplicate_facts_globally(
        new_facts, scene, current_memcell_id="incoming"
    )

    assert unique_count == 1
    assert unique_facts[0].text == "New fact"


def test_consolidate_indexes_post_dedup_state(consolidator, store, index):
    """Index should contain post-dedup facts, not stale pre-dedup facts."""
    existing = MemCell.create(
        episode="User works at Stripe.",
        atomic_facts=[AtomicFact(text="User works at Stripe")],
        foresight=[],
        source_messages=[],
    )
    consolidator.consolidate(existing)

    incoming = MemCell.create(
        episode="User works at Stripe and likes backend.",
        atomic_facts=[
            AtomicFact(text="User works at Stripe"),
            AtomicFact(text="User likes backend engineering"),
        ],
        foresight=[],
        source_messages=[],
    )

    result = consolidator.consolidate(incoming)
    stored = store.get_memcell(incoming.event_id)

    assert stored.scene_id == result["scene_id"]
    assert len(stored.atomic_facts) == 1
    assert stored.atomic_facts[0].text == "User likes backend engineering"
    assert index._doc_facts[incoming.event_id] == ["User likes backend engineering"]


def test_resolve_conflict_recency_removes_old_fact_and_persists(consolidator, store):
    """Recency resolution should retire old facts and keep audit metadata."""
    old_memcell = MemCell.create(
        episode="User is vegetarian.",
        atomic_facts=[AtomicFact(text="User is vegetarian")],
        foresight=[],
        source_messages=[],
    )
    new_memcell = MemCell.create(
        episode="User eats meat now.",
        atomic_facts=[AtomicFact(text="User eats meat now")],
        foresight=[],
        source_messages=[],
    )
    store.add_memcell(old_memcell)
    store.add_memcell(new_memcell)

    conflict = ConflictRecord.create(
        memcell_id=new_memcell.event_id,
        scene_id="scene_test",
        old_fact="User is vegetarian",
        new_fact="User eats meat now",
        metadata={
            "old_fact_source_memcell_ids": [old_memcell.event_id],
            "new_fact_memcell_id": new_memcell.event_id,
        },
    )
    store.add_conflict(conflict)

    consolidator.resolve_conflict(conflict, resolution="recency")

    updated_old = store.get_memcell(old_memcell.event_id)
    assert updated_old.atomic_facts == []
    assert "superseded_facts" in updated_old.metadata
    persisted = store.get_conflict(conflict.conflict_id)
    assert persisted.is_resolved is True
    assert persisted.resolution == "recency"


def test_resolve_conflict_persists_conflict_resolution_status(consolidator, store):
    """Resolved conflicts should no longer appear in unresolved list."""
    conflict = ConflictRecord.create(
        memcell_id="new_mem",
        scene_id="scene_test",
        old_fact="Old",
        new_fact="New",
    )
    store.add_conflict(conflict)

    consolidator.resolve_conflict(conflict, resolution="keep_both")

    unresolved = store.get_unresolved_conflicts()
    assert conflict.conflict_id not in [item.conflict_id for item in unresolved]


def test_update_user_profile_prefers_newer_existing_value(consolidator, store):
    """Older scene evidence should not overwrite a newer explicit fact."""
    profile = UserProfile(user_id="default")
    profile.update_explicit_fact(
        "job",
        "Current Job",
        timestamp=datetime.now(timezone.utc),
    )
    store.save_user_profile(profile)

    older_scene_time = datetime.now(timezone.utc) - timedelta(days=7)
    scene = MemScene.create("career", "mem_old", [0.1, 0.2, 0.3], summary="User has old role")
    scene.latest_timestamp = older_scene_time
    store.add_memscene(scene)

    llm = MagicMock()
    llm.complete_json.return_value = {
        "explicit_facts": {"job": "Old Job"},
        "implicit_traits": [],
    }
    consolidator.llm = llm

    updated = consolidator.update_user_profile(user_id="default")
    assert updated.explicit_facts["job"]["value"] == "Current Job"


def test_search_index_handles_atomic_fact_objects(index):
    """BM25 indexing should support AtomicFact objects without type errors."""
    memcell = MemCell.create(
        episode="User plays tennis.",
        atomic_facts=[AtomicFact(text="User plays tennis")],
        foresight=[],
        source_messages=[],
    )
    index.add_memcell(memcell)

    assert index._doc_facts[memcell.event_id] == ["User plays tennis"]
    results = index.search_bm25("tennis")
    assert isinstance(results, list)


def test_consolidate_conflict_first_preserves_conflicting_new_fact(store, index):
    """Conflicting new facts should survive dedup in the same consolidation pass."""

    class HighSimilarityEmbeddings:
        def embed(self, text):
            return [1.0, 0.0]

        def embed_batch(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def similarity(self, a, b):
            return 0.99

    llm = MagicMock()
    llm.complete_json.return_value = {
        "conflicts": [
            {
                "old_fact": "User is vegetarian",
                "new_fact": "User eats meat now",
                "confidence": 0.95,
                "conflict_type": "preference_change",
                "explanation": "Contradictory dietary preference",
            }
        ]
    }
    consolidator = MemSceneConsolidator(llm, HighSimilarityEmbeddings(), store, index)

    old_memcell = MemCell.create(
        episode="User is vegetarian.",
        atomic_facts=[AtomicFact(text="User is vegetarian")],
        foresight=[],
        source_messages=[],
    )
    store.add_memcell(old_memcell)
    scene = MemScene.create("health", old_memcell.event_id, [1.0, 0.0], summary="diet")
    store.add_memscene(scene)

    incoming = MemCell.create(
        episode="User now eats meat.",
        atomic_facts=[AtomicFact(text="User eats meat now")],
        foresight=[],
        source_messages=[],
    )
    scene.add_memcell(incoming.event_id)
    store.add_memscene(scene)

    consolidator.cluster_memcell = MagicMock(return_value=scene)
    result = consolidator.consolidate(incoming, auto_resolve=False)

    persisted = store.get_memcell(incoming.event_id)
    assert result["conflicts_detected"] == 1
    assert result["conflict_checked_before_dedup"] is True
    assert persisted.atomic_facts[0].text == "User eats meat now"
    assert persisted.metadata["dedup_skipped_due_to_conflict"] >= 1


def test_cluster_memcell_uses_precise_time_gap(store, index):
    """A gap above threshold by hours should not be truncated and merged."""

    class ConstantEmbeddings:
        def embed(self, text):
            return [1.0, 0.0]

        def embed_batch(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def similarity(self, a, b):
            return 1.0

    consolidator = MemSceneConsolidator(
        llm_provider=MockProvider(),
        embedding_service=ConstantEmbeddings(),
        memory_store=store,
        search_index=index,
        similarity_threshold=0.5,
        max_time_gap_days=7,
    )

    old_memcell = MemCell.create(
        episode="Old topic",
        atomic_facts=[AtomicFact(text="Old fact")],
        foresight=[],
        source_messages=[],
        timestamp=datetime.now(timezone.utc) - timedelta(days=8),
    )
    existing_scene = MemScene.create("general", old_memcell.event_id, [1.0, 0.0], summary="old")
    existing_scene.latest_timestamp = datetime.now(timezone.utc) - timedelta(days=7, hours=23)
    store.add_memscene(existing_scene)

    new_memcell = MemCell.create(
        episode="New topic",
        atomic_facts=[AtomicFact(text="New fact")],
        foresight=[],
        source_messages=[],
        timestamp=datetime.now(timezone.utc),
    )
    assigned_scene = consolidator.cluster_memcell(new_memcell)

    assert assigned_scene.scene_id != existing_scene.scene_id


def test_consolidate_updates_scene_summary_once(consolidator):
    """Scene summary should be generated exactly once per consolidate call."""
    memcell = MemCell.create(
        episode="User discussed travel plans.",
        atomic_facts=[AtomicFact(text="User plans to travel")],
        foresight=[],
        source_messages=[],
    )
    consolidator._update_scene_summary = MagicMock()

    consolidator.consolidate(memcell)

    consolidator._update_scene_summary.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
