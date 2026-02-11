"""Specific tests for EverMemOS logic changes."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core import MemoryRetriever, MemSceneConsolidator
from src.core.constants import CONTEXT_LIMITS
from src.models import AtomicFact, ConflictRecord, MemCell, MemScene
from src.storage import MemoryStore, SearchIndex


@pytest.fixture
def mock_store():
    store = MagicMock(spec=MemoryStore)
    store.default_scene_top_k = 10
    store.default_episode_top_k = 10
    return store


@pytest.fixture
def mock_index():
    return MagicMock(spec=SearchIndex)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    # Mock complete_json for sufficiency check
    llm.complete_json.return_value = {"is_sufficient": True, "reasoning": "Good"}
    return llm


@pytest.fixture
def mock_embeddings():
    emb = MagicMock()
    emb.embed.return_value = [0.1, 0.2, 0.3]
    emb.similarity.return_value = 0.9
    return emb


def test_memscene_guided_retrieval_flow(
    mock_store, mock_index, mock_llm, mock_embeddings
):
    """Verify that retrieval strictly follows the MemScene-guided pipeline."""
    retriever = MemoryRetriever(mock_llm, mock_embeddings, mock_store, mock_index)

    # Setup mocks
    # 1. Global Search Candidates
    mock_index.search_hybrid.return_value = [("evt1", 0.9), ("evt2", 0.8)]

    # 2. Scenes
    scene1 = MemScene.create("theme1", "evt1", [0.1], summary="s1")
    scene1.memcell_ids = ["evt1", "evt3"]  # evt3 is in scene but not in candidates

    mock_store.get_all_memscenes.return_value = [scene1]
    # Use constructor directly to set specific event_ids
    mock_store.get_memcells_by_ids.return_value = [
        MemCell(
            event_id="evt1",
            episode="ep1",
            atomic_facts=["fact1"],
            foresight=[],
            source_messages=[],
            timestamp=datetime.now(timezone.utc),
        ),
        MemCell(
            event_id="evt3",
            episode="ep3",
            atomic_facts=["fact3"],
            foresight=[],
            source_messages=[],
            timestamp=datetime.now(timezone.utc),
        ),
    ]

    # Execute
    result = retriever.retrieve("test query")

    # Verifications
    # 1. Should call index search (Global Search)
    mock_index.search_hybrid.assert_called()

    # 2. Should fetch scenes (implicitly verified by logic, but we can check if it fetched all scenes)
    mock_store.get_all_memscenes.assert_called()

    # 3. Should pool episodes from the SELECTED scene (evt1 and evt3)
    # The retriever logic pools ALL memcells from the selected scene
    # So we should see a fetch for IDs including evt3 (which wasn't in global results)
    # This confirms "Context Expansion"
    call_args_list = mock_store.get_memcells_by_ids.call_args_list
    # The pooling call should contain the set of IDs from the scene
    pooled_ids = set(["evt1", "evt3"])

    found_pooling = False
    for call in call_args_list:
        args, _ = call
        if set(args[0]) == pooled_ids:
            found_pooling = True
            break

    assert found_pooling, (
        "Did not find a call pooling all MemCells from the selected scene!"
    )


def test_consolidator_auto_resolution(
    mock_store, mock_index, mock_llm, mock_embeddings
):
    """Verify automatic conflict resolution."""
    consolidator = MemSceneConsolidator(
        mock_llm, mock_embeddings, mock_store, mock_index
    )

    # Setup
    memcell = MemCell.create("new ep", ["new fact"], [], source_messages=[])
    scene = MemScene.create("theme", "old_id", [0.1])

    # Mock clustering to return our scene
    consolidator.cluster_memcell = MagicMock(return_value=scene)

    # Mock conflict detection to return a conflict
    conflict = ConflictRecord.create("id", "scene_id", "old", "new")
    consolidator.detect_conflicts = MagicMock(return_value=[conflict])
    consolidator.deduplicate_facts_globally = MagicMock(return_value=(memcell.atomic_facts, 1, 1))
    consolidator._update_scene_summary = MagicMock()

    # Mock storage calls
    mock_store.get_or_create_profile.return_value = MagicMock()

    # Execute with auto_resolve=True
    consolidator.consolidate(memcell, auto_resolve=True)

    # Verify resolution was called
    assert conflict.is_resolved is True
    assert conflict.resolution == "recency"


def test_consolidator_no_auto_resolution(
    mock_store, mock_index, mock_llm, mock_embeddings
):
    """Verify auto-resolution can be disabled."""
    consolidator = MemSceneConsolidator(
        mock_llm, mock_embeddings, mock_store, mock_index
    )

    # Setup
    memcell = MemCell.create("new ep", ["new fact"], [], source_messages=[])
    scene = MemScene.create("theme", "old_id", [0.1])

    consolidator.cluster_memcell = MagicMock(return_value=scene)

    conflict = ConflictRecord.create("id", "scene_id", "old", "new")
    consolidator.detect_conflicts = MagicMock(return_value=[conflict])
    consolidator.deduplicate_facts_globally = MagicMock(return_value=(memcell.atomic_facts, 1, 1))
    consolidator._update_scene_summary = MagicMock()
    mock_store.get_or_create_profile.return_value = MagicMock()

    # Execute with auto_resolve=False
    consolidator.consolidate(memcell, auto_resolve=False)

    # Verify resolution was NOT called
    assert conflict.is_resolved is False


def test_retriever_normalizes_atomic_fact_objects(
    mock_store, mock_index, mock_llm, mock_embeddings
):
    """AtomicFact objects should be returned/serialized as strings."""
    retriever = MemoryRetriever(mock_llm, mock_embeddings, mock_store, mock_index)
    mock_index.search_hybrid.return_value = [("evt1", 0.9)]

    scene = MemScene.create("career", "evt1", [0.1], summary="career scene")
    scene.memcell_ids = ["evt1"]
    mock_store.get_all_memscenes.return_value = [scene]
    mock_store.get_user_profile.return_value = None
    mock_store.get_memcells_by_ids.return_value = [
        MemCell(
            event_id="evt1",
            episode="User discussed work",
            atomic_facts=[AtomicFact(text="User works at Google", confidence=0.9)],
            foresight=[],
            source_messages=[],
            timestamp=datetime.now(timezone.utc),
            embedding=[0.1, 0.2, 0.3],
        )
    ]

    result = retriever.retrieve("Where does the user work?")

    assert result["memcells"][0]["atomic_facts"] == ["User works at Google"]
    assert "User works at Google" in result["composed_context"]


def test_retriever_rewrites_query_when_sufficiency_output_is_malformed(
    mock_store, mock_index, mock_embeddings
):
    """Missing `is_sufficient` should trigger rewrite and second retrieval round."""
    llm = MagicMock()
    llm.complete_json.side_effect = [
        {"reasoning": "Context misses current employer", "missing_information": ["current employer"]},
        {"queries": ["current employer workplace"]},
        {"is_sufficient": True, "reasoning": "Now sufficient", "missing_information": []},
    ]

    retriever = MemoryRetriever(llm, mock_embeddings, mock_store, mock_index)
    search_queries = []

    def _search(query, top_k=10, rrf_k=60.0):
        search_queries.append(query)
        return [("evt1", 0.9)]

    mock_index.search_hybrid.side_effect = _search
    scene = MemScene.create("career", "evt1", [0.1], summary="career scene")
    scene.memcell_ids = ["evt1"]
    mock_store.get_all_memscenes.return_value = [scene]
    mock_store.get_memcells_by_ids.return_value = [
        MemCell(
            event_id="evt1",
            episode="User discussed work",
            atomic_facts=["User works at Google"],
            foresight=[],
            source_messages=[],
            timestamp=datetime.now(timezone.utc),
            embedding=[0.1, 0.2, 0.3],
        )
    ]

    result = retriever.retrieve("Where does the user work?")

    assert search_queries == ["Where does the user work?", "current employer workplace"]
    assert result["retrieval_rounds"] == 2


def test_retriever_ordering_is_deterministic_without_embeddings(mock_store, mock_index, mock_llm):
    """Without embeddings, ordering should use candidate scores then timestamp."""
    retriever = MemoryRetriever(mock_llm, None, mock_store, mock_index)
    now = datetime.now(timezone.utc)
    mock_index.search_hybrid.return_value = [("evt1", 0.9), ("evt2", 0.7)]

    scene = MemScene.create("career", "evt1", [0.1], summary="career scene")
    scene.memcell_ids = ["evt1", "evt2", "evt3"]
    mock_store.get_all_memscenes.return_value = [scene]
    mock_store.get_memcells_by_ids.return_value = [
        MemCell(
            event_id="evt3",
            episode="third",
            atomic_facts=[],
            foresight=[],
            source_messages=[],
            timestamp=now,
        ),
        MemCell(
            event_id="evt2",
            episode="second",
            atomic_facts=[],
            foresight=[],
            source_messages=[],
            timestamp=now - timedelta(days=1),
        ),
        MemCell(
            event_id="evt1",
            episode="first",
            atomic_facts=[],
            foresight=[],
            source_messages=[],
            timestamp=now - timedelta(days=2),
        ),
    ]

    result = retriever.retrieve("test query", episode_top_k=3)

    assert [m["event_id"] for m in result["memcells"]] == ["evt1", "evt2", "evt3"]


def test_retriever_uses_user_id_for_profile_lookup(
    mock_store, mock_index, mock_llm, mock_embeddings
):
    """Profile retrieval should be scoped to the provided user_id."""
    retriever = MemoryRetriever(mock_llm, mock_embeddings, mock_store, mock_index)
    mock_index.search_hybrid.return_value = []
    mock_store.get_all_memscenes.return_value = []
    mock_store.get_memcells_by_ids.return_value = []
    mock_store.get_user_profile.return_value = SimpleNamespace(
        explicit_facts={"name": "Alice"},
        implicit_traits=["organized"],
    )

    result = retriever.retrieve(
        "What do we know about the user?",
        include_profile=True,
        user_id="user-123",
    )

    mock_store.get_user_profile.assert_called_once_with("user-123")
    assert result["profile"]["explicit_facts"]["name"] == "Alice"


def test_rewrite_query_uses_configured_memcell_summary_limit(
    mock_store, mock_index, mock_embeddings
):
    """Rewrite prompt should include only configured number of memcells."""
    llm = MagicMock()
    llm.complete_json.return_value = {"queries": ["rewritten query"]}
    retriever = MemoryRetriever(llm, mock_embeddings, mock_store, mock_index)

    memcells = [
        {"episode": "episode_1", "atomic_facts": [], "timestamp": "2025-01-01T00:00:00+00:00"},
        {"episode": "episode_2", "atomic_facts": [], "timestamp": "2025-01-02T00:00:00+00:00"},
        {"episode": "episode_3", "atomic_facts": [], "timestamp": "2025-01-03T00:00:00+00:00"},
        {"episode": "episode_4", "atomic_facts": [], "timestamp": "2025-01-04T00:00:00+00:00"},
    ]

    retriever._rewrite_query(
        original_query="query",
        reasoning="reason",
        missing_information=["info"],
        foresight=[],
        memcells=memcells,
    )

    prompt = llm.complete_json.call_args[0][0][0]["content"]
    max_items = CONTEXT_LIMITS["max_memcells_in_summary"]
    for idx in range(1, max_items + 1):
        assert f"episode_{idx}" in prompt
    assert "episode_4" not in prompt
