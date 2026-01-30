"""Specific tests for EverMemOS logic changes."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.core import MemoryRetriever, MemSceneConsolidator
from src.models import ConflictRecord, MemCell, MemScene
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
    mock_store.get_or_create_profile.return_value = MagicMock()

    # Execute with auto_resolve=False
    consolidator.consolidate(memcell, auto_resolve=False)

    # Verify resolution was NOT called
    assert conflict.is_resolved is False
