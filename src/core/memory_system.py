"""Memory System Orchestrator - Ties all three phases together."""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import UserProfile
from ..storage import MemoryStore, SearchIndex
from ..utils import EmbeddingService, LLMProvider, MockEmbeddings, MockProvider, now_utc
from .consolidator import MemSceneConsolidator
from .extractor import MemCellExtractor
from .retriever import MemoryRetriever


class MemorySystem:
    """
    Main orchestrator for the EverMemOS-inspired memory system.

    Provides a unified interface for:
    - Adding conversations (Phase I + II)
    - Retrieving context (Phase III)
    - Managing the memory lifecycle

    Usage:
        system = MemorySystem()
        system.add_conversation(messages)
        result = system.retrieve("where does the user work?")
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        embedding_service: Optional[EmbeddingService] = None,
        storage_dir: str = "./memory_data",
        user_id: str = "default",
        similarity_threshold: float = 0.70,
        max_time_gap_days: int = 7,
    ):
        """
        Initialize the memory system.

        Args:
            llm_provider: LLM provider (auto-creates mock if not provided)
            embedding_service: Embedding service (auto-creates mock if not provided)
            storage_dir: Directory for persistent storage
            user_id: Default user ID for this memory system
            similarity_threshold: Threshold for MemScene clustering
            max_time_gap_days: Max days between MemCells in same scene
        """
        self.user_id = user_id
        self.created_at = now_utc()

        # Create providers if not provided (use mocks for testing)
        if llm_provider is None:
            self.llm = MockProvider()
        else:
            self.llm = llm_provider

        if embedding_service is None:
            self.embeddings = MockEmbeddings(dim=384)
        else:
            self.embeddings = embedding_service

        # Get embedding dimension from the embedding service
        embedding_dim = getattr(self.embeddings, "dim", 1536)

        # Initialize storage
        use_mock_storage = (
            embedding_service is not None
            and isinstance(embedding_service, MockEmbeddings)
        ) or (embedding_service is None and isinstance(self.embeddings, MockEmbeddings))

        if use_mock_storage:
            try:
                from tests.mock_db import (
                    MockMilvusStorageClient,
                    MockMongoStorageClient,
                )

                self.store = MemoryStore(
                    mongo_client=MockMongoStorageClient(),
                    milvus_client=MockMilvusStorageClient(),
                )
            except ImportError:
                mongo_uri = "mongodb://localhost:27017"
                if "mongodb://" in storage_dir or "mongodb+srv://" in storage_dir:
                    mongo_uri = storage_dir
                self.store = MemoryStore(
                    mongo_uri=mongo_uri, embedding_dim=embedding_dim
                )

        else:
            mongo_uri = "mongodb://localhost:27017"
            if "mongodb://" in storage_dir or "mongodb+srv://" in storage_dir:
                mongo_uri = storage_dir

            self.store = MemoryStore(mongo_uri=mongo_uri, embedding_dim=embedding_dim)

        self.index = SearchIndex(self.embeddings, memory_store=self.store)

        # Initialize phases
        self.extractor = MemCellExtractor(
            llm_provider=self.llm,
            embedding_service=self.embeddings,
        )

        self.consolidator = MemSceneConsolidator(
            llm_provider=self.llm,
            embedding_service=self.embeddings,
            memory_store=self.store,
            search_index=self.index,
            similarity_threshold=similarity_threshold,
            max_time_gap_days=max_time_gap_days,
        )

        self.retriever = MemoryRetriever(
            llm_provider=self.llm,
            embedding_service=self.embeddings,
            memory_store=self.store,
            search_index=self.index,
        )

    def add_conversation(
        self,
        messages: List[Dict[str, str]],
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Add a conversation to memory.

        Runs the full lifecycle:
        1. Extract MemCell from messages (Phase I)
        2. Consolidate into MemScene (Phase II)
        3. Detect and log conflicts
        4. Update user profile

        Args:
            messages: List of conversation messages
            timestamp: Timestamp for the conversation

        Returns:
            Dict with processing results
        """
        start_time = time.time()

        timestamp = timestamp or now_utc()

        # Phase I: Extract MemCell
        memcell = self.extractor.create_memcell(messages, timestamp)

        # Phase II: Consolidate
        result = self.consolidator.consolidate(memcell, self.user_id)

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "memcell_id": memcell.event_id,
            "scene_id": result["scene_id"],
            "theme": result["theme"],
            "conflicts_detected": result["conflicts_detected"],
            "processing_time_ms": elapsed_ms,
            "episode": memcell.episode,
            "atomic_facts": memcell.atomic_facts,
            "foresight_count": len(memcell.foresight),
            "original_facts_count": memcell.metadata.get(
                "original_facts_count", len(memcell.atomic_facts)
            ),
            "unique_facts_count": memcell.metadata.get(
                "unique_facts_count", len(memcell.atomic_facts)
            ),
        }

    def add_conversation_stream(
        self,
        messages: List[Dict[str, str]],
        flush: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Add a stream of messages, auto-detecting boundaries.

        Args:
            messages: List of conversation messages
            flush: Whether to flush pending messages at the end

        Returns:
            List of processing results for each extracted MemCell
        """
        results = []
        memcells = self.extractor.process_conversation_stream(messages, flush=flush)

        for memcell in memcells:
            result = self.consolidator.consolidate(memcell, self.user_id)
            results.append(
                {
                    "memcell_id": memcell.event_id,
                    "scene_id": result["scene_id"],
                    "theme": result["theme"],
                    "episode": memcell.episode,
                }
            )

        # Flush remaining
        if flush:
            remaining = self.extractor.flush()
            for memcell in remaining:
                result = self.consolidator.consolidate(memcell, self.user_id)
                results.append(
                    {
                        "memcell_id": memcell.event_id,
                        "scene_id": result["scene_id"],
                        "theme": result["theme"],
                        "episode": memcell.episode,
                    }
                )

        return results

    def retrieve(
        self,
        query: str,
        query_time: Optional[datetime] = None,
        include_profile: bool = True,
        include_foresight: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query.

        Args:
            query: User query
            query_time: Time for temporal filtering
            include_profile: Include user profile
            include_foresight: Include valid foresight items

        Returns:
            Dict with retrieved context
        """
        return self.retriever.retrieve(
            query=query,
            query_time=query_time,
            include_profile=include_profile,
            include_foresight=include_foresight,
        )

    def get_user_profile(self) -> Optional[UserProfile]:
        """Get the current user profile."""
        return self.store.get_user_profile(self.user_id)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        return {
            "memcell_count": len(self.store.get_all_memcells()),
            "memscene_count": len(self.store.get_all_memscenes()),
            "conflict_count": len(self.store.get_all_conflicts()),
            "unresolved_conflicts": len(self.store.get_unresolved_conflicts()),
            "index_stats": self.index.get_stats(),
        }

    def get_scenes_by_theme(self) -> Dict[str, int]:
        """Get scene counts by theme."""
        scenes = self.store.get_all_memscenes()
        counts = {}
        for scene in scenes:
            counts[scene.theme] = counts.get(scene.theme, 0) + 1
        return counts

    def save(self) -> None:
        """Save memory to disk."""
        self.store.save_to_disk()

    def load(self) -> None:
        """Load memory from disk."""
        self.store.load_from_disk()

    def clear(self) -> None:
        """Clear all memory."""
        self.store.clear()
        self.index.clear()

    def get_conflicts(self, resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get conflicts, optionally filtered by resolution status."""
        conflicts = self.store.get_all_conflicts()

        if resolved is None:
            return [c.to_dict() for c in conflicts]

        if resolved:
            return [c.to_dict() for c in conflicts if c.is_resolved]
        else:
            return [c.to_dict() for c in conflicts if not c.is_resolved]

    def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str,
        notes: str = "",
    ) -> bool:
        """
        Resolve a conflict.

        Args:
            conflict_id: ID of the conflict to resolve
            resolution: Resolution strategy ("recency", "keep_both", "manual")
            notes: Additional notes

        Returns:
            True if conflict was found and resolved
        """
        conflict = self.store.get_conflict(conflict_id)
        if not conflict:
            return False

        self.consolidator.resolve_conflict(conflict, resolution, notes)
        return True
