"""Hybrid search index with BM25 and vector search capabilities."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from ..models import MemCell
from ..utils import EmbeddingService

if TYPE_CHECKING:
    from .memory_store import MemoryStore


class SearchIndex:
    """
    Hybrid search index combining BM25 and vector similarity.

    Supports:
    - BM25 keyword search over atomic facts (In-Memory)
    - Dense vector similarity search (Delegated to MemoryStore/Milvus if available)
    - Reciprocal Rank Fusion (RRF) for combined ranking
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        memory_store: Optional["MemoryStore"] = None,
    ):
        """
        Initialize the search index.

        Args:
            embedding_service: Service for generating embeddings
            memory_store: Store to delegate vector search to (optional)
        """
        self.embeddings = embedding_service
        self.store = memory_store

        # BM25 index
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []  # Maps index to event_id
        self._doc_facts: Dict[str, List[str]] = {}  # event_id -> atomic facts

        # Batch update tracking
        self._bm25_dirty: bool = False
        self._rebuild_threshold: int = 10
        self._pending_additions: int = 0

        # In-memory Vector index (Fallback only)
        self._vectors: Dict[str, List[float]] = {}

    def add_memcell(self, memcell: MemCell) -> None:
        """Add a MemCell to the search index."""
        event_id = memcell.event_id

        # Add to BM25 index
        facts = memcell.atomic_facts
        if facts:
            self._doc_facts[event_id] = facts
            self._bm25_dirty = True
            self._pending_additions += 1

            if self._pending_additions >= self._rebuild_threshold:
                self._rebuild_bm25()
                self._pending_additions = 0

        # Add to vector index ONLY if we are NOT using external store for vectors
        # If we have a store with vector capability, it handles storage.
        use_external_vector_store = self.store and self.store._milvus_available

        if not use_external_vector_store and memcell.embedding and self.embeddings:
            self._vectors[event_id] = memcell.embedding

    def add_memcells(self, memcells: List[MemCell]) -> None:
        """Add multiple MemCells to the search index."""
        for memcell in memcells:
            self.add_memcell(memcell)
        if self._bm25_dirty:
            self._rebuild_bm25()
            self._pending_additions = 0

    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25 keyword matching."""
        if self._bm25_dirty:
            self._rebuild_bm25()
            self._bm25_dirty = False

        if not self._bm25 or not query.strip():
            return []

        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if idx < len(self._doc_ids) and scores[idx] > 0:
                results.append((self._doc_ids[idx], float(scores[idx])))
        return results

    def search_vector(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using dense vector similarity."""
        if not self.embeddings or not query.strip():
            return []

        # If we have external store with Milvus, delegate
        if self.store and self.store._milvus_available:
            query_embedding = self.embeddings.embed(query)
            # MemoryStore.search_memcells returns List[MemCell] ordered by relevance
            # But here we need (event_id, score) tuples.
            # We need to call the raw search on Milvus client or get scores from store.
            # MemoryStore.search_memcells currently returns objects.
            # Let's access the milvus client directly via store to get scores?
            # Or assume store.search_memcells returns things in order and I fake the score?
            # Better: Use store.milvus.search_memcells which returns dicts with scores.

            raw_results = self.store.milvus.search_memcells(
                query_embedding, top_k=top_k
            )
            return [(r["event_id"], r["score"]) for r in raw_results]

        # Fallback to in-memory
        return self._search_vector_in_memory(query, top_k)

    def _search_vector_in_memory(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """In-memory vector search implementation."""
        query_embedding = self.embeddings.embed(query)
        results = []
        for event_id, doc_embedding in self._vectors.items():
            similarity = self.embeddings.similarity(query_embedding, doc_embedding)
            results.append((event_id, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_hybrid(
        self, query: str, top_k: int = 10, rrf_k: float = 60.0
    ) -> List[Tuple[str, float]]:
        """Search using hybrid BM25 + vector with RRF fusion."""
        bm25_results = {
            doc_id: rank + 1
            for rank, (doc_id, _) in enumerate(self.search_bm25(query, top_k * 2))
        }
        vector_results = {
            doc_id: rank + 1
            for rank, (doc_id, _) in enumerate(self.search_vector(query, top_k * 2))
        }

        all_docs = set(bm25_results.keys()) | set(vector_results.keys())
        rrf_scores = []

        for doc_id in all_docs:
            bm25_rank = bm25_results.get(doc_id, 0)
            vector_rank = vector_results.get(doc_id, 0)

            if bm25_rank == 0 and vector_rank == 0:
                continue

            bm25_score = 1.0 / (rrf_k + bm25_rank) if bm25_rank else 0
            vector_score = 1.0 / (rrf_k + vector_rank) if vector_rank else 0

            rrf_scores.append((doc_id, bm25_score + vector_score))

        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        return rrf_scores[:top_k]

    def search_episodes(
        self, query: str, memcells: Dict[str, MemCell], top_k: int = 10
    ) -> List[Tuple[str, float, str]]:
        """Search for episodes matching the query."""
        results = self.search_hybrid(query, top_k)
        episode_results = []
        for event_id, score in results:
            # We need to fetch the MemCell
            # If provided memcells dict covers it, great.
            # Otherwise we might need to fetch from store if not in dict.
            # But this signature expects 'memcells' dict.
            # In old code, 'memcells' came from MemoryStore.get_all_memcells or similar.
            # We should try to use the passed dict or fallback to store.
            episode = None
            if event_id in memcells:
                episode = memcells[event_id].episode
            elif self.store:
                mc = self.store.get_memcell(event_id)
                if mc:
                    episode = mc.episode

            if episode:
                episode_results.append((event_id, score, episode))

        return episode_results

    def _rebuild_bm25(self) -> None:
        """Rebuild the BM25 index."""
        if not self._doc_facts:
            self._bm25 = None
            return
        self._doc_ids = list(self._doc_facts.keys())
        corpus = [" ".join(facts) for facts in self._doc_facts.values()]
        self._bm25 = BM25Okapi(corpus)

    def clear(self) -> None:
        """Clear the search index."""
        self._bm25 = None
        self._doc_ids.clear()
        self._doc_facts.clear()
        self._vectors.clear()
        self._bm25_dirty = False
        self._pending_additions = 0

    def get_stats(self) -> Dict[str, int]:
        """Get index statistics."""
        return {
            "bm25_doc_count": len(self._doc_ids),
            "vector_doc_count": len(self._vectors)
            if not (self.store and self.store._milvus_available)
            else "Externally Managed",
        }
