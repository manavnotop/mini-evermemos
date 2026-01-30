"""Phase III: Reconstructive Recollection - Intelligent retrieval from memory."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..models import ForesightItem
from ..storage import MemoryStore, SearchIndex
from ..utils import EmbeddingService, LLMProvider, now_utc


class MemoryRetriever:
    """
    Intelligent retrieval from memory.

    Implements Phase III of the EverMemOS lifecycle:
    - Hybrid BM25 + vector search
    - MemScene-guided retrieval
    - Temporal filtering for foresight
    - Agentic sufficiency checking
    - Query rewriting for improved retrieval
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_service: EmbeddingService,
        memory_store: MemoryStore,
        search_index: SearchIndex,
        default_scene_top_k: int = 10,
        default_episode_top_k: int = 10,
        max_retrieval_rounds: int = 2,
    ):
        """
        Initialize the retriever.

        Args:
            llm_provider: LLM provider for verification
            embedding_service: Service for embeddings
            memory_store: Storage for MemCells and MemScenes
            search_index: Search index for queries
            default_scene_top_k: Default number of scenes to retrieve
            default_episode_top_k: Default number of episodes to retrieve
            max_retrieval_rounds: Max rounds of retrieval + verification
        """
        self.llm = llm_provider
        self.embeddings = embedding_service
        self.store = memory_store
        self.index = search_index

        self.default_scene_top_k = default_scene_top_k
        self.default_episode_top_k = default_episode_top_k
        self.max_retrieval_rounds = max_retrieval_rounds

    def retrieve(
        self,
        query: str,
        query_time: Optional[datetime] = None,
        scene_top_k: Optional[int] = None,
        episode_top_k: Optional[int] = None,
        include_profile: bool = False,
        include_foresight: bool = True,
    ) -> Dict[str, Any]:
        """
        Main retrieval pipeline (MemScene-Guided).

        Implements reconstructive recollection:
        1. Find relevant MemCells (Global Search)
        2. Identify & Rank MemScenes (Scene Selection)
        3. Pool & Re-rank Episodes from Scenes (Context Expansion)
        4. Filter expired foresight
        5. Check sufficiency (Agentic Loop)
        6. Return composed context

        Args:
            query: User query
            query_time: Time to use for temporal filtering
            scene_top_k: Number of scenes to retrieve
            episode_top_k: Number of episodes to retrieve
            include_profile: Include user profile in results
            include_foresight: Include valid foresight items

        Returns:
            Dict with retrieved context
        """
        query_time = query_time or now_utc()
        scene_top_k = scene_top_k or self.default_scene_top_k
        episode_top_k = episode_top_k or self.default_episode_top_k

        round_num = 0
        all_retrieved_queries: List[str] = [query]
        final_context = {
            "query": query,
            "query_time": query_time.isoformat(),
            "memcells": [],
            "foresight": [],
            "profile": None,
            "retrieval_rounds": 0,
        }

        while round_num < self.max_retrieval_rounds:
            round_num += 1
            search_query = all_retrieved_queries[-1]

            # Step 1: Global Search for Signal (High Recall)
            # Retrieve more candidates to find relevant scenes
            # We fetch 5x the desired episodes to get good scene candidates
            global_candidates = self.index.search_hybrid(
                search_query, top_k=max(50, episode_top_k * 5)
            )

            # Step 2: Select MemScenes based on candidates
            selected_scenes = self.select_memscenes_from_candidates(
                global_candidates, top_k=scene_top_k
            )

            # Step 3: Pool Episodes from Scenes and Re-rank
            # Fetch ALL episodes from the selected scenes to provide full context
            pooled_memcells = self._pool_memcells_from_scenes(selected_scenes)

            # Re-rank pooled cells against the query
            reranked_memcells = self._rerank_memcells(pooled_memcells, search_query)

            # Select top K
            top_memcells = reranked_memcells[:episode_top_k]

            # Format for context
            formatted_memcells = [
                {
                    "event_id": m.event_id,
                    "score": getattr(m, "relevance_score", 0.0),
                    "episode": m.episode,
                    "atomic_facts": m.atomic_facts,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in top_memcells
            ]

            # Step 4: Filter Foresight
            # We check foresight from ALL pooled memcells or just top K?
            # Paper suggests filtering relevant foresight. We'll use Top K for precision.
            valid_foresight = []
            if include_foresight:
                for m in top_memcells:
                    for item in m.foresight:
                        if item.is_valid_at(query_time):
                            valid_foresight.append(
                                {
                                    "description": item.description,
                                    "confidence": item.confidence,
                                    "source_event": m.event_id,
                                }
                            )

            # Step 5: Sufficiency Check
            context_text = self._format_context_for_check(
                formatted_memcells, valid_foresight
            )
            is_sufficient, reasoning = self._check_sufficiency(query, context_text)

            final_context["memcells"] = formatted_memcells
            final_context["foresight"] = valid_foresight
            final_context["retrieval_rounds"] = round_num

            if is_sufficient or round_num >= self.max_retrieval_rounds:
                break

            # Step 6: Rewrite query if insufficient
            rewritten_query = self._rewrite_query(
                query,
                reasoning,
                valid_foresight,
                formatted_memcells,
            )
            all_retrieved_queries.append(rewritten_query)

        # Add user profile if requested
        if include_profile:
            profile = self.store.get_user_profile()
            if profile:
                final_context["profile"] = {
                    "explicit_facts": profile.explicit_facts,
                    "implicit_traits": profile.implicit_traits,
                }

        final_context["composed_context"] = self._compose_context(
            final_context["memcells"],
            final_context.get("foresight", []),
            final_context.get("profile"),
        )
        return final_context

    def select_memscenes_from_candidates(
        self,
        candidates: List[Tuple[str, float]],
        top_k: int = 10,
    ) -> List[Any]:
        """
        Select MemScenes based on global memcell candidates.

        Score(Scene) = Max(Score(MemCell) for MemCell in Scene)

        Args:
            candidates: List of (event_id, score) tuples
            top_k: Number of scenes to return

        Returns:
            List of MemScene objects
        """
        # Map event_id -> score
        candidate_scores = {eid: score for eid, score in candidates}

        scenes = self.store.get_all_memscenes()
        scene_scores = []

        for scene in scenes:
            max_score = 0.0
            # Check if any of the scene's memcells are in the candidates
            # Intersection check is faster for large scenes
            scene_mem_ids = set(scene.memcell_ids)
            intersection = scene_mem_ids.intersection(candidate_scores.keys())

            if intersection:
                max_score = max(candidate_scores[mid] for mid in intersection)

            if max_score > 0:
                scene_scores.append((scene, max_score))

        # Sort desc by score
        scene_scores.sort(key=lambda x: x[1], reverse=True)
        return [s for s, score in scene_scores[:top_k]]

    def _pool_memcells_from_scenes(self, scenes: List[Any]) -> List[Any]:
        """Fetch all MemCells from the given scenes."""
        all_ids = set()
        for s in scenes:
            all_ids.update(s.memcell_ids)

        if not all_ids:
            return []

        return self.store.get_memcells_by_ids(list(all_ids))

    def _rerank_memcells(self, memcells: List[Any], query: str) -> List[Any]:
        """
        Re-rank MemCells against the query.

        Uses EmbeddingService similarity if available.
        """
        if not memcells or not self.embeddings:
            # Fallback: Just return as is (random order effectively if not sorted)
            return memcells

        # Get query embedding
        query_embedding = self.embeddings.embed(query)

        scored = []
        for m in memcells:
            score = 0.0
            if m.embedding:
                # Cosine similarity (assuming normalized vectors)
                score = self.embeddings.similarity(query_embedding, m.embedding)
            else:
                # If no embedding, compute it now (slow path)
                # or skip. For now, we skip or set 0.
                pass

            # Attach score for downstream usage
            m.relevance_score = score
            scored.append(m)

        scored.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored

    def _check_sufficiency(
        self,
        query: str,
        context: str,
    ) -> Tuple[bool, str]:
        """
        Check if retrieved context is sufficient to answer the query.

        Args:
            query: User query
            context: Formatted retrieved context

        Returns:
            Tuple of (is_sufficient, reasoning)
        """
        prompt = f"""You are an expert at evaluating retrieval quality.

User Query:
{query}

Retrieved Context:
{context[:3000]}  # Limit context length

## Instructions:
Determine if the retrieved context is SUFFICIENT to answer the user's query.

## Output Format (JSON):
{{
    "is_sufficient": true or false,
    "reasoning": "Brief explanation of why this is or isn't sufficient",
    "key_information_found": ["fact 1", "fact 2"],
    "missing_information": ["what's missing 1", "what's missing 2"]
}}

## Output:"""

        try:
            response = self.llm.complete_json(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            is_sufficient = response.get("is_sufficient", True)
            reasoning = response.get("reasoning", "")

            return is_sufficient, reasoning

        except (json.JSONDecodeError, KeyError):
            # Default to sufficient on error
            return True, "Error in sufficiency check"

    def _rewrite_query(
        self,
        original_query: str,
        reasoning: str,
        foresight: List[Dict],
        memcells: List[Dict],
    ) -> str:
        """
        Rewrite query based on what's missing.

        Args:
            original_query: Original user query
            reasoning: Reasoning from sufficiency check
            foresight: Valid foresight items
            memcells: Retrieved MemCells

        Returns:
            Rewritten query string
        """
        # Extract missing information from reasoning
        prompt = f"""You are an expert at reformulating queries for memory retrieval.

Original Query:
{original_query}

Retrieval Reasoning:
{reasoning}

Valid Foresight:
{chr(10).join(f"- {f['description']}" for f in foresight)}

Retrieved Context Summary:
{chr(10).join(f"- {m['episode'][:100]}..." for m in memcells[:3])}

## Task:
Generate 2-3 reformulated queries that would help find the MISSING information.

## Strategy:
1. Pivot / Entity Association: Search for related entities/categories
2. Temporal Calculation: Use dates from foresight
3. Concept Expansion: Use synonyms and related terms
4. Constraint Relaxation: Broaden the search

## Output Format (JSON):
{{
    "queries": ["query 1", "query 2", "query 3"],
    "strategy": "What strategy was used for each query"
}}

## Output:"""

        try:
            response = self.llm.complete_json(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            queries = response.get("queries", [])
            if queries:
                return queries[0]  # Return first rewritten query

            return original_query

        except (json.JSONDecodeError, KeyError):
            return original_query

    def _format_context_for_check(
        self,
        memcells: List[Dict],
        foresight: List[Dict],
    ) -> str:
        """Format retrieved context for sufficiency checking."""
        lines = ["=== Retrieved MemCells ==="]

        for m in memcells:
            lines.append(f"[{m['timestamp']}]")
            lines.append(f"Episode: {m['episode']}")
            lines.append(f"Facts: {', '.join(m['atomic_facts'][:5])}")
            lines.append("")

        if foresight:
            lines.append("=== Valid Foresight ===")
            for f in foresight:
                lines.append(
                    f"- {f['description']} (confidence: {f['confidence']:.2f})"
                )

        return "\n".join(lines)

    def _compose_context(
        self,
        memcells: List[Dict],
        foresight: List[Dict],
        profile: Optional[Dict],
    ) -> str:
        """Compose the final context string for downstream use."""
        lines = []

        if profile:
            lines.append("=== User Profile ===")
            explicit = profile.get("explicit_facts", {})
            for key, value in explicit.items():
                lines.append(
                    f"{key}: {value['value'] if isinstance(value, dict) else value}"
                )
            if profile.get("implicit_traits"):
                lines.append(f"Traits: {', '.join(profile['implicit_traits'])}")
            lines.append("")

        lines.append("=== Relevant Memories ===")
        for m in memcells:
            lines.append(f"[{m['timestamp']}] {m['episode']}")
            if m["atomic_facts"]:
                lines.append(f"  Facts: {', '.join(m['atomic_facts'])}")

        if foresight:
            lines.append("")
            lines.append("=== Active Foresight ===")
            for f in foresight:
                lines.append(f"- {f['description']}")

        return "\n".join(lines)

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            "total_memcells": len(self.store.get_all_memcells()),
            "total_scenes": len(self.store.get_all_memscenes()),
            "index_stats": self.index.get_stats(),
        }
