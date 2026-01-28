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
        Main retrieval pipeline.

        Implements reconstructive recollection:
        1. Select relevant MemScenes
        2. Retrieve and re-rank MemCells
        3. Filter expired foresight
        4. Check sufficiency and rewrite if needed
        5. Return composed context

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

        # Track retrieval rounds for agentic loop
        round_num = 0
        all_retrieved: List[Tuple[str, float]] = []
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

            # Step 1: Search for relevant MemCells
            search_query = query
            if round_num > 1 and all_retrieved:
                # Use rewritten query from previous round
                search_query = all_retrieved[0][0] if all_retrieved else query

            retrieved = self.index.search_hybrid(search_query, top_k=episode_top_k * 2)
            all_retrieved.extend(retrieved)

            # Deduplicate by event_id
            seen = set()
            unique_retrieved = []
            for event_id, score in all_retrieved:
                if event_id not in seen:
                    seen.add(event_id)
                    unique_retrieved.append((event_id, score))

            # Get MemCell details
            memcells = []
            for event_id, score in unique_retrieved[:episode_top_k]:
                memcell = self.store.get_memcell(event_id)
                if memcell:
                    memcells.append(
                        {
                            "event_id": event_id,
                            "score": score,
                            "episode": memcell.episode,
                            "atomic_facts": memcell.atomic_facts,
                            "timestamp": memcell.timestamp.isoformat(),
                        }
                    )

            # Step 2: Filter valid foresight
            valid_foresight = []
            if include_foresight:
                for memcell in memcells:
                    event_id = memcell["event_id"]
                    full_memcell = self.store.get_memcell(event_id)
                    if full_memcell:
                        for item in full_memcell.foresight:
                            if item.is_valid_at(query_time):
                                valid_foresight.append(
                                    {
                                        "description": item.description,
                                        "confidence": item.confidence,
                                        "source_event": event_id,
                                    }
                                )

            # Step 3: Check sufficiency
            context_text = self._format_context_for_check(memcells, valid_foresight)
            is_sufficient, reasoning = self._check_sufficiency(query, context_text)

            final_context["memcells"] = memcells
            final_context["foresight"] = valid_foresight
            final_context["retrieval_rounds"] = round_num

            if is_sufficient or round_num >= self.max_retrieval_rounds:
                break

            # Step 4: Rewrite query if insufficient
            rewritten_query = self._rewrite_query(
                query,
                reasoning,
                valid_foresight,
                memcells,
            )
            # Augment with new results instead of replacing
            new_results = self.index.search_hybrid(
                rewritten_query, top_k=episode_top_k * 2
            )
            all_retrieved.extend(new_results)

        # Add user profile if requested
        if include_profile:
            profile = self.store.get_user_profile()
            if profile:
                final_context["profile"] = {
                    "explicit_facts": profile.explicit_facts,
                    "implicit_traits": profile.implicit_traits,
                }

        # Generate final composed context
        final_context["composed_context"] = self._compose_context(
            final_context["memcells"],
            final_context.get("foresight", []),
            final_context.get("profile"),
        )

        return final_context

    def select_memscenes(
        self,
        query: str,
        scene_top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Select relevant MemScenes for a query.

        Scores each scene by the max relevance of its constituent MemCells.

        Args:
            query: Search query
            scene_top_k: Number of scenes to select

        Returns:
            List of (scene_id, score) tuples
        """
        scenes = self.store.get_all_memscenes()
        if not scenes:
            return []

        scene_scores = []
        for scene in scenes:
            # Get MemCells in this scene
            memcells = self.store.get_memcells_by_scene(scene.scene_id)
            if not memcells:
                continue

            # Calculate max relevance (not average - want at least one relevant)
            max_score = 0.0
            for memcell in memcells:
                # Use hybrid search on the episode
                results = self.index.search_hybrid(query, top_k=1)
                for event_id, score in results:
                    if event_id == memcell.event_id:
                        max_score = max(max_score, score)
                        break

            if max_score > 0:
                scene_scores.append((scene.scene_id, max_score))

        # Sort by score and return top-k
        scene_scores.sort(key=lambda x: x[1], reverse=True)
        return scene_scores[:scene_top_k]

    def filter_foresight(
        self,
        foresight_items: List[ForesightItem],
        query_time: datetime,
    ) -> List[ForesightItem]:
        """
        Filter foresight items to only those valid at query_time.

        Args:
            foresight_items: All foresight items
            query_time: Time to check validity

        Returns:
            Only valid foresight items
        """
        return [f for f in foresight_items if f.is_valid_at(query_time)]

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
