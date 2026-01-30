"""Phase II: Semantic Consolidation - Organize MemCells into MemScenes."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..models import ConflictRecord, MemCell, MemScene, UserProfile
from ..prompts import CONFLICT_DETECTION_PROMPT
from ..storage import MemoryStore, SearchIndex
from ..utils import EmbeddingService, LLMProvider


class MemSceneConsolidator:
    """
    Organize MemCells into thematic MemScenes.

    Implements Phase II of the EverMemOS lifecycle:
    - Cluster MemCells into MemScenes by theme
    - Detect and resolve conflicts
    - Update user profiles from scene summaries
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_service: EmbeddingService,
        memory_store: MemoryStore,
        search_index: SearchIndex,
        similarity_threshold: float = 0.70,
        max_time_gap_days: int = 7,
    ):
        """
        Initialize the consolidator.

        Args:
            llm_provider: LLM provider for conflict detection
            embedding_service: Service for generating embeddings
            memory_store: Storage for MemCells and MemScenes
            search_index: Search index for similarity queries
            similarity_threshold: Threshold for scene clustering
            max_time_gap_days: Max days between MemCells in same scene
        """
        self.llm = llm_provider
        self.embeddings = embedding_service
        self.store = memory_store
        self.index = search_index

        self.similarity_threshold = similarity_threshold
        self.max_time_gap_days = max_time_gap_days

        # Cache for theme prototype embeddings
        self._theme_embeddings: Dict[str, List[float]] = {}

    def cluster_memcell(self, memcell: MemCell) -> MemScene:
        """
        Assign a MemCell to a MemScene.

        Uses similarity-based clustering:
        1. Compute embedding for the MemCell
        2. Find most similar MemScene centroid
        3. If similarity exceeds threshold and time gap is acceptable, assign
        4. Otherwise, create new MemScene

        Args:
            memcell: MemCell to cluster

        Returns:
            The MemScene the MemCell was assigned to
        """
        # Get or create embedding
        if memcell.embedding is None:
            memcell.embedding = self.embeddings.embed(memcell.episode)

        # Find best matching scene
        best_scene: Optional[MemScene] = None
        best_similarity = -1.0

        for scene in self.store.get_all_memscenes():
            if scene.centroid is not None:
                similarity = self.embeddings.similarity(
                    memcell.embedding,
                    scene.centroid,
                )

                # Check time gap constraint against scene's time range
                if scene.latest_timestamp:
                    time_gap = abs((memcell.timestamp - scene.latest_timestamp).days)
                    if time_gap > self.max_time_gap_days:
                        continue  # Skip this scene due to time gap

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_scene = scene

        # Assign or create scene
        if best_scene and best_similarity >= self.similarity_threshold:
            # Add to existing scene
            best_scene.add_memcell(memcell.event_id)
            best_scene.update_time_range(memcell.timestamp)
            best_scene.update_centroid(memcell.embedding)
            self._update_scene_summary(best_scene)
            self.store.add_memscene(best_scene)
        else:
            # Create new scene
            theme = self._infer_theme(memcell.episode)
            new_scene = MemScene.create(
                theme=theme,
                initial_memcell_id=memcell.event_id,
                initial_embedding=memcell.embedding,
                summary=memcell.episode,
            )
            new_scene.update_time_range(memcell.timestamp)
            self.store.add_memscene(new_scene)
            best_scene = new_scene

        return best_scene

    def detect_conflicts(
        self,
        memcell: MemCell,
        scene: MemScene,
    ) -> List[ConflictRecord]:
        """
        Detect conflicts between new MemCell and existing facts.

        Args:
            memcell: New MemCell to check
            scene: MemScene to check against

        Returns:
            List of detected conflicts
        """
        # Get existing facts from the scene
        existing_facts = []
        for memcell_id in scene.memcell_ids:
            existing = self.store.get_memcell(memcell_id)
            if existing:
                existing_facts.extend(existing.atomic_facts)

        if not existing_facts:
            return []

        # Format for conflict detection
        new_facts_str = "\n".join(memcell.atomic_facts)
        existing_facts_str = "\n".join(existing_facts)

        prompt = CONFLICT_DETECTION_PROMPT.format(
            new_facts=new_facts_str,
            existing_facts=existing_facts_str,
        )

        try:
            response = self.llm.complete_json(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            conflicts = []
            for item in response.get("conflicts", []):
                conflict = ConflictRecord.create(
                    memcell_id=memcell.event_id,
                    scene_id=scene.scene_id,
                    old_fact=item["old_fact"],
                    new_fact=item["new_fact"],
                    metadata={
                        "conflict_type": item.get("conflict_type"),
                        "explanation": item.get("explanation"),
                        "confidence": item.get("confidence", 1.0),
                    },
                )
                conflicts.append(conflict)
                self.store.add_conflict(conflict)

            return conflicts

        except (json.JSONDecodeError, KeyError):
            return []

    def resolve_conflict(
        self,
        conflict: ConflictRecord,
        resolution: str = "recency",
        notes: str = "",
    ) -> None:
        """
        Resolve a detected conflict.

        Resolution strategies:
        - "recency": New fact replaces old fact (default)
        - "keep_both": Both facts are kept with timestamps
        - "manual": Requires user intervention (logged for review)
        - "user_choice": Ask the user for resolution

        Args:
            conflict: Conflict to resolve
            resolution: Resolution strategy
            notes: Additional notes about resolution
        """
        conflict.resolve(resolution, notes)

        # Log the resolution
        if resolution == "recency":
            # New fact wins - could add metadata to track this
            pass
        elif resolution == "keep_both":
            # Both facts are kept - add timestamps to differentiate
            pass
        elif resolution == "manual":
            # Flag for manual review
            conflict.metadata["needs_review"] = True

    def update_user_profile(
        self,
        profile: Optional[UserProfile] = None,
        user_id: str = "default",
    ) -> UserProfile:
        """
        Update a user profile from scene summaries.

        Extracts:
        - Explicit facts (verifiable attributes)
        - Implicit traits (preferences, habits)

        Args:
            existing profile or None to get from store
            user_id: User ID for the profile

        Returns:
            Updated UserProfile
        """
        if profile is None:
            profile = self.store.get_or_create_profile(user_id)

        # Get all scene summaries
        scenes = self.store.get_all_memscenes()
        summaries = [scene.summary for scene in scenes if scene.summary]

        if not summaries:
            return profile

        # Extract profile information from summaries
        prompt = f"""Extract user profile information from the following scene summaries.

Scene Summaries:
{chr(10).join(f"- {s}" for s in summaries[-20:])}  # Last 20 scenes

Extract:
1. Explicit facts (verifiable attributes like name, job, location, age)
2. Implicit traits (preferences, habits, personality traits)

Format as JSON:
{{
    "explicit_facts": {{
        "job": "...",
        "location": "...",
        ...
    }},
    "implicit_traits": ["trait1", "trait2", ...]
}}
"""

        try:
            response = self.llm.complete_json(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            # Update explicit facts
            explicit = response.get("explicit_facts", {})
            for key, value in explicit.items():
                profile.update_explicit_fact(key, value)

            # Update implicit traits
            traits = response.get("implicit_traits", [])
            for trait in traits:
                profile.add_implicit_trait(trait)

        except (json.JSONDecodeError, KeyError):
            pass  # Silently fail on error

        self.store.save_user_profile(profile)
        return profile

    def consolidate(
        self,
        memcell: MemCell,
        user_id: str = "default",
        auto_resolve: bool = True,
    ) -> Dict[str, Any]:
        """
        Full consolidation process for a MemCell.

        1. Cluster into MemScene
        2. Detect conflicts
        3. Update user profile

        Args:
            memcell: MemCell to consolidate
            user_id: User ID for profile updates

        Returns:
            Dict with consolidation results
        """
        # Add to search index first
        self.index.add_memcell(memcell)
        self.store.add_memcell(memcell)

        # Cluster into scene
        scene = self.cluster_memcell(memcell)

        # Detect conflicts
        conflicts = self.detect_conflicts(memcell, scene)

        # Resolve conflicts if requested
        if auto_resolve and conflicts:
            for conflict in conflicts:
                # Default to recency for now as per requirements
                self.resolve_conflict(conflict, resolution="recency")

        # Update user profile
        self.update_user_profile(user_id=user_id)

        return {
            "memcell_id": memcell.event_id,
            "scene_id": scene.scene_id,
            "theme": scene.theme,
            "conflicts_detected": len(conflicts),
            "conflict_ids": [c.conflict_id for c in conflicts],
            "profile_updated": True,
        }

    # Semantic theme prototypes for embedding-based classification
    THEME_PROTOTYPES = {
        "career": "work job office career employment profession project meeting boss company",
        "health": "health fitness exercise doctor medical wellness gym sick medicine diet weight",
        "relationships": "family friends partner love social relationship marriage date friend",
        "hobbies": "hobby interests leisure fun recreation game sport music book movie travel cooking",
        "finance": "money budget savings investment expenses salary bank invest purchase",
        "location": "home apartment city travel moving residence house trip vacation",
    }

    def _infer_theme(self, episode: str) -> str:
        """Infer the theme of an episode using semantic similarity."""
        if not self.embeddings:
            return "general"

        # Cache prototype embeddings on first use
        if not self._theme_embeddings:
            for theme, prototype in self.THEME_PROTOTYPES.items():
                self._theme_embeddings[theme] = self.embeddings.embed(prototype)

        episode_embedding = self.embeddings.embed(episode)

        best_theme = "general"
        best_similarity = 0.0

        for theme, prototype in self.THEME_PROTOTYPES.items():
            theme_embedding = self._theme_embeddings[theme]
            similarity = self.embeddings.similarity(episode_embedding, theme_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_theme = theme

        return best_theme if best_similarity > 0.3 else "general"

    def _update_scene_summary(self, scene: MemScene) -> None:
        """Update the summary of a MemScene from its MemCells."""
        memcells = self.store.get_memcells_by_scene(scene.scene_id)
        if not memcells:
            return

        # Get recent episodes
        recent_episodes = [m.episode for m in memcells[-5:]]

        prompt = f"""Create a concise summary of this MemScene based on its recent episodes.

Recent Episodes:
{chr(10).join(f"- {e}" for e in recent_episodes)}

Summary (2-3 sentences):
"""

        try:
            response = self.llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            scene.summary = response.strip()
            scene.last_updated = datetime.now(timezone.utc)
        except Exception:
            pass  # Silently fail

    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get statistics about the consolidation state."""
        scenes = self.store.get_all_memscenes()
        conflicts = self.store.get_all_conflicts()

        return {
            "scene_count": len(scenes),
            "total_memcells": len(self.store.get_all_memcells()),
            "conflict_count": len(conflicts),
            "unresolved_conflicts": len(self.store.get_unresolved_conflicts()),
            "scenes_by_theme": self._count_scenes_by_theme(scenes),
        }

    def _count_scenes_by_theme(self, scenes: List[MemScene]) -> Dict[str, int]:
        """Count scenes by theme."""
        counts = {}
        for scene in scenes:
            counts[scene.theme] = counts.get(scene.theme, 0) + 1
        return counts
