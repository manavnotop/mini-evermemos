"""Phase II: Semantic Consolidation - Organize MemCells into MemScenes."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..models import (
    AtomicFact,
    ConflictRecord,
    ConflictResolution,
    MemCell,
    MemScene,
    UserProfile,
)
from ..prompts import CONFLICT_DETECTION_PROMPT
from ..storage import MemoryStore, SearchIndex
from ..utils import EmbeddingService, LLMProvider
from .constants import (
    CONFIDENCE_THRESHOLDS,
    DEFAULTS,
    PROFILE_LIMITS,
    SIMILARITY_THRESHOLDS,
    SUMMARY_LIMITS,
    THEME_PROTOTYPES,
    TIME_LIMITS,
)

logger = logging.getLogger(__name__)


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
        similarity_threshold: float = SIMILARITY_THRESHOLDS["scene_clustering"],
        max_time_gap_days: int = TIME_LIMITS["max_scene_time_gap_days"],
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

    def _fact_text(self, fact: Any) -> str:
        """Return fact text across AtomicFact/string/object variants."""
        if isinstance(fact, AtomicFact):
            return fact.text
        if hasattr(fact, "text"):
            return str(fact.text)
        return str(fact)

    def _normalize_fact(self, fact_text: str) -> str:
        """Normalize fact text for dedup/conflict keying."""
        return " ".join(fact_text.lower().split())

    def _normalize_trait(self, trait: Any) -> str:
        """Normalize trait text for profile deduplication."""
        return " ".join(str(trait).strip().lower().split())

    def _from_iso(self, value: str) -> Optional[datetime]:
        """Parse ISO timestamp safely."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

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
                    time_gap = (
                        abs((memcell.timestamp - scene.latest_timestamp).total_seconds())
                        / 86400.0
                    )
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
        check_cross_scene: bool = True,
        confidence_threshold: float = CONFIDENCE_THRESHOLDS["conflict_detection"],
    ) -> List[ConflictRecord]:
        """
        Detect conflicts between new MemCell and existing facts.

        Enhanced to:
        1. Check across related scenes (same theme or high similarity)
        2. Apply confidence threshold (only flag high-confidence conflicts)
        3. Include temporal context to avoid flagging temporal changes

        Args:
            memcell: New MemCell to check
            scene: MemScene to check against
            check_cross_scene: Whether to check conflicts across related scenes
            confidence_threshold: Minimum confidence to flag a conflict

        Returns:
            List of detected conflicts
        """
        # Get existing facts from the current scene, excluding incoming memcell
        existing_facts_by_norm: Dict[str, str] = {}
        fact_sources: Dict[str, List[str]] = {}
        existing_timestamps: Dict[str, str] = {}

        for existing_memcell_id in scene.memcell_ids:
            if existing_memcell_id == memcell.event_id:
                continue

            existing = self.store.get_memcell(existing_memcell_id)
            if not existing or not existing.atomic_facts:
                continue

            for fact in existing.atomic_facts:
                fact_text = self._fact_text(fact)
                normalized = self._normalize_fact(fact_text)

                if normalized not in existing_facts_by_norm:
                    existing_facts_by_norm[normalized] = fact_text
                fact_sources.setdefault(normalized, [])
                if existing_memcell_id not in fact_sources[normalized]:
                    fact_sources[normalized].append(existing_memcell_id)
                existing_timestamps[normalized] = existing.timestamp.isoformat()

        # Also check related scenes (same theme or high embedding similarity)
        detection_scope = "scene_only"
        if check_cross_scene:
            related_scenes = self._find_related_scenes(
                scene,
                similarity_threshold=SIMILARITY_THRESHOLDS["related_scenes"],
            )
            if related_scenes:
                detection_scope = "cross_scene"
            for related_scene in related_scenes:
                if related_scene.scene_id == scene.scene_id:
                    continue  # Skip current scene (already processed)

                for existing_memcell_id in related_scene.memcell_ids:
                    if existing_memcell_id == memcell.event_id:
                        continue

                    existing = self.store.get_memcell(existing_memcell_id)
                    if not existing or not existing.atomic_facts:
                        continue

                    for fact in existing.atomic_facts:
                        fact_text = self._fact_text(fact)
                        normalized = self._normalize_fact(fact_text)
                        if normalized not in existing_facts_by_norm:
                            existing_facts_by_norm[normalized] = fact_text
                        fact_sources.setdefault(normalized, [])
                        if existing_memcell_id not in fact_sources[normalized]:
                            fact_sources[normalized].append(existing_memcell_id)
                        existing_timestamps[normalized] = existing.timestamp.isoformat()

        if not existing_facts_by_norm:
            return []

        # Format for conflict detection with temporal context
        new_fact_texts = [self._fact_text(f) for f in memcell.atomic_facts]
        new_facts_str = "\n".join(new_fact_texts)
        existing_facts = list(existing_facts_by_norm.values())
        existing_facts_str = "\n".join(existing_facts)

        # Build temporal context from memcell timestamp
        temporal_context = f"New fact timestamp: {memcell.timestamp.isoformat()}"

        prompt = CONFLICT_DETECTION_PROMPT.format(
            new_facts=new_facts_str,
            existing_facts=existing_facts_str,
            temporal_context=temporal_context,
            existing_timestamps=json.dumps(
                {
                    existing_facts_by_norm[norm]: ts
                    for norm, ts in existing_timestamps.items()
                },
                indent=2,
            ),
            confidence_threshold=confidence_threshold,
        )

        try:
            response = self.llm.complete_json(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            conflicts = []
            for item in response.get("conflicts", []):
                confidence = item.get("confidence", 0.0)

                # Only flag conflicts that meet confidence threshold
                if confidence >= confidence_threshold:
                    old_fact_text = item["old_fact"]
                    old_fact_norm = self._normalize_fact(old_fact_text)
                    conflict = ConflictRecord.create(
                        memcell_id=memcell.event_id,
                        scene_id=scene.scene_id,
                        old_fact=old_fact_text,
                        new_fact=item["new_fact"],
                        metadata={
                            "conflict_type": item.get("conflict_type"),
                            "explanation": item.get("explanation"),
                            "confidence": confidence,
                            "cross_scene": item.get("cross_scene", False),
                            "temporal_context": temporal_context,
                            "old_fact_source_memcell_ids": fact_sources.get(
                                old_fact_norm, []
                            ),
                            "new_fact_memcell_id": memcell.event_id,
                            "detection_scope": detection_scope,
                        },
                    )
                    conflicts.append(conflict)
                    self.store.save_conflict(conflict)

            return conflicts

        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            logger.warning("Conflict detection failed for memcell %s: %s", memcell.event_id, exc)
            return []

    def _find_related_scenes(
        self,
        scene: MemScene,
        similarity_threshold: float = SIMILARITY_THRESHOLDS["related_scenes"],
    ) -> List[MemScene]:
        """
        Find scenes related to the given scene by theme or embedding similarity.

        Args:
            scene: Reference scene
            similarity_threshold: Minimum similarity to consider related

        Returns:
            List of related MemScene objects
        """
        related = []
        all_scenes = self.store.get_all_memscenes()

        for other_scene in all_scenes:
            if other_scene.scene_id == scene.scene_id:
                continue

            # Check if same theme
            if other_scene.theme == scene.theme:
                related.append(other_scene)
                continue

            # Check embedding similarity if centroids available
            if scene.centroid and other_scene.centroid and self.embeddings:
                similarity = self.embeddings.similarity(
                    scene.centroid, other_scene.centroid
                )
                if similarity >= similarity_threshold:
                    related.append(other_scene)

        return related

    def resolve_conflict(
        self,
        conflict: ConflictRecord,
        resolution: ConflictResolution = "recency",
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
            old_fact_norm = self._normalize_fact(conflict.old_fact)
            source_ids = conflict.metadata.get("old_fact_source_memcell_ids", [])
            replaced_by = conflict.metadata.get("new_fact_memcell_id", conflict.memcell_id)

            for source_id in source_ids:
                source_memcell = self.store.get_memcell(source_id)
                if not source_memcell:
                    continue

                original_facts = source_memcell.atomic_facts
                retained_facts = [
                    fact
                    for fact in original_facts
                    if self._normalize_fact(self._fact_text(fact)) != old_fact_norm
                ]

                if len(retained_facts) == len(original_facts):
                    continue

                source_memcell.atomic_facts = retained_facts
                superseded = source_memcell.metadata.setdefault("superseded_facts", [])
                superseded.append(
                    {
                        "old_fact": conflict.old_fact,
                        "replaced_by_memcell_id": replaced_by,
                        "conflict_id": conflict.conflict_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                self.store.upsert_memcell(source_memcell)
                self.index.add_memcell(source_memcell)
        elif resolution == "keep_both":
            # Both facts are kept - add timestamps to differentiate
            pass
        elif resolution in {"manual", "user_choice"}:
            # Flag for manual review
            conflict.metadata["needs_review"] = True

        self.store.save_conflict(conflict)

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

        # Get latest scene summaries first to keep profile updates recency-aware
        scenes = sorted(
            self.store.get_all_memscenes(),
            key=lambda s: s.latest_timestamp or s.last_updated,
            reverse=True,
        )
        scene_pairs = [
            (scene.summary, scene.latest_timestamp or scene.last_updated)
            for scene in scenes
            if scene.summary
        ]
        recent_pairs = scene_pairs[: PROFILE_LIMITS["max_scenes_for_profile"]]

        if not recent_pairs:
            return profile

        latest_scene_time = max(ts for _, ts in recent_pairs)

        # Extract profile information from summaries
        prompt = f"""Extract user profile information from the following scene summaries.

Scene Summaries:
{chr(10).join(f"- [{ts.isoformat()}] {summary}" for summary, ts in recent_pairs)}

Extract:
1. Explicit facts (verifiable attributes like name, job, location, age)
2. Implicit traits (preferences, habits, personality traits)
3. Prefer recent stable information; avoid temporary/expired states

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
                existing = profile.explicit_facts.get(key)
                existing_updated_at = None
                if isinstance(existing, dict):
                    existing_updated_at = self._from_iso(existing.get("updated_at"))

                if existing_updated_at and existing_updated_at > latest_scene_time:
                    continue

                old_value = existing.get("value") if isinstance(existing, dict) else existing
                if old_value != value:
                    history = profile.metadata.setdefault("fact_history", [])
                    history.append(
                        {
                            "key": key,
                            "old_value": old_value,
                            "new_value": value,
                            "evidence_time": latest_scene_time.isoformat(),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                profile.update_explicit_fact(key, value, timestamp=latest_scene_time)

            # Update implicit traits
            traits = response.get("implicit_traits", [])
            if not isinstance(traits, list):
                traits = []

            max_traits = PROFILE_LIMITS.get("max_implicit_traits", 50)
            existing_norm_traits = {
                self._normalize_trait(trait) for trait in profile.implicit_traits
            }
            for trait in traits:
                normalized = self._normalize_trait(trait)
                if not normalized or normalized in existing_norm_traits:
                    continue
                profile.add_implicit_trait(str(trait).strip())
                existing_norm_traits.add(normalized)

            if len(profile.implicit_traits) > max_traits:
                profile.implicit_traits = profile.implicit_traits[-max_traits:]
                profile.last_updated = datetime.now(timezone.utc)

        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            logger.warning("Profile update failed for user %s: %s", user_id, exc)

        self.store.save_user_profile(profile)
        return profile

    def deduplicate_facts_globally(
        self,
        new_facts: List,
        scene: MemScene,
        current_memcell_id: Optional[str] = None,
        protected_fact_norms: Optional[set[str]] = None,
        similarity_threshold: float = SIMILARITY_THRESHOLDS["fact_deduplication"],
    ) -> tuple[List, int, int]:
        """
        Remove facts that are semantically similar to existing facts in scene.

        This performs GLOBAL deduplication across all MemCells in the scene,
        not just within a single MemCell.

        Args:
            new_facts: Facts from the new MemCell (List[AtomicFact] or List[str])
            scene: MemScene to check against
            current_memcell_id: Current MemCell ID to exclude from existing facts
            protected_fact_norms: Fact norms that must not be removed by dedup
            similarity_threshold: Cosine similarity threshold for duplicates

        Returns:
            Tuple of (unique_facts, original_count, unique_count)
        """
        original_count = len(new_facts)
        protected_fact_norms = protected_fact_norms or set()

        if not new_facts or not self.embeddings:
            return new_facts, original_count, original_count

        # Gather existing scene facts excluding incoming memcell
        existing_text_by_norm: Dict[str, str] = {}
        for memcell_id in scene.memcell_ids:
            if current_memcell_id and memcell_id == current_memcell_id:
                continue

            existing = self.store.get_memcell(memcell_id)
            if not existing or not existing.atomic_facts:
                continue

            for fact in existing.atomic_facts:
                fact_text = self._fact_text(fact)
                fact_norm = self._normalize_fact(fact_text)
                if fact_norm not in existing_text_by_norm:
                    existing_text_by_norm[fact_norm] = fact_text

        # Local exact dedup inside incoming memcell (keep higher confidence)
        unique_new_facts: List[Any] = []
        best_by_norm: Dict[str, Tuple[Any, float]] = {}
        order: List[str] = []
        for fact in new_facts:
            fact_text = self._fact_text(fact)
            fact_norm = self._normalize_fact(fact_text)
            confidence = getattr(fact, "confidence", 1.0)
            if fact_norm not in best_by_norm:
                best_by_norm[fact_norm] = (fact, confidence)
                order.append(fact_norm)
            elif confidence > best_by_norm[fact_norm][1]:
                best_by_norm[fact_norm] = (fact, confidence)

        for fact_norm in order:
            unique_new_facts.append(best_by_norm[fact_norm][0])

        if not existing_text_by_norm:
            unique_count = len(unique_new_facts)
            return unique_new_facts, original_count, unique_count

        existing_items = list(existing_text_by_norm.items())
        existing_texts = [text for _, text in existing_items]
        existing_embeddings = self.embeddings.embed_batch(existing_texts)
        existing_pairs = [
            (fact_norm, emb) for (fact_norm, _), emb in zip(existing_items, existing_embeddings)
        ]

        # Semantic dedup against existing scene facts using fact-level embeddings
        unique_facts = []
        new_texts = [self._fact_text(f) for f in unique_new_facts]
        new_embeddings = self.embeddings.embed_batch(new_texts)
        for new_fact, new_emb, new_text in zip(unique_new_facts, new_embeddings, new_texts):
            new_norm = self._normalize_fact(new_text)
            if new_norm in protected_fact_norms:
                unique_facts.append(new_fact)
                continue
            if new_norm in existing_text_by_norm:
                continue

            duplicate = False
            for existing_norm, existing_emb in existing_pairs:
                if existing_norm == new_norm:
                    duplicate = True
                    break
                similarity = self.embeddings.similarity(new_emb, existing_emb)
                if similarity > similarity_threshold:
                    duplicate = True
                    break

            if not duplicate:
                unique_facts.append(new_fact)

        unique_count = len(unique_facts)

        return unique_facts, original_count, unique_count

    def consolidate(
        self,
        memcell: MemCell,
        user_id: str = DEFAULTS["user_id"],
        auto_resolve: bool = True,
    ) -> Dict[str, Any]:
        """
        Full consolidation process for a MemCell.

        1. Cluster into MemScene
        2. Deduplicate facts globally
        3. Detect conflicts
        4. Update user profile

        Args:
            memcell: MemCell to consolidate
            user_id: User ID for profile updates

        Returns:
            Dict with consolidation results
        """
        # Ensure embedding is available before scene assignment
        if memcell.embedding is None:
            memcell.embedding = self.embeddings.embed(memcell.episode)

        # Cluster into scene
        scene = self.cluster_memcell(memcell)
        memcell.scene_id = scene.scene_id

        # Detect conflicts BEFORE dedup to avoid suppressing contradictory updates.
        conflicts = self.detect_conflicts(memcell, scene)
        protected_conflict_facts = {
            self._normalize_fact(conflict.new_fact) for conflict in conflicts
        }

        # Resolve conflicts if requested
        if auto_resolve and conflicts:
            for conflict in conflicts:
                # Default to recency for now as per requirements
                self.resolve_conflict(conflict, resolution="recency")

        # Deduplicate facts globally within the scene, preserving conflict facts.
        unique_facts, orig_count, unique_count = self.deduplicate_facts_globally(
            memcell.atomic_facts,
            scene,
            current_memcell_id=memcell.event_id,
            protected_fact_norms=protected_conflict_facts,
        )
        memcell.atomic_facts = unique_facts

        # Update metadata with dedup stats
        memcell.metadata["original_facts_count"] = orig_count
        memcell.metadata["unique_facts_count"] = unique_count
        memcell.metadata["deduplicated_count"] = orig_count - unique_count
        memcell.metadata["conflict_checked_before_dedup"] = True
        memcell.metadata["dedup_skipped_due_to_conflict"] = len(
            protected_conflict_facts
        )
        memcell.metadata["scene_id"] = scene.scene_id

        # Persist MemCell only after deduplication and scene assignment
        self.store.upsert_memcell(memcell)
        self.index.add_memcell(memcell)

        # Update/persist scene summary after MemCell is stored
        self._update_scene_summary(scene)
        self.store.add_memscene(scene)

        # Update user profile
        self.update_user_profile(user_id=user_id)

        return {
            "memcell_id": memcell.event_id,
            "scene_id": scene.scene_id,
            "theme": scene.theme,
            "conflicts_detected": len(conflicts),
            "conflict_ids": [c.conflict_id for c in conflicts],
            "profile_updated": True,
            "original_facts_count": orig_count,
            "unique_facts_count": unique_count,
            "conflict_checked_before_dedup": True,
            "dedup_skipped_due_to_conflict": len(protected_conflict_facts),
            "dedup_rate": ((orig_count - unique_count) / orig_count * 100)
            if orig_count > 0
            else 0,
        }

    def _infer_theme(self, episode: str) -> str:
        """Infer the theme of an episode using semantic similarity."""
        if not self.embeddings:
            return DEFAULTS["theme"]

        # Cache prototype embeddings on first use
        if not self._theme_embeddings:
            for theme, prototype in THEME_PROTOTYPES.items():
                self._theme_embeddings[theme] = self.embeddings.embed(prototype)

        episode_embedding = self.embeddings.embed(episode)

        best_theme = DEFAULTS["theme"]
        best_similarity = 0.0

        for theme, prototype in THEME_PROTOTYPES.items():
            theme_embedding = self._theme_embeddings[theme]
            similarity = self.embeddings.similarity(episode_embedding, theme_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_theme = theme

        return (
            best_theme
            if best_similarity > SIMILARITY_THRESHOLDS["theme_classification"]
            else DEFAULTS["theme"]
        )

    def _update_scene_summary(self, scene: MemScene) -> None:
        """Update the summary of a MemScene from its MemCells."""
        memcells = self.store.get_memcells_by_scene(scene.scene_id)
        if not memcells:
            return

        # Get recent episodes
        recent_episodes = [
            m.episode for m in memcells[-SUMMARY_LIMITS["max_recent_episodes"] :]
        ]

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
        except Exception as exc:
            logger.warning("Scene summary update failed for scene %s: %s", scene.scene_id, exc)

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
