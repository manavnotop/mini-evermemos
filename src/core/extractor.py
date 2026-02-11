"""Phase I: Episodic Trace Formation - Extract MemCells from conversations."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..models import ForesightItem, MemCell
from ..prompts import BOUNDARY_DETECTION_PROMPT, UNIFIED_EXTRACTION_PROMPT
from ..utils import (
    EmbeddingService,
    LLMProvider,
    infer_validity_duration,
    parse_datetime,
)
from .constants import (
    LLM_TEMPERATURES,
    MESSAGE_LIMITS,
    SIMILARITY_THRESHOLDS,
)


class MemCellExtractor:
    """
    Extract MemCells from conversation streams.

    Implements Phase I of the EverMemOS lifecycle:
    - Detect semantic boundaries in conversations (LLM-only, per paper)
    - Synthesize episodes from dialogue history
    - Extract atomic facts with confidence scores
    - Identify foresight with validity intervals
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_service: Optional[EmbeddingService] = None,
        boundary_threshold: float = SIMILARITY_THRESHOLDS["scene_clustering"],
        max_messages_per_episode: int = MESSAGE_LIMITS["max_messages_per_episode"],
    ):
        """
        Initialize the MemCell extractor.

        Args:
            llm_provider: LLM provider for extraction
            embedding_service: Service for generating embeddings (used for deduplication only)
            boundary_threshold: Similarity threshold for embedding backstop boundary detection
            max_messages_per_episode: Maximum messages before forced split
        """
        self.llm = llm_provider
        self.embeddings = embedding_service
        self.boundary_threshold = boundary_threshold
        self.max_messages_per_episode = max_messages_per_episode

        # State management
        self._episode_history: List[Dict[str, Any]] = []  # Raw history for synthesis
        self._last_episode_embedding: Optional[List[float]] = None

    def detect_boundary(
        self,
        messages: List[Dict[str, Any]],
        force_split: bool = False,
    ) -> Tuple[bool, str]:
        """
        Detect if a semantic boundary should occur.

        Uses LLM-only detection per paper specification.

        Args:
            messages: Messages to evaluate for boundary
            force_split: Force a split regardless of semantic analysis

        Returns:
            Tuple of (is_boundary, reason)
        """
        if force_split:
            return True, "Forced split"

        if not messages:
            return False, "No messages to evaluate"

        # Build context from prior history + candidate turn
        previous_summary = (
            self._format_episode_history(messages[:-1]) if len(messages) > 1 else ""
        )
        recent_context = self._format_episode_history(messages[-1:])

        prompt = BOUNDARY_DETECTION_PROMPT.format(
            turn_count=len(messages),
            recent_context=recent_context,
            previous_summary=previous_summary,
        )

        try:
            response = self.llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURES["default"],
            )
            result = json.loads(response)

            is_boundary = result.get("is_boundary", False)
            reason = result.get("reason", "No reason provided")

            # Embedding backstop: if topic drifts far from last finalized episode,
            # split even when LLM misses a boundary.
            if (
                not is_boundary
                and self.embeddings
                and self._last_episode_embedding
                and len(messages) > 1
            ):
                candidate_text = self._format_episode_history(messages)
                candidate_embedding = self.embeddings.embed(candidate_text)
                similarity = self.embeddings.similarity(
                    candidate_embedding, self._last_episode_embedding
                )
                if similarity < self.boundary_threshold:
                    return True, (
                        "Boundary via embedding drift "
                        f"(sim={similarity:.3f} < {self.boundary_threshold:.3f})"
                    )

            return is_boundary, reason

        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: no boundary on error
            return False, f"Error in boundary detection: {e}"

    def extract_episode(
        self,
        episode_history: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Extract MemCell components from episode history via unified LLM call.

        Single LLM call per paper: "prompt the LLM over the rewritten Episode E
        to output a constrained schema of Atomic Facts and Foresight signals"

        Args:
            episode_history: Raw accumulated messages
            timestamp: Reference timestamp for the episode

        Returns:
            Dict containing episode, atomic_facts (with confidence), foresight,
            key_entities, and topics
        """
        # Format history for prompt
        history_text = self._format_episode_history(episode_history)
        current_time = timestamp or datetime.now(timezone.utc)

        # Single unified LLM call
        prompt = UNIFIED_EXTRACTION_PROMPT.format(
            episode_history=history_text,
            current_time=current_time.isoformat(),
        )

        try:
            response = self.llm.complete_json(
                [{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURES["default"],
            )

            # Parse unified response
            return self._parse_unified_response(response, current_time)

        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            AttributeError,
            ValueError,
        ):
            # Fallback: simple extraction
            return self._fallback_extraction(episode_history, current_time)

    def _parse_unified_response(
        self,
        response: Any,
        current_time: datetime,
    ) -> Dict[str, Any]:
        """Parse unified LLM response into MemCell components."""
        if not isinstance(response, dict):
            response = {}

        # Parse atomic facts with confidence
        atomic_facts = []
        raw_facts = response.get("atomic_facts", [])
        if not isinstance(raw_facts, list):
            raw_facts = []

        for fact_data in raw_facts:
            if isinstance(fact_data, dict):
                text = fact_data.get("text", fact_data.get("fact", ""))
                confidence = fact_data.get("confidence", 1.0)
            elif isinstance(fact_data, str):
                text = fact_data
                confidence = 1.0
            elif hasattr(fact_data, "text"):
                text = str(getattr(fact_data, "text", ""))
                confidence = getattr(fact_data, "confidence", 1.0)
            else:
                continue

            text = str(text).strip()
            if not text:
                continue

            atomic_facts.append(
                {
                    "text": text,
                    "confidence": confidence,
                }
            )

        # Parse foresight
        foresight_items = []
        raw_foresight = response.get("foresight", [])
        if not isinstance(raw_foresight, list):
            raw_foresight = []

        for item in raw_foresight:
            if not isinstance(item, dict):
                continue

            description = str(item.get("description", "")).strip()
            if not description:
                continue

            start_time = None
            end_time = None

            if item.get("start_time"):
                try:
                    start_time = parse_datetime(str(item["start_time"]))
                except (TypeError, ValueError):
                    start_time = None
            if item.get("end_time"):
                try:
                    end_time = parse_datetime(str(item["end_time"]))
                except (TypeError, ValueError):
                    end_time = None
            elif description:
                # Infer end_time from description
                duration = infer_validity_duration(
                    description, str(item.get("reasoning", ""))
                )
                end_time = current_time + duration

            # Ensure end_time is after start_time
            if start_time and end_time and end_time < start_time:
                end_time = start_time + infer_validity_duration(
                    description
                )

            foresight_items.append(
                ForesightItem(
                    description=description,
                    start_time=start_time or current_time,
                    end_time=end_time,
                    confidence=item.get("confidence", 1.0),
                )
            )

        return {
            "episode": response.get("episode", ""),
            "atomic_facts": atomic_facts,
            "foresight": foresight_items,
            "key_entities": response.get("key_entities", []),
            "topics": response.get("topics", []),
        }

    def _fallback_extraction(
        self,
        episode_history: List[Dict[str, Any]],
        current_time: datetime,
    ) -> Dict[str, Any]:
        """Fallback extraction when LLM JSON parsing fails."""
        # Create simple episode from last user message
        user_msgs = [m for m in episode_history if m.get("role") == "user"]
        episode = (
            f"User discussed: {user_msgs[-1].get('content', '')[:200]}"
            if user_msgs
            else f"Conversation with {len(episode_history)} messages"
        )

        return {
            "episode": episode,
            "atomic_facts": [],
            "foresight": [],
            "key_entities": [],
            "topics": [],
        }

    def _deduplicate_facts(
        self,
        facts: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], int, int]:
        """
        Remove semantically duplicate facts using embeddings.
        Keep higher confidence version when duplicates found.

        Returns:
            Tuple of (unique_facts, original_count, unique_count)
        """
        original_count = len(facts)

        if original_count <= 1 or not self.embeddings:
            return facts, original_count, original_count

        # Extract texts for embedding
        fact_texts = [f["text"] for f in facts]
        embeddings = self.embeddings.embed_batch(fact_texts)

        unique_facts = []
        unique_embeddings = []

        for fact, embedding in zip(facts, embeddings):
            is_duplicate = False
            duplicate_idx = -1

            for idx, unique_embedding in enumerate(unique_embeddings):
                similarity = self.embeddings.similarity(embedding, unique_embedding)
                if similarity > SIMILARITY_THRESHOLDS["fact_local_dedup"]:
                    is_duplicate = True
                    duplicate_idx = idx
                    break

            if not is_duplicate:
                unique_facts.append(fact)
                unique_embeddings.append(embedding)
            else:
                # Keep the one with higher confidence
                existing_conf = unique_facts[duplicate_idx].get("confidence", 1.0)
                new_conf = fact.get("confidence", 1.0)
                if new_conf > existing_conf:
                    unique_facts[duplicate_idx] = fact

        return unique_facts, original_count, len(unique_facts)

    def create_memcell(
        self,
        episode_history: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None,
    ) -> MemCell:
        """
        Create a MemCell from episode history.

        This is the main entry point for Phase I.

        Args:
            episode_history: Raw accumulated conversation history
            timestamp: Timestamp for the MemCell

        Returns:
            MemCell instance
        """
        # Single extraction call
        resolved_timestamp = timestamp or self._resolve_episode_timestamp(
            episode_history
        )
        extraction = self.extract_episode(episode_history, resolved_timestamp)

        # Deduplicate atomic facts (keeping higher confidence duplicates)
        atomic_facts, original_facts_count, unique_facts_count = (
            self._deduplicate_facts(extraction["atomic_facts"])
        )

        metadata = {
            "key_entities": extraction.get("key_entities", []),
            "topics": extraction.get("topics", []),
            "original_facts_count": original_facts_count,
            "unique_facts_count": unique_facts_count,
        }

        return MemCell.create(
            episode=extraction["episode"],
            atomic_facts=atomic_facts,
            foresight=extraction["foresight"],
            source_messages=episode_history,
            metadata=metadata,
            timestamp=resolved_timestamp,
        )

    def process_conversation_stream(
        self,
        messages: List[Dict[str, Any]],
        flush: bool = False,
    ) -> List[MemCell]:
        """
        Process a stream of messages and extract MemCells.

        Uses a sliding window approach with episode history buffer.
        Boundary detection happens BEFORE adding new message to history.

        Args:
            messages: New messages to process
            flush: Force flush all pending messages as MemCells

        Returns:
            List of extracted MemCells
        """
        memcells = []

        for message in messages:
            # Check boundary BEFORE adding to history
            if self._episode_history:
                is_boundary, _ = self.detect_boundary(self._episode_history + [message])
            else:
                is_boundary = False

            if is_boundary and self._episode_history:
                # Synthesize and create MemCell from complete history
                memcell = self.create_memcell(self._episode_history)
                memcells.append(memcell)
                self._update_last_episode_embedding(self._episode_history)
                # Start fresh history with current message
                self._episode_history = [message]
            else:
                # Accumulate in history
                self._episode_history.append(message)

            # Force split based on current episode size (not global message count)
            if len(self._episode_history) >= self.max_messages_per_episode:
                # Force split
                memcell = self.create_memcell(self._episode_history)
                memcells.append(memcell)
                self._update_last_episode_embedding(self._episode_history)
                self._episode_history = []

        # Handle flush: process remaining history
        if flush and self._episode_history:
            memcell = self.create_memcell(self._episode_history)
            memcells.append(memcell)
            self._update_last_episode_embedding(self._episode_history)
            self._episode_history = []

        return memcells

    def flush(self) -> List[MemCell]:
        """Flush any pending messages as MemCells."""
        memcells = []
        if self._episode_history:
            memcell = self.create_memcell(self._episode_history)
            memcells.append(memcell)
            self._update_last_episode_embedding(self._episode_history)
            self._episode_history = []
        return memcells

    def _format_episode_history(self, history: List[Dict[str, Any]]) -> str:
        """Format episode history for LLM prompt."""
        formatted = []
        for msg in history[-MESSAGE_LIMITS["max_messages_in_context"] :]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > MESSAGE_LIMITS["max_message_length"]:
                content = content[: MESSAGE_LIMITS["max_message_length"]] + "..."
            formatted.append(f"[{role}]: {content}")
        return "\n".join(formatted)

    def _resolve_episode_timestamp(
        self,
        episode_history: List[Dict[str, Any]],
    ) -> datetime:
        """Derive episode timestamp from latest message timestamp when available."""
        for message in reversed(episode_history):
            timestamp = message.get("timestamp")
            if not timestamp:
                continue

            if isinstance(timestamp, datetime):
                return (
                    timestamp
                    if timestamp.tzinfo is not None
                    else timestamp.replace(tzinfo=timezone.utc)
                )

            if isinstance(timestamp, str):
                return parse_datetime(timestamp)

        return datetime.now(timezone.utc)

    def _update_last_episode_embedding(
        self, episode_history: List[Dict[str, Any]]
    ) -> None:
        """Store embedding of the last finalized episode for boundary backstop."""
        if not self.embeddings or not episode_history:
            return
        episode_text = self._format_episode_history(episode_history)
        self._last_episode_embedding = self.embeddings.embed(episode_text)
