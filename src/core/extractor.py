"""Phase I: Episodic Trace Formation - Extract MemCells from conversations."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..models import ForesightItem, MemCell
from ..prompts import (
    BOUNDARY_DETECTION_PROMPT,
    EPISODE_EXTRACTION_PROMPT,
    FORESIGHT_EXTRACTION_PROMPT,
)
from ..utils import (
    EmbeddingService,
    LLMProvider,
    infer_validity_duration,
    parse_datetime,
)


class MemCellExtractor:
    """
    Extract MemCells from conversation streams.

    Implements Phase I of the EverMemOS lifecycle:
    - Detect semantic boundaries in conversations
    - Synthesize episodes from dialogue
    - Extract atomic facts
    - Identify foresight with validity intervals
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_service: Optional[EmbeddingService] = None,
        boundary_threshold: float = 0.7,
        max_messages_per_episode: int = 50,
    ):
        """
        Initialize the MemCell extractor.

        Args:
            llm_provider: LLM provider for extraction
            embedding_service: Service for generating embeddings
            boundary_threshold: Similarity threshold for boundary detection
            max_messages_per_episode: Maximum messages before forced split
        """
        self.llm = llm_provider
        self.embeddings = embedding_service
        self.boundary_threshold = boundary_threshold
        self.max_messages_per_episode = max_messages_per_episode

        # Sliding window state
        self._pending_messages: List[Dict[str, Any]] = []
        self._last_topic_embedding: Optional[List[float]] = None

    def detect_boundary(
        self,
        messages: List[Dict[str, Any]],
        force_split: bool = False,
    ) -> Tuple[bool, str]:
        """
        Detect if a semantic boundary should occur before the given messages.

        Args:
            messages: New messages to evaluate
            force_split: Force a split regardless of semantic analysis

        Returns:
            Tuple of (is_boundary, reason)
        """
        if force_split or len(self._pending_messages) >= self.max_messages_per_episode:
            return True, "Forced split due to message count limit"

        if not self._pending_messages:
            return False, "First episode, no boundary"

        # Build context
        recent_context = self._format_messages(messages)
        previous_summary = self._format_messages(self._pending_messages)

        prompt = BOUNDARY_DETECTION_PROMPT.format(
            turn_count=len(messages),
            recent_context=recent_context,
            previous_summary=previous_summary,
        )

        try:
            response = self.llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            result = json.loads(response)

            is_boundary = result.get("is_boundary", False)
            reason = result.get("reason", "No reason provided")

            # Update topic embedding if boundary detected
            if is_boundary and self.embeddings:
                combined = recent_context + " " + previous_summary
                self._last_topic_embedding = self.embeddings.embed(combined)

            return is_boundary, reason

        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: no boundary on error
            return False, f"Error in boundary detection: {e}"

    def extract_episode(
        self,
        messages: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Extract an episode from a batch of messages.

        Args:
            messages: Messages to extract from
            timestamp: Reference timestamp for the episode

        Returns:
            Dict containing episode, facts, and foresight
        """
        conversation = self._format_messages(messages)

        # Extract episode
        episode_prompt = EPISODE_EXTRACTION_PROMPT.format(conversation=conversation)

        try:
            episode_response = self.llm.complete_json(
                [{"role": "user", "content": episode_prompt}],
                temperature=0.0,
            )

            episode = episode_response.get("episode", "")
            if not episode:
                # Fallback: create a simple summary
                user_msgs = [m for m in messages if m.get("role") == "user"]
                episode = (
                    f"User discussed: {user_msgs[-1].get('content', '')[:200]}"
                    if user_msgs
                    else "Conversation occurred"
                )

            # Extract atomic facts from episode
            facts = self._extract_atomic_facts(episode)

            # Extract foresight
            foresight_items = self._extract_foresight(
                episode, timestamp or datetime.now(timezone.utc)
            )

            return {
                "episode": episode,
                "atomic_facts": facts,
                "foresight": foresight_items,
                "key_entities": episode_response.get("key_entities", []),
                "topics": episode_response.get("topics", []),
            }

        except json.JSONDecodeError:
            # Fallback on JSON error
            return {
                "episode": f"Conversation with {len(messages)} messages",
                "atomic_facts": [],
                "foresight": [],
                "key_entities": [],
                "topics": [],
            }

    def _extract_atomic_facts(self, episode: str) -> List[str]:
        """Extract atomic facts from an episode."""
        prompt = f"""Extract atomic facts from the following episode. Atomic facts are discrete, verifiable statements.

Episode: {episode}

Format each fact as a concise statement. Examples:
- "User works at Google"
- "User is a software engineer"
- "User lives in San Francisco"
- "User has a dog named Max"

Output as JSON array of strings:
"""

        try:
            response = self.llm.complete_json(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            if isinstance(response, dict) and "facts" in response:
                return response["facts"]
            elif isinstance(response, list):
                return response

            return []

        except (json.JSONDecodeError, KeyError):
            return []

    def _deduplicate_facts(self, facts: List[str]) -> tuple[List[str], int, int]:
        """
        Remove semantically duplicate facts using embeddings.

        Returns:
            Tuple of (unique_facts, original_count, unique_count)
        """
        original_count = len(facts)

        if original_count <= 1 or not self.embeddings:
            return facts, original_count, original_count

        # Compute embeddings for all facts ONCE
        embeddings = self.embeddings.embed_batch(facts)

        unique_facts = []
        unique_embeddings = []

        for i, (fact, embedding) in enumerate(zip(facts, embeddings)):
            is_duplicate = False
            for unique_embedding in unique_embeddings:
                # Compute cosine similarity
                dot = sum(a * b for a, b in zip(embedding, unique_embedding))
                norm_a = sum(a * a for a in embedding) ** 0.5
                norm_b = sum(b * b for b in unique_embedding) ** 0.5
                similarity = dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

                if similarity > 0.95:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_facts.append(fact)
                unique_embeddings.append(embedding)

        return unique_facts, original_count, len(unique_facts)

    def _extract_foresight(
        self,
        episode: str,
        current_time: datetime,
    ) -> List[ForesightItem]:
        """Extract foresight items from an episode."""
        prompt = FORESIGHT_EXTRACTION_PROMPT.format(
            episode=episode,
            current_time=current_time.isoformat(),
        )

        try:
            response = self.llm.complete_json(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            items = []
            for item in response.get("foresight_items", []):
                start_time = None
                end_time = None

                if item.get("start_time"):
                    start_time = parse_datetime(item["start_time"])
                if item.get("end_time"):
                    end_time = parse_datetime(item["end_time"])
                elif item.get("reasoning"):
                    # Infer from reasoning
                    duration = infer_validity_duration(
                        item.get("description", ""), item.get("reasoning", "")
                    )
                    end_time = current_time + duration

                # Ensure end_time is after start_time
                if start_time and end_time and end_time < start_time:
                    end_time = start_time + infer_validity_duration(
                        item.get("description", "")
                    )

                items.append(
                    ForesightItem(
                        description=item["description"],
                        start_time=start_time,
                        end_time=end_time,
                        confidence=item.get("confidence", 1.0),
                    )
                )

            return items

        except (json.JSONDecodeError, KeyError):
            return []

    def create_memcell(
        self,
        messages: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None,
    ) -> MemCell:
        """
        Create a MemCell from messages.

        This is the main entry point for Phase I.

        Args:
            messages: Conversation messages
            timestamp: Timestamp for the MemCell

        Returns:
            MemCell instance
        """
        # Extract episode, facts, and foresight
        extraction = self.extract_episode(messages, timestamp)

        # Deduplicate atomic facts
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
            source_messages=messages,
            metadata=metadata,
            timestamp=timestamp,
        )

    def process_conversation_stream(
        self,
        messages: List[Dict[str, Any]],
        flush: bool = False,
    ) -> List[MemCell]:
        """
        Process a stream of messages and extract MemCells.

        Uses a sliding window approach to detect boundaries.

        Args:
            messages: New messages to process
            flush: Force flush all pending messages as MemCells

        Returns:
            List of extracted MemCells
        """
        memcells = []

        for message in messages:
            self._pending_messages.append(message)

            # Check for boundary
            is_boundary, _ = self.detect_boundary([message], force_split=False)

            if is_boundary or flush:
                # Extract MemCell from pending messages
                if self._pending_messages:
                    memcell = self.create_memcell(self._pending_messages)
                    memcells.append(memcell)
                    self._pending_messages = []

        return memcells

    def flush(self) -> List[MemCell]:
        """Flush any pending messages as MemCells."""
        memcells = []
        if self._pending_messages:
            memcell = self.create_memcell(self._pending_messages)
            memcells.append(memcell)
            self._pending_messages = []
        return memcells

    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages as a readable string."""
        formatted = []
        for msg in messages[-20:]:  # Limit to last 20 for context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 500:
                content = content[:500] + "..."
            formatted.append(f"[{role}]: {content}")
        return "\n".join(formatted)
