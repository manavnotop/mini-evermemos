"""Constants for the core memory system.

This module contains all threshold values, limits, and configuration constants
to avoid magic numbers scattered throughout the codebase.
"""

# Similarity thresholds for clustering and deduplication
SIMILARITY_THRESHOLDS = {
    "scene_clustering": 0.70,  # Minimum similarity for MemCell to join existing MemScene
    "related_scenes": 0.65,  # Minimum similarity to consider scenes related
    "fact_deduplication": 0.92,  # Cosine similarity threshold for duplicate facts (global)
    "fact_local_dedup": 0.95,  # Cosine similarity threshold for duplicate facts (local)
    "theme_classification": 0.30,  # Minimum similarity for theme classification
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "conflict_detection": 0.7,  # Minimum confidence to flag a conflict
}

# Time-related constants
TIME_LIMITS = {
    "max_scene_time_gap_days": 7,  # Maximum days between MemCells in same scene
}

# Message processing limits
MESSAGE_LIMITS = {
    "max_messages_per_episode": 12,  # Force split after this many messages
    "max_messages_in_context": 20,  # Limit context to last N messages
    "max_message_length": 500,  # Truncate messages longer than this
}

# Search and retrieval defaults
SEARCH_DEFAULTS = {
    "rrf_k": 60.0,  # Reciprocal Rank Fusion constant
    "scene_top_k": 10,  # Default number of scenes to retrieve
    "episode_top_k": 10,  # Default number of episodes to retrieve
    "global_candidates_multiplier": 5,  # Multiply episode_top_k for global search
    "max_retrieval_rounds": 2,  # Maximum rounds of retrieval + verification
}

# Profile extraction limits
PROFILE_LIMITS = {
    "max_scenes_for_profile": 20,  # Number of recent scenes to use for profile extraction
}

# Summary generation limits
SUMMARY_LIMITS = {
    "max_recent_episodes": 5,  # Number of recent episodes to use for scene summary
}

# Temperature settings for LLM calls
LLM_TEMPERATURES = {
    "default": 0.0,  # Use deterministic output by default
}

# Context formatting limits
CONTEXT_LIMITS = {
    "max_context_length": 3000,  # Limit context for sufficiency checking
    "max_facts_per_memcell": 5,  # Limit facts shown per memcell in context
    "max_memcells_in_summary": 3,  # Number of memcells to show in rewrite context
}

# Theme prototypes for semantic classification
THEME_PROTOTYPES = {
    "career": "work job office career employment profession project meeting boss company",
    "health": "health fitness exercise doctor medical wellness gym sick medicine diet weight",
    "relationships": "family friends partner love social relationship marriage date friend",
    "hobbies": "hobby interests leisure fun recreation game sport music book movie travel cooking",
    "finance": "money budget savings investment expenses salary bank invest purchase",
    "location": "home apartment city travel moving residence house trip vacation",
}

# Default values
DEFAULTS = {
    "user_id": "default",
    "theme": "general",
    "similarity_threshold": 0.70,
    "max_time_gap_days": 7,
}
