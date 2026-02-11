"""Unified extraction prompt for Phase I Episodic Trace Formation."""

UNIFIED_EXTRACTION_PROMPT = """You are an expert memory system extracting structured information from conversation history.

TASK:
Given the episode history below, synthesize a coherent narrative and extract atomic facts and foresight signals.

Episode History:
{episode_history}

Current Time: {current_time}

OUTPUT FORMAT - Return valid JSON:
{{
  "episode": "Third-person narrative of what happened...",
  "atomic_facts": [
    {{
      "text": "User works at Google",
      "confidence": 0.95
    }},
    {{
      "text": "User lives in San Francisco", 
      "confidence": 0.88
    }}
  ],
  "foresight": [
    {{
      "description": "User is taking antibiotics",
      "start_time": "2026-01-15T10:00:00Z",
      "end_time": "2026-01-22T10:00:00Z",
      "confidence": 0.92
    }}
  ],
  "key_entities": ["Google", "San Francisco", "antibiotics"],
  "topics": ["career", "location", "health"]
}}

GUIDELINES:
- Episode: Concise third-person summary (1-3 sentences)
- Atomic Facts: Discrete, verifiable statements with confidence 0.0-1.0
- Foresight: Time-bounded future states with validity windows
- Key Entities: Proper nouns and significant terms
- Topics: Thematic categories (career, health, relationships, hobbies, etc.)
"""
