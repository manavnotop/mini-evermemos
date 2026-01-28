"""Prompt for detecting conflicts between memory facts."""

CONFLICT_DETECTION_PROMPT = """You are an expert at detecting contradictions in memory records.

Your task is to compare new information against existing facts and identify any CONFLICTS.

## New Information (atomic facts):
{new_facts}

## Existing Facts in Memory:
{existing_facts}

## Instructions:
Analyze the new information for contradictions with existing facts.

A conflict exists when:
1. DIRECT CONTRADICTION: "I'm vegetarian" vs "I love eating steak"
2. TEMPORAL CONFLICT: "I moved to NYC" vs "I live in Boston" (one must be outdated)
3. INCOMPATIBLE STATES: "I'm allergic to nuts" vs "I had peanut butter for breakfast"

NON-conflicts include:
1. COMPLEMENTARY: "I work at Google" + "I'm a software engineer" (compatible)
2. AGGREGATION: "I like pizza" + "I like pasta" (both can be true)
3. DETAIL ADDITION: "I live in California" + "I live in San Francisco" (specific to general)

## Output Format (JSON):
{{
    "conflicts": [
        {{
            "old_fact": "The existing fact being contradicted",
            "new_fact": "The new fact that conflicts",
            "conflict_type": "direct_contradiction | temporal_conflict | incompatible_states",
            "confidence": 0.0 to 1.0,
            "explanation": "Why these are conflicting"
        }}
    ],
    "no_conflicts": "Empty string if conflicts found, or explanation if none"
}}

## Output:"""
