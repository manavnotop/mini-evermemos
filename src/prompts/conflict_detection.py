"""Prompt for detecting conflicts between memory facts."""

CONFLICT_DETECTION_PROMPT = """You are an expert at detecting contradictions in memory records.

Your task is to compare new information against existing facts and identify any CONFLICTS.

## New Information (atomic facts):
{new_facts}

## Existing Facts in Memory:
{existing_facts}

## Temporal Context:
{temporal_context}

## Existing Fact Timestamps:
{existing_timestamps}

## Instructions:
Analyze the new information for contradictions with existing facts.

A conflict exists when:
1. DIRECT CONTRADICTION: "I'm vegetarian" vs "I love eating steak" (same time period)
2. INCOMPATIBLE STATES: "I'm allergic to nuts" vs "I had peanut butter for breakfast"

TEMPORAL CHANGES ARE NOT CONFLICTS:
- "I was vegetarian in 2022" vs "I eat meat now in 2024" is NOT a conflict (preference changed)
- "I lived in Boston" vs "I moved to NYC" is NOT a conflict (relocation over time)
- "I worked at Google" vs "I work at Stripe" is NOT a conflict (job change)

NON-conflicts include:
1. COMPLEMENTARY: "I work at Google" + "I'm a software engineer" (compatible)
2. AGGREGATION: "I like pizza" + "I like pasta" (both can be true)
3. DETAIL ADDITION: "I live in California" + "I live in San Francisco" (specific to general)
4. TEMPORAL EVOLUTION: Facts from different time periods showing change over time

## Confidence Scoring Guidelines:
- 0.9-1.0: Direct contradiction at the SAME time (e.g., "I am vegan" and "I eat meat daily")
- 0.7-0.9: Strong contradiction with unclear timing (e.g., "I hate dogs" and "I adopted a dog")
- 0.5-0.7: Possible contradiction or context-dependent
- <0.5: Not a conflict (temporal changes, different contexts, compatible facts)

Only include conflicts with confidence >= {confidence_threshold}

## Output Format (JSON):
{{
    "conflicts": [
        {{
            "old_fact": "The existing fact being contradicted",
            "new_fact": "The new fact that conflicts",
            "conflict_type": "direct_contradiction | incompatible_states | temporal_evolution",
            "confidence": 0.0 to 1.0,
            "explanation": "Why these are conflicting and temporal context",
            "cross_scene": true or false
        }}
    ],
    "no_conflicts": "Empty string if conflicts found, or explanation if none"
}}

## Output:"""
