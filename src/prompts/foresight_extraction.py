"""Prompt for extracting foresight (time-bounded information) from episodes."""

FORESIGHT_EXTRACTION_PROMPT = """You are an expert at identifying time-bounded information in conversational data.

Your task is to extract all FORESIGHT items from the given episode - these are forward-looking statements about future states, plans, or temporary conditions.

## Episode:
{episode}

## Current Time Reference: {current_time}

## What to Extract:
Look for information that:
1. Is about a FUTURE state or condition
2. Has an implied or explicit VALIDITY PERIOD (when it starts and ends)
3. Will become OUTDATED after a certain date

Examples of foresight:
- "I'm traveling to Paris next week" -> Valid during trip, expires after
- "I'm on antibiotics until Friday" -> Expires after the course ends
- "My deadline is March 15th" -> Expires after the deadline passes
- "I'm visiting family for 2 weeks" -> Valid during visit, expires after

Examples of NON-foresight (stable information):
- "I work at Google" (stable fact)
- "My name is John" (permanent)
- "I graduated from MIT" (past event)

## Validity Period Guidelines:
- If a specific end date is mentioned, use that
- If a duration is mentioned (e.g., "for 2 weeks"), calculate end date
- If no end is clear, use your best judgment based on context (default to 30 days)
- Start time is typically "now" or explicitly stated

## Output Format (JSON):
{{
    "foresight_items": [
        {{
            "description": "Clear description of the future state",
            "start_time": "ISO datetime or null if implicit (now)",
            "end_time": "ISO datetime when this expires or becomes irrelevant",
            "confidence": 0.0 to 1.0,
            "reasoning": "Brief explanation of the validity period"
        }}
    ],
    "no_foresight": "Explanation if no foresight items found, or empty string"
}}

## Output:"""
