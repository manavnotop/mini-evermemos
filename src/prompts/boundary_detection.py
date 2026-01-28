"""Prompt for detecting semantic boundaries in conversations."""

BOUNDARY_DETECTION_PROMPT = """You are an expert at detecting topic shifts in conversational data.

Your task is to determine if a semantic boundary (topic shift) has occurred between the recent conversation turns and the previous turns.

## Recent Conversation (last {turn_count} turns):
{recent_context}

## Previous Context Summary:
{previous_summary}

## Instructions:
Analyze the conversation flow and determine if a NEW TOPIC or CONTEXT SHIFT has occurred.

Consider the following indicators of a topic shift:
- New person, place, or event being discussed
- Change in conversation purpose or goal
- Significant time gap mentioned (days, weeks, etc.)
- Completely different subject matter

Consider the following as CONTINUATION of the same topic:
- Elaborating on the same subject
- Asking follow-up questions
- Same people/events being discussed
- Gradual natural flow of conversation

## Output Format (JSON):
{{
    "is_boundary": true or false,
    "reason": "Brief explanation of why this is or isn't a boundary",
    "new_topic": "Description of the new topic if boundary detected",
    "confidence": 0.0 to 1.0
}}

## Output:"""
