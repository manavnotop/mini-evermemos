"""Prompt for extracting episodes from conversations."""

EPISODE_EXTRACTION_PROMPT = """You are an expert at synthesizing conversation transcripts into coherent episodic summaries.

Your task is to convert the given conversation turns into a concise third-person narrative episode.

## Conversation:
{conversation}

## Instructions:
Create a single, coherent episode that:
1. Is written in third person (e.g., "The user discussed...", "They mentioned...")
2. Captures the key events, decisions, or information exchanged
3. Resolves coreferences (e.g., "it" -> the previously mentioned thing)
4. Is concise but complete (2-4 sentences typically)
5. Focuses on factual content, not emotional subtext

## Output Format (JSON):
{{
    "episode": "The synthesized episode as a third-person narrative",
    "key_entities": ["list", "of", "key", "entities", "mentioned"],
    "topics": ["list", "of", "topics", "covered"]
}}

## Output:"""
