# Prompt templates for LLM operations
from .boundary_detection import BOUNDARY_DETECTION_PROMPT
from .conflict_detection import CONFLICT_DETECTION_PROMPT
from .episode_extraction import EPISODE_EXTRACTION_PROMPT
from .foresight_extraction import FORESIGHT_EXTRACTION_PROMPT

__all__ = [
    "BOUNDARY_DETECTION_PROMPT",
    "EPISODE_EXTRACTION_PROMPT",
    "FORESIGHT_EXTRACTION_PROMPT",
    "CONFLICT_DETECTION_PROMPT",
]
