"""LLM provider abstraction for the memory system."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion for the given messages."""
        pass

    @abstractmethod
    def complete_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate a JSON completion for the given messages."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible LLM provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use for completions
            base_url: Base URL for OpenAI-compatible API
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: uv add openai"
            )

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate a completion using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "text"},
        )
        return response.choices[0].message.content

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate a JSON completion using OpenAI."""
        import json

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            return {"raw": content}


class MockProvider(LLMProvider):
    """Mock LLM provider for testing without API calls."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        """
        Initialize the mock provider.

        Args:
            responses: Dict mapping prompt patterns to mock responses
        """
        self.responses = responses or {}

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> str:
        """Return a mock completion."""
        prompt = messages[-1].get("content", "") if messages else ""
        for pattern, response in self.responses.items():
            if pattern.lower() in prompt.lower():
                return response
        return "Mock response"

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Return a mock JSON completion."""
        return {"result": "mock_json_response"}


def get_llm_provider(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    **kwargs,
) -> LLMProvider:
    """
    Factory function to get an LLM provider.

    Args:
        provider: Provider name ("openai" or "mock")
        api_key: API key for the provider
        model: Model to use
        **kwargs: Additional provider-specific arguments

    Returns:
        LLMProvider instance
    """
    if provider == "openai":
        return OpenAIProvider(api_key=api_key, model=model, **kwargs)
    elif provider == "mock":
        return MockProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
