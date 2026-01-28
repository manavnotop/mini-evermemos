"""Embedding service for the memory system."""

import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        pass

    @abstractmethod
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        pass


class SentenceTransformerEmbeddings(EmbeddingService):
    """Embedding service using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: uv add sentence-transformers"
            )

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))


class OpenAIEmbeddings(EmbeddingService):
    """Embedding service using OpenAI text-embedding models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
    ):
        """
        Initialize the OpenAI embedding service.

        Args:
            api_key: OpenAI API key
            model: Embedding model to use
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

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        # OpenAI has a max batch size of 2048
        batch_size = 100  # Conservative limit
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend([data.embedding for data in response.data])

        return all_embeddings

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))


class MockEmbeddings(EmbeddingService):
    """Mock embedding service for testing without API calls."""

    # Keywords that map to specific dimensions for semantic similarity
    THEME_KEYWORDS = {
        "career": [
            "work",
            "job",
            "office",
            "project",
            "meeting",
            "boss",
            "company",
            "career",
        ],
        "health": [
            "health",
            "doctor",
            "exercise",
            "gym",
            "sick",
            "medicine",
            "diet",
            "weight",
        ],
        "relationships": [
            "friend",
            "family",
            "partner",
            "love",
            "relationship",
            "marriage",
            "date",
        ],
        "hobbies": [
            "hobby",
            "game",
            "sport",
            "music",
            "book",
            "movie",
            "travel",
            "cooking",
        ],
        "finance": [
            "money",
            "bank",
            "invest",
            "salary",
            "budget",
            "expense",
            "purchase",
        ],
        "location": ["home", "apartment", "city", "travel", "move", "trip", "vacation"],
    }

    def __init__(self, dim: int = 1536):
        """Initialize mock embeddings with given dimension."""
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        """Return a mock embedding based on text content (keyword-based)."""
        text_lower = text.lower()

        # Create a deterministic embedding based on theme keywords
        embedding = np.zeros(self.dim)

        # Map keywords to specific dimensions
        dim_per_keyword = self.dim // len(self.THEME_KEYWORDS)

        for theme_idx, (theme, keywords) in enumerate(self.THEME_KEYWORDS.items()):
            start_dim = theme_idx * dim_per_keyword
            for kw_idx, kw in enumerate(keywords):
                if kw in text_lower:
                    embedding[start_dim + kw_idx] = 1.0

        # If no keywords found, use text hash for some variation
        if np.all(embedding == 0):
            import hashlib

            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.RandomState(hash_val)
            embedding = rng.randn(self.dim) * 0.1

        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return mock embeddings for a batch of texts."""
        return [self.embed(text) for text in texts]

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))


def get_embedding_service(
    service: str = "sentence-transformers",
    **kwargs,
) -> EmbeddingService:
    """
    Factory function to get an embedding service.

    Args:
        service: Service name ("sentence-transformers", "openai", or "mock")
        **kwargs: Additional service-specific arguments

    Returns:
        EmbeddingService instance
    """
    if service == "sentence-transformers":
        model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
        return SentenceTransformerEmbeddings(model_name=model_name)
    elif service == "openai":
        return OpenAIEmbeddings(**kwargs)
    elif service == "mock":
        return MockEmbeddings(**kwargs)
    else:
        raise ValueError(f"Unknown embedding service: {service}")
