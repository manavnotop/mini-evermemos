# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mini-evermemos is an EverMemOS-inspired memory system for AI companions. It implements a three-phase memory lifecycle:
1. **Trace Formation** - Extract atomic memory units (MemCells) from conversations
2. **Semantic Consolidation** - Group into thematic clusters (MemScenes), detect conflicts
3. **Reconstructive Recollection** - Temporal-aware retrieval with foresight expiration

## Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_extractor.py -v

# Run specific test
uv run pytest tests/test_extractor.py::test_memcell_creation -v

# Run with coverage
uv run pytest --cov=src tests/

# Run interactive demo
uv run python demo.py

# Run scale evaluation
uv run python scale_evaluation.py

# Lint code
uv run ruff check src/ tests/
```

## Architecture

```
src/
├── core/           # Business logic (extractor, consolidator, retriever)
├── models/         # Data models (MemCell, MemScene, Conflict, ForesightItem)
├── storage/        # MemoryStore, SearchIndex (MongoDB + Milvus + BM25 hybrid)
├── utils/          # LLM provider, embeddings, datetime utilities
└── prompts/        # LLM prompt templates
```

**Key classes:**
- `MemorySystem` (src/core/memory_system.py) - Main orchestrator
- `MemCell` - Atomic memory unit with serialization
- `MemScene` - Thematic grouping of MemCells
- `ConflictRecord` - Detected memory conflicts

**Three-phase flow:**
1. `extractor.py` - Boundary detection, episode/facts/foresight extraction
2. `consolidator.py` - Clustering (0.70 similarity threshold), conflict detection
3. `retriever.py` - Hybrid BM25 + vector search, temporal filtering

## Key Conventions

- **Dataclass models**: All use `to_dict()`/`from_dict()` for serialization
- **Mock providers**: `MockProvider` and `MockEmbeddings` for testing without real APIs
- **Temporal filtering**: 7-day max gap between MemCells in same scene
- **Foresight expiry**: 30-day default (LLM-inferred)
- **Storage**: MongoDB (structured data) + Milvus (vectors) + BM25 (text search)

## Testing Notes

- Tests use mock storage clients by default (tests/mock_db.py) to avoid requiring MongoDB/Milvus
- Tests that use `MemorySystem()` directly work because it detects `MockEmbeddings` and uses mock storage
- For tests requiring real storage, inject clients via `MemoryStore(mongo_client=..., milvus_client=...)`
- Embedding dimension must match between embedding service and Milvus client (384 for sentence-transformers, 1536 for OpenAI)

## Dependencies

- Python 3.12+
- MongoDB for structured data storage
- Milvus for vector similarity search
- OpenAI API or sentence-transformers for embeddings
