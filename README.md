# EverMemOS: Self-Organizing Memory for AI Companions

<div align="center">

**An implementation of the EverMemOS memory operating system for structured long-horizon reasoning.**

[Key Features](#key-features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Evaluation](#evaluation)

</div>

---

## Overview

Existing AI memory systems often fail because they treat memories as static text logs, leading to contradictions, stale information, and noisy retrieval. **EverMemOS** effectively solves these problems by organizing memory into a lifecycle: **Trace Formation → Consolidation → Reconstructive Recollection**.

This implementation focuses on:
1.  **Structured Memory**: Converting raw text into **MemCells** (Episodes, Atomic Facts, Foresight).
2.  **Thematic Organization**: Clustering related memories into **MemScenes** (e.g., "Health", "Career").
3.  **Conflict Resolution**: actively detecting and resolving contradictions (e.g., "I'm vegetarian" vs "I love steak").
4.  **Temporal Awareness**: respecting validity windows for information (e.g., "antibiotics for 2 weeks").

## Key Features

- **🧠 MemCells**: Atomic units of memory containing verified facts and time-bounded foresight.
- **🏙️ MemScenes**: Dynamic clustering of related memories to improve retrieval context.
- **⚔️ Conflict Detection**: LLM-based detection of contradictions prevents silent data corruption.
- **⏳ Temporal Filtering**: Automatically filters out expired "foresight" (plans, temporary states).
- **🔎 Smart Retrieval**: Hybrid search (BM25 + Embeddings) with sufficiency checks and query rewriting.

## Installation

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Docker (for MongoDB and Milvus)

### Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/mini-evermemos.git
    cd mini-evermemos
    ```

2.  **Start Infrastructure**
    ```bash
    docker-compose up -d
    ```

3.  **Install Dependencies**
    ```bash
    uv venv
    source .venv/bin/activate
    uv sync
    ```

4.  **Configure Environment**
    Create a `.env` file with your API keys:
    ```bash
    OPENAI_API_KEY=sk-...
    # Optional: MongoDB/Milvus settings if not using defaults
    ```

## Usage

### Basic Usage

```python
from src.core import MemorySystem
from src.utils import get_llm_provider, OpenAIEmbeddings

# Initialize
llm = get_llm_provider("openai", model="gpt-4o-mini")
system = MemorySystem(llm_provider=llm, embedding_service=OpenAIEmbeddings())

# Add a conversation
messages = [
    {"role": "user", "content": "I'm starting a new job at Google next Monday!"},
    {"role": "assistant", "content": "That's exciting! Good luck."}
]
system.add_conversation(messages)

# Retrieve context
result = system.retrieve("Where does the user work?")
print(result["composed_context"])
```

### Scale Evaluation

To verify system performance scaling (100 -> 500+ messages), run the included evaluation script:

```bash
uv run python scale_evaluation.py --locomo --limit 600
```

This will:
1.  Ingest the **LoCoMo** dataset.
2.  Pause at **100, 200, 300, and 500** messages.
3.  Log metrics (MemCells, Scenes, Conflicts, Latency) to `scale_eval_results_TIMESTAMP.json`.

## Architecture

The system follows the **EverMemOS** 3-phase lifecycle:

### Phase 1: Episodic Trace Formation
*   **Input**: Raw conversation streams.
*   **Process**: LLM extracts "MemCells".
*   **Output**: Structured objects with `episode` (summary), `atomic_facts` (list), and `foresight` (time-bounded info).

### Phase 2: Semantic Consolidation
*   **Input**: New MemCells.
*   **Process**: 
    *   Find similar "MemScene" (or create new).
    *   Detect conflicts with existing scene facts.
    *   Update distinct User Profile (traits vs facts).
*   **Output**: Updated Knowledge Graph / Vector Store.

### Phase 3: Reconstructive Recollection
*   **Input**: User Query.
*   **Process**:
    *   **Scene Selection**: Pick most relevant scenes.
    *   **Filtering**: Remove expired foresight (e.g., old travel plans).
    *   **Sufficiency Check**: LLM checks if retrieved info answers the query. If not, it rewrites the query and tries again.
*   **Output**: Highly relevant composed context.

## Benchmark & Performance

Evaluation on the **LoCoMo** dataset demonstrates:
*   **Conflict Detection**: Successfully flags contradictions that naive RAG misses.
*   **Deduplication**: Reduces storage volume by identifying duplicate facts.
*   **Scalability**: Maintenance of retrieval latency (~P90 < 2s) even as memory grows.

### Locomo10 Evaluation Results

| Checkpoint (Msgs) | MemCells | Scenes | Accuracy | Latency (P90) |
|-------------------|----------|--------|----------|---------------|
| 108               | 6        | 6      | 54.1%    | ~9.1s         |
| 215               | 10       | 7      | 54.3%    | ~10.6s        |
| ~300              | 14       | 11     | 42.6%    | ~8.4s         |
| ~500              | 24       | 18     | 23.8%    | ~9.0s         |

## License

MIT
