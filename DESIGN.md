# EverMemOS Technical Design Document

## 1. System Architecture

The system implements the **EverMemOS** lifecycle, consisting of three distinct phases: **Trace Formation**, **Semantic Consolidation**, and **Reconstructive Recollection**.


---

## 2. Core Data Structures

### 2.1 MemCell (Atomic Memory Unit)
The fundamental unit of storage, designed to be immutable once created (except for metadata updates).

```json
{
  "event_id": "uuid4",
  "episode": "User mentioned they are starting a new keto diet today.",
  "atomic_facts": [
    "User started keto diet",
    "User is aiming for low-carb intake"
  ],
  "foresight": [
    {
      "description": "User on keto diet",
      "start_time": "2023-10-01T10:00:00Z",
      "end_time": "2023-11-01T10:00:00Z",
      "confidence": 0.9
    }
  ],
  "embedding": [0.01, -0.45, ...],
  "timestamp": "2023-10-01T10:00:00Z"
}
```

### 2.2 MemScene (Thematic Cluster)
A dynamic cluster representing a "thread" of memory (e.g., "Health", "Career").

```json
{
  "scene_id": "uuid4",
  "theme": "health",
  "summary": "User's ongoing journey with various diets and fitness routines.",
  "memcell_ids": ["uuid-1", "uuid-2"],
  "centroid": [0.02, -0.41, ...],
  "latest_timestamp": "2023-10-05T12:00:00Z"
}
```

### 2.3 UserProfile (Long-term Persona)
Maintained separately to track explicit facts and implicit traits derived from memory scenes.

```json
{
  "user_id": "default",
  "explicit_facts": {
    "job": {
      "value": "Software Engineer",
      "confidence": 0.95,
      "source_scenes": ["scene-1"]
    }
  },
  "implicit_traits": [
    "Prefers morning meetings",
    "Health-conscious"
  ],
  "last_updated": "2023-10-06T09:00:00Z"
}
```

---

## 3. Algorithm Implementation Details

### 3.1 Topic Boundary Detection
Located in `src/core/extractor.py`.

*   **Logic**: Sliding Window + LLM Check.
*   **Window**: Buffer of accumulated messages.
*   **Trigger**:
    1.  **Semantic Check**: `BOUNDARY_DETECTION_PROMPT` compares `current_buffer` vs `previous_episode_summary`.
    2.  **Hard Limit**: `max_messages_per_episode` (default: 50) forces a flush to prevent context overflow.
*   **Input**: Stream of messages.
*   **Output**: Boolean flag (`is_boundary`) and reason.
*   **Tradeoff**: We prioritized **semantic coherence** over speed. A pure token-counter would be faster but would split "My favorite movie is..." and "...The Matrix" into separate cells.

### 3.2 Semantic Consolidation & Scene Management
Located in `src/core/consolidator.py`.

*   **Clustering Metric**: Cosine Similarity via `EmbeddingService`.
*   **Theme Prototypes**: Uses `THEME_PROTOTYPES` (e.g., "health" -> "wellness gym diet") to bootstrap scene matching even with empty scenes.
*   **Threshold**: `0.70` (empirically chosen).
    *   `> 0.70`: Add to existing scene.
    *   `< 0.70`: Spawn new scene (inferred theme).
*   **Time Horizon**: `max_time_gap_days` (default: 7). Even if semantically similar, if the last update was > 7 days ago, we prefer starting a new "chapter" (scene) to keep contexts temporally tight.
*   **Summarization**: Each scene maintains a running summary updated via LLM after new MemCells are added.

### 3.3 Conflict Detection Strategy
Located in `src/core/consolidator.py`.

*   **Scope**: Intra-Scene only. We only check for conflicts within the assigned `MemScene` (e.g., "Health"). This avoids false positives (e.g., "I love running" in 2020 vs "I hate running" in 2024—context matters).
*   **Extraction**:
    1.  Flatten all `atomic_facts` from the Scene.
    2.  Compare new `atomic_facts` against flattened list using `CONFLICT_DETECTION_PROMPT`.
*   **Resolution Default**: **Recency**.
    *   *Why*: In a companion context, user preferences evolve. "I am vegan" (today) should supersede "I eat meat" (last year).
    *   *Safety*: We do not delete the old MemCell. We log a `ConflictRecord` so the history is preserved for debugging.

### 3.4 User Profile Management
Located in `src/core/consolidator.py`.

*   **Process**: Triggered after Scene Consolidation.
*   **Input**: Recent `MemScene` summaries.
*   **Extraction**: LLM extracts:
    1.  **Explicit Facts**: Verifiable attributes (Name, Job, Location).
    2.  **Implicit Traits**: Recurring patterns or preferences.
*   **Storage**: Updates the `UserProfile` object, which provides high-level context during retrieval.

### 3.4 Temporal Awareness (Foresight)
Located in `src/core/extractor.py` and `src/core/retriever.py`.

*   **Extraction**: LLM infers `start_time` and `end_time` from natural language.
    *   *Explicit*: "I'm travelling for 2 weeks" -> `duration=14 days`.
    *   *Implicit*: "I have a cold" -> `heuristic=7 days` (tunable).
*   **Retrieval Filtering**:
    ```python
    valid = (item.start_time <= query_time) and (query_time <= item.end_time)
    ```
    This ensures that querying "What is the user's health status?" *after* the flu has passed does not return "User has a cold".

---

## 4. Reconstructive Retrieval Pipeline

Located in `src/core/retriever.py`.

This is an **agentic** pipeline, not just vector search.

1.  **Hybrid Search (Recall)**:
    *   Retrieves `5 * top_k` candidates using `index.search_hybrid()` (BM25 + Vector).
    *   We deliberately over-fetch candidates (`max(50, episode_top_k * 5)`) to ensure we catch enough signals to identify relevant *Scenes*.
2.  **Scene Selection (Precision)**:
    *   Identifying the "Active Scene" is crucial.
    *   Score(Scene) = Max(Score(MemCells in Scene)).
    *   We select the top 3-5 Scenes.
3.  **Context Expansion**:
    *   We fetch **ALL** MemCells from the selected scenes. This provides the LLM with the full narrative arc of that topic, not just the isolated chunks that matched the vector query.
4.  **Sufficiency Check (Agentic Loop)**:
    *   LLM evaluates: "Does this context answer the query?"
    *   **If No**: Uses reasoning to rewrite the query (e.g., "Find user's location" -> "Check user's moving plans from last month").
    *   **Max Rounds**: 2 (to cap latency).

## 5. Tradeoffs & Limitations

*   **Latency vs. Quality**: The agentic retrieval loop (Sufficiency Check) adds ~2-3s per hop. Total latency is ~7s. We accept this for higher precision.
*   **Granularity**: Atomic facts can sometimes be *too* atomic, losing nuance. We mitigate this by always including the full `episode` text in the retrieval context.
*   **Scalability**: While storage scales sub-linearly (good), the "Context Expansion" step (loading full scenes) could become a bottleneck if a single scene grows to huge sizes (e.g., "General Chat" scene). We mitigate this with `max_time_gap_days` to force scene splitting.
