# EverMemOS Technical Design (Updated)

## 1. Goal and Paper Alignment

This system implements the EverMemOS lifecycle from the paper:

`Trace Formation -> Semantic Consolidation -> Reconstructive Recollection`

It is designed to solve the task requirements:

- Extract structured memory units (MemCells) from dialogue.
- Organize memory into thematic scenes (MemScenes).
- Detect and resolve contradictions explicitly (no silent overwrite).
- Apply temporal validity to foresight so expired states are filtered out.
- Use hybrid retrieval (BM25 + dense embeddings) with an agentic sufficiency loop.

## 2. End-to-End Architecture

### 2.1 Phase I: Episodic Trace Formation

Input stream is segmented into episodes, then each episode is converted into one MemCell containing:

- `episode`: rewritten third-person summary.
- `atomic_facts`: structured facts with confidence.
- `foresight`: time-bounded forward-looking items (`start_time`, `end_time`, `confidence`).
- `metadata`: entities/topics and dedup stats.

Boundary detection is LLM-first with an embedding drift backstop, plus hard splitting by message limit.

### 2.2 Phase II: Semantic Consolidation

Each MemCell is embedded and assigned to a MemScene:

- Join existing scene if semantic similarity and time-gap constraints pass.
- Otherwise create a new scene with inferred theme.

Then:

- Conflicts are detected (scene + related-scene scope).
- Auto-resolution defaults to `recency`.
- Global fact deduplication runs after conflict detection (conflict facts are protected from dedup removal).
- Scene summary and user profile are updated.

### 2.3 Phase III: Reconstructive Recollection

Given query `q` at time `t_query`:

1. Global hybrid retrieval over MemCells using RRF (BM25 + dense).
2. Score scenes by max candidate score among their MemCells.
3. Pool all MemCells from selected scenes, re-rank, keep top episodes.
4. Filter foresight by validity interval at `t_query`.
5. Run sufficiency check; rewrite query and retry (max rounds capped).

## 3. Core Data Structures

### 3.1 MemCell

```json
{
  "event_id": "uuid",
  "scene_id": "uuid",
  "episode": "User discussed starting antibiotics for two weeks.",
  "atomic_facts": [{"text": "User is on antibiotics", "confidence": 0.91}],
  "foresight": [{
    "description": "User should avoid alcohol during antibiotics",
    "start_time": "2023-07-01T10:00:00Z",
    "end_time": "2023-07-15T10:00:00Z",
    "confidence": 0.88
  }],
  "embedding": [0.0, 0.0, "..."],
  "timestamp": "2023-07-01T10:00:00Z",
  "metadata": {
    "original_facts_count": 4,
    "unique_facts_count": 3,
    "deduplicated_count": 1
  }
}
```

### 3.2 MemScene

```json
{
  "scene_id": "uuid",
  "theme": "health",
  "summary": "Recent episodes about temporary illness and treatment constraints.",
  "memcell_ids": ["event_1", "event_2"],
  "centroid": [0.0, 0.0, "..."],
  "latest_timestamp": "2023-07-05T08:00:00Z"
}
```

### 3.3 ConflictRecord

- Captures `old_fact`, `new_fact`, confidence, source MemCell IDs, detection scope, and resolution.
- Keeps audit trail for explainability/debugging.

### 3.4 UserProfile

- `explicit_facts`: stable, verifiable fields (job, location, etc.) with history.
- `implicit_traits`: recurring preferences/behavior traits.

## 4. Technical Configuration (Exact)

### 4.1 Models and Temperature

- LLM provider: OpenAI-compatible.
- LLM model in evaluations: `gpt-4o-mini`.
- Embedding model (default OpenAI backend): `text-embedding-3-small`.
- Temperature: `0.0` (deterministic) for extraction, conflict detection, profile update, scene summary, sufficiency, and query rewrite.

### 4.2 Embedding Dimensions

- Runtime embedding dimension in storage is derived as:
  - `embedding_dim = getattr(embedding_service, "dim", 1536)`.
- For `OpenAIEmbeddings`, `dim` attribute is not set in code, so Milvus defaults to `1536`.
- For sentence-transformers backend, dimension is model-dependent and read from `get_sentence_embedding_dimension()`.

### 4.3 Retrieval and Consolidation Hyperparameters

- Scene clustering threshold: `0.70` (LoCoMo eval uses `0.60` override).
- Related-scene threshold (cross-scene conflict scope): `0.65`.
- Fact dedup threshold (global): `0.92`.
- Fact dedup threshold (local): `0.95`.
- Theme classification threshold: `0.30`.
- Max scene time gap: `7` days (LoCoMo eval uses `30` days override).
- Max messages per episode: `12`.
- Retrieval:
  - `scene_top_k = 10`
  - `episode_top_k = 10`
  - RRF constant `k = 60`
  - Global candidate multiplier `5`
  - Max retrieval rounds `2`

## 5. Infrastructure (MongoDB + Milvus)

### 5.1 Storage split

- MongoDB stores structured records:
  - collections: `memcells`, `memscenes`, `conflicts`, `profiles`.
  - indexes:
    - MemCell: `event_id` (unique), `scene_id`, `timestamp`
    - MemScene: `scene_id` (unique), `theme`
    - Conflict: `conflict_id` (unique)
    - Profile: `user_id` (unique)
- Milvus stores vectors:
  - `memcells` collection: `event_id`, `embedding`, `scene_id`, `timestamp`
  - `memscenes` collection: `scene_id`, `embedding`, `theme`

### 5.2 Vector index details

- Index type: `HNSW`
- Metric: `COSINE`
- Build params: `M=8`, `efConstruction=64`
- Search params: `ef=64`

### 5.3 Local docker stack

- Milvus standalone: `milvusdb/milvus:v2.3.0`
- etcd: `quay.io/coreos/etcd:v3.5.0`
- minio: `minio/minio:RELEASE.2023-03-20T20-16-18Z`
- MongoDB: `mongo:latest`

## 6. Scale Evaluation Results (from JSON files)

Two result sets exist:

- Baseline run: `scale_eval_results_1769750745.json`
- Improved run: `eval_v2.json`

### 6.1 Raw checkpoint metrics

| Messages | File | MemCells | Scenes | Conflicts | Dedup Rate | Retrieval Accuracy | P50 (ms) | P90 (ms) | Avg (ms) | Valid QA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 108 | scale_eval_results_1769750745.json | 6 | 6 | 0 | 0.00% | 54.10% | 7509 | 9057 | 7509 | 61 |
| 215 | scale_eval_results_1769750745.json | 10 | 7 | 0 | 0.00% | 54.29% | 8807 | 10650 | 8807 | 105 |
| 306 | scale_eval_results_1769750745.json | 14 | 11 | 0 | 0.00% | 42.65% | 7140 | 8436 | 7140 | 136 |
| 519 | scale_eval_results_1769750745.json | 24 | 18 | 0 | 0.00% | 23.84% | 7658 | 9012 | 7658 | 302 |
| 108 | eval_v2.json | 6 | 1 | 0 | 9.68% | 73.77% | 867 | 1216 | 953 | 61 |
| 215 | eval_v2.json | 10 | 1 | 0 | 8.33% | 68.57% | 808 | 1536 | 926 | 105 |
| 306 | eval_v2.json | 14 | 1 | 1 | 6.35% | 72.79% | 773 | 1165 | 870 | 136 |
| 519 | eval_v2.json | 24 | 3 | 1 | 6.42% | 70.76% | 770 | 1053 | 828 | 236 |

`eval_v2.json` also includes checkpoint query-time anchors:

- 108: `2023-07-06T20:18:00+00:00`
- 215: `2023-07-20T20:56:00+00:00`
- 306: `2023-08-25T13:33:00+00:00`
- 519: `2023-10-22T09:55:00+00:00`

### 6.2 v2 vs baseline deltas

| Messages | Accuracy Delta | P90 Latency Delta | Scenes Delta | Conflicts Delta | Dedup Delta |
|---|---:|---:|---:|---:|---:|
| 108 | +19.67 pts | -7840 ms | -5 | 0 | +9.68 pts |
| 215 | +14.29 pts | -9114 ms | -6 | 0 | +8.33 pts |
| 306 | +30.15 pts | -7271 ms | -10 | +1 | +6.35 pts |
| 519 | +46.92 pts | -7959 ms | -15 | +1 | +6.42 pts |

## 7. Interpretation Against Task Requirements

### 7.1 What is working

- Structured extraction is stable across scale (`6 -> 24` MemCells as message count grows).
- Conflict handling appears in larger checkpoints (`1` conflict by 306 and 519 in v2).
- Deduplication is active in v2 (`~6-10%`), absent in baseline (`0%`).
- Retrieval relevance remains high at 500+ messages in v2 (`70.76%`), while baseline degrades sharply.
- Latency in v2 stays around ~0.8-1.5s P90 across checkpoints, satisfying scale requirement.

### 7.2 Tradeoffs observed

- Very aggressive scene consolidation in v2 (1-3 scenes total) improves retrieval speed and accuracy on this dataset but can reduce thematic granularity.
- Baseline over-fragmented scenes (6/7/11/18), likely causing retrieval noise and lower accuracy.

## 8. Current Limitations and Next Technical Steps

- Profile quality is benchmarked indirectly; add explicit profile regression metrics.
- Temporal foresight validity is implemented but should be stress-tested with more synthetic expiry-heavy conversations.
- Add ablations per task rubric:
  - no conflict resolution
  - no foresight filtering
  - no scene-guided retrieval
- Add explicit infra benchmark dimensions:
  - Mongo read/write p95 under concurrent ingestion
  - Milvus recall@k vs latency across larger vector counts
