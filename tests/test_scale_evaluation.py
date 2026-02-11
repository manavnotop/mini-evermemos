"""Tests for LoCoMo scale evaluation helpers and wiring."""

import json
from datetime import datetime, timezone

from scale_evaluation import (
    _compute_latency_metrics,
    extract_locomo_sessions_with_stats,
    run_locomo_eval,
)


def test_extract_locomo_sessions_infers_user_role_without_caroline() -> None:
    """User role should be inferred from earliest speaker, not a hardcoded name."""
    data = [
        {
            "conversation": {
                "session_1": [
                    {"speaker": "Gina", "text": "Hi", "dia_id": "D1:1"},
                    {"speaker": "Jon", "text": "Hello", "dia_id": "D1:2"},
                ],
                "session_1_date_time": "1:00 pm on 1 May, 2023",
            },
            "qa": [],
        }
    ]

    sessions, qa_pairs, _ = extract_locomo_sessions_with_stats(data)

    assert len(sessions) == 1
    assert sessions[0]["user_speaker"] == "Gina"
    assert sessions[0]["messages"][0]["role"] == "user"
    assert sessions[0]["messages"][1]["role"] == "assistant"
    assert qa_pairs == []


def test_extract_locomo_sessions_avoids_cross_dialogue_evidence_collisions() -> None:
    """Same dia_id across dialogues must not make later QA prematurely valid."""
    data = [
        {
            "conversation": {
                "session_1": [
                    {"speaker": "A", "text": "First", "dia_id": "D1:1"},
                    {"speaker": "B", "text": "Reply", "dia_id": "D1:2"},
                ],
                "session_1_date_time": "1:00 pm on 1 May, 2023",
            },
            "qa": [
                {"question": "q1", "answer": "a1", "evidence": ["D1:1"]},
            ],
        },
        {
            "conversation": {
                "session_1": [
                    {"speaker": "C", "text": "Other", "dia_id": "D1:1"},
                    {"speaker": "D", "text": "Other reply", "dia_id": "D1:2"},
                ],
                "session_1_date_time": "1:00 pm on 2 May, 2023",
            },
            "qa": [
                {"question": "q2", "answer": "a2", "evidence": ["D1:1"]},
            ],
        },
    ]

    _, qa_pairs, _ = extract_locomo_sessions_with_stats(data, limit_messages=2)

    assert len(qa_pairs) == 1
    assert qa_pairs[0]["question"] == "q1"
    assert qa_pairs[0]["dialogue_idx"] == 0


def test_extract_locomo_sessions_skips_no_evidence_qas() -> None:
    """QAs without evidence should be skipped and counted in stats."""
    data = [
        {
            "conversation": {
                "session_1": [
                    {"speaker": "A", "text": "First", "dia_id": "D1:1"},
                ],
                "session_1_date_time": "1:00 pm on 1 May, 2023",
            },
            "qa": [
                {"question": "has evidence", "answer": "a", "evidence": ["D1:1"]},
                {"question": "missing evidence", "answer": "a", "evidence": []},
            ],
        }
    ]

    _, qa_pairs, stats = extract_locomo_sessions_with_stats(data, limit_messages=1)

    assert len(qa_pairs) == 1
    assert qa_pairs[0]["question"] == "has evidence"
    assert stats["skipped_no_evidence_qa_count"] == 1


def test_compute_latency_metrics_uses_true_median() -> None:
    """P50 should be median, not mean."""
    metrics = _compute_latency_metrics([1.0, 1.0, 100.0])

    assert metrics["latency_p50"] == 1.0
    assert metrics["latency_avg"] == 34.0


def test_run_locomo_eval_passes_checkpoint_eval_time_to_retrieve(
    monkeypatch, tmp_path
) -> None:
    """run_locomo_eval should call retrieve with checkpoint-derived query_time."""

    class FakeLLM:
        def complete_json(self, messages):
            return {"match": True}

    class FakeEmbeddings:
        dim = 3

    class FakeMemorySystem:
        instances = []

        def __init__(self, *args, **kwargs):
            self.retrieve_calls = []
            FakeMemorySystem.instances.append(self)

        def clear(self):
            return None

        def add_conversation(self, messages, timestamp=None):
            return {
                "original_facts_count": len(messages),
                "unique_facts_count": len(messages),
            }

        def get_memory_stats(self):
            return {"memcell_count": 1, "memscene_count": 1, "conflict_count": 0}

        def retrieve(self, query, query_time=None):
            self.retrieve_calls.append(query_time)
            return {"composed_context": "context"}

    messages = []
    for i in range(100):
        speaker = "Gina" if i % 2 == 0 else "Jon"
        messages.append({"speaker": speaker, "text": f"m{i}", "dia_id": f"D1:{i+1}"})

    data = [
        {
            "conversation": {
                "session_1": messages,
                "session_1_date_time": "1:00 pm on 8 May, 2023",
            },
            "qa": [
                {"question": "q", "answer": "a", "evidence": ["D1:1"]},
            ],
        }
    ]

    input_path = tmp_path / "locomo_test.json"
    output_path = tmp_path / "results.json"
    input_path.write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(
        "scale_evaluation.get_llm_provider",
        lambda *_args, **_kwargs: FakeLLM(),
    )
    monkeypatch.setattr("src.utils.OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr("src.core.MemorySystem", FakeMemorySystem)

    run_locomo_eval(
        data_path=str(input_path),
        limit_messages=100,
        provider="openai",
        output_file=str(output_path),
    )

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(output["checkpoints"]) == 1
    assert "eval_query_time" in output["checkpoints"][0]

    eval_query_time = datetime.fromisoformat(
        output["checkpoints"][0]["eval_query_time"]
    )
    assert FakeMemorySystem.instances[0].retrieve_calls
    assert FakeMemorySystem.instances[0].retrieve_calls[0] == eval_query_time
    assert eval_query_time.tzinfo == timezone.utc
