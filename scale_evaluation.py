"""
Scale evaluation for the memory system using Locomo10 dataset.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from statistics import mean, median, quantiles
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils import get_llm_provider, now_utc


def load_locomo_data(path: str) -> List[Dict[str, Any]]:
    """Load Locomo dataset."""
    with open(path, "r") as f:
        return json.load(f)


def parse_session_date(date_str: str) -> datetime:
    """Parse Locomo session date string (e.g. '1:56 pm on 8 May, 2023')."""
    try:
        # Normalize: '1:56 pm on 8 May, 2023' -> '8 May 2023 1:56 pm'
        parts = date_str.split(" on ")
        if len(parts) == 2:
            time_part = parts[0]
            date_part = parts[1]
            full_str = f"{date_part} {time_part}"
            dt = datetime.strptime(full_str, "%d %B, %Y %I:%M %p")
            return dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return now_utc()


def _evidence_key(dialogue_idx: int, dia_id: str) -> str:
    """Create a dialogue-scoped evidence key to avoid cross-dialogue collisions."""
    return f"{dialogue_idx}:{dia_id}"


def _parse_evidence_tokens(evidence: List[str]) -> List[str]:
    """Parse QA evidence entries into clean dia_id tokens."""
    tokens: List[str] = []
    for item in evidence:
        for token in str(item).split(";"):
            cleaned = token.strip()
            if cleaned:
                tokens.append(cleaned)
    return tokens


def _infer_user_speaker(
    sorted_session_nums: List[int],
    sessions_map: Dict[int, Dict[str, Any]],
) -> Optional[str]:
    """Infer the user speaker as the first speaker in the earliest non-empty session."""
    for sess_num in sorted_session_nums:
        messages = sessions_map[sess_num].get("messages", [])
        if messages:
            return messages[0].get("speaker")
    return None


def _compute_latency_metrics(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute latency aggregates for reporting."""
    if not latencies_ms:
        return {"latency_p50": 0.0, "latency_p90": 0.0, "latency_avg": 0.0}

    avg_latency = mean(latencies_ms)
    p50_latency = median(latencies_ms)
    p90_latency = (
        quantiles(latencies_ms, n=10)[8]
        if len(latencies_ms) >= 2
        else avg_latency
    )
    return {
        "latency_p50": p50_latency,
        "latency_p90": p90_latency,
        "latency_avg": avg_latency,
    }


def extract_locomo_sessions_with_stats(
    data: List[Dict[str, Any]],
    limit_messages: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Extract chronological sessions and relevant QA pairs, plus extraction stats.

    Returns:
        (sessions, qa_pairs, stats)
        sessions: List of {"messages": [...], "timestamp": datetime, ...metadata}
        qa_pairs: List of QA dicts valid for the extracted range
        stats: {"skipped_no_evidence_qa_count": int}
    """
    all_sessions: List[Dict[str, Any]] = []
    valid_qa: List[Dict[str, Any]] = []
    processed_evidence_keys: Set[str] = set()
    message_count = 0
    skipped_no_evidence_qa_count = 0

    for dialogue_idx, entry in enumerate(data):
        conversation = entry.get("conversation", {})
        qa_list = entry.get("qa", [])
        dialogue_id = f"locomo_{dialogue_idx}"

        # 1. Extract and sort sessions
        sessions_map: Dict[int, Dict[str, Any]] = {}
        for key, value in conversation.items():
            if (
                key.startswith("session_")
                and isinstance(value, list)
                and "date_time" not in key
            ):
                date_key = f"{key}_date_time"
                date_str = conversation.get(date_key, "")
                timestamp = parse_session_date(date_str)

                try:
                    sess_num = int(key.split("_")[1])
                except ValueError:
                    sess_num = 9999

                sessions_map[sess_num] = {
                    "messages": value,
                    "timestamp": timestamp,
                    "id": key,
                }

        sorted_keys = sorted(sessions_map.keys())
        user_speaker = _infer_user_speaker(sorted_keys, sessions_map)

        for sess_num in sorted_keys:
            if limit_messages is not None and message_count >= limit_messages:
                break

            session_data = sessions_map[sess_num]
            raw_msgs = session_data["messages"]

            formatted_msgs = []
            for msg in raw_msgs:
                if limit_messages is not None and message_count >= limit_messages:
                    break

                speaker = msg.get("speaker")
                role = "user" if speaker == user_speaker else "assistant"

                formatted_msgs.append(
                    {
                        "role": role,
                        "content": msg.get("text", ""),
                    }
                )

                dia_id = msg.get("dia_id")
                if dia_id:
                    processed_evidence_keys.add(_evidence_key(dialogue_idx, dia_id))
                message_count += 1

            if formatted_msgs:
                all_sessions.append(
                    {
                        "messages": formatted_msgs,
                        "timestamp": session_data["timestamp"],
                        "dialogue_idx": dialogue_idx,
                        "dialogue_id": dialogue_id,
                        "session_num": sess_num,
                        "user_speaker": user_speaker,
                    }
                )

        # 2. Filter QA pairs based on processed dialogue evidence
        if limit_messages is not None:
            for qa in qa_list:
                evidence_tokens = _parse_evidence_tokens(qa.get("evidence", []))

                if not evidence_tokens:
                    skipped_no_evidence_qa_count += 1
                    continue

                is_valid = True
                for dia_id in evidence_tokens:
                    namespaced_key = _evidence_key(dialogue_idx, dia_id)
                    if namespaced_key not in processed_evidence_keys:
                        is_valid = False
                        break

                if is_valid:
                    qa_with_meta = dict(qa)
                    qa_with_meta["dialogue_idx"] = dialogue_idx
                    qa_with_meta["dialogue_id"] = dialogue_id
                    valid_qa.append(qa_with_meta)
        else:
            for qa in qa_list:
                qa_with_meta = dict(qa)
                qa_with_meta["dialogue_idx"] = dialogue_idx
                qa_with_meta["dialogue_id"] = dialogue_id
                valid_qa.append(qa_with_meta)

        if limit_messages is not None and message_count >= limit_messages:
            break

    stats = {
        "skipped_no_evidence_qa_count": skipped_no_evidence_qa_count,
    }
    return all_sessions, valid_qa, stats


def extract_locomo_sessions(
    data: List[Dict[str, Any]], limit_messages: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract chronological sessions and relevant QA pairs.

    Returns:
        (sessions, qa_pairs)
        sessions: List of {"messages": [...], "timestamp": datetime}
        qa_pairs: List of QA dicts valid for the extracted range
    """
    all_sessions, valid_qa, _ = extract_locomo_sessions_with_stats(data, limit_messages)
    return all_sessions, valid_qa


def run_locomo_eval(
    data_path: str,
    limit_messages: Optional[int] = None,
    provider: str = "openai",
    output_file: Optional[str] = None,
    fast_ingest: bool = False,
    fast_retrieval: bool = False,
    qa_max: Optional[int] = None,
    progress_every: int = 10,
) -> None:
    """Run evaluation on Locomo dataset with incremental checkpoints."""
    from src.core import MemorySystem
    from src.utils import OpenAIEmbeddings

    print(f"\n{'=' * 60}")
    print("LOCOMO INCREMENTAL EVALUATION")
    print("Target Checkpoints: 100, 200, 300, 500+ messages")
    print(f"{'=' * 60}")

    # 1. Initialize System
    llm = get_llm_provider(provider, model="gpt-4o-mini")
    llm_call_stats = {"complete": 0, "complete_json": 0}

    # Wrap LLM methods to expose per-stage call counts in logs.
    original_complete = getattr(llm, "complete", None)
    original_complete_json = getattr(llm, "complete_json", None)

    if original_complete is not None:
        def counted_complete(*args, **kwargs):
            llm_call_stats["complete"] += 1
            return original_complete(*args, **kwargs)

        llm.complete = counted_complete

    if original_complete_json is not None:
        def counted_complete_json(*args, **kwargs):
            llm_call_stats["complete_json"] += 1
            return original_complete_json(*args, **kwargs)

        llm.complete_json = counted_complete_json

    embeddings = OpenAIEmbeddings()

    # Initialize with LoCoMo-optimized hyperparameters
    # Based on EverMemOS paper Table 6 and empirical tuning
    system = MemorySystem(
        llm_provider=llm,
        embedding_service=embeddings,
        storage_dir="./locomo_mem_data",
        # More permissive than default 0.70 for better scene formation.
        similarity_threshold=0.60,
        max_time_gap_days=30,  # LoCoMo spans months, not days
    )
    system.clear()  # Start fresh
    if fast_ingest:
        # Disable per-message boundary LLM calls; each session becomes one episode.
        system.extractor.detect_boundary = lambda messages, force_split=False: (
            False,
            "fast_ingest_disabled_boundary_detection",
        )
        system.extractor.max_messages_per_episode = 10_000
        print("[CONFIG] fast_ingest enabled: boundary detection calls disabled.")

    if fast_retrieval:
        # Disable LLM sufficiency/rewrite loop in retrieval; keep judge call in eval.
        system.retriever.max_retrieval_rounds = 1
        system.retriever._check_sufficiency = lambda query, context: (
            True,
            "fast_retrieval_disabled_sufficiency",
            [],
        )
        print("[CONFIG] fast_retrieval enabled: retrieval sufficiency calls disabled.")

    # 2. Load Data
    data = load_locomo_data(data_path)
    # Load ALL sessions initially, we will control limit manually during ingestion
    all_sessions, _ = extract_locomo_sessions(data, limit_messages=None)

    print(f"Loaded {len(all_sessions)} sessions from {data_path}")
    if limit_messages is not None and limit_messages >= 100:
        print(
            "[INFO] limit >= 100 will trigger checkpoint QA evaluation, "
            "which can add many LLM calls."
        )

    # 3. Incremental Ingestion & Evaluation
    CHECKPOINTS = [100, 200, 300, 500]

    # If a limit is set, we might not reach all checkpoints, but we still use them.
    # If limit is smaller than a checkpoint, we just stop there.

    current_checkpoint_idx = 0
    total_msgs_ingested = 0

    # Stats accumulators
    total_original_facts = 0
    total_unique_facts = 0

    results_by_checkpoint = []
    ingested_timestamps: List[datetime] = []

    # We iterate session by session
    run_start = time.time()
    for session_idx, session in enumerate(all_sessions):
        msgs = session["messages"]
        timestamp = session["timestamp"]
        dialogue_id = session.get("dialogue_id", "unknown")
        session_num = session.get("session_num", -1)

        # Ingest this session
        print(
            f"[INGEST] Session {session_idx + 1}/{len(all_sessions)} "
            f"({dialogue_id}, session_{session_num}, {len(msgs)} msgs) ..."
        )
        ingest_t0 = time.time()
        ingest_calls_before = dict(llm_call_stats)

        # Update system
        result = system.add_conversation(msgs, timestamp=timestamp)
        ingest_ms = (time.time() - ingest_t0) * 1000
        ingest_complete_calls = (
            llm_call_stats["complete"] - ingest_calls_before["complete"]
        )
        ingest_complete_json_calls = (
            llm_call_stats["complete_json"] - ingest_calls_before["complete_json"]
        )
        print(
            f"[INGEST] done in {ingest_ms:.0f}ms | "
            f"memcells={result.get('memcell_count', 0)} "
            f"conflicts={result.get('conflicts_detected', 0)} "
            f"dedup={result.get('dedup_rate', 0):.1f}% | "
            f"llm_calls={ingest_complete_calls + ingest_complete_json_calls} "
            f"(text={ingest_complete_calls}, json={ingest_complete_json_calls})"
        )

        # Update stats
        total_original_facts += result.get("original_facts_count", 0)
        total_unique_facts += result.get("unique_facts_count", 0)
        total_msgs_ingested += len(msgs)
        ingested_timestamps.append(timestamp)

        while (
            current_checkpoint_idx < len(CHECKPOINTS)
            and total_msgs_ingested >= CHECKPOINTS[current_checkpoint_idx]
        ):
            checkpoint_target = CHECKPOINTS[current_checkpoint_idx]

            # PERFORM SNAPSHOT EVALUATION
            print(
                f"\n\n>>> REACHED CHECKPOINT: {total_msgs_ingested} messages "
                f"(Target: {checkpoint_target})"
            )

            # Snapshot Stats
            dedup_rate = 0.0
            if total_original_facts > 0:
                dedup_rate = (1 - (total_unique_facts / total_original_facts)) * 100

            sys_stats = system.get_memory_stats()

            # Snapshot Retrieval Performance
            print("Running Snapshot Retrieval Evaluation...")
            checkpoint_latencies: List[float] = []
            checkpoint_correct = 0
            checkpoint_eval_count = 0
            checkpoint_qa_results = []
            checkpoint_eval_time = (
                max(ingested_timestamps) if ingested_timestamps else now_utc()
            )

            _, valid_qa_pairs, extraction_stats = extract_locomo_sessions_with_stats(
                data, limit_messages=total_msgs_ingested
            )

            if qa_max is not None and qa_max >= 0:
                valid_qa_pairs = valid_qa_pairs[:qa_max]

            print(
                f"Evaluating on {len(valid_qa_pairs)} valid QA pairs "
                f"(qa_max={qa_max if qa_max is not None else 'all'})..."
            )
            if extraction_stats["skipped_no_evidence_qa_count"] > 0:
                print(
                    "Skipped "
                    f"{extraction_stats['skipped_no_evidence_qa_count']} "
                    "QA items with no evidence."
                )
            if fast_retrieval:
                print("[CHECKPOINT] Estimated LLM calls per QA: ~1 (judge only)")
            else:
                print(
                    "[CHECKPOINT] Estimated LLM calls per QA: "
                    "~2-4 (retrieve + judge)"
                )

            qa_eval_t0 = time.time()
            qa_calls_before = dict(llm_call_stats)
            for i, qa in enumerate(valid_qa_pairs):
                if (
                    i == 0
                    or (progress_every > 0 and (i + 1) % progress_every == 0)
                    or i == len(valid_qa_pairs) - 1
                ):
                    elapsed = time.time() - qa_eval_t0
                    done = i + 1
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = max(len(valid_qa_pairs) - done, 0)
                    eta_s = remaining / rate if rate > 0 else 0
                    print(
                        f"[CHECKPOINT] QA {done}/{len(valid_qa_pairs)} "
                        f"| elapsed={elapsed:.1f}s "
                        f"| eta~{eta_s:.1f}s"
                    )
                q_text = qa["question"]
                gold = str(qa.get("answer", "N/A"))

                t0 = time.time()
                retrieval = system.retrieve(q_text, query_time=checkpoint_eval_time)
                t1 = time.time()
                lat_ms = (t1 - t0) * 1000
                checkpoint_latencies.append(lat_ms)

                context = retrieval.get("composed_context", "")

                # LLM Verify
                verify_prompt = f"""Question: {q_text}
Golden Answer: {gold}
Retrieved Context: {context}
Does the retrieved context contain the information necessary
to answer the question matching the golden answer?
Answer JSON: {{"match": true/false}}"""
                try:
                    check = llm.complete_json(
                        [{"role": "user", "content": verify_prompt}]
                    )
                    is_match = check.get("match", False)
                except Exception:
                    is_match = False

                if is_match:
                    checkpoint_correct += 1
                checkpoint_eval_count += 1

                checkpoint_qa_results.append(
                    {"question": q_text, "match": is_match, "latency_ms": lat_ms}
                )
            print(
                f"[CHECKPOINT] QA evaluation complete in "
                f"{(time.time() - qa_eval_t0):.1f}s"
            )
            qa_complete_calls = llm_call_stats["complete"] - qa_calls_before["complete"]
            qa_complete_json_calls = (
                llm_call_stats["complete_json"] - qa_calls_before["complete_json"]
            )
            print(
                "[CHECKPOINT] LLM calls in QA stage: "
                f"{qa_complete_calls + qa_complete_json_calls} "
                f"(text={qa_complete_calls}, json={qa_complete_json_calls})"
            )

            # Checkpoint Metrics
            latency_metrics = _compute_latency_metrics(checkpoint_latencies)
            accuracy = (
                (checkpoint_correct / checkpoint_eval_count * 100)
                if checkpoint_eval_count
                else 0
            )

            snapshot = {
                "message_count": total_msgs_ingested,
                "timestamp": now_utc().isoformat(),
                "eval_query_time": checkpoint_eval_time.isoformat(),
                "metrics": {
                    "memcells": sys_stats.get("memcell_count", 0),
                    "scenes": sys_stats.get("memscene_count", 0),
                    "conflicts": sys_stats.get("conflict_count", 0),
                    "dedup_rate": dedup_rate,
                    "retrieval_accuracy": accuracy,
                    "valid_qa_count": checkpoint_eval_count,
                    "latency_p50": latency_metrics["latency_p50"],
                    "latency_p90": latency_metrics["latency_p90"],
                    "latency_avg": latency_metrics["latency_avg"],
                    "skipped_no_evidence_qa_count": extraction_stats[
                        "skipped_no_evidence_qa_count"
                    ],
                },
                "qa_results": checkpoint_qa_results,
            }
            results_by_checkpoint.append(snapshot)

            # Print Summary Table for this Checkpoint
            print("-" * 40)
            print(f"SNAPSHOT @ {total_msgs_ingested} MSGS")
            print("-" * 40)
            print(f"MemCells    : {snapshot['metrics']['memcells']}")
            print(f"Scenes      : {snapshot['metrics']['scenes']}")
            print(f"Conflicts   : {snapshot['metrics']['conflicts']}")
            print(f"Dedup Rate  : {dedup_rate:.1f}%")
            print(
                "Accuracy    : "
                f"{accuracy:.1f}% ({checkpoint_correct}/{checkpoint_eval_count})"
            )
            print(
                "Latency     : "
                f"P50={latency_metrics['latency_p50']:.0f}ms, "
                f"Avg={latency_metrics['latency_avg']:.0f}ms, "
                f"P90={latency_metrics['latency_p90']:.0f}ms"
            )
            print("-" * 40)

            current_checkpoint_idx += 1

        # Global limit break
        if limit_messages and total_msgs_ingested >= limit_messages:
            break

    print(f"\n\nEvaluation Complete. Processed {total_msgs_ingested} messages.")
    print(f"Total runtime: {(time.time() - run_start):.1f}s")
    print(
        "[TOTAL] LLM calls: "
        f"{llm_call_stats['complete'] + llm_call_stats['complete_json']} "
        f"(text={llm_call_stats['complete']}, json={llm_call_stats['complete_json']})"
    )

    # Save Results
    final_output_file = output_file or f"scale_eval_results_{int(time.time())}.json"

    output_data = {
        "config": {
            "dataset": data_path,
            "provider": provider,
            "model": "gpt-4o-mini",
            "checkpoints_target": CHECKPOINTS,
        },
        "checkpoints": results_by_checkpoint,
    }

    with open(final_output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Full results saved to {final_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--locomo", action="store_true", help="Run Locomo evaluation")
    parser.add_argument(
        "--path", default="data/locomo10.json", help="Path to Locomo data"
    )
    parser.add_argument("--limit", type=int, help="Limit number of messages")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--output", help="Path to save results JSON")
    parser.add_argument(
        "--fast-ingest",
        action="store_true",
        help="Disable boundary-detection LLM calls during ingestion.",
    )
    parser.add_argument(
        "--fast-retrieval",
        action="store_true",
        help="Disable retrieval sufficiency/rewrite LLM calls (judge still runs).",
    )
    parser.add_argument(
        "--qa-max",
        type=int,
        help="Max QA items evaluated per checkpoint (default: all).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print QA progress every N items.",
    )
    args = parser.parse_args()

    if args.locomo:
        run_locomo_eval(
            args.path,
            args.limit,
            args.provider,
            args.output,
            fast_ingest=args.fast_ingest,
            fast_retrieval=args.fast_retrieval,
            qa_max=args.qa_max,
            progress_every=args.progress_every,
        )
    else:
        # Legacy/Synthetic mode (kept for compatibility or default run)
        print("Please use --locomo flag to run the Locomo evaluation.")
