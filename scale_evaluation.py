
"""
Scale evaluation for the memory system using Locomo10 dataset.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.core import MemorySystem
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
    all_sessions = []
    valid_qa = []
    processed_dia_ids = set()
    message_count = 0

    # Locomo structure is a list, but usually just one big object with 'conversation'
    # We iterate properly just in case
    for entry in data:
        conversation = entry.get("conversation", {})
        qa_list = entry.get("qa", [])

        # 1. Extract and sort sessions
        sessions_map = {}
        for key, value in conversation.items():
            if (
                key.startswith("session_")
                and isinstance(value, list)
                and "date_time" not in key
            ):
                # Found a session list (e.g. "session_1")
                # Try to find corresponding date
                date_key = f"{key}_date_time"
                date_str = conversation.get(date_key, "")
                timestamp = parse_session_date(date_str)

                # Extract session number for sorting
                try:
                    sess_num = int(key.split("_")[1])
                except ValueError:
                    sess_num = 9999

                sessions_map[sess_num] = {
                    "messages": value,
                    "timestamp": timestamp,
                    "id": key,
                }

        # Sort by session number
        sorted_keys = sorted(sessions_map.keys())

        for key in sorted_keys:
            if limit_messages is not None and message_count >= limit_messages:
                break

            session_data = sessions_map[key]
            raw_msgs = session_data["messages"]

            # Format for MemorySystem
            formatted_msgs = []
            for msg in raw_msgs:
                if limit_messages is not None and message_count >= limit_messages:
                    break

                formatted_msgs.append(
                    {
                        "role": "user"
                        if msg["speaker"] == "Caroline"
                        else "assistant",
                        "content": msg["text"],
                    }
                )
                processed_dia_ids.add(msg["dia_id"])
                message_count += 1

            if formatted_msgs:
                all_sessions.append(
                    {"messages": formatted_msgs, "timestamp": session_data["timestamp"]}
                )

        # 2. Filter QA pairs based on processed dialogue
        # A QA is valid if ALL its evidence points to processed lines
        if limit_messages is not None:
            for qa in qa_list:
                evidence = qa.get("evidence", [])
                # If evidence matches processed IDs
                # Note: evidence is ["D1:3", "D18:5"]
                # We need to clean them or check basic containment
                # Locomo ID format in evidence matches "dia_id" in text?
                # Yes: "D1:3" matches "dia_id": "D1:3"

                if not evidence:
                    continue  # specific logic for no-evidence? Assume usually needs context.

                is_valid = True
                for ev in evidence:
                    # Handle multiple evidence separated by semicolon if any
                    sub_evs = ev.split(";")
                    for sub in sub_evs:
                        clean_id = sub.strip()
                        if clean_id not in processed_dia_ids:
                            is_valid = False
                            break
                    if not is_valid:
                        break

                if is_valid:
                    valid_qa.append(qa)
        else:
            valid_qa.extend(qa_list)

        if limit_messages is not None and message_count >= limit_messages:
            break

    return all_sessions, valid_qa


def run_locomo_eval(
    data_path: str,
    limit_messages: Optional[int] = None,
    provider: str = "openai",
    output_file: Optional[str] = None,
) -> None:
    """Run evaluation on Locomo dataset with incremental checkpoints."""
    print(f"\n{'=' * 60}")
    print(f"LOCOMO INCREMENTAL EVALUATION")
    print(f"Target Checkpoints: 100, 200, 300, 500+ messages")
    print(f"{'=' * 60}")

    # 1. Initialize System
    llm = get_llm_provider(provider, model="gpt-4o-mini")

    from src.utils import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()

    system = MemorySystem(
        llm_provider=llm, embedding_service=embeddings, storage_dir="./locomo_mem_data"
    )
    system.clear()  # Start fresh

    # 2. Load Data
    data = load_locomo_data(data_path)
    # Load ALL sessions initially, we will control limit manually during ingestion
    all_sessions, all_qa_pairs = extract_locomo_sessions(data, limit_messages=None)

    print(f"Loaded {len(all_sessions)} sessions from {data_path}")

    # 3. Incremental Ingestion & Evaluation
    CHECKPOINTS = [100, 200, 300, 500]

    # If a limit is set, we might not reach all checkpoints, but we still use them.
    # If limit is smaller than a checkpoint, we just stop there.

    current_checkpoint_idx = 0
    total_msgs_ingested = 0

    # Track which dia_ids (message IDs) we have processed to filter valid QAs
    processed_dia_ids = set()

    # Stats accumulators
    total_original_facts = 0
    total_unique_facts = 0
    total_conflicts_detected = 0

    results_by_checkpoint = []

    start_time = time.time()

    # We iterate session by session
    for session_idx, session in enumerate(all_sessions):
        msgs = session["messages"]
        timestamp = session["timestamp"]

        # Ingest this session
        print(f"Ingesting Session {session_idx + 1} ({len(msgs)} msgs)...", end="\r")

        # Update system
        result = system.add_conversation(msgs, timestamp=timestamp)

        # Update stats
        total_original_facts += result.get("original_facts_count", 0)
        total_unique_facts += result.get("unique_facts_count", 0)
        total_conflicts_detected += result.get("conflicts_detected", 0)

        total_msgs_ingested += len(msgs)

        while (
            current_checkpoint_idx < len(CHECKPOINTS)
            and total_msgs_ingested >= CHECKPOINTS[current_checkpoint_idx]
        ):
            checkpoint_target = CHECKPOINTS[current_checkpoint_idx]

            # PERFORM SNAPSHOT EVALUATION
            print(
                f"\n\n>>> REACHED CHECKPOINT: {total_msgs_ingested} messages (Target: {checkpoint_target})"
            )

            # Snapshot Stats
            dedup_rate = 0.0
            if total_original_facts > 0:
                dedup_rate = (1 - (total_unique_facts / total_original_facts)) * 100

            sys_stats = system.get_memory_stats()

            # Snapshot Retrieval Performance
            print("Running Snapshot Retrieval Evaluation...")
            checkpoint_latencies = []
            checkpoint_correct = 0
            checkpoint_eval_count = 0
            checkpoint_qa_results = []

            _, valid_qa_pairs = extract_locomo_sessions(
                data, limit_messages=total_msgs_ingested
            )

            print(f"Evaluating on {len(valid_qa_pairs)} valid QA pairs...")

            for i, qa in enumerate(valid_qa_pairs):
                print(f"Evaluating QA {i + 1}/{len(valid_qa_pairs)}...", end="\r")
                q_text = qa["question"]
                gold = str(qa.get("answer", "N/A"))

                t0 = time.time()
                retrieval = system.retrieve(q_text)
                t1 = time.time()
                lat_ms = (t1 - t0) * 1000
                checkpoint_latencies.append(lat_ms)

                context = retrieval.get("composed_context", "")

                # LLM Verify
                verify_prompt = f"""Question: {q_text}
Golden Answer: {gold}
Retrieved Context: {context}
Does the retrieved context contain the information necessary to answer the question matching the golden answer? 
Answer JSON: {{"match": true/false}}"""
                try:
                    check = llm.complete_json(
                        [{"role": "user", "content": verify_prompt}]
                    )
                    is_match = check.get("match", False)
                except:
                    is_match = False

                if is_match:
                    checkpoint_correct += 1
                checkpoint_eval_count += 1

                checkpoint_qa_results.append(
                    {"question": q_text, "match": is_match, "latency_ms": lat_ms}
                )
            print()  # Clear progress line

            # Checkpoint Metrics
            import statistics

            avg_lat = (
                statistics.mean(checkpoint_latencies) if checkpoint_latencies else 0
            )
            p90_lat = (
                statistics.quantiles(checkpoint_latencies, n=10)[8]
                if len(checkpoint_latencies) >= 2
                else avg_lat
            )
            accuracy = (
                (checkpoint_correct / checkpoint_eval_count * 100)
                if checkpoint_eval_count
                else 0
            )

            snapshot = {
                "message_count": total_msgs_ingested,
                "timestamp": now_utc().isoformat(),
                "metrics": {
                    "memcells": sys_stats.get("memcell_count", 0),
                    "scenes": sys_stats.get("memscene_count", 0),
                    "conflicts": sys_stats.get("conflict_count", 0),
                    "dedup_rate": dedup_rate,
                    "retrieval_accuracy": accuracy,
                    "valid_qa_count": checkpoint_eval_count,
                    "latency_p50": avg_lat,
                    "latency_p90": p90_lat,
                    "latency_avg": avg_lat,
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
                f"Accuracy    : {accuracy:.1f}% ({checkpoint_correct}/{checkpoint_eval_count})"
            )
            print(f"Latency     : Avg={avg_lat:.0f}ms, P90={p90_lat:.0f}ms")
            print("-" * 40)

            current_checkpoint_idx += 1

        # Global limit break
        if limit_messages and total_msgs_ingested >= limit_messages:
            break

    print(f"\n\nEvaluation Complete. Processed {total_msgs_ingested} messages.")

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
    args = parser.parse_args()

    if args.locomo:
        run_locomo_eval(args.path, args.limit, args.provider, args.output)
    else:
        # Legacy/Synthetic mode (kept for compatibility or default run)
        print("Please use --locomo flag to run the Locomo evaluation.")
