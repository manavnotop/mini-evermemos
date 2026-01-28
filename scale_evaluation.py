#!/usr/bin/env python3
"""
Scale evaluation for the memory system.

Tests performance and quality at various message counts:
- 100 messages: Basic extraction, scene formation, simple retrieval
- 200 messages: Conflict detection, deduplication
- 300 messages: Foresight expiry, profile evolution
- 500+ messages: Performance and relevance
"""

import random
import time
from typing import Any, Dict, List

from src.core import MemorySystem


def generate_synthetic_conversation(
    topic: str,
    num_messages: int = 5,
    include_foresight: bool = True,
) -> List[Dict[str, str]]:
    """Generate synthetic conversation on a topic."""
    templates = {
        "career": [
            ("user", "I've been working at {company} as a {role}."),
            ("assistant", "That sounds interesting! What do you do there?"),
            ("user", "I mainly focus on {task}."),
            ("user", "I've been there for {years} years now."),
        ],
        "health": [
            ("user", "I've started {exercise} {frequency}."),
            ("assistant", "How's that going for you?"),
            ("user", "It's been {outcome}! I feel much better."),
            ("user", "I try to {habit} every day."),
        ],
        "hobbies": [
            ("user", "I've picked up {hobby} as a new hobby."),
            ("assistant", "What got you interested in that?"),
            ("user", "I find it {adjective} and relaxing."),
            ("user", "I practice {frequency}."),
        ],
        "food": [
            ("user", "I've been trying {cuisine} food lately."),
            ("assistant", "What's your favorite dish?"),
            ("user", "I really love {dish}!"),
            ("user", "I cook {dish} at home {frequency}."),
        ],
        "travel": [
            ("user", "I'm planning a trip to {destination}."),
            ("assistant", "When are you going?"),
            ("user", "I'll be there {timeframe}."),
            ("user", "I'm excited to {activity} there."),
        ],
    }

    topic_templates = templates.get(topic, templates["career"])
    messages = []

    for role, content_template in topic_templates[:num_messages]:
        content = content_template.format(
            company=random.choice(["Google", "Meta", "Amazon", "Startup", "Microsoft"]),
            role=random.choice(["engineer", "designer", "manager", "analyst"]),
            task=random.choice(
                [
                    "backend systems",
                    "machine learning",
                    "user experience",
                    "data analysis",
                ]
            ),
            years=random.choice(["1", "2", "3", "5"]),
            exercise=random.choice(["running", "yoga", "gym", "swimming"]),
            frequency=random.choice(["every day", "3 times a week", "on weekends"]),
            outcome=random.choice(["great", "amazing", "really good"]),
            habit=random.choice(["stretching", "meditating", "walking"]),
            hobby=random.choice(["guitar", "photography", "painting", "hiking"]),
            adjective=random.choice(["creative", "mindful", "challenging", "fun"]),
            cuisine=random.choice(["Italian", "Japanese", "Mexican", "Indian"]),
            dish=random.choice(["pizza", "sushi", "tacos", "curry"]),
            destination=random.choice(["Paris", "Tokyo", "Mexico", "India"]),
            timeframe=random.choice(["next month", "in spring", "this summer"]),
            activity=random.choice(["try local food", "see the sights", "take photos"]),
        )
        messages.append({"role": role, "content": content})

    return messages


def generate_foresight_conversation(
    topic: str,
    days_until_start: int = 0,
    duration_days: int = 7,
) -> List[Dict[str, str]]:
    """Generate conversation with foresight element."""
    templates = {
        "health": "I'm starting a {treatment} for {condition}. It lasts {duration}.",
        "travel": "I'm going to {destination} for {duration}. I leave in {start}.",
        "work": "I'm starting a new {project} at work. It runs for {duration}.",
    }

    template = templates.get(topic, templates["health"])
    content = template.format(
        treatment=random.choice(["antibiotics", "vitamin regimen", "physical therapy"]),
        condition=random.choice(["infection", "deficiency", "injury"]),
        destination=random.choice(["Paris", "Tokyo", "New York"]),
        project=random.choice(["product launch", "migration", "research"]),
        duration=f"{duration_days} days" if duration_days < 30 else "a month",
        start=f"{days_until_start} days",
    )

    return [{"role": "user", "content": content}]


def run_scale_test(message_count: int) -> Dict[str, Any]:
    """Run a scale test with the given message count."""
    print(f"\n{'=' * 60}")
    print(f"Running scale test with {message_count} messages")
    print(f"{'=' * 60}")

    start_time = time.time()
    system = MemorySystem()

    # Generate messages across topics
    topics = ["career", "health", "hobbies", "food", "travel"]
    messages_per_topic = message_count // len(topics)

    total_messages = 0
    memcells_created = 0
    scenes_created = 0
    conflicts_detected = 0
    original_facts_count = 0
    unique_facts_count = 0

    for topic in topics:
        for i in range(messages_per_topic):
            # Add some foresight conversations occasionally
            if random.random() < 0.1:
                conv = generate_foresight_conversation(topic)
            else:
                conv = generate_synthetic_conversation(topic, num_messages=3)

            result = system.add_conversation(conv)

            total_messages += len(conv)
            memcells_created += 1
            scenes_created += result.get("scene_id", "").startswith("scene_") and 1 or 0
            conflicts_detected += result.get("conflicts_detected", 0)
            original_facts_count += result.get("original_facts_count", 0)
            unique_facts_count += result.get("unique_facts_count", 0)

    # Let system settle
    time.sleep(0.1)

    # Test retrieval
    test_queries = [
        "Where does the user work?",
        "What are the user's health habits?",
        "What hobbies does the user have?",
        "Is the user traveling anywhere?",
    ]

    retrieval_results = []
    for query in test_queries:
        query_start = time.time()
        result = system.retrieve(query)
        query_latency_ms = (time.time() - query_start) * 1000

        # Extract relevance scores from retrieved memcells
        relevance_scores = [m.get("score", 0) for m in result.get("memcells", [])]

        retrieval_results.append(
            {
                "query": query,
                "memcells_retrieved": len(result.get("memcells", [])),
                "retrieval_time_ms": query_latency_ms,
                "relevance_scores": relevance_scores,
                "avg_relevance_score": sum(relevance_scores) / len(relevance_scores)
                if relevance_scores
                else 0,
            }
        )

    retrieval_time = sum(r["retrieval_time_ms"] for r in retrieval_results) / 1000

    # Get final stats
    stats = system.get_memory_stats()

    elapsed_time = time.time() - start_time

    # Calculate deduplication rate
    deduplication_rate = (
        (original_facts_count - unique_facts_count) / original_facts_count
        if original_facts_count > 0
        else 0
    )

    return {
        "message_count": message_count,
        "total_messages_processed": total_messages,
        "memcells_created": stats["memcell_count"],
        "memscenes_created": stats["memscene_count"],
        "conflicts_detected": stats["conflict_count"],
        "original_facts_count": original_facts_count,
        "unique_facts_count": unique_facts_count,
        "deduplication_rate": deduplication_rate,
        "elapsed_time_seconds": elapsed_time,
        "retrieval_time_seconds": retrieval_time,
        "throughput_messages_per_second": total_messages / elapsed_time
        if elapsed_time > 0
        else 0,
        "retrieval_results": retrieval_results,
        "scenes_by_theme": system.get_scenes_by_theme(),
    }


def print_scale_report(report: Dict[str, Any]) -> None:
    """Print a scale test report."""
    print(f"\n{'=' * 60}")
    print(f"SCALE TEST RESULTS: {report['message_count']} messages")
    print(f"{'=' * 60}")

    print("\nMemory Statistics:")
    print(f"  MemCells created: {report['memcells_created']}")
    print(f"  MemScenes created: {report['memscenes_created']}")
    print(f"  Conflicts detected: {report['conflicts_detected']}")

    print("\nDeduplication:")
    print(f"  Original facts: {report['original_facts_count']}")
    print(f"  Unique facts: {report['unique_facts_count']}")
    print(f"  Deduplication rate: {report['deduplication_rate'] * 100:.1f}%")

    print("\nPerformance:")
    print(f"  Total time: {report['elapsed_time_seconds']:.2f}s")
    print(f"  Throughput: {report['throughput_messages_per_second']:.1f} messages/sec")

    print("\nRetrieval Quality:")
    for result in report["retrieval_results"]:
        avg_score = result.get("avg_relevance_score", 0)
        scores = result.get("relevance_scores", [])
        print(
            f"  '{result['query']}': {result['memcells_retrieved']} memcells, "
            f"latency: {result['retrieval_time_ms']:.1f}ms, "
            f"avg relevance: {avg_score:.3f}"
        )
        if scores:
            print(f"    Relevance scores: {[f'{s:.3f}' for s in scores[:5]]}")

    print("\nScene Distribution:")
    for theme, count in sorted(report["scenes_by_theme"].items()):
        print(f"  {theme}: {count} scenes")


def main():
    """Run scale evaluations at multiple message counts."""
    print("\n" + "=" * 60)
    print("  MEMORY SYSTEM SCALE EVALUATION")
    print("=" * 60)

    # Message counts to test
    test_counts = [100, 200, 300, 500]

    reports = []

    for count in test_counts:
        try:
            report = run_scale_test(count)
            reports.append(report)
            print_scale_report(report)
        except Exception as e:
            print(f"Error at {count} messages: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    print(
        f"\n{'Messages':<12} {'MemCells':<10} {'Scenes':<10} {'Conflicts':<12} {'Time (s)':<12}"
    )
    print("-" * 60)
    for report in reports:
        print(
            f"{report['message_count']:<12} "
            f"{report['memcells_created']:<10} "
            f"{report['memscenes_created']:<10} "
            f"{report['conflicts_detected']:<12} "
            f"{report['elapsed_time_seconds']:<12.2f}"
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
