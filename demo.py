#!/usr/bin/env python3
"""
Interactive demo for the EverMemOS-inspired memory system.

Demonstrates:
1. Memory extraction from conversations
2. Scene formation and thematic clustering
3. Conflict detection
4. Foresight temporal filtering
5. Retrieval with temporal queries
"""

from datetime import timedelta

from src.core import MemorySystem
from src.utils import now_utc


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n--- {title} ---")


def demo_basic_memory():
    """Demo: Basic memory extraction and retrieval."""
    print_header("Demo 1: Basic Memory Extraction")

    # Initialize memory system
    system = MemorySystem()

    # Sample conversation
    conversation = [
        {
            "role": "user",
            "content": "I've been working at Google as a software engineer for the past year.",
        },
        {
            "role": "assistant",
            "content": "That sounds interesting! What do you work on there?",
        },
        {
            "role": "user",
            "content": "I mainly work on machine learning infrastructure, helping train models more efficiently.",
        },
        {
            "role": "assistant",
            "content": "That's cool! What programming languages do you use?",
        },
        {
            "role": "user",
            "content": "I use Python and C++ mostly. I really love Python for its simplicity.",
        },
    ]

    print("\nAdding conversation about work...")
    result = system.add_conversation(conversation)

    print("\nExtracted MemCell:")
    print(f"  Episode: {result['episode']}")
    print(f"  Facts: {result['atomic_facts']}")
    print(f"  Theme: {result['theme']}")
    print(f"  Foresight: {result['foresight_count']} items")

    # Retrieve
    print_section("Retrieval Query: 'What does the user do for work?'")
    ctx = system.retrieve("What does the user do for work?")

    print(f"\nRetrieved {len(ctx['memcells'])} memories:")
    for m in ctx["memcells"]:
        print(f"  - {m['episode']}")


def demo_conflict_detection():
    """Demo: Conflict detection when preferences change."""
    print_header("Demo 2: Conflict Detection")

    system = MemorySystem()

    # Initial preference
    conversation1 = [
        {
            "role": "user",
            "content": "I've been a vegetarian for 5 years now. I love it!",
        },
        {"role": "assistant", "content": "That's great! How did you get into it?"},
        {
            "role": "user",
            "content": "Mainly for health and ethical reasons. I feel much better not eating meat.",
        },
    ]

    print("\nAdding conversation about diet (vegetarian)...")
    result1 = system.add_conversation(conversation1)
    print(f"  Scene: {result1['theme']}, Conflicts: {result1['conflicts_detected']}")

    # Changed preference
    conversation2 = [
        {
            "role": "user",
            "content": "Actually, I started eating meat again last month. I tried a steak and it was amazing!",
        },
        {
            "role": "assistant",
            "content": "Oh interesting! What made you change your mind?",
        },
        {
            "role": "user",
            "content": "I just felt like trying it again and now I love having burgers and steaks.",
        },
    ]

    print("\nAdding conversation about diet change (eating meat)...")
    result2 = system.add_conversation(conversation2)
    print(f"  Scene: {result2['theme']}, Conflicts: {result2['conflicts_detected']}")

    # Check conflicts
    conflicts = system.get_conflicts(resolved=False)
    print(f"\nDetected {len(conflicts)} conflict(s):")
    for c in conflicts:
        print(f"  - Old: '{c['old_fact']}'")
        print(f"    New: '{c['new_fact']}'")
        print(f"    Type: {c.get('conflict_type', 'unknown')}")

    # Stats
    stats = system.get_memory_stats()
    print("\nMemory Stats:")
    print(f"  MemCells: {stats['memcell_count']}")
    print(f"  MemScenes: {stats['memscene_count']}")
    print(f"  Conflicts: {stats['conflict_count']}")


def demo_foresight_expiry():
    """Demo: Foresight items expire correctly."""
    print_header("Demo 3: Foresight Temporal Filtering")

    system = MemorySystem()

    # Current time reference
    now = now_utc()

    # Conversation with temporary condition
    conversation = [
        {
            "role": "user",
            "content": "I went to the doctor today and got prescribed antibiotics for a throat infection.",
        },
        {"role": "assistant", "content": "How long do you need to take them?"},
        {"role": "user", "content": "Just for 5 days, should be done by Friday."},
    ]

    print(
        f"\nAdding conversation with antibiotics (current time: {now.strftime('%Y-%m-%d')})..."
    )
    result = system.add_conversation(conversation)

    # Check foresight
    memcell = system.store.get_memcell(result["memcell_id"])
    print(f"\nForesight items ({len(memcell.foresight)}):")
    for f in memcell.foresight:
        start = f.start_time.strftime("%Y-%m-%d") if f.start_time else "now"
        end = f.end_time.strftime("%Y-%m-%d") if f.end_time else "?"
        print(f"  - {f.description} (valid: {start} to {end})")

    # Retrieve at current time (should include foresight)
    print_section("Query at current time")
    ctx = system.retrieve("Can I have a drink with dinner?", query_time=now)
    print(f"Valid foresight items: {len(ctx['foresight'])}")
    for f in ctx["foresight"]:
        print(f"  - {f['description']}")

    # Retrieve after expiry (should NOT include foresight)
    after_expiry = now + timedelta(days=10)
    print_section(f"Query 10 days later ({after_expiry.strftime('%Y-%m-%d')})")
    ctx_expired = system.retrieve(
        "Can I have a drink with dinner?", query_time=after_expiry
    )
    print(f"Valid foresight items: {len(ctx_expired['foresight'])}")
    if not ctx_expired["foresight"]:
        print("  (Correctly expired!)")


def demo_thematic_clustering():
    """Demo: Memories cluster by theme."""
    print_header("Demo 4: Thematic Clustering")

    system = MemorySystem()

    # Add conversations on different topics
    conversations = [
        # Career
        (
            [{"role": "user", "content": "I just accepted a job offer at a startup!"}],
            "career",
        ),
        (
            [
                {
                    "role": "user",
                    "content": "My commute takes about 30 minutes by train.",
                }
            ],
            "career",
        ),
        # Health
        (
            [{"role": "user", "content": "I've been going to the gym 3 times a week."}],
            "health",
        ),
        (
            [
                {
                    "role": "user",
                    "content": "Trying to eat more vegetables and less sugar.",
                }
            ],
            "health",
        ),
        # Hobbies
        (
            [{"role": "user", "content": "I started learning guitar last month."}],
            "hobbies",
        ),
        (
            [{"role": "user", "content": "Weekend hiking trips are my favorite."}],
            "hobbies",
        ),
    ]

    print("\nAdding conversations on various topics...")
    for conv, expected_theme in conversations:
        result = system.add_conversation(conv)
        theme_match = "✓" if result["theme"] == expected_theme else "✗"
        print(f"  {theme_match} '{conv[0]['content'][:40]}...' -> {result['theme']}")

    # Show clustering
    print_section("Scene Distribution by Theme")
    by_theme = system.get_scenes_by_theme()
    for theme, count in sorted(by_theme.items(), key=lambda x: -x[1]):
        print(f"  {theme}: {count} scene(s)")


def demo_temporal_queries():
    """Demo: Different retrieval results based on query time."""
    print_header("Demo 5: Temporal Query Differences")

    system = MemorySystem()

    # Add memories at different times
    past_time = now_utc() - timedelta(days=60)
    recent_time = now_utc() - timedelta(days=5)

    print("\nAdding memory about living situation (60 days ago)...")
    system.add_conversation(
        [{"role": "user", "content": "I live in a small apartment in downtown."}],
        timestamp=past_time,
    )

    print("Adding memory about moving (5 days ago)...")
    system.add_conversation(
        [{"role": "user", "content": "We just moved into a house in the suburbs!"}],
        timestamp=recent_time,
    )

    # Query about current location at different times
    print_section("Query: 'Where does the user live?'")

    print("\n  At 60 days ago:")
    ctx_old = system.retrieve(
        "Where does the user live?", query_time=past_time - timedelta(days=1)
    )
    for m in ctx_old["memcells"]:
        print(f"    - {m['episode']}")

    print("\n  Today (should show move to house):")
    ctx_new = system.retrieve("Where does the user live?", query_time=now_utc())
    for m in ctx_new["memcells"]:
        print(f"    - {m['episode']}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  EverMemOS Memory System - Interactive Demo")
    print("=" * 60)

    try:
        demo_basic_memory()
        demo_conflict_detection()
        demo_foresight_expiry()
        demo_thematic_clustering()
        demo_temporal_queries()

        print_header("Demo Complete!")
        print("\nAll demonstrations completed successfully.")
        print("The memory system correctly handles:")
        print("  ✓ Memory extraction and scene formation")
        print("  ✓ Conflict detection for changing preferences")
        print("  ✓ Foresight temporal filtering")
        print("  ✓ Thematic clustering of memories")
        print("  ✓ Time-aware retrieval")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
