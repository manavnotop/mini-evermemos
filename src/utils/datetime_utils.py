"""Timezone-aware datetime utilities for the memory system."""

from datetime import datetime, timedelta, timezone
from typing import Optional

from dateutil import parser as date_parser


def now_utc() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


def parse_datetime(dt_str: str) -> datetime:
    """Parse a datetime string to a timezone-aware datetime."""
    if not dt_str:
        return now_utc()

    try:
        parsed = date_parser.parse(dt_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (ValueError, TypeError):
        return now_utc()


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a datetime to a string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime(format_str)


def days_between(start: datetime, end: datetime) -> int:
    """Calculate the number of days between two datetimes."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    delta = end - start
    return delta.days


def is_within_interval(
    query_time: datetime, start: Optional[datetime], end: Optional[datetime]
) -> bool:
    """
    Check if query_time falls within the [start, end] interval.

    Args:
        query_time: The time to check
        start: Start of interval (inclusive, can be None for no start limit)
        end: End of interval (inclusive, can be None for no end limit)

    Returns:
        True if query_time is within the interval
    """
    if start and query_time < start:
        return False
    if end and query_time > end:
        return False
    return True


# Default foresight validity duration when not explicitly specified
def default_foresight_expiry_days() -> int:
    """Return the default number of days for foresight expiry."""
    return 30  # 30 days as per PRD


def infer_validity_duration(description: str, context: str = "") -> timedelta:
    """
    Infer the validity duration for a foresight item based on its description.

    This uses heuristics to determine appropriate expiry times:
    - "today", "tomorrow" -> 1-2 days
    - "this week", "next week" -> 7-14 days
    - "this month", "next month" -> 30 days
    - "until [date]" -> parse the date
    - Default -> 30 days

    Args:
        description: The foresight description
        context: Additional context that might help infer duration

    Returns:
        timedelta representing the validity duration
    """
    text = (description + " " + context).lower()

    # Immediate/short-term
    if any(word in text for word in ["today", "tonight", "right now"]):
        return timedelta(days=1)
    if any(word in text for word in ["tomorrow", "in the morning"]):
        return timedelta(days=2)
    if any(word in text for word in ["this week", "next week"]):
        return timedelta(days=7)

    # Medium-term
    if any(word in text for word in ["this month", "next month"]):
        return timedelta(days=30)
    if any(word in text for word in ["until friday", "until monday", "until weekend"]):
        return timedelta(days=7)
    if any(word in text for word in ["until tomorrow"]):
        return timedelta(days=2)

    # Treatment/medication patterns
    if any(
        word in text for word in ["antibiotics", "medication", "medicine", "treatment"]
    ):
        if "week" in text:
            return timedelta(
                days=7 * int(text.split("week")[0].split()[-1])
                if any(c.isdigit() for c in text)
                else 14
            )
        if "day" in text:
            return timedelta(
                days=int(text.split("day")[0].split()[-1])
                if any(c.isdigit() for c in text)
                else 7
            )
        return timedelta(days=14)  # Typical antibiotic course

    # Travel patterns
    if any(word in text for word in ["travel", "trip", "vacation", "holiday"]):
        if "week" in text:
            return timedelta(
                days=7 * int(text.split("week")[0].split()[-1])
                if any(c.isdigit() for c in text)
                else 7
            )
        if "month" in text:
            return timedelta(days=30)
        return timedelta(days=14)

    # Event-based
    if any(word in text for word in ["appointment", "meeting", "conference", "event"]):
        return timedelta(days=1)

    # Default: 30 days
    return timedelta(days=default_foresight_expiry_days())
