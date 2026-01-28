"""Tests for foresight temporal filtering."""

from datetime import datetime, timedelta, timezone

import pytest

from src.core import MemorySystem
from src.models import ForesightItem
from src.utils import is_within_interval, parse_datetime


class TestForesightValidity:
    """Test ForesightItem validity checking."""

    def test_valid_foresight(self):
        """Test foresight within validity period."""
        now = datetime.now(timezone.utc)

        foresight = ForesightItem(
            description="On antibiotics",
            start_time=now - timedelta(days=2),
            end_time=now + timedelta(days=3),
        )

        assert foresight.is_valid_at(now) is True

    def test_expired_foresight(self):
        """Test expired foresight."""
        now = datetime.now(timezone.utc)

        foresight = ForesightItem(
            description="On antibiotics",
            start_time=now - timedelta(days=10),
            end_time=now - timedelta(days=5),
        )

        assert foresight.is_valid_at(now) is False

    def test_future_foresight(self):
        """Test not-yet-started foresight."""
        now = datetime.now(timezone.utc)

        foresight = ForesightItem(
            description="Traveling next week",
            start_time=now + timedelta(days=7),
            end_time=now + timedelta(days=14),
        )

        assert foresight.is_valid_at(now) is False

    def test_no_end_time(self):
        """Test foresight with no end time."""
        now = datetime.now(timezone.utc)

        foresight = ForesightItem(
            description="Permanent preference",
            start_time=now - timedelta(days=1),
            end_time=None,
        )

        assert foresight.is_valid_at(now) is True

    def test_no_start_time(self):
        """Test foresight with no start time."""
        now = datetime.now(timezone.utc)

        foresight = ForesightItem(
            description="Some condition",
            start_time=None,
            end_time=now + timedelta(days=1),
        )

        assert foresight.is_valid_at(now) is True


class TestForesightExtraction:
    """Test foresight extraction in memory system."""

    def test_foresight_expiry_in_retrieval(self):
        """Test that foresight expires in retrieval results."""
        system = MemorySystem()

        now = datetime.now(timezone.utc)

        # Add memory with foresight
        system.add_conversation(
            [
                {"role": "user", "content": "I'm on antibiotics until Friday."},
            ],
            timestamp=now - timedelta(days=1),
        )

        # Retrieve now - should include foresight
        system.retrieve("Can I drink alcohol?", query_time=now)

        # Retrieve after "Friday" - should not include foresight
        future_time = now + timedelta(days=10)
        result_future = system.retrieve("Can I drink alcohol?", query_time=future_time)

        # Foresight should be filtered out
        assert result_future["foresight"] == []


class TestDatetimeUtils:
    """Test datetime utility functions."""

    def test_is_within_interval(self):
        """Test interval checking."""
        now = datetime.now(timezone.utc)

        # Within interval
        assert (
            is_within_interval(now, now - timedelta(days=1), now + timedelta(days=1))
            is True
        )

        # Before interval
        assert (
            is_within_interval(
                now - timedelta(days=10),
                now - timedelta(days=1),
                now + timedelta(days=1),
            )
            is False
        )

        # After interval
        assert (
            is_within_interval(
                now + timedelta(days=10),
                now - timedelta(days=1),
                now + timedelta(days=1),
            )
            is False
        )

        # None intervals (no bounds)
        assert is_within_interval(now, None, None) is True
        assert is_within_interval(now, now - timedelta(days=1), None) is True
        assert is_within_interval(now, None, now + timedelta(days=1)) is True

    def test_parse_datetime(self):
        """Test datetime parsing."""
        dt = parse_datetime("2025-01-15T10:30:00Z")
        assert dt.year == 2025
        assert dt.month == 1
        assert dt.day == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
