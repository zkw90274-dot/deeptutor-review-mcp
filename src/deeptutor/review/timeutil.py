"""One timezone/day-boundary implementation for queue, service and export.

Keeping this in one module avoids the classic drift where two screens disagree
about what "tomorrow" means because one used UTC and the other the plan zone.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except Exception:
        # Specs are validated at ingestion, so this only covers damaged legacy
        # rows. Fall back to UTC to keep the queue usable, but leave a trace:
        # a silent fallback moves day boundaries by hours.
        logger.warning("unknown timezone %r; falling back to UTC", name)
        return ZoneInfo("UTC")


def local_date(timestamp: float, tz: ZoneInfo) -> date:
    return datetime.fromtimestamp(timestamp, tz=tz).date()


def day_start(day: date, tz: ZoneInfo) -> float:
    return datetime.combine(day, dtime.min, tzinfo=tz).timestamp()


def iso(timestamp: float | None, tz: ZoneInfo) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=tz).isoformat()


def tomorrow_window(now: float, timezone: str) -> tuple[str, float, float]:
    """Return ``(tomorrow_iso_date, start_ts, end_ts)`` in the plan timezone."""
    tz = zone(timezone)
    tomorrow = local_date(now, tz) + timedelta(days=1)
    return tomorrow.isoformat(), day_start(tomorrow, tz), day_start(tomorrow + timedelta(days=1), tz)
