"""Shared freshness rules for serving and notification consumers."""
import math
from datetime import datetime, timezone

LIVE_MAX_AGE_SECONDS = 120
FORECAST_MAX_AGE_SECONDS = 45 * 60


def timestamp_age(value, now):
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return None
        return (now - dt).total_seconds()
    except (TypeError, ValueError, AttributeError):
        return None


def is_fresh(value, now, max_age=FORECAST_MAX_AGE_SECONDS):
    age = timestamp_age(value, now)
    return age is not None and 0 <= age <= max_age


def valid_live_pct(body, now=None):
    now = now or datetime.now(timezone.utc)
    value = body.get("capacity_pct")
    if (body.get("source") == "cache_stale" or body.get("sensor_ok") is False or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or not is_fresh(body.get("recorded_at"), now, LIVE_MAX_AGE_SECONDS)):
        return None
    return round(value)
