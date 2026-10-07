"""Python mirror of api/_sensor.js for the manual scraper escape hatch.

Shared fixtures enforce parity. Two consecutive interior quarter-hour readings
at or below 25% confirm both observations invalid; raw counts are preserved.
"""
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from academic_calendar import get_open_hours

PT = ZoneInfo('America/Los_Angeles')
FLOOR_COUNT = 150 * .25


def _low(count):
    return isinstance(count, (int, float)) and not isinstance(count, bool) and math.isfinite(count) and 0 <= count <= FLOOR_COUNT


def _interior(t):
    start, end = get_open_hours(t.strftime('%A'), t.date())
    minutes = t.hour * 60 + t.minute
    return end > start and start * 60 + 15 <= minutes <= end * 60 - 15


def _slot(t):
    return t.hour * 4 + t.minute // 15


def evaluate_sensor_reading(rows, count, now):
    now = now.astimezone(PT)
    if not _low(count) or not _interior(now):
        return {'stalled': False}
    prior = []
    for row in rows or []:
        try:
            t = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')).astimezone(PT)
        except (ValueError, TypeError, KeyError):
            continue
        if t < now and t.date() == now.date() and _slot(t) < _slot(now):
            prior.append((t, row))
    prior.sort(key=lambda p: p[0], reverse=True)
    if not prior:
        return {'stalled': False}
    t, last = prior[0]
    if _slot(t) != _slot(now) - 1 or not _interior(t) or not _low(last.get('people_count')):
        return {'stalled': False}
    timestamps = list(dict.fromkeys(row['timestamp'] for t, row in prior
        if _slot(t) == _slot(now) - 1 and _low(row.get('people_count'))))
    return {'stalled': True, 'since': last['timestamp'], 'timestamps': timestamps}


def is_sensor_stalled(sb, count, now):
    if not _low(count) or not _interior(now.astimezone(PT)):
        return {'stalled': False}
    try:
        rows = (sb.table('capacity_log').select('timestamp,people_count')
                .gte('timestamp', (now - timedelta(minutes=45)).isoformat())
                .lte('timestamp', now.isoformat()).order('timestamp', desc=True)
                .limit(12).execute().data)
        return evaluate_sensor_reading(rows, count, now)
    except Exception as exc:
        print(f'WARNING: sensor lookback failed: {exc}')
        return {'stalled': False}
