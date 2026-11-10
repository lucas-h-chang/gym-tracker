"""Read-only paired shadow scoring; python3 shadow_health.py [--days 30].

First 30 open days are the locked prospective holdout. The rolling window is
separate ongoing monitoring. Today, pilot days, reconstructed rows, closing
zeros, flagged actuals, and unmatched candidate slots never enter the score.
"""
import argparse
import math
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from statistics import mean
from zoneinfo import ZoneInfo

from dateutil.parser import isoparse

from academic_calendar import classify_date, get_open_hours
from forecast_health import horizon_label, load_env, recent_open_dates
from shadow_forecast import ARMS, EXPERIMENT_ID, HOLDOUT_OPEN_DAYS, HOLDOUT_START, POLICY_HASH
from supabase_io import client, execute_read

PT = ZoneInfo("America/Los_Angeles")


def holdout_dates():
    dates, day = [], HOLDOUT_START
    while len(dates) < HOLDOUT_OPEN_DAYS:
        lo, hi = get_open_hours(day.strftime("%A"), day)
        if lo < hi:
            dates.append(day)
        day += timedelta(days=1)
    return dates


def read_window(sb, table, columns, start, end, order):
    """Time-bounded, stably paginated reads; stop only on an empty page."""
    lower = datetime.combine(start, time.min, PT).isoformat()
    upper = datetime.combine(end + timedelta(days=1), time.min, PT).isoformat()
    rows, offset = [], 0
    while True:
        def query():
            q = sb.table(table).select(columns).gte(order, lower).lt(order, upper)
            if table == "prediction_snapshots":
                q = q.eq("source", "live").eq("model", "carry")
            q = q.order(order)
            if table == "prediction_snapshots":
                q = q.order("id")
            return q.range(offset, offset + 199)
        page = execute_read(query).data
        if not page:
            return rows
        rows.extend(page)
        offset += len(page)


def actual_slots(rows):
    values = defaultdict(list)
    for row in rows:
        if row.get("sensor_ok") is False or row.get("people_count") is None:
            continue
        # PostgREST trims fractional zeros; Python 3.10 fromisoformat rejects
        # fractional precision other than three or six digits.
        ts = isoparse(row["timestamp"]).astimezone(PT)
        lo, hi = get_open_hours(ts.strftime("%A"), ts.date())
        slot = min(95, math.floor((ts.hour * 60 + ts.minute) / 15 + 0.5))
        count = float(row["people_count"])
        if lo * 4 <= slot < hi * 4 and math.isfinite(count) and count >= 0:
            values[(ts.date().isoformat(), slot)].append(count / 150 * 100)
    return {key: mean(items) for key, items in values.items()}


def paired_points(snapshots, actuals, dates):
    points, valid_snapshots, expected = [], 0, 0
    for row in snapshots:
        if row["date"] not in dates:
            continue
        now = isoparse(row["computed_at"]).astimezone(PT)
        lo, hi = get_open_hours(now.strftime("%A"), now.date())
        # End-of-day runs with no future open slot have no testable forecast.
        if not any(datetime.combine(now.date(), time(s // 4, s % 4 * 15), PT) > now
                   for s in range(lo * 4, hi * 4)):
            continue
        expected += 1
        shadow = (row.get("metadata") or {}).get("shadow") or {}
        if (shadow.get("id") != EXPERIMENT_ID or shadow.get("policy_hash") != POLICY_HASH
                or shadow.get("status") != "ok"):
            continue
        forecasts = shadow.get("forecasts", {})
        if any(arm not in forecasts for arm in ARMS):
            continue
        shared = set.intersection(*(set(forecasts[arm]) for arm in ARMS))
        valid_snapshots += 1
        for key in sorted(shared, key=int):
            slot = int(key)
            slot_at = datetime.combine(now.date(), time(slot // 4, slot % 4 * 15), PT)
            actual = actuals.get((row["date"], slot))
            if actual is None or slot_at <= now or not lo * 4 <= slot < hi * 4:
                continue
            predictions = {arm: float(forecasts[arm][key]) for arm in ARMS}
            if not all(math.isfinite(v) for v in predictions.values()):
                continue
            points.append({"date": row["date"], "horizon": (slot_at - now).total_seconds() / 3600,
                           "actual": actual, **predictions})
    return points, valid_snapshots, expected


def arm_stats(points, arm):
    if not points:
        return None
    by_day = defaultdict(list)
    for p in points:
        by_day[p["date"]].append(abs(p["production"] - p["actual"]) - abs(p[arm] - p["actual"]))
    return {"mae": mean(abs(p[arm] - p["actual"]) for p in points),
            "bias": mean(p[arm] - p["actual"] for p in points),
            "day_gain": mean(mean(gains) for gains in by_day.values()),
            "days_won": sum(mean(gains) > 0 for gains in by_day.values())}


def render(title, dates, points, valid, expected):
    days = len({p["date"] for p in points})
    lines = [f"\n{title}: {dates[0]} through {dates[-1]}",
             f"Coverage: {days}/{len(dates)} complete open days scored; "
             f"{valid}/{expected} eligible live publications have shadows; {len(points):,} paired points.",
             "Arm                 MAE     Bias   Daily gain vs production   Days won"]
    for arm in ARMS:
        stat = arm_stats(points, arm)
        if stat:
            lines.append(f"{arm:<18} {stat['mae']:6.2f} {stat['bias']:+8.2f} "
                         f"{stat['day_gain']:+12.2f} pp                 {stat['days_won']}/{days}")
        else:
            lines.append(f"{arm:<18} no scored points yet")
    lines.append("Horizon MAE (hours from publication): production | stronger trail | weaker carry | both")
    for label in ("0–1h", "1–3h", "3–6h", "6h+"):
        bucket = [p for p in points if horizon_label(p["horizon"]) == label]
        maes = [f"{arm_stats(bucket, arm)['mae']:.2f}" if bucket else "N/A" for arm in ARMS]
        lines.append(f"  {label}: " + " | ".join(maes))
    phases = sorted({classify_date(date.fromisoformat(p["date"])) for p in points})
    for phase in phases:
        bucket = [p for p in points if classify_date(date.fromisoformat(p["date"])) == phase]
        gains = [f"{arm_stats(bucket, arm)['day_gain']:+.2f}" for arm in ARMS[1:]]
        lines.append(f"  {phase} daily gains (trail | carry | both): " + " | ".join(gains))
    lines.append("Positive daily gain means lower error, giving each scored day equal weight.")
    lines.append("Observational only: no automatic promotion; a 30-day result does not replace the full-year gate.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be positive")
    load_env()
    today = datetime.now(PT).date()
    print(f"SHADOW EXPERIMENT {EXPERIMENT_ID}\nPolicy SHA256: {POLICY_HASH}")
    locked = holdout_dates()
    print(f"Locked holdout: {locked[0]} through {locked[-1]} ({HOLDOUT_OPEN_DAYS} open days).")
    windows = [("LOCKED HOLDOUT", [d for d in locked if d < today]),
               ("ONGOING WINDOW", [d for d in recent_open_dates(args.days, today - timedelta(days=1))
                                    if d >= HOLDOUT_START])]
    if not any(dates for _, dates in windows):
        print("Waiting for the first complete prospective day. Pilot rows are excluded.")
        return
    sb, failures, cache = client(), [], {}
    for title, dates in windows:
        if not dates:
            continue
        key = (dates[0], dates[-1])
        if key not in cache:
            snapshots = read_window(sb, "prediction_snapshots", "id,date,computed_at,metadata",
                                    dates[0], dates[-1], "computed_at")
            readings = read_window(sb, "capacity_log", "timestamp,people_count,sensor_ok",
                                  dates[0], dates[-1], "timestamp")
            cache[key] = paired_points(snapshots, actual_slots(readings),
                                      {d.isoformat() for d in dates})
        points, valid, expected = cache[key]
        print(render(title, dates, points, valid, expected))
        # Surface broken collection independently of the successful serving job.
        scored_dates = {p["date"] for p in points}
        if not expected or valid / expected < 0.95 or len(scored_dates) != len(dates):
            failures.append(f"{title}: missing days or <95% shadow coverage")
    if failures:
        raise SystemExit("\nCOLLECTION WARNING: " + "; ".join(failures))


if __name__ == "__main__":
    main()
