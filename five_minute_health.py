"""Read-only cadence experiment: python3 five_minute_health.py --days 14.

Phase 1 measures missed occupancy changes and scores the frozen production
forecast against two explicit reference forecasts: hold the latest 5-minute
count, or hold the latest quarter-hour count. These are persistence benchmarks,
not a newly trained carry model. Nothing here publishes or refits anything.
"""
import argparse
import math
from collections import defaultdict
from datetime import datetime, time, timedelta
from statistics import mean
from zoneinfo import ZoneInfo

from dateutil.parser import isoparse

from academic_calendar import get_open_hours
from forecast_health import load_env, recent_open_dates
from supabase_io import client, execute_read

PT = ZoneInfo("America/Los_Angeles")
TABLE = "capacity_samples_5m"
EXPERIMENT = "cadence-5m-v1"
TIMING_TOLERANCE = 90  # actual reading must land within 90s after its tick
ARMS = ("production", "hold_5m", "hold_15m")


def normalize(rows):
    """Keep true zeros, reject invalid/closed samples; never relabel a late tick."""
    samples = {}
    for row in rows:
        if row.get("experiment") != EXPERIMENT:
            continue
        tick, observed = isoparse(row["tick_at"]).astimezone(PT), isoparse(row["observed_at"]).astimezone(PT)
        count = row.get("people_count")
        lo, hi = get_open_hours(tick.strftime("%A"), tick.date())
        delay = (observed - tick).total_seconds()
        if (tick.minute % 5 or tick.second or tick.microsecond
                or not 0 <= delay <= TIMING_TOLERANCE
                or not lo <= tick.hour + tick.minute / 60 < hi
                or isinstance(count, bool) or not isinstance(count, (int, float))
                or not math.isfinite(count) or count < 0):
            continue
        samples[tick] = {**row, "tick": tick, "observed": observed,
                         "actual": count / 150 * 100, "valid": True}
    return samples


def flag_outages(samples):
    """Analysis-only sensor heuristic using ELAPSED TIME, not number of rows.

    Like production, <=25% in the interior can indicate a dead counter. Require
    a continuous run lasting >=15 real minutes, with <=7.5m between readings.
    Exclude that entire confirmed run, including its initial observations.
    Short low runs and opening/closing lows remain genuine observations.
    This does not diagnose sensor truth or write flags to either table.
    """
    run = []
    for tick, row in sorted(samples.items()):
        lo, hi = get_open_hours(tick.strftime("%A"), tick.date())
        minute = tick.hour * 60 + tick.minute
        low = row["actual"] <= 25 and lo * 60 + 15 <= minute <= hi * 60 - 15
        if not low:
            run = []
            continue
        if run and (tick.date() != run[-1]["tick"].date()
                    or (row["observed"] - run[-1]["observed"]).total_seconds() > 450):
            run = []
        run.append(row)
        if (row["observed"] - run[0]["observed"]).total_seconds() >= 900:
            for item in run:
                item["valid"] = False
    return samples


def production_at(control, issue, target):
    """Linear interpolation of the captured production chart, no extrapolation.

    Quarter-hour points remain the actual production forecast. Off-grid values
    are explicitly interpolated benchmarks, not independent model outputs.
    """
    if control.get("status") != "ok" or not control.get("computed_at"):
        return None
    published = isoparse(control["computed_at"]).astimezone(PT)
    if published.date() != issue.date() or not 0 <= (issue - published).total_seconds() <= 2700:
        return None
    values = {}
    for key, value in (control.get("base") or {}).items():
        slot = float(key)
        if slot.is_integer() and math.isfinite(float(value)):
            values[int(slot)] = float(value)
    for point in control.get("preds") or []:
        slot = float(point["x"]) * 4
        if abs(slot - round(slot)) < 1e-6 and math.isfinite(float(point["y"])):
            values[round(slot)] = float(point["y"])
    slot = target.hour * 4 + target.minute / 15
    lo, hi = math.floor(slot), math.ceil(slot)
    if lo not in values or hi not in values:
        return None
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (slot - lo)


def paired_points(samples):
    points = []
    for tick, issue in sorted(samples.items()):
        if not issue["valid"]:
            continue
        coarse_tick = tick.replace(minute=tick.minute // 15 * 15)
        coarse = samples.get(coarse_tick)
        if not coarse or not coarse["valid"] or coarse["observed"] > issue["observed"]:
            continue
        for lead in (15, 30, 60):
            target = tick + timedelta(minutes=lead)
            actual = samples.get(target)
            if (not actual or not actual["valid"] or target.date() != tick.date()
                    or target <= issue["observed"]):
                continue
            production = production_at(issue["production"], issue["observed"], target)
            if production is None:
                continue
            points.append({"date": tick.date(), "lead": lead,
                           "target_hour": target.hour + target.minute / 60,
                           "actual": actual["actual"], "production": production,
                           "hold_5m": issue["actual"], "hold_15m": coarse["actual"]})
    return points


def stats(points, arm):
    by_day = defaultdict(list)
    for p in points:
        by_day[p["date"]].append(abs(p["production"] - p["actual"]) - abs(p[arm] - p["actual"]))
    return {"mae": mean(abs(p[arm] - p["actual"]) for p in points),
            "bias": mean(p[arm] - p["actual"] for p in points),
            "gain": mean(mean(values) for values in by_day.values()),
            "wins": sum(mean(values) > 0 for values in by_day.values())}


def summarize(label, values):
    if not values:
        return f"{label}: no paired readings yet"
    ordered = sorted(values)
    p90 = ordered[max(0, math.ceil(len(ordered) * .9) - 1)]
    return f"{label}: mean {mean(values):.2f}pp | p90 {p90:.2f}pp | max {max(values):.2f}pp | n={len(values)}"


def render(rows, dates):
    samples = flag_outages(normalize(rows))
    expected = 0
    for day in dates:
        lo, hi = get_open_hours(day.strftime("%A"), day)
        expected += (hi - lo) * 12
    valid = sum(row["valid"] for row in samples.values())
    controls = sum(row["production"].get("status") == "ok" for row in samples.values())
    lines = [f"FIVE-MINUTE COLLECTION — {EXPERIMENT}", f"Window: {dates[0]} through {dates[-1]}",
             f"On-time samples: {len(samples)}/{expected}; sensor-screened samples: {valid}; captured controls marked ok: {controls}/{len(samples)}.",
             "Sensor screening is a <=25% sustained-low heuristic, not verified sensor ground truth."]
    changes, hidden = [], []
    for tick, row in samples.items():
        if not row["valid"]:
            continue
        previous = samples.get(tick - timedelta(minutes=5))
        if previous and previous["valid"] and previous["tick"].date() == tick.date():
            changes.append(abs(row["actual"] - previous["actual"]))
        anchor = samples.get(tick.replace(minute=tick.minute // 15 * 15))
        if tick.minute % 15 and anchor and anchor["valid"]:
            hidden.append(abs(row["actual"] - anchor["actual"]))
    lines.extend([summarize("5-minute occupancy changes", changes),
                  summarize("Changes hidden by holding the last quarter-hour reading", hidden),
                  "", "Paired forecast error (percentage points; lower is better).",
                  "Off-grid production forecasts are linearly interpolated from captured chart points.",
                  "Leads are approximately 15/30/60 minutes from sample time (<=90s tick delay)."])
    points = paired_points(samples)
    for lead in (15, 30, 60):
        for label, lo, hi in [("all hours", 0, 24), ("noon–5pm", 12, 17), ("5–9pm", 17, 21), ("9pm–close", 21, 24)]:
            bucket = [p for p in points if p["lead"] == lead and lo <= p["target_hour"] < hi]
            if not bucket:
                continue
            days = len({p["date"] for p in bucket})
            lines.append(f"\n+{lead}m {label}: {len(bucket)} paired points / {days} days")
            lines.append("  Forecast       MAE    Bias   Equal-day gain vs production  Days won")
            for arm in ARMS:
                s = stats(bucket, arm)
                lines.append(f"  {arm:<12} {s['mae']:6.2f} {s['bias']:+7.2f} {s['gain']:+12.2f}pp            {s['wins']}/{days}")
    lines.extend(["", "hold_5m/hold_15m simply predict that the latest count stays unchanged.",
                  "These reference forecasts test freshness; neither is a trained five-minute model.",
                  "Descriptive pilot only. No policy tuning, production change, or automatic promotion."])
    healthy = bool(expected and len(samples) / expected >= .95
                   and samples and controls / len(samples) >= .95 and points)
    lines.append("Collection: " + ("HEALTHY" if healthy else "WATCH — missing samples, controls, or paired targets"))
    return "\n".join(lines), healthy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14, help="complete open days, excluding any partial first collection day")
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be positive")
    load_env()
    sb = client()
    first = execute_read(lambda: sb.table(TABLE).select("tick_at").order("tick_at").limit(1)).data
    if not first:
        raise SystemExit("COLLECTION WARNING: no five-minute samples. Collection has not been verified active.")
    first_tick = isoparse(first[0]["tick_at"]).astimezone(PT)
    lo, _ = get_open_hours(first_tick.strftime("%A"), first_tick.date())
    first_full = first_tick.date() if first_tick.hour == lo and first_tick.minute == 0 else first_tick.date() + timedelta(days=1)
    dates = [d for d in recent_open_dates(args.days, datetime.now(PT).date() - timedelta(days=1)) if d >= first_full]
    if not dates:
        print("Waiting for the first complete collection day. Partial launch day and today are excluded.")
        return
    lower = datetime.combine(dates[0], time.min, PT).isoformat()
    upper = datetime.combine(dates[-1] + timedelta(days=1), time.min, PT).isoformat()
    rows, offset = [], 0
    while True:
        page = execute_read(lambda: sb.table(TABLE).select("tick_at,observed_at,people_count,experiment,production")
                            .gte("tick_at", lower).lt("tick_at", upper).order("tick_at")
                            .range(offset, offset + 199)).data
        if not page:
            break
        rows.extend(page)
        offset += len(page)
    report, healthy = render(rows, dates)
    print(report)
    if not healthy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
