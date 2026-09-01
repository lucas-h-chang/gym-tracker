"""
today_builder.py — today's forecast → Supabase today_summary. Runs every 15 min.

WHAT THIS DOES
--------------
Takes the base curve for today (the `predictions` table) and corrects its LEVEL
using today's readings so far. The curve keeps the shape of the day — it is
built on years of history and smoothed — and today's data only moves it up or
down. See carry_model.py for the model and
handoffs/SPEC_TODAY_BUILDER_REWRITE.md for the full write-up.

WHAT IT REPLACED, AND WHY
-------------------------
Until 2026-08-31 this file ran a K=5 similarity nowcast and blended it against
the base curve on a stopwatch: weight 0.9 at the last observed slot, ramping to
0 over two hours. That ramp was a bug, not a forecast. Whenever the two models
disagreed by D, the *act of handing off* injected a slope of D * (0.9 / 2.0) per
hour into the drawn line. On 2026-08-31 they sat 21pp apart and the chart
climbed ~2.4pp per 15 minutes for reasons unrelated to anyone arriving at the
gym — freezing the nowcast completely flat still produced a 16-point climb.

It also meant that past two hours, today's data had NO influence at all: 71.5%
of all forecasts served are more than two hours ahead, and for every one of them
the deployed correction contributed 0.005pp, i.e. nothing.

Scored by replay_day.py over 357 days at this exact 15-minute cadence
(423,793 predictions), the replacement beats the old model at every horizon and
every hour of the day: 9.268 vs 9.973 MAE, against a 10.549 do-nothing baseline.

THE WIRE FORMAT AND WHY w = 1.0
-------------------------------
Every consumer (docs/index.html, RSFApp2.0's GymLogic.swift,
send_workout_notifications.py) computes (1 - w) * base + w * y. Publishing the
finished number with w = 1.0 makes that an identity, so all three render exactly
this forecast with no client change — and it repairs a live defect where
notifications blended with the scalar blend_weight while web and iOS used
per-point w, so a push and the website disagreed about the same hour.

An empty list is a valid output: below carry_model.MIN_OBSERVED readings there
is not enough evidence to correct anything, and every client then falls back to
the bare base curve, which is the right answer rather than a guess.
"""
import os
import json
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from supabase import create_client

import carry_model as km
from academic_calendar import get_open_hours

PT  = ZoneInfo("America/Los_Angeles")
now = datetime.now(PT)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

CARRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "carry.json")


def _pt_iso(d, t):
    """ISO8601 for a PT wall-clock (date, time)."""
    return datetime.combine(d, t, tzinfo=PT).isoformat()


def _slot_label(slot):
    h, m = slot // 4, (slot % 4) * 15
    return f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def load_carry():
    """The coefficients built weekly by build_carry.py, or None if missing."""
    try:
        with open(CARRY_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARNING: {CARRY_PATH} missing — serving the base curve uncorrected")
        return None


def fetch_today_rows():
    """Today's capacity_log rows (a few dozen)."""
    return (
        sb.table("capacity_log")
        .select("timestamp,percent_full")
        .gte("timestamp", _pt_iso(now.date(), time.min))
        .order("timestamp")
        .limit(2000)
        .execute()
        .data
    )


def fetch_today_predictions():
    """Today's deployed baseline from `predictions` -> {slot: pct}.

    This is the curve PLUS predictions_builder's 28-day trailing correction,
    which is what carry.json was fitted against. Reading the raw curve here
    instead would double-count that layer on every day it fires.
    """
    rows = (
        sb.table("predictions")
        .select("slot_ts,pct")
        .gte("slot_ts", _pt_iso(now.date(), time.min))
        .lt("slot_ts", _pt_iso(now.date() + timedelta(days=1), time.min))
        .order("slot_ts")
        .execute()
        .data
    )
    out = {}
    for r in rows:
        ts = datetime.fromisoformat(r["slot_ts"]).astimezone(PT)
        out[ts.hour * 4 + ts.minute // 15] = float(r["pct"])
    return out


def actuals_by_slot(today_rows):
    """Today's readings so far -> {slot: mean percent_full}, up to now."""
    if not today_rows:
        return {}
    df = pd.DataFrame(today_rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601').dt.tz_convert(PT)
    # Round to NEAREST quarter-hour, matching academic_calendar.slot_of and every
    # display consumer. Flooring would file a 10:40 scrape at 10:30.
    df['slot'] = ((df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60) * 4).round().astype(int)
    df = df[df['slot'] <= round((now.hour + now.minute / 60) * 4)]
    return df.groupby('slot')['percent_full'].mean().to_dict()


def compute_level_correction(actuals, base, carry):
    """Base curve plus a fitted level correction, for every remaining slot.

    Returns [{x, y, w, label}] with w = 1.0, or [] when there is not yet enough
    evidence to say anything (see carry_model.MIN_OBSERVED).
    """
    if not carry or not base or not actuals:
        return []

    open_h, close_h = get_open_hours(now.strftime('%A'), now.date())
    lo, hi = open_h * 4, close_h * 4
    obs = sorted(s for s in actuals if lo <= s < hi and s in base)
    if not obs:
        return []

    gaps = km.compute_gaps(obs, [actuals[s] for s in obs], [base[s] for s in obs])
    if gaps is None:
        print(f"  only {len(obs)} usable readings (need {km.MIN_OBSERVED}) — "
              "serving the base curve uncorrected")
        return []
    gap_day, gap_recent, gap_last, last_slot, n_obs = gaps
    print(f"  {n_obs} readings; gaps  day {gap_day:+.1f} / hour {gap_recent:+.1f} "
          f"/ last {gap_last:+.1f}")

    corrected = km.apply_to_day(carry, max(obs), last_slot,
                                (gap_day, gap_recent, gap_last), base, lo, hi,
                                n_obs=n_obs)
    preds = [
        {'x': s / 4, 'y': round(v, 1), 'w': 1.0, 'label': _slot_label(s)}
        for s, v in sorted(corrected.items())
    ]
    # Closing zero, so the chart drops to 0 at close like every finished day.
    ch_label = f"{close_h % 12 or 12}:00 {'AM' if close_h < 12 else 'PM'}"
    preds.append({'x': float(close_h), 'y': 0.0, 'w': 1.0, 'label': ch_label})
    return preds


def main():
    # Skip entirely when the RSF is closed — nothing to forecast.
    open_h, close_h = get_open_hours(now.strftime('%A'), now.date())
    now_hour = now.hour + now.minute / 60
    if now_hour < open_h or now_hour >= close_h:
        print(f"[{now.isoformat()}] RSF closed (open {open_h}:00-{close_h}:00); "
              "skipping today_summary build.")
        return

    print("Computing today's level correction...")
    preds = compute_level_correction(
        actuals_by_slot(fetch_today_rows()), fetch_today_predictions(), load_carry())

    sb.table("today_summary").upsert({
        "date":             now.strftime('%Y-%m-%d'),
        "similarity_preds": preds,
        # w = 1.0 on every point makes each client's (1-w)*base + w*y an
        # identity. The scalar is kept at 1.0 for the one consumer that still
        # reads it (send_workout_notifications.py) so it agrees with the rest.
        "blend_weight":     1.0,
        "computed_at":      now.isoformat(),
    }).execute()

    print(f"[{now.isoformat()}] today_summary updated: {len(preds)} slots")


if __name__ == "__main__":
    main()
