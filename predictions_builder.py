"""
predictions_builder.py — compute 90-day curve-model predictions → Supabase predictions table.
Runs daily at midnight PT via daily.yml.
"""
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client

import numpy as np

import curve_model as cm
import nowcast as nw
from academic_calendar import (
    classify_date, is_summer_day, is_closed_day, get_open_hours, slot_of,
)
from supabase_io import parse_supabase_timestamps, paginated_fetch

PT  = ZoneInfo("America/Los_Angeles")
now = datetime.now(PT)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

BATCH_SIZE = 500

# is_summer_day/get_open_hours/SUMMER_RANGES live in academic_calendar.py
# (consolidated 2026-07-21, see CLAUDE.md).

# The trailing-residual layer (window length, shrinkage, horizon decay) lives in
# nowcast.py, which carry_data.py imports too. It used to be implemented here
# and hand-mirrored there, with nothing enforcing that the copies agreed.


def load_curves():
    with open('models/curves.json') as f:
        return json.load(f)


def build_trailing(table):
    """
    Fetch the trailing window of actuals, difference them against the curve, and
    return a nowcast.Trailing ready to answer "how far off is the gym running?"

    THE CLEANING HAS TO MATCH THE OTHER SIDE OF THE SUBTRACTION
    -----------------------------------------------------------
    A residual is `actual - curve_pred`, and `curve_pred` comes out of a table
    built by curve_model.prepare_slots, which drops full-facility closure days
    and expects its caller to have already dropped rows where the counter was
    dead. This function used to do neither, so it differenced clean predictions
    against dirty actuals. Two measured consequences:

      - A closed building reads 0-2 people, which looks like a ~-40pp residual.
        Twelve closure days since 2024 fed a later cell this way; the Christmas
        and New Year closures are the worst, each landing in up to three
        early-January winter-break cells and dragging those forecasts down.
      - `percent_full > 0` was filtering out genuinely empty readings. That is
        the same bug curve_model documented at length when it removed its own
        `> 5` filter: a nearly empty gym at 7:00 or 22:45 is the truth, not
        noise, and dropping it biases the residual upward at exactly the edges
        of the day where this layer is most visible.

    Both sides are now cleaned identically: closure days out, sensor_ok=False
    out, genuine zeros kept.
    """
    lo = (now - timedelta(days=nw.WINDOW_DAYS)).isoformat()
    hi = now.isoformat()

    rows = paginated_fetch(sb, "capacity_log", "timestamp,percent_full,sensor_ok",
                           gte=lo, lte=hi, order="timestamp")
    if not rows:
        print("  No recent readings; serving the base curve uncorrected")
        return None

    df = pd.DataFrame(rows)
    df['timestamp']    = parse_supabase_timestamps(df['timestamp'])
    df['percent_full'] = df['percent_full'].astype(float)
    df = df.dropna(subset=['timestamp', 'percent_full'])

    # sensor_ok defaults to true for rows written before migration 008, so this
    # is a no-op on the historical span and a real filter on anything recent.
    if 'sensor_ok' in df.columns:
        df = df[df['sensor_ok'] != False]

    df['date'] = df['timestamp'].dt.date
    df = df[~df['date'].map(is_closed_day)].reset_index(drop=True)
    if df.empty:
        print("  No usable recent readings; serving the base curve uncorrected")
        return None

    # Nearest-boundary slot (academic_calendar.slot_of), matching
    # curve_model.prepare_slots, so a 09:54 scrape lands on the 10:00 slot the
    # prediction side actually generates rather than the 09:45 one it never does.
    df['slot'] = slot_of(df['timestamp']).astype(int)
    df = df.groupby(['date', 'slot'], as_index=False)['percent_full'].mean()

    df['curve_pred'] = cm.predict(table, list(zip(df['date'], df['slot'])))
    df = df.dropna(subset=['curve_pred'])
    if df.empty:
        print("  No recent readings matched a curve; serving the base curve uncorrected")
        return None

    dates = sorted(df['date'].unique())
    index = {d: i for i, d in enumerate(dates)}
    resid = np.full((len(dates), nw.SLOTS_PER_DAY), np.nan)
    for d, s, a, c in zip(df['date'], df['slot'], df['percent_full'], df['curve_pred']):
        resid[index[d], s] = a - c

    segment, regime, dow = nw.day_keys(dates, classify_date, is_summer_day)
    observed = int(np.isfinite(resid).sum())
    print(f"  Trailing residuals: {observed:,} readings over {len(dates)} days "
          f"({dates[0]} -> {dates[-1]})")
    return nw.Trailing(dates, resid, segment, regime, dow)


def compute_predictions(table, trailing, days=91):
    """Build (slot_ts ISO string, pct) for every open 15-min slot over the next N days."""
    slot_ts, dates_slots, corrections = [], [], []

    for offset in range(days):
        d        = now.date() + timedelta(days=offset)
        day_name = pd.Timestamp(d).day_name()
        open_h, close_h = get_open_hours(day_name, d)
        if open_h >= close_h:            # full-facility closure day
            continue

        # One ladder fit per target day, reused across that day's slots. Decay
        # toward 0 as the horizon grows: a trailing window tracks "this stretch
        # is running hot" for about a week, and a backtest showed an undecayed
        # correction improved days 1-7 but dragged the 90-day average below the
        # raw curve.
        if trailing is None:
            day_corr = np.zeros(nw.SLOTS_PER_DAY)
        else:
            day_corr = nw.decay(offset) * trailing.correction(
                d,
                nw.correction_segment(classify_date(d)),
                is_summer_day(d),
                d.weekday(),
            )

        for h in range(open_h, close_h):
            for m in (0, 15, 30, 45):
                # Store as PT-aware ISO timestamp for Supabase TIMESTAMPTZ
                dt = datetime(d.year, d.month, d.day, h, m, tzinfo=PT)
                slot = h * 4 + m // 15
                slot_ts.append(dt.isoformat())
                dates_slots.append((d, slot))
                corrections.append(day_corr[slot])

    print(f"  Predicting {len(dates_slots):,} slots from curve table...")
    preds = cm.predict(table, dates_slots)

    # Applied to every open slot, not just evenings: a "running hot/cool" stretch
    # is an all-day phenomenon, and a backtest on the week-aware base showed
    # all-hours strictly dominates evening-only (~2.5% better on forecast days
    # 1-7, evenings unchanged).
    records, n_moved = [], 0
    for ts, p, c in zip(slot_ts, preds, corrections):
        if p != p:  # NaN -> no curve matched this (phase, dow, slot)
            continue
        if abs(c) >= 0.05:
            n_moved += 1
        records.append({
            "slot_ts": ts,
            "pct":     round(min(max(float(p + c), 0.0), 100.0), 1),
        })
    pct_moved = 100.0 * n_moved / len(records) if records else 0.0
    print(f"  Trailing correction moved {n_moved:,}/{len(records):,} slots ({pct_moved:.0f}%)")
    return records


def main():
    print("Loading curve table...")
    table = load_curves()

    print("Building trailing-residual ladder...")
    trailing = build_trailing(table)

    print("Computing predictions (today + 90 days)...")
    records = compute_predictions(table, trailing, days=91)
    print(f"  {len(records):,} slots computed")

    print("Upserting to Supabase predictions table...")
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        sb.table("predictions").upsert(batch, on_conflict="slot_ts").execute()
        print(f"  Upserted {min(i + BATCH_SIZE, len(records))}/{len(records)}")

    # Purge stale in-horizon rows: upsert only ever adds/overwrites slots we generate
    # today, it never removes ones from an earlier run whose open/close hours no
    # longer match (e.g. a date that used to be generated with academic-year hours
    # and is now correctly summer-hours-only keeps its old post-close rows forever
    # otherwise). Diff today's generated slot set against what's actually in the
    # table over the same horizon and delete anything left over.
    horizon_start = datetime(now.year, now.month, now.day, 0, 0, tzinfo=PT)
    horizon_end   = horizon_start + timedelta(days=91)
    generated_instants = {datetime.fromisoformat(r["slot_ts"]) for r in records}

    existing, offset = [], 0
    while True:
        batch = (
            sb.table("predictions")
            .select("slot_ts")
            .gte("slot_ts", horizon_start.isoformat())
            .lt("slot_ts", horizon_end.isoformat())
            .range(offset, offset + 8999)
            .execute()
            .data
        )
        existing.extend(batch)
        if len(batch) < 9000:
            break
        offset += 9000

    stale = [
        r["slot_ts"] for r in existing
        if datetime.fromisoformat(r["slot_ts"]) not in generated_instants
    ]
    for i in range(0, len(stale), BATCH_SIZE):
        sb.table("predictions").delete().in_("slot_ts", stale[i:i + BATCH_SIZE]).execute()
    print(f"  Purged {len(stale)} stale in-horizon rows")

    # Purge stale far-future rows left over from when we generated 180 days, so the
    # table stays bounded to the ~90-day horizon we now compute. The +93-day margin sits
    # beyond the clients' +92-day fetch bound, so this can never delete a slot that's
    # still viewable, even accounting for PT/UTC boundary fuzz.
    purge_from = (now.date() + timedelta(days=93)).isoformat()
    sb.table("predictions").delete().gte("slot_ts", purge_from).execute()
    print(f"  Purged any predictions on/after {purge_from}")

    print(f"[{now.isoformat()}] predictions table updated: {len(records)} rows")


if __name__ == "__main__":
    main()
