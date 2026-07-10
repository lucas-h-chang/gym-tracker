"""
day_profiles_builder.py — precompute per-day hourly-average profiles → Supabase day_profiles.
Runs daily via daily.yml.

Why: today_builder.py (every 15 min) used to re-download the entire capacity_log (~10 MB) just to
re-derive per-day hourly averages for its similarity nowcast — ~19 GB/month, over the free tier
(Finding E). Those daily averages only change once/day, so we materialize them here once/day and let
today_builder read the small, pre-aggregated slice it needs.

Scope: only dates >= DATA_CUTOFF (drop the unrepresentative COVID era, 2020-2021). Keep this boundary
in sync with train.py when T20 applies the same cutoff to the ML model.
"""
import os
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from supabase import create_client

PT  = ZoneInfo("America/Los_Angeles")
now = datetime.now(PT)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

# Fixed data boundary: keep 2022 onward, drop COVID-era 2020-2021.
# Shared with today_builder.py (and, later, train.py via T20) — keep in sync.
DATA_CUTOFF = date(2022, 1, 1)

BATCH_SIZE = 500

SPRING_BREAKS = [
    ('2021-03-20', '2021-03-28'),
    ('2022-03-19', '2022-03-27'),
    ('2023-03-25', '2023-04-02'),
    ('2024-03-23', '2024-03-31'),
    ('2025-03-22', '2025-03-30'),
    ('2026-03-21', '2026-03-29'),
    ('2027-03-20', '2027-03-28'),
    ('2028-03-25', '2028-04-02'),
]


def is_semester_day(d):
    month, dom = d.month, d.day
    is_summer = month in [6, 7, 8]
    is_winter = (month == 12 and dom >= 16) or (month == 1 and dom <= 12)
    is_sb = any(
        pd.Timestamp(s).date() <= d <= pd.Timestamp(e).date()
        for s, e in SPRING_BREAKS
    )
    return not (is_summer or is_winter or is_sb)


def _pt_iso(d, t):
    """ISO8601 for a PT wall-clock (date, time)."""
    return datetime.combine(d, t, tzinfo=PT).isoformat()


def fetch_rows(gte_iso, lte_iso=None):
    """Paginated capacity_log fetch over a timestamp window (inclusive)."""
    BATCH  = 9000
    offset = 0
    rows   = []
    while True:
        q = (
            sb.table("capacity_log")
            .select("timestamp,percent_full")
            .gte("timestamp", gte_iso)
        )
        if lte_iso is not None:
            q = q.lte("timestamp", lte_iso)
        batch = q.range(offset, offset + BATCH - 1).order("timestamp").execute().data
        rows.extend(batch)
        if len(batch) < BATCH:
            break
        offset += BATCH
        print(f"  Fetched {len(rows):,} rows...")
    return rows


def latest_profiled_date():
    """Most recent date already in day_profiles, or None if the table is empty."""
    data = (
        sb.table("day_profiles")
        .select("date")
        .order("date", desc=True)
        .limit(1)
        .execute()
        .data
    )
    return pd.Timestamp(data[0]["date"]).date() if data else None


def build_records(rows, last_complete):
    """Aggregate raw rows into (date, hour_slot) profiles for dates <= last_complete."""
    if not rows:
        return []

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601').dt.tz_convert(PT)
    df['date']      = df['timestamp'].dt.date
    df['hour_slot'] = ((df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60) * 4).round() / 4

    # Never emit today (partial) or anything past the last complete day.
    df = df[df['date'] <= last_complete]
    if df.empty:
        return []

    grouped = (
        df.groupby(['date', 'hour_slot'])['percent_full']
        .mean()
        .reset_index()
    )

    # Per-date attributes computed once (not per row).
    attrs = {
        d: (pd.Timestamp(d).day_name(), bool(is_semester_day(d)))
        for d in grouped['date'].unique()
    }

    records = []
    for _, r in grouped.iterrows():
        day_name, is_sem = attrs[r['date']]
        records.append({
            'date':        r['date'].isoformat(),
            'day_name':    day_name,
            'is_semester': is_sem,
            'hour_slot':   float(r['hour_slot']),
            'avg_pct':     float(r['percent_full']),   # unrounded, for parity with today_builder
        })
    return records


def main():
    last_complete = now.date() - timedelta(days=1)
    have_through  = latest_profiled_date()

    if have_through is None:
        print(f"day_profiles empty — backfilling from {DATA_CUTOFF} through {last_complete}...")
        rows = fetch_rows(_pt_iso(DATA_CUTOFF, time.min))
    else:
        # Re-include have_through in case it was written while still partial.
        start = max(have_through, DATA_CUTOFF)
        if start > last_complete:
            print(f"day_profiles already current through {have_through}; nothing to do.")
            return
        print(f"Incremental update {start} → {last_complete} (have through {have_through})...")
        rows = fetch_rows(_pt_iso(start, time.min), _pt_iso(last_complete, time.max))

    print(f"  {len(rows):,} raw rows fetched")
    records = build_records(rows, last_complete)
    print(f"  {len(records):,} profile rows computed")

    if not records:
        print(f"[{now.isoformat()}] day_profiles: no new complete days to write.")
        return

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        sb.table("day_profiles").upsert(batch).execute()
        print(f"  Upserted {min(i + BATCH_SIZE, len(records))}/{len(records)}")

    print(f"[{now.isoformat()}] day_profiles updated: {len(records)} rows through {last_complete}")


if __name__ == "__main__":
    main()
