"""
predictions_builder.py — compute 90-day curve-model predictions → Supabase
predictions table. Runs daily at midnight PT via daily.yml.

Reads the pre-built models/curves.json (see build_curves.py, run weekly by
build_curves.yml) instead of training or loading a pickle — inference is a
dict lookup + weighted sum, no sklearn import.
"""
import os
import json
import pandas as pd
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client

import curve_model as cm

PT  = ZoneInfo("America/Los_Angeles")
now = datetime.now(PT)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

BATCH_SIZE = 500

SUMMER_RANGES = [
    (date(2024, 5, 10), date(2024, 8, 24)),
    (date(2025, 5, 16), date(2025, 8, 23)),
    (date(2026, 5, 15), date(2026, 8, 22)),
    (date(2027, 5, 14), date(2027, 8, 21)),
]


def is_summer_day(d):
    return any(s <= d <= e for s, e in SUMMER_RANGES)


def get_open_hours(day_name, d):
    summer = is_summer_day(d)
    if day_name == 'Saturday':
        return 8, 18
    if day_name == 'Sunday':
        return 8, (20 if summer else 23)
    return 7, (20 if summer else 23)


def load_curves():
    with open('models/curves.json') as f:
        return json.load(f)


def compute_predictions(table, days=91):
    """Build (slot_ts ISO string, pct) for every open 15-min slot over the next N days."""
    dates_slots = []
    slot_ts     = []

    for offset in range(days):
        d        = now.date() + timedelta(days=offset)
        day_name = pd.Timestamp(d).day_name()
        open_h, close_h = get_open_hours(day_name, d)
        for h in range(open_h, close_h):
            for m in (0, 15, 30, 45):
                # Store as PT-aware ISO timestamp for Supabase TIMESTAMPTZ
                dt = datetime(d.year, d.month, d.day, h, m, tzinfo=PT)
                slot_ts.append(dt.isoformat())
                dates_slots.append((d, h * 4 + m // 15))

    print(f"  Predicting {len(dates_slots):,} slots...")
    preds = cm.predict(table, dates_slots)

    records = []
    for ts, p in zip(slot_ts, preds):
        records.append({
            "slot_ts": ts,
            "pct":     round(min(max(float(p), 0.0), 100.0), 1),
        })
    return records


def main():
    print("Loading curve table...")
    table = load_curves()

    print("Computing predictions (today + 90 days)...")
    records = compute_predictions(table, days=91)
    print(f"  {len(records):,} slots computed")

    print("Upserting to Supabase predictions table...")
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        sb.table("predictions").upsert(batch, on_conflict="slot_ts").execute()
        print(f"  Upserted {min(i + BATCH_SIZE, len(records))}/{len(records)}")

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
