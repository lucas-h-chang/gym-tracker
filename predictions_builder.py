"""
predictions_builder.py — compute 90-day RF+MLP predictions → Supabase predictions table.
Runs daily at midnight PT via daily.yml.
"""
import os
import pickle
import pandas as pd
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client

from train import engineer_features

PT  = ZoneInfo("America/Los_Angeles")
now = datetime.now(PT)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

BATCH_SIZE = 500

# Summer-hours windows: derived from train.py summer-break ranges, end-shifted by -3 days
# (RSF flips back to academic hours ~3 days before classes resume).
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


def load_model():
    with open('models/rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    return rf


def compute_predictions(rf, days=91):
    """Build (slot_ts ISO string, pct) for every open 15-min slot over the next N days."""
    timestamps = []
    slot_ts    = []

    for offset in range(days):
        d        = now.date() + timedelta(days=offset)
        day_name = pd.Timestamp(d).day_name()
        open_h, close_h = get_open_hours(day_name, d)
        for h in range(open_h, close_h):
            for m in (0, 15, 30, 45):
                # Store as PT-aware ISO timestamp for Supabase TIMESTAMPTZ
                dt = datetime(d.year, d.month, d.day, h, m, tzinfo=PT)
                slot_ts.append(dt.isoformat())
                timestamps.append(pd.Timestamp(f'{d} {h:02d}:{m:02d}'))

    df = pd.DataFrame({
        'timestamp':    timestamps,
        'people_count': [100] * len(timestamps),
        'percent_full': [66.7] * len(timestamps),
    })

    print(f"  Engineering features for {len(df):,} slots...")
    X, _ = engineer_features(df)

    preds = rf.predict(X)

    # Single-model schema: one `pct` per slot (the Random Forest prediction, capped at 100).
    return [
        {
            "slot_ts": ts,
            "pct":     round(min(float(p), 100.0), 1),
        }
        for ts, p in zip(slot_ts, preds)
    ]


def main():
    print("Loading model...")
    rf = load_model()

    print("Computing predictions (today + 90 days)...")
    records = compute_predictions(rf, days=91)
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
