"""
predictions_builder.py — compute 90-day RF predictions → Supabase predictions table.
Runs daily at midnight PT via daily.yml.
"""
import os
import pickle
import pandas as pd
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client

from train import engineer_features, parse_supabase_timestamps

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


CORRECTION_DAYS     = 28   # trailing window for residual computation
CORRECTION_MIN_N    = 3    # min observations per (is_break, dow, hour) cell to apply correction
CORRECTION_HOUR_MIN = 17   # only correct evening slots (5 PM onwards)


def load_model():
    with open('models/rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    return rf


def build_evening_correction(rf):
    """
    Fetch the last CORRECTION_DAYS days of actuals, re-predict with RF, and return a
    dict keyed by (is_break, dow, hour, minute) → mean residual (pp).
    minute is rounded to the nearest 15-min boundary (0/15/30/45) so scraped
    timestamps (which land off-quarter) align with the prediction slots.
    """
    lo = (now - timedelta(days=CORRECTION_DAYS)).isoformat()
    hi = now.isoformat()

    rows, offset = [], 0
    while True:
        batch = (
            sb.table("capacity_log")
            .select("timestamp,percent_full")
            .gte("timestamp", lo)
            .lte("timestamp", hi)
            .order("timestamp")
            .range(offset, offset + 8999)
            .execute()
            .data
        )
        rows.extend(batch)
        if len(batch) < 9000:
            break
        offset += 9000

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    df['timestamp']    = parse_supabase_timestamps(df['timestamp'])
    df['percent_full'] = df['percent_full'].astype(float)
    df = df[df['percent_full'] > 0].dropna().reset_index(drop=True)

    X, _ = engineer_features(df)
    df['rf_pred']  = rf.predict(X)
    df['residual'] = df['percent_full'] - df['rf_pred']
    df['dow']      = df['timestamp'].dt.dayofweek
    df['hour']     = df['timestamp'].dt.hour
    df['minute']   = (df['timestamp'].dt.minute // 15) * 15  # round to 0/15/30/45
    df['is_break'] = X['is_break'].values.astype(int)

    correction = {}
    for (ib, dow, hr, mn), g in df.groupby(['is_break', 'dow', 'hour', 'minute']):
        if len(g) >= CORRECTION_MIN_N:
            correction[(int(ib), int(dow), int(hr), int(mn))] = g['residual'].mean()

    n_cells = len(correction)
    print(f"  Evening correction: {len(df):,} recent rows → {n_cells} (is_break, dow, hour, minute) cells")
    return correction


def compute_predictions(rf, correction, days=91):
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

    preds    = rf.predict(X)
    is_break = X['is_break'].values.astype(int)
    dow      = df['timestamp'].dt.dayofweek.values
    hour     = df['timestamp'].dt.hour.values
    minute   = df['timestamp'].dt.minute.values

    records = []
    for ts, p, ib, dw, hr, mn in zip(slot_ts, preds, is_break, dow, hour, minute):
        if hr >= CORRECTION_HOUR_MIN:
            p += correction.get((ib, dw, hr, mn), 0.0)
        records.append({
            "slot_ts": ts,
            "pct":     round(min(max(float(p), 0.0), 100.0), 1),
        })
    return records


def main():
    print("Loading model...")
    rf = load_model()

    print("Building evening correction table...")
    correction = build_evening_correction(rf)

    print("Computing predictions (today + 90 days)...")
    records = compute_predictions(rf, correction, days=91)
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
