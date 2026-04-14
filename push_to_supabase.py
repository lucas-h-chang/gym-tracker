"""
push_to_supabase.py — runs after scraper.py in the GitHub Action.

1. Pushes the latest capacity reading to Supabase capacity_log.
2. Computes 180-day hourly predictions (RF + MLP) and upserts to Supabase predictions.
3. Computes weekly averages for all date-range × semester combinations.

The iOS app reads all three tables directly via the Supabase REST API.

Required Supabase tables:
  -- predictions (add rf_pct if not exists):
  ALTER TABLE predictions ADD COLUMN IF NOT EXISTS rf_pct INTEGER;

  -- weekly_averages (recreate with new schema):
  DROP TABLE IF EXISTS weekly_averages;
  CREATE TABLE weekly_averages (
    day_of_week   TEXT    NOT NULL,
    hour          INTEGER NOT NULL,
    range_type    TEXT    NOT NULL DEFAULT 'all_time',
    semester_only BOOLEAN NOT NULL DEFAULT true,
    avg_pct       REAL    NOT NULL,
    PRIMARY KEY (day_of_week, hour, range_type, semester_only)
  );
  ALTER TABLE weekly_averages ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "public read" ON weekly_averages FOR SELECT USING (true);
"""

import os
import sqlite3
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from supabase import create_client
from predict import predict

PT = ZoneInfo("America/Los_Angeles")

sb = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)


def push_latest_occupancy(conn):
    row = conn.execute(
        "SELECT timestamp, people_count, percent_full FROM capacity_log "
        "ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    if not row:
        print("[supabase] No capacity data to push.")
        return
    ts, people_count, percent_full = row
    sb.table("capacity_log").insert({
        "timestamp":    ts,
        "people_count": people_count,
        "percent_full": percent_full,
    }).execute()
    print(f"[supabase] Pushed occupancy: {percent_full}% at {ts}")


RSF_HOURS = {
    "Saturday": (8, 18),
    "Sunday":   (8, 23),
}
RSF_HOURS_DEFAULT = (7, 23)  # Mon–Fri


def push_predictions(days=180):
    now = datetime.now(PT)
    computed_at = now.strftime("%Y-%m-%d %H:%M:%S")
    today = now.date()

    rows = []
    for day_offset in range(days):
        d = today + timedelta(days=day_offset)
        day_name = d.strftime("%A")
        open_h, close_h = RSF_HOURS.get(day_name, RSF_HOURS_DEFAULT)

        for hour in range(open_h, close_h):
            ts_str = f"{d} {hour:02d}:00"
            rf_pct, mlp_pct = predict(ts_str)
            rows.append({
                "hour_ts":       ts_str,
                "predicted_pct": min(round(mlp_pct), 100),   # MLP (backward compat)
                "rf_pct":        min(round(rf_pct), 100),     # Random Forest
                "computed_at":   computed_at,
            })

    # Upsert in batches of 500 (Supabase limit per request)
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        sb.table("predictions").upsert(batch, on_conflict="hour_ts").execute()
        print(f"[supabase] Upserted predictions {i + 1}–{i + len(batch)} of {len(rows)}")

    print(f"[supabase] Done — {len(rows)} predictions across {days} days")


_BREAK_RANGES = [
    (date(2020, 12, 19), date(2021,  1, 18)),
    (date(2021,  3, 22), date(2021,  3, 26)),
    (date(2021,  5, 14), date(2021,  8, 24)),
    (date(2021, 12, 18), date(2022,  1, 17)),
    (date(2022,  3, 21), date(2022,  3, 25)),
    (date(2022,  5, 13), date(2022,  8, 23)),
    (date(2022, 12, 17), date(2023,  1, 16)),
    (date(2023,  3, 27), date(2023,  3, 31)),
    (date(2023,  5, 12), date(2023,  8, 22)),
    (date(2023, 12, 16), date(2024,  1, 14)),
    (date(2024,  3, 25), date(2024,  3, 29)),
    (date(2024,  5, 10), date(2024,  8, 27)),
    (date(2024, 12, 21), date(2025,  1, 19)),
    (date(2025,  3, 24), date(2025,  3, 28)),
    (date(2025,  5, 16), date(2025,  8, 26)),
    (date(2025, 12, 20), date(2026,  1, 20)),
    (date(2026,  3, 23), date(2026,  3, 27)),
    (date(2026,  5, 15), date(2026,  8, 25)),
]


def push_weekly_averages(conn):
    df = pd.read_sql_query("SELECT timestamp, percent_full FROM capacity_log", conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day_of_week"] = df["timestamp"].dt.strftime("%A")
    df["hour"]        = df["timestamp"].dt.hour
    df["date"]        = df["timestamp"].dt.date

    def is_break(d):
        return any(start <= d <= end for start, end in _BREAK_RANGES)

    df["is_break"] = df["date"].apply(is_break)

    now = datetime.now(PT).date()
    date_ranges = {
        "last_week":      now - timedelta(days=7),
        "last_month":     now - timedelta(days=30),
        "last_6_months":  now - timedelta(days=180),
        "last_year":      now - timedelta(days=365),
        "all_time":       None,
    }

    all_rows = []
    for range_type, cutoff in date_ranges.items():
        for semester_only in [True, False]:
            filtered = df.copy()
            if cutoff is not None:
                filtered = filtered[filtered["date"] >= cutoff]
            if semester_only:
                filtered = filtered[~filtered["is_break"]]

            if filtered.empty:
                continue

            avg = (
                filtered.groupby(["day_of_week", "hour"])["percent_full"]
                .mean()
                .round(1)
                .reset_index()
                .rename(columns={"percent_full": "avg_pct"})
            )
            avg["range_type"]    = range_type
            avg["semester_only"] = semester_only
            all_rows.extend(avg.to_dict(orient="records"))

    # Upsert all combos
    batch_size = 500
    for i in range(0, len(all_rows), batch_size):
        batch = all_rows[i:i + batch_size]
        sb.table("weekly_averages").upsert(
            batch, on_conflict="day_of_week,hour,range_type,semester_only"
        ).execute()
        print(f"[supabase] Upserted weekly_averages {i + 1}–{i + len(batch)} of {len(all_rows)}")

    print(f"[supabase] Done — {len(all_rows)} weekly average rows ({len(date_ranges) * 2} combos)")


def main():
    conn = sqlite3.connect("gym_history.db")
    push_latest_occupancy(conn)

    # Weekly averages only need recomputing once a day
    now = datetime.now(PT)
    if now.hour == 0 or not predictions_exist_for_today():
        push_predictions(days=180)
        push_weekly_averages(conn)
    else:
        print("[supabase] Skipping daily recompute (already done today)")

    conn.close()
    print("[supabase] Done.")


def predictions_exist_for_today():
    today = datetime.now(PT).strftime("%Y-%m-%d")
    result = sb.table("predictions").select("hour_ts").eq("hour_ts", f"{today} 12:00").execute()
    return len(result.data) > 0


if __name__ == "__main__":
    main()
