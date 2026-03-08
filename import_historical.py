import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

# ==========================================
# WHAT THIS SCRIPT DOES
# ==========================================
# Reads the historical RSF data file (rsfdata.txt) and loads it into
# our gym_history.db database so it can be used by the app and ML model.
#
# The file has ~279,000 rows covering Nov 2020 – Mar 2026 at 10-minute intervals.
# Format: timestamp_UTC, count, min_in_interval, max_in_interval
#
# Run this script ONCE locally. After that, the DB is the source of truth.

HISTORICAL_FILE = "/Users/lucaschang/Downloads/rsfdata.txt"
DB_FILE = "gym_history.db"
MAX_CAPACITY = 150
PACIFIC = ZoneInfo("America/Los_Angeles")

# RSF open hours (Pacific time) by day of week
# We use these to filter out closed-hours noise and staff-only readings
RSF_HOURS = {
    0: (7, 23),   # Monday
    1: (7, 23),   # Tuesday
    2: (7, 23),   # Wednesday
    3: (7, 23),   # Thursday
    4: (7, 23),   # Friday
    5: (8, 18),   # Saturday
    6: (8, 23),   # Sunday
}

def is_rsf_open(dt_pacific):
    """Return True if the gym is open at the given Pacific datetime."""
    day = dt_pacific.weekday()  # 0=Monday, 6=Sunday
    hour = dt_pacific.hour
    open_hour, close_hour = RSF_HOURS[day]
    return open_hour <= hour < close_hour


def setup_db(conn):
    """
    Ensure the capacity_log table exists and has a UNIQUE constraint on timestamp.
    The UNIQUE constraint is what makes INSERT OR IGNORE work — if a row with the
    same timestamp already exists (from our scraper), it gets silently skipped.
    """
    cursor = conn.cursor()

    # Create table if it doesn't exist (same schema as before)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS capacity_log (
            timestamp TEXT,
            people_count INTEGER,
            percent_full INTEGER
        )
    ''')

    # Add a unique index on timestamp if it doesn't already exist.
    # This is what prevents duplicate rows when we import overlapping data.
    cursor.execute('''
        CREATE UNIQUE INDEX IF NOT EXISTS idx_timestamp_unique
        ON capacity_log (timestamp)
    ''')

    conn.commit()


def main():
    conn = sqlite3.connect(DB_FILE)
    setup_db(conn)
    cursor = conn.cursor()

    imported = 0
    skipped_hours = 0   # outside RSF open hours
    skipped_dupes = 0   # already in DB (overlap with scraper data)
    errors = 0

    print(f"Reading {HISTORICAL_FILE}...")

    with open(HISTORICAL_FILE, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # ---- PARSE THE LINE ----
            # Each line looks like: 2022-12-11T16:00:00+00:00,1,1,24
            # We only need the first two columns (timestamp and count).
            try:
                parts = line.split(",")
                timestamp_utc_str = parts[0]
                count = int(parts[1])
            except (IndexError, ValueError):
                errors += 1
                continue

            # ---- CONVERT UTC → PACIFIC ----
            # The timestamps in the file are UTC (the +00:00 at the end).
            # ZoneInfo("America/Los_Angeles") automatically handles
            # PST (UTC-8, Nov–Mar) vs PDT (UTC-7, Mar–Nov) — no manual math needed.
            try:
                dt_utc = datetime.fromisoformat(timestamp_utc_str)
                dt_pacific = dt_utc.astimezone(PACIFIC)
            except ValueError:
                errors += 1
                continue

            # ---- FILTER: RSF OPEN HOURS ONLY ----
            # Rows outside open hours are either the gym being closed, a staff member,
            # or sensor noise (like the count=1 readings we saw at 7 AM).
            if not is_rsf_open(dt_pacific):
                skipped_hours += 1
                continue

            # ---- CALCULATE PERCENT FULL ----
            # Our scraper uses a max capacity of 150, so we match that here.
            percent_full = int((count / MAX_CAPACITY) * 100)

            # Cap at 100% in case of sensor overcounts (we saw max=154 in the data)
            percent_full = min(percent_full, 100)

            # ---- FORMAT TIMESTAMP ----
            # Store as Pacific time string, matching the format our scraper uses:
            # "YYYY-MM-DD HH:MM:SS"
            timestamp_str = dt_pacific.strftime("%Y-%m-%d %H:%M:%S")

            # ---- INSERT INTO DB ----
            # INSERT OR IGNORE means: if a row with this exact timestamp already
            # exists (e.g. from our scraper), skip it silently. No error, no duplicate.
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO capacity_log (timestamp, people_count, percent_full)
                    VALUES (?, ?, ?)
                ''', (timestamp_str, count, percent_full))

                if cursor.rowcount == 1:
                    imported += 1
                else:
                    skipped_dupes += 1
            except sqlite3.Error as e:
                errors += 1
                continue

            # Commit in batches of 10,000 to avoid holding everything in memory
            if imported % 10000 == 0 and imported > 0:
                conn.commit()
                print(f"  {imported:,} rows imported so far...")

    # Final commit for the last batch
    conn.commit()
    conn.close()

    # ---- SUMMARY ----
    print("\n=== Import Complete ===")
    print(f"  Imported:              {imported:,} rows")
    print(f"  Skipped (closed hrs):  {skipped_hours:,} rows")
    print(f"  Skipped (duplicates):  {skipped_dupes:,} rows")
    print(f"  Errors:                {errors:,} rows")
    print(f"  Total lines processed: {line_num:,}")

    # Show the date range now in the DB
    conn2 = sqlite3.connect(DB_FILE)
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM capacity_log")
    min_ts, max_ts, total = cursor2.fetchone()
    conn2.close()

    print(f"\n=== Database Summary ===")
    print(f"  Total rows in DB: {total:,}")
    print(f"  Earliest entry:   {min_ts}")
    print(f"  Latest entry:     {max_ts}")


if __name__ == "__main__":
    main()
