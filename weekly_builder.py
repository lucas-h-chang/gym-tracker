"""
weekly_builder.py — compute weekly pattern averages → Supabase weekly_averages table.
Runs daily at midnight PT via daily.yml.
"""
import os
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client

PT  = ZoneInfo("America/Los_Angeles")
now = datetime.now(PT)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

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

SUMMER_RANGES = [
    (date(2024, 5, 10), date(2024, 8, 24)),
    (date(2025, 5, 16), date(2025, 8, 23)),
    (date(2026, 5, 15), date(2026, 8, 22)),
    (date(2027, 5, 14), date(2027, 8, 21)),
]

RANGE_TYPE_MAP = {
    'last_week':       timedelta(days=7),
    'last_month':      timedelta(days=30),
    'last_6_months':   timedelta(days=182),
    'last_year':       timedelta(days=365),
    'all_time':        None,
    'this_semester':   None,  # computed dynamically below
}

BATCH_SIZE = 500


def is_summer_day(d):
    return any(s <= d <= e for s, e in SUMMER_RANGES)


def get_open_hours(day_name, d):
    summer = is_summer_day(d)
    if day_name == 'Saturday':
        return 8, 18
    if day_name == 'Sunday':
        return 8, (20 if summer else 23)
    return 7, (20 if summer else 23)


def is_semester_day(d):
    month, dom = d.month, d.day
    is_summer = month in [6, 7, 8]
    is_winter = (month == 12 and dom >= 16) or (month == 1 and dom <= 12)
    is_sb = any(
        pd.Timestamp(s).date() <= d <= pd.Timestamp(e).date()
        for s, e in SPRING_BREAKS
    )
    return not (is_summer or is_winter or is_sb)


def get_semester_start(today):
    d = today
    while is_semester_day(d):
        d -= timedelta(days=1)
    return d + timedelta(days=1)


def fetch_all_history():
    """Fetch all capacity_log from Supabase (paginated)."""
    BATCH  = 9000
    offset = 0
    rows   = []
    while True:
        batch = (
            sb.table("capacity_log")
            .select("timestamp,percent_full")
            .range(offset, offset + BATCH - 1)
            .order("timestamp")
            .execute()
            .data
        )
        rows.extend(batch)
        if len(batch) < BATCH:
            break
        offset += BATCH
        print(f"  Fetched {len(rows):,} rows...")
    return rows


def compute_weekly_averages(df):
    df['day_of_week']  = df['timestamp'].dt.day_name()
    df['hour_numeric'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60

    DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    semester_start = get_semester_start(now.date())
    semester_days  = max((now.date() - semester_start).days, 1)

    cutoffs = {
        'last_week':     (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0),
        'last_month':    (now - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0),
        'last_6_months': (now - timedelta(days=182)).replace(hour=0, minute=0, second=0, microsecond=0),
        'last_year':     (now - timedelta(days=365)).replace(hour=0, minute=0, second=0, microsecond=0),
        'all_time':      None,
        'this_semester': (now - timedelta(days=semester_days)).replace(hour=0, minute=0, second=0, microsecond=0),
    }

    records = []

    for range_type, cutoff in cutoffs.items():
        if cutoff is not None:
            # Strip timezone from cutoff for comparison with naive timestamps
            cutoff_naive = cutoff.replace(tzinfo=None)
            range_df = df[df['timestamp_naive'] >= cutoff_naive].copy()
        else:
            range_df = df.copy()

        for semester_only in [True, False]:
            if semester_only:
                month     = range_df['timestamp'].dt.month
                dom       = range_df['timestamp'].dt.day
                date_only = range_df['timestamp'].dt.date

                is_summer = month.isin([6, 7, 8])
                is_winter = ((month == 12) & (dom >= 16)) | ((month == 1) & (dom <= 12))
                is_sb     = np.zeros(len(range_df), dtype=bool)
                for start, end in SPRING_BREAKS:
                    is_sb |= (
                        (date_only >= pd.Timestamp(start).date()) &
                        (date_only <= pd.Timestamp(end).date())
                    )
                filtered = range_df[~(is_summer | is_winter | is_sb)]
            else:
                filtered = range_df

            # Per-row open/close bounds based on each row's date — summer dates
            # close earlier than academic-year dates, so filter row-by-row.
            row_dates  = filtered['timestamp'].dt.date
            row_summer = row_dates.apply(is_summer_day).to_numpy()
            row_days   = filtered['day_of_week'].to_numpy()
            row_open   = np.where(np.isin(row_days, ['Saturday', 'Sunday']), 8, 7)
            row_close  = np.where(
                row_days == 'Saturday', 18,
                np.where(row_summer, 20, 23),
            )
            filtered = filtered.assign(_open_h=row_open, _close_h=row_close)

            for day in DAYS:
                academic_close = 18 if day == 'Saturday' else 23
                day_data = filtered[
                    (filtered['day_of_week'] == day) &
                    (filtered['hour_numeric'] >= filtered['_open_h']) &
                    (filtered['hour_numeric'] <  filtered['_close_h'])
                ].copy()

                day_data['hour_slot'] = (day_data['hour_numeric'] * 4).round() / 4

                avg = day_data.groupby('hour_slot').agg(
                    avg_pct=('percent_full', 'mean'),
                ).reset_index()

                # Use the actual max closing time from the data so summer-only
                # ranges place the 0% at 20:00 (summer close) instead of 23:00
                # (academic close), which caused a long diagonal tail on the chart.
                chart_close = int(day_data['_close_h'].max()) if len(day_data) > 0 else academic_close

                # Drop any bin that rounded up to the close hour (e.g. a 22:58
                # reading binning to 23.0) so the synthetic close-zero we add
                # below doesn't collide with it on the primary key.
                avg = avg[avg['hour_slot'] < chart_close]
                closing = pd.DataFrame([{'hour_slot': float(chart_close), 'avg_pct': 0.0}])
                avg     = pd.concat([avg, closing], ignore_index=True)
                avg     = avg.sort_values('hour_slot')

                for _, row in avg.iterrows():
                    records.append({
                        'day_of_week':   day,
                        'hour_slot':     float(row['hour_slot']),
                        'range_type':    range_type,
                        'semester_only': semester_only,
                        'avg_pct':       round(float(row['avg_pct']), 1),
                    })

    return records


def main():
    print("Fetching all history from Supabase...")
    rows = fetch_all_history()
    print(f"  {len(rows):,} rows loaded")

    df = pd.DataFrame(rows)
    # Keep a naive copy for timezone-naive cutoff comparisons
    df['timestamp']       = pd.to_datetime(df['timestamp'], format='ISO8601').dt.tz_convert(PT)
    df['timestamp_naive'] = df['timestamp'].dt.tz_localize(None)

    print("Computing weekly averages...")
    records = compute_weekly_averages(df)
    print(f"  {len(records):,} records computed")

    print("Truncating weekly_averages table...")
    sb.table("weekly_averages").delete().neq("day_of_week", "").execute()

    print("Inserting weekly averages...")
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        sb.table("weekly_averages").insert(batch).execute()
        print(f"  Inserted {min(i + BATCH_SIZE, len(records))}/{len(records)}")

    print(f"[{now.isoformat()}] weekly_averages updated: {len(records)} rows")


if __name__ == "__main__":
    main()
