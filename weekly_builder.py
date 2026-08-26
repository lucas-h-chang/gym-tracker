"""
weekly_builder.py — compute weekly pattern averages → Supabase weekly_averages table.
Runs daily at midnight PT via daily.yml.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client

from academic_calendar import (
    is_summer_day,
    get_open_hours,
    is_semester_day,
    is_closed_day,
)
from supabase_io import paginated_fetch

PT  = ZoneInfo("America/Los_Angeles")
now = datetime.now(PT)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

# SPRING_BREAKS/SUMMER_RANGES/is_summer_day/get_open_hours/is_semester_day live
# in academic_calendar.py (consolidated 2026-07-21 — see CLAUDE.md). SPRING_BREAKS
# is now the same SPRING_BREAK_RANGES date-tuple list used elsewhere in the
# codebase (aliased so the rest of this file's `pd.Timestamp(start)` calls below
# are unchanged — pd.Timestamp accepts a date object exactly as it accepted the
# original ISO strings).

RANGE_TYPE_MAP = {
    'last_week':       timedelta(days=7),
    'last_month':      timedelta(days=30),
    'last_6_months':   timedelta(days=182),
    'last_year':       timedelta(days=365),
    'all_time':        None,
    'this_semester':   None,  # computed dynamically below
}

BATCH_SIZE = 500


def get_semester_start(today):
    d = today
    while is_semester_day(d):
        d -= timedelta(days=1)
    return d + timedelta(days=1)


def fetch_all_history():
    """Fetch all capacity_log from Supabase (paginated).

    sensor_ok is selected so main() can drop readings taken while the RSF's
    counter was dead (migration 008). Unlike the ML path, this builder has no
    people_count floor -- it averages raw percent_full -- so a stalled day
    would otherwise sink the "vs usual <Day>s" baseline for that weekday
    permanently. Filtering happens in pandas rather than here because
    paginated_fetch has no .eq() hook (see its docstring).
    """
    return paginated_fetch(
        sb, "capacity_log", "timestamp,percent_full,sensor_ok", order="timestamp"
    )


DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def period_type(d):
    """Partition a date into 'summer' / 'semester' / 'break' for the Today
    comparison card ("compared to usual <Day>s").

    is_summer_day peels off the summer session first; the remaining days split
    on is_semester_day — in-session ('semester') vs a winter/spring/holiday
    break ('break'). These three must NOT be pooled: per academic_calendar.py,
    a winter-break Tuesday 7pm runs ~6% while a summer-break Tuesday 7pm runs
    ~78%, so a single "usual" baseline would match none of them.
    """
    if is_summer_day(d):
        return 'summer'
    if not is_semester_day(d):
        return 'break'
    return 'semester'


def _emit_day_records(filtered, range_type, semester_only, records):
    """Bucket `filtered` rows into per-day, per-15-min-slot averages (plus the
    synthetic closing-zero row) and append them to `records`. Shared by both
    the windowed range_types and the period-typed comparison slices."""
    # A slice with no rows publishes nothing. Falling through would still emit
    # one synthetic 0%-at-close point per weekday, i.e. a fabricated flat
    # baseline that reads on the chart as "the gym is always empty", which is
    # worse than the frontend's own no-data state for that range.
    if filtered.empty:
        print(f"  no rows for range_type={range_type}, semester_only={semester_only}"
              ", publishing no baseline for it")
        return

    # Per-row open/close bounds based on each row's date — summer dates close
    # earlier than academic-year dates, so filter row-by-row.
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

        # Use the actual max closing time from the data so summer-only ranges
        # place the 0% at 20:00 (summer close) instead of 23:00 (academic
        # close), which caused a long diagonal tail on the chart.
        chart_close = int(day_data['_close_h'].max()) if len(day_data) > 0 else academic_close

        # Drop any bin that rounded up to the close hour (e.g. a 22:58 reading
        # binning to 23.0) so the synthetic close-zero we add below doesn't
        # collide with it on the primary key.
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


def compute_weekly_averages(df):
    df['day_of_week']  = df['timestamp'].dt.day_name()
    df['hour_numeric'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60

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
                # Precise in-session gate: is_semester_day uses the exact
                # academic-calendar break ranges, not month cutoffs, so
                # first-week-of-fall and the semester-boundary days are kept
                # instead of being dropped as "summer/winter". Classify each
                # distinct date once, then map back onto the rows.
                date_only   = range_df['timestamp'].dt.date
                sem_by_date = {dd: is_semester_day(dd) for dd in date_only.unique()}
                # .astype(bool) is load-bearing, not defensive tidying: on an
                # EMPTY range_df, .map() returns an empty float64 Series, which
                # pandas reads as a COLUMN selector rather than a row mask and
                # hands back a zero-column frame, so the failure surfaces
                # later, inside _emit_day_records, pointing at the wrong line.
                # This fired for real on 2026-08-24/25, when the whole
                # this_semester window was Caltopia closure days and main() had
                # already dropped every row in it.
                filtered    = range_df[date_only.map(sem_by_date).astype(bool)]
            else:
                filtered = range_df

            _emit_day_records(filtered, range_type, semester_only, records)

    # Period-typed, all-time comparison slices: "compared to usual <Day>s"
    # picks whichever of these matches TODAY's period, so a summer Thursday is
    # compared against every summer Thursday on record (not against in-session
    # Thursdays). All-time window for a stable, always-populated baseline —
    # year-over-year drift (~8pp) sits inside the reading noise floor (~15pp),
    # while a this-year-only split would leave many period×weekday×hour cells
    # near-empty. semester_only is not meaningful here (the period IS the
    # filter), so it's stored False.
    ptype_by_date = {dd: period_type(dd) for dd in df['timestamp'].dt.date.unique()}
    period_series = df['timestamp'].dt.date.map(ptype_by_date)
    for ptype, range_type in [('summer',   'all_summers'),
                              ('semester', 'all_semesters'),
                              ('break',    'all_breaks')]:
        _emit_day_records(df[period_series == ptype], range_type, False, records)

    return records


def main():
    print("Fetching all history from Supabase...")
    rows = fetch_all_history()
    print(f"  {len(rows):,} rows loaded")

    df = pd.DataFrame(rows)

    # Drop hardware-outage readings. Tolerates the column being absent so this
    # still runs against a database where 008 hasn't been applied yet.
    if 'sensor_ok' in df.columns:
        before = len(df)
        df = df[df['sensor_ok'] != False].drop(columns=['sensor_ok'])  # noqa: E712
        if before != len(df):
            print(f"  dropped {before - len(df):,} rows flagged sensor_ok = false")

    # Keep a naive copy for timezone-naive cutoff comparisons
    df['timestamp']       = pd.to_datetime(df['timestamp'], format='ISO8601').dt.tz_convert(PT)
    df['timestamp_naive'] = df['timestamp'].dt.tz_localize(None)

    # Drop full-facility closure days (Caltopia). Density keeps reporting 0-2
    # people while the building is shut, and those readings would otherwise
    # become part of "usual Sundays" for every year on record.
    #
    # This cannot ride on academic_calendar.get_open_hours()' empty-interval
    # short-circuit the way the other gates do: _emit_day_records() re-derives
    # the open/close bounds inline with numpy for speed, so it never calls that
    # function. main() is the one choke point every range_type flows through,
    # so the exclusion goes here, before any slice is cut — and after the
    # timestamp parse above, so the 300k ISO strings are only parsed once.
    ts_dates       = df['timestamp'].dt.date
    closed_by_date = {d: is_closed_day(d) for d in ts_dates.unique()}
    # .astype(bool) for the same reason as in compute_weekly_averages below.
    closed_mask    = ts_dates.map(closed_by_date).astype(bool)
    if closed_mask.any():
        print(f"  dropped {int(closed_mask.sum()):,} rows on RSF closure days")
        df = df[~closed_mask]

    print("Computing weekly averages...")
    records = compute_weekly_averages(df)
    print(f"  {len(records):,} records computed")

    # Truncate-then-insert. This intentionally does NOT use upsert(on_conflict=...):
    # that would require (day_of_week, hour_slot, range_type, semester_only) to be a
    # declared unique constraint in Postgres, which is not defined anywhere in this
    # repo and could not be verified against the live DB. If that constraint is
    # missing, an upsert errors at runtime and this daily job silently stops updating
    # (weekly_averages is not covered by freshness.yml). The brief empty-table window
    # at midnight is the accepted cost of not depending on an unverified constraint.
    # To switch to upsert-then-purge, first confirm/add that unique constraint.
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
