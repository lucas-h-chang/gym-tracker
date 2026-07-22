-- 003_weekly_averages_view.sql
--
-- Replaces the `weekly_averages` TABLE (rebuilt nightly, truncate-then-insert,
-- by weekly_builder.py) with a plain, always-fresh VIEW of the same name and
-- same columns, computed live from capacity_log.
--
-- Requires 001_is_semester_day.sql to already be applied.
--
-- Source of truth translated: gym-tracker/weekly_builder.py, in particular
-- compute_weekly_averages() and get_semester_start(). Read that file
-- alongside this one -- the comments below cite the exact Python lines each
-- SQL block reproduces.
--
-- IMPORTANT: unlike day_profiles_builder.py, weekly_builder.py's
-- fetch_all_history() has NO DATA_CUTOFF filter -- it reads the entire
-- capacity_log table, COVID-era 2020-2021 included. This view does the same
-- (no date floor) to match exactly.
--
-- IMPORTANT: weekly_builder.py imports `is_summer_day` from
-- academic_calendar.py, which is keyed off SUMMER_RANGES (the *open-hours*
-- calendar: 2024-2027 only, end dates shifted -3 days from the break
-- calendar) -- NOT off SUMMER_BREAK_RANGES (the *is_semester_day* calendar:
-- 2021-2027). These are two genuinely different lists in the Python source.
-- A local is_summer_day() SQL function using SUMMER_RANGES is defined below
-- for that reason; do not merge it with is_semester_day()'s break ranges.
--
-- Columns exposed (must match weekly_builder.py's six emitted fields):
--   day_of_week    -- English weekday, e.g. 'Monday'
--   hour_slot      -- double precision, quarter-hour bin
--   range_type     -- one of 'last_week','last_month','last_6_months',
--                     'last_year','all_time','this_semester'
--   semester_only  -- boolean
--   avg_pct        -- double precision, rounded to 1 decimal
--
-- ============================================================================
-- WHAT DID NOT TRANSLATE 1:1 -- READ THIS BEFORE TRUSTING THE VIEW
-- ============================================================================
-- Two pieces of weekly_builder.py's output ARE reproduced below (they did
-- turn out to be expressible in a plain view -- see the CTEs), but with one
-- honest caveat each:
--
-- 1. get_semester_start()'s Python is an UNBOUNDED backward walk ("keep
--    stepping back a day at a time until you hit a non-in-session day").
--    SQL has no unbounded loop in a plain view, so `semester_start` below
--    searches back at most 400 days. Every break recurs well within 400 days
--    of any other (longest real gap between break days is ~120 days, August
--    to mid-December), so this produces IDENTICAL results to the Python for
--    every date the academic calendar actually covers (2020-2028, per
--    academic_calendar.py's SEM_STARTS/BREAK_RANGES). It only diverges if
--    the server clock is ever outside that maintained range, at which point
--    the Python version would also degrade (it would walk back indefinitely
--    since is_semester_day() is vacuously true before 2020 / after 2028).
--    Not a translation gap in practice; flagged for completeness.
--
-- 2. Floating-point rounding: Python's round(x, 1) uses round-half-to-even;
--    Postgres round(numeric, 1) uses round-half-away-from-zero. These only
--    differ on an exact x.x5 tie, which a float mean of real percent_full
--    samples essentially never lands on exactly. Cosmetic, not a design gap.
--
-- Everything else -- the six range windows, the semester_only gate, the
-- per-row open/close-hour filtering (including the Saturday/summer/academic
-- distinction), and the synthetic hour_slot=close_hour / avg_pct=0 row per
-- (day_of_week, range_type, semester_only) -- IS implemented below, translated
-- faithfully. See RUNBOOK.md for how to validate this against the old table
-- before dropping it.
-- ============================================================================

create or replace function is_summer_day(d date)
returns boolean
language sql
immutable
as $$
  -- academic_calendar.SUMMER_RANGES (the open-hours calendar -- NOT
  -- SUMMER_BREAK_RANGES; see the file header above).
  select exists (
    select 1
    from (
      values
        (date '2024-05-10', date '2024-08-24'),
        (date '2025-05-16', date '2025-08-23'),
        (date '2026-05-15', date '2026-08-22'),
        (date '2027-05-14', date '2027-08-21')
    ) as ranges(start_date, end_date)
    where d between ranges.start_date and ranges.end_date
  );
$$;

comment on function is_summer_day(date) is
  'Translated from academic_calendar.py::is_summer_day() / SUMMER_RANGES '
  '(open-hours calendar, 2024-2027, distinct from the SUMMER_BREAK_RANGES '
  'used by is_semester_day()). Keep in sync with academic_calendar.py.';


create or replace view weekly_averages as
with today_pt as (
  -- Python: now = datetime.now(PT); all cutoffs below are relative to
  -- now.date(). Re-evaluated on every query (this view has no cron lag).
  select (now() at time zone 'America/Los_Angeles')::date as today_date
),
semester_start as (
  -- SQL equivalent of weekly_builder.get_semester_start(today):
  --   d = today
  --   while is_semester_day(d): d -= 1 day
  --   return d + 1 day
  -- i.e. "the day after the most recent NOT-in-session day on/before today."
  -- Bounded to a 400-day backward search -- see the file header note above.
  select
    coalesce(
      (
        select max(t.today_date - n)
        from today_pt t, generate_series(0, 400) as n
        where not is_semester_day(t.today_date - n)
      ) + 1,
      (select today_date from today_pt) - 400
    ) as start_date
),
range_defs(range_type, cutoff_ts) as (
  -- Python cutoffs dict: (now - timedelta(days=N)).replace(hour=0, minute=0,
  -- second=0, microsecond=0) -- i.e. midnight PT on (today - N days).
  select 'last_week',      (t.today_date - 7)::timestamp   from today_pt t
  union all
  select 'last_month',     (t.today_date - 30)::timestamp  from today_pt t
  union all
  select 'last_6_months',  (t.today_date - 182)::timestamp from today_pt t
  union all
  select 'last_year',      (t.today_date - 365)::timestamp from today_pt t
  union all
  select 'all_time',       null::timestamp
  union all
  -- semester_days = max((today - semester_start).days, 1)
  -- cutoff = midnight PT on (today - semester_days)
  select 'this_semester',
         (t.today_date - greatest((t.today_date - s.start_date), 1))::timestamp
  from today_pt t, semester_start s
),
days(day_of_week) as (
  values ('Monday'), ('Tuesday'), ('Wednesday'), ('Thursday'), ('Friday'), ('Saturday'), ('Sunday')
),
sems(semester_only) as (
  values (true), (false)
),
dims as (
  -- The full 6 range_types x 7 days x 2 semester_only = 84-row combo space
  -- weekly_builder.py's nested loops iterate over -- every combo gets at
  -- least the synthetic closing row below, even with zero matching data.
  select r.range_type, r.cutoff_ts, d.day_of_week, s.semester_only
  from range_defs r cross join days d cross join sems s
),
base as (
  -- df['day_of_week']  = df['timestamp'].dt.day_name()
  -- df['hour_numeric'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
  select
    (c.timestamp at time zone 'America/Los_Angeles')                      as ts_pt,
    (c.timestamp at time zone 'America/Los_Angeles')::date                as pt_date,
    trim(to_char(c.timestamp at time zone 'America/Los_Angeles', 'Day'))  as day_of_week,
    extract(hour   from (c.timestamp at time zone 'America/Los_Angeles'))
      + extract(minute from (c.timestamp at time zone 'America/Los_Angeles')) / 60.0
                                                                            as hour_numeric,
    c.percent_full
  from capacity_log c
),
base_flagged as (
  select
    b.*,
    is_semester_day(b.pt_date) as is_sem_day,
    -- row_open  = 8 if day in (Saturday, Sunday) else 7
    -- row_close = 18 if Saturday, else 20 if is_summer_day(date) else 23
    (case when b.day_of_week in ('Saturday', 'Sunday') then 8 else 7 end)  as row_open_h,
    (case when b.day_of_week = 'Saturday' then 18
          when is_summer_day(b.pt_date) then 20
          else 23 end)                                                     as row_close_h,
    (round(b.hour_numeric * 4) / 4)::double precision                      as hour_slot
  from base b
),
matched as (
  -- filtered = range_df[semester_only gate]; then per-row open/close bound;
  -- then day_data = filtered[day_of_week == day & open <= hour < close].
  -- The day_of_week equality join folds all three steps into one filter.
  select
    dm.range_type,
    dm.semester_only,
    bf.day_of_week,
    bf.hour_slot,
    bf.percent_full,
    bf.row_close_h
  from dims dm
  join base_flagged bf
    on bf.day_of_week = dm.day_of_week
   and (dm.cutoff_ts is null or bf.ts_pt >= dm.cutoff_ts)
   and (not dm.semester_only or bf.is_sem_day)
   and bf.hour_numeric >= bf.row_open_h
   and bf.hour_numeric <  bf.row_close_h
),
chart_close as (
  -- chart_close = int(day_data['_close_h'].max()) if len(day_data) > 0 else
  --               (18 if day == 'Saturday' else 23)
  select
    dm.range_type,
    dm.semester_only,
    dm.day_of_week,
    coalesce(
      max(m.row_close_h),
      case when dm.day_of_week = 'Saturday' then 18 else 23 end
    ) as close_h
  from dims dm
  left join matched m
    on m.range_type    = dm.range_type
   and m.semester_only = dm.semester_only
   and m.day_of_week   = dm.day_of_week
  group by dm.range_type, dm.semester_only, dm.day_of_week
),
slot_avgs as (
  -- avg = day_data.groupby('hour_slot').agg(avg_pct=('percent_full','mean'))
  -- avg = avg[avg['hour_slot'] < chart_close]   -- drops bins that rounded
  --                                                up onto the close hour
  select
    m.range_type,
    m.semester_only,
    m.day_of_week,
    m.hour_slot,
    round(avg(m.percent_full)::numeric, 1)::double precision as avg_pct
  from matched m
  join chart_close cc
    on cc.range_type    = m.range_type
   and cc.semester_only = m.semester_only
   and cc.day_of_week   = m.day_of_week
  where m.hour_slot < cc.close_h
  group by m.range_type, m.semester_only, m.day_of_week, m.hour_slot
),
closing_rows as (
  -- closing = pd.DataFrame([{'hour_slot': float(chart_close), 'avg_pct': 0.0}])
  select
    range_type,
    semester_only,
    day_of_week,
    close_h::double precision as hour_slot,
    0.0::double precision     as avg_pct
  from chart_close
)
select day_of_week, hour_slot, range_type, semester_only, avg_pct from slot_avgs
union all
select day_of_week, hour_slot, range_type, semester_only, avg_pct from closing_rows;

comment on view weekly_averages is
  'Live replacement for the weekly_averages table formerly rebuilt nightly '
  'by weekly_builder.py (now in legacy/). Same columns, computed from '
  'capacity_log on every query. See migrations/003_weekly_averages_view.sql '
  'for the line-by-line translation notes and the two documented, '
  'non-blocking divergences (bounded semester_start search; round() tie '
  'behavior).';
