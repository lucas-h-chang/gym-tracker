-- 002_day_profiles_view.sql
--
-- Replaces the `day_profiles` TABLE (populated nightly by
-- day_profiles_builder.py) with a plain, always-fresh VIEW of the same name
-- and same columns, computed live from capacity_log.
--
-- Requires 001_is_semester_day.sql to already be applied.
--
-- Source of truth translated: gym-tracker/day_profiles_builder.py, in
-- particular build_records():
--
--   df['timestamp'] = tz_convert(PT)
--   df['date']      = df['timestamp'].dt.date
--   df['hour_slot'] = ((df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60) * 4).round() / 4
--   df = df[df['date'] <= last_complete]              # last_complete = now.date() - 1 day (PT)
--   grouped = df.groupby(['date', 'hour_slot'])['percent_full'].mean()
--   day_name    = pd.Timestamp(d).day_name()
--   is_semester = is_semester_day(d)
--   avg_pct     = float(...)   # UNROUNDED, for parity with today_builder
--
-- and the DATA_CUTOFF constant (date(2022, 1, 1)) that gates the backfill.
--
-- Columns exposed (must match what today_builder.py::fetch_candidates()
-- selects: "date,hour_slot,avg_pct" plus .eq() filters on "day_name" and
-- "is_semester" -- all five columns below are present):
--   date         -- PT calendar date
--   day_name     -- English weekday, e.g. 'Monday' (trimmed)
--   is_semester  -- boolean, via is_semester_day()
--   hour_slot    -- double precision, quarter-hour bin
--   avg_pct      -- double precision, unrounded mean of percent_full
--
-- Live-view semantics note: unlike the old nightly table, "last_complete"
-- here is evaluated at query time (today's PT date is always excluded, not
-- just as of the last midnight cron run). This is a strict improvement --
-- there is no window where the view is stale relative to a batch job that
-- hasn't run yet -- and does not change the value of any date that Python
-- would have considered "complete".

create or replace view day_profiles as
with base as (
  select
    (timestamp at time zone 'America/Los_Angeles')          as ts_pt,
    (timestamp at time zone 'America/Los_Angeles')::date     as pt_date,
    (round((
      extract(hour   from (timestamp at time zone 'America/Los_Angeles'))
      + extract(minute from (timestamp at time zone 'America/Los_Angeles')) / 60.0
    ) * 4) / 4)::double precision                              as hour_slot,
    percent_full
  from capacity_log
)
select
  pt_date                                       as date,
  trim(to_char(pt_date, 'Day'))                  as day_name,
  is_semester_day(pt_date)                       as is_semester,
  hour_slot,
  avg(percent_full)::double precision            as avg_pct
from base
where pt_date >= date '2022-01-01'                                          -- DATA_CUTOFF
  and pt_date <  (now() at time zone 'America/Los_Angeles')::date            -- exclude today (partial day)
group by pt_date, hour_slot;

comment on view day_profiles is
  'Live replacement for the day_profiles table formerly populated by '
  'day_profiles_builder.py (now in legacy/). Same columns, computed from '
  'capacity_log on every query. See migrations/002_day_profiles_view.sql '
  'for the line-by-line translation notes.';
