-- 009_caltopia_closures.sql
--
-- Corrects 008. The 2026-08-23 readings were not a broken sensor: the RSF was
-- CLOSED for Caltopia, so 0-2 people was the accurate count of a shut building.
--
-- 008's schema is all still wanted — a real hardware stall remains a real
-- failure mode and `sensor_ok` is the right guard for it. What was wrong was
-- 008's backfill, which labelled a closure as a hardware fault. Both the flag
-- and the closure calendar exclude the same rows from the same averages, but
-- only one of them is true, and a flag that means two different things is
-- worthless the next time the hardware actually does die.
--
-- So: hand those rows back to `sensor_ok = true`, and exclude them the honest
-- way instead — because the gym was closed.
--
-- SQL equivalent of academic_calendar.py::CLOSURES / is_closed_day(). Copy new
-- tuples from that file; do not retype from memory.

-- ── 1. Undo 008's mislabelling ────────────────────────────────────────────
-- Scoped to the closure window. Any row in it that 008 (or api/scrape.js
-- running before this deploy) marked as a sensor fault was really a closed
-- gym. Step 3 is what actually keeps these out of the baselines.
update capacity_log
   set sensor_ok = true
 where (timestamp at time zone 'America/Los_Angeles')::date
         between date '2026-08-23' and date '2026-08-25'
   and sensor_ok = false;

-- ── 2. The closure calendar ───────────────────────────────────────────────
-- Caltopia is Cal's start-of-year org fair; it takes over the RSF, which shuts
-- for it. Every fall semester begins on a Wednesday, so these are the Sunday
-- and Monday before instruction (start-3, start-2). Fall 2026 also closed the
-- Tuesday — observed, not assumed to recur.
create or replace function is_rsf_closed_day(d date)
returns boolean
language sql
immutable
as $$
  select exists (
    select 1
    from (values
      (date '2021-08-22', date '2021-08-23'),
      (date '2022-08-21', date '2022-08-22'),
      (date '2023-08-20', date '2023-08-21'),
      (date '2024-08-25', date '2024-08-26'),
      (date '2025-08-24', date '2025-08-25'),
      (date '2026-08-23', date '2026-08-25'),
      (date '2027-08-22', date '2027-08-23')
    ) as r(start_date, end_date)
    where d between r.start_date and r.end_date
  );
$$;

comment on function is_rsf_closed_day(date) is
  'SQL mirror of academic_calendar.py::is_closed_day(). Days the RSF is shut '
  'entirely (Caltopia). Density still reports 0-2 people on these days, so '
  'they must be excluded from any occupancy average.';

-- ── 3. day_profiles excludes closure days ─────────────────────────────────
-- 002's view with 008's sensor_ok predicate plus the closure predicate. Both
-- are needed and they are not redundant: sensor_ok covers "the hardware lied",
-- is_rsf_closed_day covers "the building was shut". This matters most for
-- today_builder.py's similarity matcher, which picks historical days that look
-- like today — a closed Sunday is a near-perfect match for a quiet morning and
-- would drag the whole nowcast toward zero.
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
  where sensor_ok                                                            -- 008
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
  and not is_rsf_closed_day(pt_date)                                         -- 009
group by pt_date, hour_slot;

comment on view day_profiles is
  'Live replacement for the day_profiles table formerly populated by '
  'day_profiles_builder.py (now in legacy/). Same columns, computed from '
  'capacity_log on every query, excluding readings flagged sensor_ok = false '
  '(008) and full-facility closure days (009). See migrations/'
  '002_day_profiles_view.sql, 008_sensor_outage_flag.sql, 009_caltopia_closures.sql.';
