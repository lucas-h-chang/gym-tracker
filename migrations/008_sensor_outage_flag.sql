-- 008_sensor_outage_flag.sql
--
-- Adds a per-reading "was the hardware alive?" flag, so a dead RSF counter
-- stops silently averaging into the baselines the site shows.
--
-- WHY A FLAG RATHER THAN DROPPING THE ROWS
-- Two options were considered for the 2026-08-23 outage (Density reported 0-2
-- people all Sunday morning while the hardware was down):
--
--   (a) have api/scrape.js refuse to insert while stalled. Rejected: the stall
--       detector reads capacity_log to find its run of floor readings, so
--       suppressing the writes starves the detector of its own input and the
--       outage "ends" the moment we stop recording it. It also destroys the
--       forensic record of when the sensor died.
--
--   (b) filter <= 5 people at aggregation time, the way
--       curve_model.prepare_slots() already does. Rejected: that retroactively
--       changes what every historical baseline means. The opening slot is
--       legitimately at the floor — all_summers has Sunday 8:00 AM at 1% — so
--       a blanket <=5 filter would delete the open-slot point from the weekly
--       chart for every day in history.
--
-- The flag does neither. Rows are still written (detector stays fed, forensics
-- intact) and only rows actually judged stalled are excluded downstream, so
-- the legitimate 1% opening readings survive untouched.
--
-- Note capacity_log's ML consumers (curve_model.prepare_slots via
-- build_curves.py, and train.py) already drop people_count <= 5 and were never
-- exposed to this outage. The exposed consumers are day_profiles (feeds
-- today_builder.py's similarity matcher) and weekly_builder.py (feeds the
-- "vs usual <Day>s" comparison card) — both of which aggregate raw
-- percent_full with no floor. Those are the two this migration protects.

-- ── 1. The flag ───────────────────────────────────────────────────────────
alter table capacity_log
  add column if not exists sensor_ok boolean not null default true;

-- Partial index: the flagged set is tiny and every query is "exclude the bad
-- ones", so indexing only the false rows keeps this near-free on a ~300k-row
-- table.
create index if not exists capacity_log_sensor_not_ok_idx
  on capacity_log (timestamp)
  where sensor_ok = false;

-- The display cache carries the flag too, so a cache hit in
-- api/live-capacity.js can still tell the clients the sensor is down.
alter table live_capacity
  add column if not exists sensor_ok boolean not null default true;

-- ── 2. Backfill the 2026-08-23 outage ─────────────────────────────────────
-- Scoped to that single PT day and to floor readings only. Deliberately NOT a
-- general "flag every <=5 row ever" — see option (b) above.
update capacity_log
   set sensor_ok = false
 where (timestamp at time zone 'America/Los_Angeles')::date = date '2026-08-23'
   and people_count <= 5;

-- ── 3. day_profiles honours the flag ──────────────────────────────────────
-- Verbatim 002_day_profiles_view.sql with one added predicate. See that file
-- for the line-by-line translation notes from day_profiles_builder.py.
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
group by pt_date, hour_slot;

comment on view day_profiles is
  'Live replacement for the day_profiles table formerly populated by '
  'day_profiles_builder.py (now in legacy/). Same columns, computed from '
  'capacity_log on every query, excluding readings flagged sensor_ok = false. '
  'See migrations/002_day_profiles_view.sql and 008_sensor_outage_flag.sql.';
