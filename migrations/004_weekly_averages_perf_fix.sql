-- 004_weekly_averages_perf_fix.sql
--
-- Fixes: `weekly_averages` started returning Postgres error 57014
-- ("canceling statement due to statement timeout") on every request as of
-- 2026-07-23, one day after 003_weekly_averages_view.sql went live.
--
-- Root cause: 003's `weekly_averages` view joined ALL of `capacity_log`
-- (182,924 rows as of this writing, growing ~15k/month) against an 84-row
-- dimension table (6 range_types x 7 weekdays x 2 semester_only) matched on
-- day_of_week ALONE, with the per-range time cutoff applied only AFTER that
-- join. That's a ~12x fan-out to ~2.2M intermediate rows, recomputed from a
-- full unindexed scan on every single page load. It was flagged as the
-- expected failure mode in SPEC_VIEWS_MIGRATION.md's "operational notes"
-- section when the view was written -- it just took a day of data growth to
-- cross the statement timeout.
--
-- Fix, in two independent parts (do both):
--
-- 1. An index on capacity_log(timestamp). Cheap, safe, always correct to
--    have -- capacity_log is append-only and every view/query that touches
--    it filters or orders by timestamp.
--
-- 2. Rewrite `weekly_averages` so the five range types with a real cutoff
--    (last_week/last_month/last_6_months/last_year/this_semester) filter
--    capacity_log on the RAW indexed `timestamp` column (as an index range
--    scan) BEFORE any timezone/day-of-week math runs, instead of computing
--    day-of-week/hour bucketing over the full unfiltered table first and
--    filtering by cutoff afterward. `all_time` still does one full scan --
--    unavoidable, it's supposed to cover all history -- but now it's the
--    only branch that does, instead of every branch effectively paying for
--    it via the shared 12x join.
--
--    The join against the day dimension is also dropped: each capacity_log
--    row already carries its own day_of_week once computed, so matching it
--    against a `days` table was pure overhead. Only `semester_only` (2 rows)
--    still needs a cross join. Fan-out per row goes from 12x to 2x, and only
--    for rows that already survived the per-range cutoff filter.
--
-- Every downstream column, filter, and the synthetic "closing zero" row
-- logic (chart_close / slot_avgs / closing_rows) is UNCHANGED from
-- 003_weekly_averages_view.sql -- this is a performance rewrite of how
-- `matched` gets built, not a behavior change. Re-run RUNBOOK.md's Step 4c
-- spot checks after applying this to confirm output is identical.

create index if not exists idx_capacity_log_timestamp on capacity_log (timestamp);

create or replace view weekly_averages as
with today_pt as (
  select (now() at time zone 'America/Los_Angeles')::date as today_date
),
semester_start as (
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
  -- Still the full 6 x 7 x 2 = 84-row combo space, used below only to
  -- guarantee every combo gets its synthetic closing row -- same role it
  -- played in 003, unrelated to the matched-rows fan-out fix.
  select r.range_type, d.day_of_week, s.semester_only
  from range_defs r cross join days d cross join sems s
),
-- One UNION ALL branch per range type instead of one join against
-- range_defs, so Postgres plans each branch independently: an index range
-- scan on capacity_log(timestamp) for the 5 bounded ranges, one plain scan
-- for all_time. cutoff_ts is a naive PT wall-clock value (matches
-- range_defs above); converting it with `at time zone` turns it into the
-- timestamptz instant that capacity_log.timestamp is actually stored as, so
-- the comparison can use the index created above.
raw_rows as (
  select r.range_type, c.timestamp, c.percent_full
  from range_defs r
  join capacity_log c
    on c.timestamp >= (r.cutoff_ts at time zone 'America/Los_Angeles')
  where r.range_type <> 'all_time'
  union all
  select 'all_time', c.timestamp, c.percent_full
  from capacity_log c
),
base as (
  select
    rr.range_type,
    (rr.timestamp at time zone 'America/Los_Angeles')                      as ts_pt,
    (rr.timestamp at time zone 'America/Los_Angeles')::date                as pt_date,
    trim(to_char(rr.timestamp at time zone 'America/Los_Angeles', 'Day'))  as day_of_week,
    extract(hour   from (rr.timestamp at time zone 'America/Los_Angeles'))
      + extract(minute from (rr.timestamp at time zone 'America/Los_Angeles')) / 60.0
                                                                            as hour_numeric,
    rr.percent_full
  from raw_rows rr
),
base_flagged as (
  select
    b.*,
    is_semester_day(b.pt_date) as is_sem_day,
    (case when b.day_of_week in ('Saturday', 'Sunday') then 8 else 7 end)  as row_open_h,
    (case when b.day_of_week = 'Saturday' then 18
          when is_summer_day(b.pt_date) then 20
          else 23 end)                                                     as row_close_h,
    (round(b.hour_numeric * 4) / 4)::double precision                      as hour_slot
  from base b
),
matched as (
  select
    bf.range_type,
    s.semester_only,
    bf.day_of_week,
    bf.hour_slot,
    bf.percent_full,
    bf.row_close_h
  from base_flagged bf
  cross join sems s
  where (not s.semester_only or bf.is_sem_day)
    and bf.hour_numeric >= bf.row_open_h
    and bf.hour_numeric <  bf.row_close_h
),
chart_close as (
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
  'Live view over capacity_log. See 003_weekly_averages_view.sql for the '
  'line-by-line translation from weekly_builder.py, and '
  '004_weekly_averages_perf_fix.sql for why matched-rows is built via '
  'per-range UNION ALL + an index on capacity_log(timestamp) instead of a '
  'blanket join against the day dimension.';

-- grant/notify are required again -- CREATE OR REPLACE VIEW does not reset
-- existing grants, but run these anyway in case this is applied fresh:
grant select on weekly_averages to anon, authenticated;
notify pgrst, 'reload schema';
