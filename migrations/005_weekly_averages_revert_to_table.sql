-- 005_weekly_averages_revert_to_table.sql
--
-- Reverts `weekly_averages` from a live Postgres VIEW (003 + 004) back to a
-- plain TABLE rebuilt nightly by weekly_builder.py (via daily.yml), which is
-- how it worked before the views migration.
--
-- Why (full diagnosis in handoffs/SPEC_WEEKLY_AVERAGES_REDESIGN.md):
--   The 003/004 view recomputes a 6-range x 7-weekday x 2-semester_only
--   aggregation over the ENTIRE capacity_log history on every request. The
--   captured `explain (analyze, buffers)` shows ~13s, an unused timestamp
--   index (the open-hours filter forces a Seq Scan, so the per-range cutoff
--   can't be an index range scan and is applied as a nested-loop join filter
--   dropping 855,905 rows), per-row is_semester_day/is_summer_day calls, a
--   220k->357k row fan-out from overlapping ranges, and a 12 MB on-disk
--   sort. It completes for the 2min dashboard role but 57014s against the
--   REST `anon` role's much shorter statement_timeout -- and only grows
--   ~15k rows/month. It is not fixable as a live per-request view; the work
--   has to move off the request path.
--
--   Precomputing nightly into a table makes reads instant, is immune to
--   statement_timeout no matter how big capacity_log gets, needs zero client
--   changes (web + iOS already `select * from weekly_averages`), and keeps
--   all 84 combos so the (hidden) iOS WeeklyView and the personal artifact
--   still work. day_profiles is unaffected -- it stays a cheap live view.
--
-- This is deliberately the RUNBOOK.md rollback for weekly_averages
-- (drop the view, restore the original renamed table) generalized to also
-- cover the case where the original `weekly_averages_old` table was already
-- dropped. Safe to run once whichever state the DB is in.

-- The current `weekly_averages` is the 003/004 VIEW -- drop it so the name
-- is free for the table.
drop view if exists weekly_averages;

do $$
begin
  if exists (
    select 1 from pg_class
    where relname = 'weekly_averages_old' and relkind = 'r'
  ) then
    -- The views migration only RENAMED the original nightly table to
    -- `weekly_averages_old` (RUNBOOK Step 3); its destructive drop (Step 5)
    -- was never run. Restore it as-is -- this keeps the last snapshot
    -- weekly_builder.py wrote, so the comparison card works immediately,
    -- and tonight's daily run refreshes it.
    raise notice 'weekly_averages_old found: renaming it back to weekly_averages (data preserved)';
    alter table weekly_averages_old rename to weekly_averages;
  else
    -- The original table was already dropped. Recreate it empty with the
    -- exact columns weekly_builder.py inserts and both clients read
    -- (WeeklyRow in Swift / the JS comparison card). The next daily run
    -- (or a manual `python weekly_builder.py`) populates it.
    raise notice 'weekly_averages_old not found: creating a fresh empty weekly_averages table';
    create table weekly_averages (
      day_of_week   text             not null,
      hour_slot     double precision not null,
      range_type    text             not null,
      semester_only boolean          not null,
      avg_pct       double precision not null
    );
  end if;
end $$;

-- Same read access the pre-migration table had, and tell PostgREST to pick
-- up the swapped object (its schema cache won't see it otherwise).
grant select on weekly_averages to anon, authenticated;
notify pgrst, 'reload schema';

-- After running this:
--   1. If it created a fresh empty table, run `python weekly_builder.py`
--      once (or trigger the "Daily Data Build" workflow via workflow_dispatch)
--      so the table isn't empty until midnight PT.
--   2. Confirm the site no longer 57014s:
--        select count(*), count(distinct range_type) from weekly_averages;
--      -- expect ~3000-3500 rows across 6 range_types.
--   3. 003_weekly_averages_view.sql and 004_weekly_averages_perf_fix.sql are
--      now retired -- do not re-apply them.
