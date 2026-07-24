-- 006_lock_down_history.sql
--
-- Stop the public (anon / publishable key) from bulk-downloading the full
-- occupancy history. Today two copies of that history are world-readable
-- through the REST API using the key embedded in docs/index.html:
--   * capacity_log  — ~183k raw 15-min readings back to 2022 (the ML training data)
--   * day_profiles  — ~99k per-day hourly summaries derived from capacity_log
--
-- The website only ever needs the last ~36h of capacity_log (the "actual so far
-- today" line, index.html's `capacity_log?timestamp=gte.<yesterday>T12:00:00Z`)
-- and never reads day_profiles (it is backend-only input to today_builder.py).
--
-- Every server script (scraper.py, today_builder.py, predictions_builder.py,
-- weekly_builder.py, build_curves.py, train.py) connects with SUPABASE_SERVICE_KEY
-- = the service_role, which BYPASSES row-level security and keeps its grants, so
-- scrapes / builds / training are completely unaffected by anything below.
--
-- Net effect for the anon key after this runs:
--   capacity_log  : may read only rows newer than 3 days (all the chart needs)
--   day_profiles  : no access at all
--   predictions / weekly_averages / today_summary : unchanged (derived outputs,
--                   forecasts and aggregates rather than the raw backlog)
--
-- Rollback is at the bottom of this file.

begin;

-- 1) capacity_log: limit anon/authenticated reads to a rolling 3-day window -----

alter table public.capacity_log enable row level security;

-- Remove any pre-existing policies first, so no leftover permissive rule (e.g. a
-- `using (true)` SELECT policy) keeps the full history readable — RLS policies
-- are OR'd together, so a single permissive rule would re-open everything.
do $$
declare pol record;
begin
  for pol in
    select policyname
    from pg_policies
    where schemaname = 'public' and tablename = 'capacity_log'
  loop
    execute format('drop policy %I on public.capacity_log', pol.policyname);
  end loop;
end $$;

-- Public site (anon) may read only the recent window the chart actually uses.
create policy capacity_log_recent_anon
  on public.capacity_log
  for select
  to anon
  using (timestamp > now() - interval '3 days');

-- Mirror it for logged-in users (harmless if you have none today). Keeps the
-- door open for a future authenticated experience without re-exposing history.
create policy capacity_log_recent_auth
  on public.capacity_log
  for select
  to authenticated
  using (timestamp > now() - interval '3 days');

-- Note: no INSERT/UPDATE/DELETE policies are created, so writes by anon/
-- authenticated remain denied (unchanged from today). service_role bypasses RLS.

-- 2) day_profiles: remove public API access entirely (backend-only view) -------

revoke select on public.day_profiles from anon;
revoke select on public.day_profiles from authenticated;
-- anon/authenticated inherit PUBLIC grants, so a blanket grant to PUBLIC would
-- keep the view readable even after the two revokes above. Revoke there too,
-- then re-assert the backend role's access explicitly so today_builder.py (which
-- reads day_profiles as service_role) is guaranteed to keep working.
revoke select on public.day_profiles from public;
grant  select on public.day_profiles to service_role;

commit;

-- PostgREST caches schema + permissions; tell it to reload so these take effect
-- immediately rather than on its next periodic refresh.
notify pgrst, 'reload schema';

-- ─────────────────────────────────────────────────────────────────────────────
-- ROLLBACK (paste and run if you need to undo this):
--
--   begin;
--   -- capacity_log: drop the windowed policies and turn RLS back off, which
--   -- restores the previous "anon can read all rows via its SELECT grant" state.
--   drop policy if exists capacity_log_recent_anon on public.capacity_log;
--   drop policy if exists capacity_log_recent_auth on public.capacity_log;
--   alter table public.capacity_log disable row level security;
--   -- day_profiles: hand read access back to the API roles.
--   grant select on public.day_profiles to anon, authenticated;
--   commit;
--   notify pgrst, 'reload schema';
--
-- (This only restores the prior *permissions*; it does not recreate any custom
-- pre-existing capacity_log policies, since this repo has none on record. If the
-- table had bespoke policies before, recreate them from your own notes.)
-- ─────────────────────────────────────────────────────────────────────────────
