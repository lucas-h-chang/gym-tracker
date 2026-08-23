-- 007_weekly_averages_anon_read.sql
--
-- Restores public (anon / publishable key) read access to `weekly_averages`,
-- which powers the "compared to usual <Day>s" card on the website and in
-- RSFApp2.0's TodayView.
--
-- Symptom: the card renders "—" instead of a verdict, on every day and every
-- period type.
--
-- Diagnosis: the nightly build is healthy. The 2026-08-13 "Daily Data Build"
-- run logged `weekly_averages updated: 4547 rows`, written by weekly_builder.py
-- as service_role. But the same table read with the anon key returns HTTP 200
-- with an empty array and `content-range: */0`. A 200-with-zero-rows (rather
-- than the 401 that `day_profiles` returns after 006 revoked its grant) is the
-- signature of row-level security being ENABLED on the table with no SELECT
-- policy granting anon anything: the table grant still exists, so PostgREST
-- authorizes the request, then RLS filters every row away. index.html's
-- weekly_averages fetch therefore succeeds, `data.weekly` ends up `{}`,
-- computeDayComparisonAt() finds no baseline, and the card falls through to
-- its "—" branch.
--
-- 006_lock_down_history.sql deliberately touched only capacity_log (windowed
-- RLS) and day_profiles (grant revoked) and explicitly left weekly_averages
-- unchanged, so RLS was turned on here out of band — most likely by accepting
-- the Supabase dashboard's "RLS disabled in public schema" security-advisor
-- suggestion, which enables RLS without creating any policy.
--
-- Fix: keep RLS on (so the advisor stays quiet and the table matches the rest
-- of the schema) and add the explicit read-everything SELECT policy that
-- matches what this table is: a fully derived, non-sensitive aggregate that
-- both public clients already read in full via `select *`. No INSERT/UPDATE/
-- DELETE policies are created, so anon still cannot write; weekly_builder.py
-- connects as service_role and bypasses RLS either way.

begin;

alter table public.weekly_averages enable row level security;

-- Drop any pre-existing policies first so re-running this can't stack
-- duplicates, and so a leftover restrictive rule can't keep rows hidden.
do $$
declare pol record;
begin
  for pol in
    select policyname
    from pg_policies
    where schemaname = 'public' and tablename = 'weekly_averages'
  loop
    execute format('drop policy %I on public.weekly_averages', pol.policyname);
  end loop;
end $$;

create policy weekly_averages_read_anon
  on public.weekly_averages
  for select
  to anon
  using (true);

create policy weekly_averages_read_auth
  on public.weekly_averages
  for select
  to authenticated
  using (true);

-- The table grant should already be there from 005, but re-assert it so this
-- file is sufficient on its own.
grant select on public.weekly_averages to anon, authenticated;

commit;

-- PostgREST caches schema + permissions; tell it to reload so this takes
-- effect immediately rather than on its next periodic refresh.
notify pgrst, 'reload schema';

-- Verify (expect ~4500 rows across 9 range_types, including all_summers /
-- all_semesters / all_breaks):
--   select count(*), count(distinct range_type) from weekly_averages;
-- and from a terminal, with the anon key, expect a non-empty array:
--   curl -s "$SUPABASE_URL/rest/v1/weekly_averages?select=avg_pct&day_of_week=eq.Thursday&range_type=eq.all_summers&limit=3" \
--     -H "apikey: <anon key>" -H "Authorization: Bearer <anon key>"
--
-- ─────────────────────────────────────────────────────────────────────────────
-- ROLLBACK (paste and run if you need to undo this):
--
--   begin;
--   drop policy if exists weekly_averages_read_anon on public.weekly_averages;
--   drop policy if exists weekly_averages_read_auth on public.weekly_averages;
--   alter table public.weekly_averages disable row level security;
--   commit;
--   notify pgrst, 'reload schema';
--
-- (Disabling RLS also restores public reads, since the table grant remains —
-- it just re-triggers the dashboard's security-advisor warning.)
-- ─────────────────────────────────────────────────────────────────────────────
