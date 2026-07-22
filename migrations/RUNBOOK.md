RUNBOOK — swap day_profiles / weekly_averages from tables to views
====================================================================

Read this whole document before running anything.
Nobody but you can run this against the live database — I cannot execute SQL
against Supabase, so every step below is something you paste into the
Supabase SQL editor and eyeball the result of, in order.
Do not skip the validation steps.
Do not drop the `_old` tables until you have personally looked at the
comparison queries and they look right.

Everything here is non-destructive until the final "drop table" step, and
even that has a documented rollback below.


Step 0 — before you start
--------------------------

Open these three files in `gym-tracker/migrations/` so you can paste their
contents in the next steps:

- `001_is_semester_day.sql`
- `002_day_profiles_view.sql`
- `003_weekly_averages_view.sql`

You do not need to read them line by line right now (the code comments in
each file explain the translation in detail) — just have them ready to paste.


Step 1 — create the `is_semester_day` function
-------------------------------------------------

In the Supabase SQL editor, paste and run the entire contents of
`001_is_semester_day.sql`.

Sanity check immediately after:

```sql
-- Should be false (during winter break) and true (a normal Tuesday):
select is_semester_day('2026-01-05') as should_be_false,
       is_semester_day('2026-02-10') as should_be_true;
```

If both values come back as expected, continue. If not, stop here — do not
proceed to steps 2/3 with a broken calendar function.


Step 2 — swap `day_profiles`
------------------------------

```sql
alter table day_profiles rename to day_profiles_old;
```

Then paste and run the entire contents of `002_day_profiles_view.sql`
(it starts with `create or replace view day_profiles as ...`).

Grant the same read access the old table had, and tell PostgREST to notice
the new view (Supabase's REST API caches the schema; without this the
website/app will keep 404ing or returning stale results after the swap):

```sql
grant select on day_profiles to anon, authenticated;
notify pgrst, 'reload schema';
```


Step 3 — swap `weekly_averages`
----------------------------------

```sql
alter table weekly_averages rename to weekly_averages_old;
```

Then paste and run the entire contents of `003_weekly_averages_view.sql`
(this also (re)creates a small `is_summer_day(date)` helper function used
only by this view — see the comments at the top of that file for why it is
a different date list than `is_semester_day`'s).

```sql
grant select on weekly_averages to anon, authenticated;
notify pgrst, 'reload schema';
```


Step 4 — validate BEFORE dropping anything
----------------------------------------------

Run all of these and actually look at the numbers. They are meant to catch a
translation mistake before it reaches the live site.

**4a. Row counts** (expect them close; see notes below each query for what
"close" means for that table):

```sql
select 'day_profiles (view)'      as src, count(*) from day_profiles
union all
select 'day_profiles_old (table)' as src, count(*) from day_profiles_old;
```

`day_profiles` should match `day_profiles_old` almost exactly for any date
range up through whenever `day_profiles_old` was last built. The view can
only be *ahead* of the old table (it includes yesterday even if the nightly
job hadn't run yet today), never behind, so if the view has strictly more
rows than the old table that's expected, not a bug — if it has fewer, that's
a real problem.

```sql
select 'weekly_averages (view)'      as src, count(*) from weekly_averages
union all
select 'weekly_averages_old (table)' as src, count(*) from weekly_averages_old;
```

These should be in the same ballpark (a handful of rows apart is normal —
see 4c) but not necessarily identical, because the view's six range windows
are computed against the *live* "now," while `weekly_averages_old` is a
snapshot from whenever `weekly_builder.py` last ran. A large gap (dozens of
rows, or an entire `range_type`/day missing) is a real problem.

**4b. Spot-check `day_profiles` values** — pick a handful of recent
weekdays and compare the view against the old table directly:

```sql
select * from day_profiles     where date = '2026-07-15' order by hour_slot;
select * from day_profiles_old where date = '2026-07-15' order by hour_slot;
```

`avg_pct` should match to several decimal places (both are the unrounded
mean of `percent_full` for that date/hour_slot). `day_name` and
`is_semester` should match exactly.

Also confirm weekday spelling/casing (a locale gotcha called out in
`002_day_profiles_view.sql`):

```sql
select distinct day_name from day_profiles order by 1;
-- expect exactly: Friday, Monday, Saturday, Sunday, Thursday, Tuesday, Wednesday
```

**4c. Spot-check `weekly_averages` values** — pick a few
(day_of_week, range_type, semester_only) combos you can reason about:

```sql
select * from weekly_averages
where day_of_week = 'Monday' and range_type = 'last_month' and semester_only = false
order by hour_slot;

select * from weekly_averages_old
where day_of_week = 'Monday' and range_type = 'last_month' and semester_only = false
order by hour_slot;
```

Things to check by eye:
- The curve shapes should look the same (same rough peak time/height).
- `avg_pct` values should be close (small drift is expected/normal — the
  view's cutoff is "30 days before right now," the old table's cutoff was
  "30 days before whenever the nightly job last ran," so the exact set of
  underlying rows differs by up to a day).
- The LAST row for each (day_of_week, range_type, semester_only) combo
  should have `avg_pct = 0` and `hour_slot` equal to that day's closing hour
  (18 for Saturday, 20 or 23 for other days depending on summer/academic —
  see `003_weekly_averages_view.sql`'s comments). This is the synthetic
  "closing zero" row; if it's missing or at the wrong hour for a
  summer-heavy vs. academic-heavy range, that's the piece most likely to
  have a translation bug — look closely here.
- Try this for at least one Saturday row and one summer-inclusive
  `range_type` (e.g. `last_6_months` or `all_time`) since those exercise the
  close-hour branching the hardest.

If anything in 4a-4c looks wrong, **do not drop the old tables.** Go to the
rollback section below, and treat the migration as not-yet-safe to ship.


Step 5 — drop the old tables (only after Step 4 looks right)
-------------------------------------------------------------

```sql
drop table day_profiles_old;
drop table weekly_averages_old;
```

There is no undo past this point (short of a Supabase point-in-time restore
or backup). Only run this once you've actually eyeballed Step 4's output
and it looked correct — not just "the query ran without an error."


Rollback (if something looks wrong before Step 5)
---------------------------------------------------

Per view, independently:

```sql
-- day_profiles
drop view day_profiles;
alter table day_profiles_old rename to day_profiles;

-- weekly_averages
drop view weekly_averages;
alter table weekly_averages_old rename to weekly_averages;
```

This puts the original table back exactly as it was (nothing was deleted),
and `today_builder.py` / the daily cron will keep working against the
restored table. `weekly_builder.py` and `day_profiles_builder.py` (now in
`gym-tracker/legacy/`) are still present and importable if you need to
temporarily move them back into `daily.yml` while you debug the SQL — the
translation notes in `legacy/README.md` point back to the exact view files
to re-diff against.

The `is_semester_day` and `is_summer_day` functions are harmless to leave in
place even after a rollback (nothing else depends on them existing, and
nothing breaks if they're there unused).


What I could not verify myself
----------------------------------

I cannot connect to Supabase or run SQL against your database from here, so
none of the SQL in `migrations/` has been executed against real data. I
read it back against `academic_calendar.py`, `day_profiles_builder.py`, and
`weekly_builder.py` line by line and I'm confident in the translation, but
Step 4 above is not optional — it is the actual test.
