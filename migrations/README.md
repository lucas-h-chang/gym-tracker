# migrations/

Applied-state ledger for the Supabase schema. **Migrations are not tracked by any
tool** — there is no `supabase migration` state table here. Each file is pasted into
the Supabase SQL editor by hand, so this ledger is the only record of what actually
ran.

Keep it accurate. Add a row when you add a file, and update the row when you apply it.

## Status

All twelve were **verified applied on 2026-09-09** by probing the live database
through the public REST API. The "verified by" column is the check that was run, so
anyone can repeat it.

| # | File | State | Verified by |
|---|------|-------|-------------|
| 001 | `is_semester_day.sql` | ✅ applied | `rpc/is_semester_day('2026-02-10')` → `true` |
| 002 | `day_profiles_view.sql` | ✅ applied | `day_profiles` responds `42501 permission denied for view` (exists, backend-only) |
| 003 | `weekly_averages_view.sql` | ⚪ applied, then reverted | Superseded by 005. Its `is_summer_day()` SQL function still exists but nothing live reads it |
| 004 | `weekly_averages_perf_fix.sql` | ⚪ applied, then reverted | Superseded by 005 |
| 005 | `weekly_averages_revert_to_table.sql` | ✅ applied | `weekly_averages` returns rows to anon; it is a table again |
| 006 | `lock_down_history.sql` | ✅ applied | anon reads `capacity_log` but only a ~3-day window; full history needs the service key |
| 007 | `weekly_averages_anon_read.sql` | ✅ applied | anon `select` on `weekly_averages` returns 200 |
| 008 | `sensor_outage_flag.sql` | ✅ applied | `capacity_log` rows carry `sensor_ok` |
| 009 | `caltopia_closures.sql` | ✅ applied | `rpc/is_rsf_closed_day('2026-08-23')` → `true` |
| 010 | `holiday_closures.sql` | ✅ applied | `is_rsf_closed_day` → `true` for 2026-11-26, 2026-12-24, 2026-12-25, 2027-01-01 |
| 011 | `device_tokens_anon_upsert.sql` | ✅ applied | `device_tokens` responds `42501 permission denied` (exists, backend-only) |
| 012 | `prediction_snapshots.sql` | ✅ applied | `prediction_snapshots` responds `42501 permission denied` (exists, backend-only) |

> `42501 permission denied` means the object **exists** and is locked down.
> A missing object returns `42P01 relation does not exist`. That difference is what
> makes this ledger checkable from outside the database.

## Calendar parity

`009`/`010` define `is_rsf_closed_day()` and `001` defines `is_semester_day()`. These
are **hand-written SQL mirrors of `academic_calendar.py`** and are the one set of
mirrors `sync_calendar.py` does **not** generate, because applied migrations are
history and must not be rewritten. So adding a new academic year needs a new
migration by hand, on top of the one-file Python edit.

Parity was spot-checked on 2026-09-09 across five dates and SQL matched Python on all
of them. There is no automated test for this, which is the gap `test_calendar_mirrors.py`
closes for every other mirror.

## Running a new migration

Paste the file into the Supabase SQL editor and run it, then update this ledger.

Note the role timeout difference: the SQL editor allows ~2 minutes, but the REST anon
role is capped near 3 seconds. A query that succeeds in the editor can still return
`57014` on the live site. That is exactly how `003`/`004` failed, and why
`weekly_averages` is a nightly table instead of a view.

## `RUNBOOK.md`

Historical. It is the step-by-step procedure for one specific job — the 2026-07 swap
of `day_profiles` and `weekly_averages` from tables to views — and that job is done
(and half of it was reverted by `005`). It is **not** a general runbook for this
directory. Kept because its validation queries and rollback steps are a good model
for the next risky migration.
