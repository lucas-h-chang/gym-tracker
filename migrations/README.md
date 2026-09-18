# migrations/

Applied-state ledger for the Supabase schema. **Migration 016 introduces `public.schema_migrations`** for the verified forward
baseline (013–016). It was applied on 2026-09-18. Historical 001–012 were applied manually;
do not replay them wholesale, because some were subsequently reverted. This file records
verified state alongside the database-owned ledger.

Keep it accurate. Add a row when you add a file, and update the row when you apply it.

## Status

The first twelve were **verified applied on 2026-09-09** by probing the live database
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
| 013 | `snapshot_bare_curve.sql` | ✅ applied | Verified 2026-09-16 in production: `prediction_snapshots.curve` and `prediction_accuracy.curve_pct` both exist |
| 014 | `reliable_publication.sql` | ✅ applied 2026-09-18 | Function body hashes match migration; anon/authenticated execution denied, service_role allowed; internal tables have RLS |
| 015 | `accuracy_publication_horizon.sql` | ✅ applied 2026-09-18 | View replacement succeeded; migration 016 verified baseline expansion; anon/authenticated reads denied |
| 016 | `migration_history.sql` | ✅ applied 2026-09-18 | Ledger lists 013–016; RLS enabled, backend-only access verified |
| 017 | `weekly_safeupdate.sql` | ✅ applied 2026-09-18 | Explicit non-null weekday predicate satisfies Supabase safeupdate without disabling the guard; transaction/rollback SQL test passes |

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

Apply forward files in order and update this ledger after verification. After 016, include a
`schema_migrations` entry in each new migration's transaction. The 016 rows for 013–015 adopt a
verified baseline; their `applied_at` is the adoption time, not a claim about the original 013 run.

**2026-09-18:** Lucas explicitly approved production migrations and deployment. Migrations
014–016 were applied in order and verified in the production SQL editor. The first weekly
publication encountered Supabase safeupdate (unqualified DELETE rejected); it rolled back
without data loss. Forward migration 017 adds an explicit predicate on the table’s NOT NULL
weekday column. Existing migrations remain unchanged and permissions are preserved.
See `handoffs/FABLE_REVIEW.md` in the parent workspace for the exact rollout and rollback sequence.

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
