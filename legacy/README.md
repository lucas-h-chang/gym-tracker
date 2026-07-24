# legacy/

Retired artifacts from the pre-curve-model era, moved here 2026-07-21 during the
RF-retirement cleanup (see `handoffs/HANDOFF_MODEL_REDESIGN.md` and
`handoffs/SPEC_CURVE_MODEL.md`), plus `day_profiles_builder.py`, retired
2026-07-22 when `day_profiles` became a live Postgres view
(see `handoffs/SPEC_VIEWS_MIGRATION.md`).

> **Note (2026-07-23):** `weekly_builder.py` was moved back OUT of here to the
> `gym-tracker/` root and runs again in `daily.yml`. The `weekly_averages` view
> (`003`/`004`) was reverted to a nightly table because it 57014'd — see
> `handoffs/SPEC_WEEKLY_AVERAGES_REDESIGN.md`. Only `day_profiles` stays a view.

## What's actually in here

- `predictions_cache.json` — old local prediction cache, made obsolete when
  predictions moved to the Supabase `predictions` table.
  Already gitignored before this move (see `.gitignore` history); confirmed
  zero code references anywhere in the repo before moving it.
- `weekly_cache.json` — same story, for the Supabase `weekly_averages` table.
- `day_profiles_builder.py` — used to rebuild the `day_profiles` table nightly
  (incremental upsert) via `daily.yml`. Superseded by the `day_profiles` VIEW
  in `migrations/002_day_profiles_view.sql`, translated line-by-line from this
  file. No longer run; `today_builder.py` now reads the view directly. Kept as
  the translation's reference source.

## What did NOT move here, and why

The original cleanup plan called for moving `train.py`, `test_model_sanity.py`,
`test_features.py`, `models/rf_model.pkl`, and `models/feature_names.pkl` here
too, since the Random Forest they implement is no longer read at inference —
`predictions_builder.py` reads `models/curves.json` via `curve_model.py` instead.

That move did not happen. Grepping the repo first (per the cleanup's own rule:
"only move/delete if nothing on a live read path references it") found that
`backtest.py` — the rolling-origin evaluation harness that is the actual gate
for every curve-model tuning decision (see `SPEC_CURVE_MODEL.md` §5, §6) —
still does:

```python
from train import engineer_features, parse_supabase_timestamps
...
with open("models/rf_model.pkl", "rb") as f: ...
with open("models/feature_names.pkl", "rb") as f: ...
```

to compute the RF baseline column in every backtest report (`backtest_report.json`),
exactly as `HANDOFF_MODEL_REDESIGN.md` §7 intended ("archive RF code ... don't
delete — it's the comparison baseline").

> The two RF-specific one-off analysis scripts that shared this dependency —
> `eval_model.py` and `compare_cutoffs.py` — were deleted 2026-07-23 as orphaned
> (nothing referenced them; `backtest.py`'s baseline column supersedes them).
> `backtest.py` remains the sole live reader that keeps `train.py` here.

Moving `train.py` out of the project root would have broken `backtest.py` without a
messier fix (package-ifying `legacy/`, or duplicating `engineer_features`), which
is a bigger change than "mechanical de-duplication and dead-code removal" should
make. So `train.py`, `test_model_sanity.py`, `test_features.py`,
`models/rf_model.pkl`, and `models/feature_names.pkl` all stay in the project
root, unmoved, exactly as they were — they're frozen (nothing retrains them now
that `train.yml` is deleted) but still live-read by the backtest/eval tooling.

If a future pass wants to actually relocate them, the clean way is to make
`legacy/` an importable package (or fetch the RF baseline as read-only historical
data instead of live-loading the pickle) and update `backtest.py` accordingly —
that's a real refactor, not a move, and belongs in its own change with its own
verification.
