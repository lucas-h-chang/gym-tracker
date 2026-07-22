# legacy/

Retired artifacts from the pre-curve-model era, moved here 2026-07-21 during the
RF-retirement cleanup (see `handoffs/HANDOFF_MODEL_REDESIGN.md` and
`handoffs/SPEC_CURVE_MODEL.md`), plus the two nightly builder scripts retired
2026-07-22 when `day_profiles` and `weekly_averages` became live Postgres views
(see `handoffs/SPEC_VIEWS_MIGRATION.md`).

## What's actually in here

- `predictions_cache.json` — old local prediction cache, made obsolete when
  predictions moved to the Supabase `predictions` table.
  Already gitignored before this move (see `.gitignore` history); confirmed
  zero code references anywhere in the repo before moving it.
- `weekly_cache.json` — same story, for the Supabase `weekly_averages` table.
- `weekly_builder.py` — used to rebuild the `weekly_averages` table nightly
  (truncate-then-insert) via `daily.yml`. Superseded by the `weekly_averages`
  VIEW in `migrations/003_weekly_averages_view.sql`, which was translated
  line-by-line from this file. No longer run; kept as that translation's
  reference source. If the view's SQL ever needs updating, re-derive it from
  this file, not from memory.
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
delete — it's the comparison baseline"). `eval_model.py` and `compare_cutoffs.py`
(RF-specific analysis scripts) have the same live dependency on `train.py`.

Moving `train.py` out of the project root would have broken all three without a
messier fix (package-ifying `legacy/`, or duplicating `engineer_features`), which
is a bigger change than "mechanical de-duplication and dead-code removal" should
make. So `train.py`, `test_model_sanity.py`, `test_features.py`,
`models/rf_model.pkl`, and `models/feature_names.pkl` all stay in the project
root, unmoved, exactly as they were — they're frozen (nothing retrains them now
that `train.yml` is deleted) but still live-read by the backtest/eval tooling.

If a future pass wants to actually relocate them, the clean way is to make
`legacy/` an importable package (or fetch the RF baseline as read-only historical
data instead of live-loading the pickle) and update `backtest.py` /
`eval_model.py` / `compare_cutoffs.py` accordingly — that's a real refactor, not
a move, and belongs in its own change with its own verification.
