# legacy/

Frozen artifacts from earlier versions of the pipeline. **Nothing in here runs on a
schedule, and nothing on a serving path imports from here.** Kept for provenance:
each file is the reference source for something that replaced it.

Before deleting anything here, grep the repo. The rule this directory follows is
"only move or delete if nothing on a live read path references it."

## The Random Forest (retired from inference 2026-07-21, moved here 2026-09-09)

- `train.py` — trained the single `RandomForestRegressor` that served predictions
  until the curve model replaced it. Frozen since 2026-07-13; `train.yml` is deleted,
  so nothing retrains it.
- `rf_model.pkl`, `feature_names.pkl` — the frozen fitted model and the exact feature
  list it was fitted on.
- `metrics.json` — its last recorded scores. Nothing reads this file. The live
  equivalent is `models/curve_metrics.json`.
- `test_features.py`, `test_model_sanity.py` — cover `engineer_features()` and the
  pickle. Removed from `ci.yml` in the same change; they were the only reason CI
  installed scikit-learn.

**Why these moved when an earlier pass decided they could not.** The blocker was
`backtest.py`, which imported `engineer_features` and loaded the pickle to compute an
RF baseline column in every report. That column stopped being meaningful once the
pickle froze: scoring a freshly built curve against a July artifact measures drift in
the dead model, not quality in the live one. So the column was deleted rather than the
import rerouted, which removed the last live reader and let these files move without
package-ifying `legacy/` or duplicating `engineer_features`.

**Running the archive.** These files resolve their artifacts relative to their own
directory, not `models/`, so a re-run cannot overwrite a live artifact. `pytest.ini`
excludes `legacy/` from default collection because scikit-learn is no longer in
`requirements.txt`. To run it anyway:

```
pip install scikit-learn
python3 -m pytest legacy/
```

## The pre-Supabase caches (moved here 2026-07-21)

- `predictions_cache.json` — local prediction cache, obsolete once predictions moved
  to the Supabase `predictions` table.
- `weekly_cache.json` — same story, for `weekly_averages`.

Both are gitignored, so they exist on disk only.

## `day_profiles_builder.py` (retired 2026-07-22)

Rebuilt the `day_profiles` table nightly via `daily.yml`. Superseded by the
`day_profiles` VIEW in `migrations/002_day_profiles_view.sql`, which was translated
line by line from this file. Kept as that translation's reference source.

Its consumer is gone too: `today_builder.py` stopped reading `day_profiles` on
2026-08-31 when the similarity nowcast was replaced by the fitted level correction.

## Deleted outright, not archived

- `eval_model.py`, `compare_cutoffs.py` (2026-07-23) — RF-specific one-offs, superseded
  by `backtest.py`'s baseline column.
- `nowcast_carry.py` + `nowcast_carry_report.json` (2026-09-09) — the exploratory
  measurement that established the carry structure. It self-marked as superseded; its
  conclusions are written up in `handoffs/SPEC_TODAY_BUILDER_REWRITE.md`.
- `blend_sweep.py` (2026-09-09) — swept `blend_window_days` for a blend that no longer
  exists.
- `accuracy_check.py` (2026-09-09) — ad-hoc MAE table, orphaned since 2026-07-13 and
  superseded by the `prediction_accuracy` view (`migrations/012`).

## Related reading

`handoffs/HANDOFF_MODEL_REDESIGN.md` (why the RF was replaced),
`handoffs/SPEC_CURVE_MODEL.md` (what replaced it),
`handoffs/SPEC_TODAY_BUILDER_REWRITE.md` (the within-day correction),
`handoffs/SPEC_VIEWS_MIGRATION.md` (the `day_profiles` view).
