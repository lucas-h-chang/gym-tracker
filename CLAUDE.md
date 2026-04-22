# CLAUDE.md — RSF Gym Tracker

## What this project is
A machine learning web app that predicts occupancy at UC Berkeley's RSF weight room. Students use it to decide when to visit. Live at Vercel (static HTML). Future: SwiftUI iOS app using the same Supabase backend.

## Architecture
**No Python runs at request time.** All ML inference happens in GitHub Actions ahead of time. The frontend reads directly from Supabase on page load.

```
GitHub Actions (every 15 min) — NO git commits
  → scraper.py        fetch live occupancy from Density.io API → Supabase capacity_log
  → today_builder.py  similarity predictions → Supabase today_summary

GitHub Actions (daily, midnight PT = 8:00 UTC)
  → predictions_builder.py  180-day RF+MLP predictions → Supabase predictions
  → weekly_builder.py       weekly averages → Supabase weekly_averages

GitHub Actions (1st of month, 4 AM UTC)
  → train.py          retrain both ML models from Supabase data, commit models/
```

## Deployed site
- **Vercel** serves `docs/` as the output directory (`vercel.json`)
- `docs/index.html` — static HTML/JS frontend, no framework
- Frontend queries Supabase REST API directly on page load (no data.json)
- `api/subscribe.js` — Vercel serverless function for SMS subscription management

## Key files
| File | Purpose |
|------|---------|
| `scraper.py` | Fetches real-time occupancy from Density.io API → Supabase capacity_log |
| `today_builder.py` | Similarity predictions for today → Supabase today_summary (every 15 min) |
| `predictions_builder.py` | 180-day RF+MLP predictions → Supabase predictions (daily) |
| `weekly_builder.py` | Weekly pattern averages → Supabase weekly_averages (daily) |
| `train.py` | Trains Random Forest + PyTorch MLP from Supabase data, saves to models/ |
| `predict.py` | One-off model inference + backtesting helper |
| `test_features.py` | Tests for feature engineering |
| `test_model_sanity.py` | Sanity checks for trained models |
| `validate_semester_feature.py` | Validates academic calendar feature flags |

## Deleted files (do not recreate)
- `build_static.py` — replaced by predictions_builder.py + weekly_builder.py
- `push_to_supabase.py` — replaced by individual builder scripts
- `migrate_sqlite_to_supabase.py` — one-time migration, done

## Supabase
**Single source of truth for all data.**

Project URL: `https://njhxcwcvyorwqlfacnal.supabase.co`

Tables:
| Table | Description |
|-------|-------------|
| `capacity_log` | Raw occupancy readings (177k+ rows, source of truth) |
| `predictions` | 180-day RF+MLP predictions, rebuilt daily |
| `weekly_averages` | Pre-aggregated weekly patterns (~4200 rows), rebuilt daily |
| `today_summary` | Similarity-based today predictions, rebuilt every 15 min |
| `subscribers` | SMS notification subscribers |
| `subscriber_windows` | Per-subscriber custom alert windows |

All tables have RLS enabled with public read policy (anon key is safe in frontend).

Env vars: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` (GitHub Actions secrets)

**Key gotchas:**
- Max rows set to 9998 in Supabase dashboard (Data API settings)
- Python builders use `BATCH = 9000` (must be < max_rows so paginator continues)
- Timestamps from Supabase have microseconds — use `format='ISO8601'` in pd.to_datetime
- URL timezone offset `+00:00` must be written as `Z` in query params (+ decodes as space)

## ML models
Two models trained on ~177k rows (Nov 2020 – present), chronological 80/20 split:

**Random Forest** (`models/rf_model.pkl`)
- 20 trees, max_depth=15
- MAE: ±9.42%
- Top features: `is_break` (50%), `hour_numeric` (25%), `week_of_year` (13%)

**PyTorch MLP** (`models/pytorch_model.pt`)
- Architecture: Input(15) → Linear(64) → ReLU → Dropout(0.2) → Linear(32) → ReLU → Dropout(0.2) → Linear(1)
- 200 epochs, Adam optimizer, early stopping (patience=20)
- MAE: ±9.93%

Model artifacts in `models/`: `rf_model.pkl`, `pytorch_model.pt`, `scaler.pkl`, `model_config.pkl`, `feature_names.pkl`, `metrics.json`

## Feature engineering (15 features)
Defined in `train.py::engineer_features()`, used by `predictions_builder.py` and `predict.py`:
- `hour_numeric` — hour as float (7.5 = 7:30 AM)
- `week_of_year` — 1–52
- `is_weekend` — binary
- `day_Monday` … `day_Sunday` — one-hot (7 columns)
- **Berkeley academic calendar flags** (hardcoded through 2028):
  - `is_finals`, `is_dead_week`, `is_first_week`, `is_break`, `is_holiday`

## Database
**Supabase** — cloud postgres, primary data store
- All reads/writes go through Supabase. SQLite is gone.

`gym_history.db` and `docs/data.json` are in `.gitignore` — do not commit them.

## RSF operating hours
- Mon–Fri: 7 AM – 11 PM
- Saturday: 8 AM – 6 PM
- Sunday: 8 AM – 11 PM
- Max capacity: 150 people

## Notifications (`notifier.py`)
Infrastructure is built but Twilio SMS is currently disabled. Three alert types:
1. **Quiet alerts** — capacity spike during normally low-traffic periods
2. **Window alerts** — custom time windows per subscriber
3. **Daily digest** — morning summary with best visit times

## GitHub Actions workflows
- `.github/workflows/scrape.yml` — triggered every 15 min via cron-job.org webhook; runs scraper.py + today_builder.py; **no git commit**
- `.github/workflows/daily.yml` — midnight PT cron; runs predictions_builder.py + weekly_builder.py; **no git commit**
- `.github/workflows/train.yml` — 1st of month at 4 AM UTC; reads from Supabase, commits updated models/ to git

## What's NOT in this project
- **No Streamlit** — `app.py` was deleted
- No React, no build step, no bundler — the frontend is plain HTML/JS
- No SQLite — all data is in Supabase
- No data.json — frontend queries Supabase directly
