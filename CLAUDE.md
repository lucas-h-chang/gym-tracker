# CLAUDE.md — Bear Meter

ML web app predicting occupancy at UC Berkeley's RSF weight room. No Python at request time — all inference runs in GitHub Actions ahead of time. Frontend is plain HTML/JS served by Vercel, reads directly from Supabase.

## Key files
| File | Purpose |
|------|---------|
| `scraper.py` | Fetches live occupancy from Density.io API → Supabase capacity_log (every 15 min) |
| `today_builder.py` | Similarity predictions for today → Supabase today_summary (every 15 min) |
| `predictions_builder.py` | 180-day RF+MLP predictions → Supabase predictions (daily) |
| `weekly_builder.py` | Weekly pattern averages → Supabase weekly_averages (daily) |
| `train.py` | Trains Random Forest + PyTorch MLP from Supabase data, commits models/ |
| `predict.py` | One-off model inference + backtesting helper |
| `docs/index.html` | Static frontend — queries Supabase REST API directly on page load |
| `api/subscribe.js` | Vercel serverless function for SMS subscription management |
| `api/live-capacity.js` | Vercel serverless function — cached live-% proxy (checks `live_capacity` row, falls through to Density on stale cache) |

## ML models
Two models, ~177k rows (Nov 2020–present), chronological 80/20 split:

**Random Forest** — 20 trees, max_depth=15, MAE ±9.42%
- Top features: `is_break` (50%), `hour_numeric` (25%), `week_of_year` (13%)

**PyTorch MLP** — Input(15)→Linear(64)→ReLU→Dropout(0.2)→Linear(32)→ReLU→Dropout(0.2)→Linear(1), MAE ±9.93%

## Feature engineering (15 features)
Defined in `train.py::engineer_features()`, used by `predictions_builder.py` and `predict.py`:
- `hour_numeric`, `week_of_year`, `is_weekend`, `day_Monday`…`day_Sunday` (one-hot)
- Berkeley academic calendar flags (hardcoded through 2028): `is_finals`, `is_dead_week`, `is_first_week`, `is_break`, `is_holiday`

## RSF operating hours
- Academic year — Mon–Fri: 7 AM – 11 PM · Sat: 8 AM – 6 PM · Sun: 8 AM – 11 PM
- Summer — Mon–Fri: 7 AM – 8 PM · Sat: 8 AM – 6 PM (unchanged) · Sun: 8 AM – 8 PM
- Max capacity: 150
- Summer windows live in `SUMMER_RANGES` (duplicated in `scraper.py`, `predictions_builder.py`, `today_builder.py`, `weekly_builder.py`, `docs/index.html`, and `RSFApp2.0/.../TimeUtils.swift`). Derived from `train.py` summer-break ranges with end date shifted −3 days (RSF flips back to academic hours ~3 days before classes resume). Keep all six copies in sync when adding a new year.
- Cron-job.org fires the scrape webhook through academic-year hours year-round. `scraper.py` self-gates and exits early when the gym is actually closed (e.g., summer evenings), so cron does not need seasonal adjustment.

## Live capacity (Density.io) — fan-out cache
- Clients (web + iOS) never call Density directly. They call `https://rsfnow.com/api/live-capacity` instead.
- `api/live-capacity.js` checks the `live_capacity` Supabase row (id=1). If `recorded_at` is <30s old, returns cached. Otherwise calls Density, upserts, returns fresh.
- Vercel edge cache (`Cache-Control: s-maxage=30, stale-while-revalidate=60`) coalesces simultaneous client requests so Density gets ~2 calls/min regardless of user count.
- `DENSITY_TOKEN` env var lives only in: Vercel project settings + `DENSITY_TOKEN` GHA secret (consumed by `scraper.py` via `scrape.yml`). Not in any committed source file.
- `scraper.py` is the only thing that still calls Density directly — it writes raw counts to `capacity_log` for ML training. Different concern from the live-% pill.