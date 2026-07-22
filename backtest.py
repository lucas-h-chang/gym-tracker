"""
backtest.py — rolling-origin evaluation harness for the curve model vs two
baselines (equal-weight lookup, deployed RF). See SPEC_CURVE_MODEL.md §6.

Origins: 1st of each month, 2024-01 -> 2026-06. Per origin: build a curve
table on data strictly before that origin, predict every open 15-min slot
for the next 90 days, and join against the actual per-(date,slot) mean
(observed later in the same capacity_log history). This replaces a single
70/10/20 split, which let the RF train on zero recent data and made the
test window 40% break.

Run:  python3 backtest.py
Requires SUPABASE_URL and SUPABASE_SERVICE_KEY (or a read key) env vars.
Writes backtest_report.json next to this script.
"""
import os
import json
import pickle
import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

import curve_model as cm
from academic_calendar import (
    classify_date, days_to_sem_start, days_to_sem_end, get_open_hours,
)
from train import engineer_features, parse_supabase_timestamps

MAX_CAPACITY = 150

# is_summer_day/get_open_hours/SUMMER_RANGES live in academic_calendar.py
# (consolidated 2026-07-21 — see CLAUDE.md).


def open_slots_grid(start_date, days):
    """List of (date, slot) for every open 15-min slot over `days` days from start_date."""
    out = []
    for offset in range(days):
        d = start_date + timedelta(days=offset)
        day_name = pd.Timestamp(d).day_name()
        open_h, close_h = get_open_hours(day_name, d)
        for h in range(open_h, close_h):
            for m in (0, 15, 30, 45):
                out.append((d, h * 4 + m // 15))
    return out


def fetch_capacity_log():
    from supabase import create_client
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_ANON_KEY"]
    sb = create_client(os.environ["SUPABASE_URL"], key)

    print("Loading data from Supabase capacity_log...")
    BATCH, offset, rows = 9000, 0, []
    while True:
        batch = (
            sb.table("capacity_log")
            .select("timestamp,people_count")
            .range(offset, offset + BATCH - 1)
            .order("timestamp")
            .execute()
            .data
        )
        rows.extend(batch)
        if len(batch) < BATCH:
            break
        offset += BATCH
        print(f"  Fetched {len(rows):,} rows...")

    df = pd.DataFrame(rows)
    df['timestamp'] = parse_supabase_timestamps(df['timestamp'])
    df['people_count'] = df['people_count'].astype(float)
    return df


ORIGINS = [d for d in (date(y, m, 1) for y in (2024, 2025, 2026) for m in range(1, 13))
           if date(2024, 1, 1) <= d <= date(2026, 6, 1)]

TUNE_ORIGINS = [d for d in ORIGINS if d < date(2025, 7, 1)]
HOLDOUT_ORIGINS = [d for d in ORIGINS if d >= date(2025, 7, 1)]


def hour_bucket(slot):
    h = slot // 4
    if h < 9:
        return "open-9am"
    if h < 12:
        return "9-12"
    if h < 17:
        return "12-5pm"
    if h < 21:
        return "5-9pm"
    return "9pm-close"


def segment_for_date(d, ramp_days=10):
    """regular / first_week / ramp (break within ramp_days of a boundary) / finals_dead / break_deep / holiday.

    "break" is season-specific in the model (winter_break/spring_break/
    summer_break_<month> — see academic_calendar.classify_date) but pooled
    back into one "break_deep"/"ramp" bucket here purely for report readability.
    """
    phase = classify_date(d)
    if phase in ("winter_break", "spring_break") or phase.startswith("summer_break_"):
        dts = days_to_sem_start(d)
        dte = days_to_sem_end(d)
        if (1 <= dts <= ramp_days) or (1 <= -dte <= ramp_days):
            return "ramp"
        return "break_deep"
    if phase in ("finals", "dead_week"):
        return "finals_dead"
    return phase  # regular, first_week, holiday


def load_rf():
    with open("models/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    # The deployed pickle may predate feature additions to engineer_features
    # (e.g. days_to_sem_start/end) — feature_names.pkl records what it was
    # actually fit on, so we can reindex engineer_features()'s output to match
    # instead of erroring or silently retraining.
    try:
        with open("models/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
    except FileNotFoundError:
        feature_names = None
    return rf, feature_names


def rf_predict_grid(rf, feature_names, grid):
    ts = [pd.Timestamp(f"{d} {slot // 4:02d}:{(slot % 4) * 15:02d}") for d, slot in grid]
    df = pd.DataFrame({'timestamp': ts, 'people_count': 100.0, 'percent_full': 66.7})
    X, _ = engineer_features(df)
    if feature_names is not None and list(X.columns) != list(feature_names):
        X = X[feature_names]
    return rf.predict(X)


def run_backtest(full_slots, params, origins, rf=None, rf_feature_names=None, verbose=True):
    """
    full_slots: output of curve_model.prepare_slots on the FULL history (used
      both as the training source, sliced per-origin by build_table's
      build_date cutoff, and as the source of ground truth).
    Returns a long DataFrame: origin, date, slot, actual, curve, equal, rf.
    """
    actual_map = {
        (row.date.date(), int(row.slot)): float(row.percent_full)
        for row in full_slots.itertuples(index=False)
    }

    records = []
    for origin in origins:
        train_slots = full_slots[full_slots['date'] < pd.Timestamp(origin)]
        if train_slots.empty:
            continue

        table = cm.build_table(train_slots, params, build_date=origin)
        eq_table = cm.build_table(train_slots, {**params, 'halflife_days': 1_000_000}, build_date=origin)

        grid = open_slots_grid(origin, 90)
        actuals = [actual_map.get((d, s)) for d, s in grid]
        keep = [i for i, a in enumerate(actuals) if a is not None]
        if not keep:
            continue
        grid_k = [grid[i] for i in keep]
        actual_k = [actuals[i] for i in keep]

        bw = params.get('blend_window_days')
        curve_preds = cm.predict(table, grid_k, blend_window_days=bw)
        eq_preds = cm.predict(eq_table, grid_k, blend_window_days=bw)
        rf_preds = rf_predict_grid(rf, rf_feature_names, grid_k) if rf is not None else np.full(len(grid_k), np.nan)

        for (d, slot), a, cpred, eqpred, rfpred in zip(grid_k, actual_k, curve_preds, eq_preds, rf_preds):
            records.append({
                "origin": origin, "date": d, "slot": slot, "actual": a,
                "curve": cpred, "equal": eqpred, "rf": rfpred,
            })
        if verbose:
            print(f"  origin {origin}: {len(grid_k):,} scored slots")

    return pd.DataFrame(records)


def _mae(df, pred_col):
    return round(float((df[pred_col] - df['actual']).abs().mean()), 3)


def _p90(df, pred_col):
    return round(float((df[pred_col] - df['actual']).abs().quantile(0.9)), 3)


def report(records, model_cols=("curve", "equal", "rf")):
    records = records.dropna(subset=['actual']).copy()
    records['hour_bucket'] = records['slot'].apply(hour_bucket)
    records['segment'] = records['date'].apply(segment_for_date)
    records['hour'] = records['slot'] // 4

    out = {}
    for col in model_cols:
        sub = records.dropna(subset=[col])
        if sub.empty:
            out[col] = None
            continue

        by_hour_bucket = {hb: _mae(g, col) for hb, g in sub.groupby('hour_bucket')}
        by_segment = {seg: _mae(g, col) for seg, g in sub.groupby('segment')}

        # Good-day rate (spec-literal, SPEC_CURVE_MODEL.md §6): collapse to
        # hourly means per (origin, date, hour), day passes if EVERY open
        # hour is within 10pp (hard max). Also report a percentile variant:
        # with a documented ~15pp within-cell noise floor (see
        # HANDOFF_MODEL_REDESIGN.md §2.4) and ~14-16 open hours/day, a hard
        # max across that many correlated-but-noisy hourly buckets can fail
        # almost every day even for the true conditional mean — the
        # literal metric may not discriminate between models as intended.
        # Report both; see backtest_report.json for which one the Step 3
        # gate was actually evaluated against.
        hourly = sub.groupby(['origin', 'date', 'hour'])[[col, 'actual']].mean().reset_index()
        hourly['err'] = (hourly[col] - hourly['actual']).abs()
        by_day = hourly.groupby(['origin', 'date'])['err']
        good_day_rate = round(float((by_day.max() <= 10).mean() * 100), 1)
        good_day_rate_p90 = round(float((by_day.apply(lambda e: (e <= 10).mean()) >= 0.9).mean() * 100), 1)

        morning = sub[(sub['hour'] >= 7) & (sub['hour'] < 9)]
        evening = sub[(sub['hour'] >= 17) & (sub['hour'] < 22)]
        p90_morning = _p90(morning, col) if not morning.empty else None
        p90_evening = _p90(evening, col) if not evening.empty else None

        # Per-day shape correlation, median across (origin, date) with >=4 slots.
        def day_corr(g):
            if len(g) < 4 or g[col].std() == 0 or g['actual'].std() == 0:
                return np.nan
            return g[col].corr(g['actual'])

        corrs = sub.groupby(['origin', 'date']).apply(day_corr, include_groups=False)
        shape_corr_median = round(float(corrs.median()), 3)

        # Max slot-to-slot jump, excluding first/last 2 slots of each day.
        def max_jump(g):
            g = g.sort_values('slot')
            if len(g) < 5:
                return np.nan
            trimmed = g.iloc[2:-2]
            return trimmed[col].diff().abs().max()

        jumps = sub.groupby(['origin', 'date']).apply(max_jump, include_groups=False)
        max_slot_jump = round(float(jumps.max()), 2) if not jumps.dropna().empty else None

        out[col] = {
            "overall_mae": _mae(sub, col),
            "mae_by_hour_bucket": by_hour_bucket,
            "mae_by_segment": by_segment,
            "good_day_rate_pct": good_day_rate,
            "good_day_rate_p90_pct": good_day_rate_p90,
            "p90_hourly_error_7_9am": p90_morning,
            "p90_hourly_error_5_10pm": p90_evening,
            "shape_correlation_median": shape_corr_median,
            "max_slot_jump_pp": max_slot_jump,
            "n": int(len(sub)),
        }
    return out


def print_report(title, rep):
    print(f"\n=== {title} ===")
    for model, m in rep.items():
        if m is None:
            print(f"  {model}: no data")
            continue
        print(f"  --- {model} ---")
        print(f"    overall MAE: {m['overall_mae']}   good-day rate: {m['good_day_rate_pct']}% "
              f"(p90 variant: {m['good_day_rate_p90_pct']}%)   "
              f"shape corr (median): {m['shape_correlation_median']}   n={m['n']:,}")
        print(f"    P90 hourly error  7-9am: {m['p90_hourly_error_7_9am']}   5-10pm: {m['p90_hourly_error_5_10pm']}")
        print(f"    max slot-to-slot jump (excl. open/close): {m['max_slot_jump_pp']}pp")
        print(f"    MAE by hour bucket: {m['mae_by_hour_bucket']}")
        print(f"    MAE by segment: {m['mae_by_segment']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-json", default=None, help='JSON overrides, e.g. \'{"halflife_days":120}\'')
    parser.add_argument("--origins", choices=["all", "tune", "holdout"], default="all")
    parser.add_argument("--no-rf", action="store_true", help="skip the RF baseline (models/rf_model.pkl)")
    args = parser.parse_args()

    params = cm.DEFAULT_PARAMS.copy()
    if args.params_json:
        params.update(json.loads(args.params_json))

    origins = {"all": ORIGINS, "tune": TUNE_ORIGINS, "holdout": HOLDOUT_ORIGINS}[args.origins]

    raw = fetch_capacity_log()
    full_slots = cm.prepare_slots(raw)
    print(f"Prepared {len(full_slots):,} (date, slot) rows spanning "
          f"{full_slots['date'].min().date()} -> {full_slots['date'].max().date()}")

    rf, rf_feature_names = (None, None) if args.no_rf else load_rf()

    print(f"\nRunning backtest over {len(origins)} origins with params={params} ...")
    records = run_backtest(full_slots, params, origins, rf=rf, rf_feature_names=rf_feature_names)

    rep = report(records)
    print_report(f"Backtest report ({args.origins} origins, params={params})", rep)

    with open("backtest_report.json", "w") as f:
        json.dump({"params": params, "origins": args.origins, "report": rep}, f, indent=2, default=str)
    print("\nSaved -> backtest_report.json")
