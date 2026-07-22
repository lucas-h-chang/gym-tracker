#!/usr/bin/env python3
"""
Evaluate the current saved model without retraining.

What this does, step by step:
  1. Loads models/rf_model.pkl (the trained Random Forest) and feature_names.pkl
  2. Pulls all rows from Supabase capacity_log — same query as train.py
  3. Applies the same cleaning filter (people_count > 5, drop nulls)
  4. Rebuilds the calendar features with engineer_features()
  5. Cuts the same 80/20 chronological split so the test set is identical
     to what the model was trained and evaluated on
  6. Runs two predictors on the test set:
       - baseline: groupby mean per (day-of-week, 15-min slot, is_break)
       - RF model: the saved pickle
  7. Computes aggregate MAE/RMSE + segmented metrics + within_10pp
  8. Overwrites models/metrics.json with the full results

Run with:
    export SUPABASE_URL="..."
    export SUPABASE_SERVICE_KEY="..."
    python3 eval_model.py
"""
import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(__file__))
from train import engineer_features, parse_supabase_timestamps

MAX_CAPACITY = 150


# ── Step 1: load the saved model ──────────────────────────────────────────────

print("Step 1 — loading saved model...")
with open("models/rf_model.pkl", "rb") as f:
    rf = pickle.load(f)
with open("models/feature_names.pkl", "rb") as f:
    saved_feature_names = pickle.load(f)
print(f"  RF loaded  ({rf.n_estimators} trees, max_depth={rf.max_depth})")
print(f"  Features:  {saved_feature_names}")


# ── Step 2: pull data from Supabase ──────────────────────────────────────────

print("\nStep 2 — loading data from Supabase capacity_log...")
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

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
    print(f"  ...{len(rows):,} rows fetched")

print(f"  {len(rows):,} total rows from Supabase")


# ── Step 3: clean ─────────────────────────────────────────────────────────────

print("\nStep 3 — cleaning...")
df = pd.DataFrame(rows)
df["timestamp"]    = parse_supabase_timestamps(df["timestamp"])
df["people_count"] = df["people_count"].astype(float)
df = df[df["people_count"] > 5].dropna()
df = df.sort_values("timestamp").reset_index(drop=True)
df["percent_full"] = (df["people_count"] / MAX_CAPACITY) * 100
print(f"  {len(df):,} rows after removing noise/zeros")
print(f"  Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")


# ── Step 4: feature engineering ───────────────────────────────────────────────

print("\nStep 4 — engineering calendar features (this takes a moment)...")
X_all, feature_names = engineer_features(df)
y_all = df["percent_full"].values

# Verify features match the saved model — if someone added/removed a feature
# since the last retrain, predictions would be silently wrong.
if feature_names != saved_feature_names:
    print(f"\n  WARNING: feature mismatch!")
    print(f"  Saved model expects: {saved_feature_names}")
    print(f"  Current code gives:  {feature_names}")
    print("  The model was trained on different features — retrain before trusting these numbers.")


# ── Step 5: same 80/20 chronological split ────────────────────────────────────

print("\nStep 5 — reconstructing train/test split (80/20 chronological)...")
split = int(len(X_all) * 0.8)
X_fit,  X_test = X_all.iloc[:split],  X_all.iloc[split:]
y_fit,  y_test = y_all[:split],        y_all[split:]
df_fit, df_test = df.iloc[:split],     df.iloc[split:]

print(f"  Fit (train+val): {len(X_fit):,} rows  "
      f"({df_fit['timestamp'].iloc[0].date()} → {df_fit['timestamp'].iloc[-1].date()})")
print(f"  Test:            {len(X_test):,} rows  "
      f"({df_test['timestamp'].iloc[0].date()} → {df_test['timestamp'].iloc[-1].date()})")


# ── Step 6: baseline predictions ─────────────────────────────────────────────

print("\nStep 6 — computing baseline (groupby mean per weekday/slot/is_break)...")
key_cols = ["dow_num", "slot", "is_break"]
base_df = pd.DataFrame({
    "dow_num":      df["timestamp"].dt.dayofweek.values,
    "slot":         (df["timestamp"].dt.hour * 4 + df["timestamp"].dt.minute // 15).values,
    "is_break":     X_all["is_break"].values,
    "percent_full": y_all,
})
base_train = base_df.iloc[:split]
base_test  = base_df.iloc[split:]
lookup     = base_train.groupby(key_cols)["percent_full"].mean()
base_preds = base_test.set_index(key_cols).index.map(lookup)
base_preds = pd.Series(base_preds).fillna(base_train["percent_full"].mean()).to_numpy(dtype=float)

base_mae  = mean_absolute_error(y_test, base_preds)
base_rmse = mean_squared_error(y_test, base_preds) ** 0.5
print(f"  Baseline MAE: {base_mae:.2f}%")


# ── Step 7: RF predictions + all metrics ──────────────────────────────────────

print("\nStep 7 — running RF predictions and computing metrics...")
rf_preds = rf.predict(X_test)
rf_mae   = mean_absolute_error(y_test, rf_preds)
rf_rmse  = mean_squared_error(y_test, rf_preds) ** 0.5

# Segmented metrics
h  = X_test["hour_numeric"].values
wd = X_test["is_weekend"].values
br = X_test["is_break"].values

eve_peak = (h >= 17) & (h < 21) & (wd == 0) & (br == 0)
off_peak = ~eve_peak & (wd == 0) & (br == 0)
wknd     = wd == 1
brk      = br == 1

def seg_mae(mask):
    if mask.sum() == 0:
        return {"n": 0, "mae": None, "bias": None, "baseline_mae": None}
    a, r, b = y_test[mask], rf_preds[mask], base_preds[mask]
    return {
        "n":            int(mask.sum()),
        "mae":          round(float(mean_absolute_error(a, r)), 2),
        "bias":         round(float((r - a).mean()), 2),
        "baseline_mae": round(float(mean_absolute_error(a, b)), 2),
    }

segmented = {
    "evening_peak_weekday": seg_mae(eve_peak),
    "weekday_off_peak":     seg_mae(off_peak),
    "weekend":              seg_mae(wknd),
    "break_periods":        seg_mae(brk),
}

within_10pp = round(float((np.abs(rf_preds - y_test) <= 10).mean() * 100), 1)

feature_importances = dict(zip(feature_names, rf.feature_importances_))


# ── Step 8: print results and update metrics.json ─────────────────────────────

print("\n" + "=" * 55)
print("RESULTS")
print("=" * 55)
print(f"{'Model':<24} {'RMSE':>8} {'MAE':>8}")
print(f"{'-'*42}")
print(f"{'Baseline (groupby mean)':<24} {base_rmse:>7.2f}% {base_mae:>7.2f}%")
print(f"{'Random Forest':<24} {rf_rmse:>7.2f}% {rf_mae:>7.2f}%")

print(f"\n{'Segment':<28} {'N':>6}  {'MAE':>6}  {'Bias':>7}  {'Baseline':>8}")
print(f"{'-'*58}")
for seg, m in segmented.items():
    if m["n"] > 0:
        print(f"{seg:<28} {m['n']:>6,}  {m['mae']:>5.2f}%  {m['bias']:>+6.2f}%  {m['baseline_mae']:>7.2f}%")

print(f"\nWithin ±10pp: {within_10pp:.1f}%")

metrics = {
    "trained_at":   json.load(open("models/metrics.json"))["trained_at"],  # keep original
    "evaluated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "date_range":   f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}",
    "training_rows": int(len(X_fit)),
    "test_rows":     int(len(X_test)),
    "rf":            {"rmse": round(rf_rmse, 2),   "mae": round(rf_mae, 2)},
    "baseline":      {"rmse": round(base_rmse, 2), "mae": round(base_mae, 2)},
    "segmented":     segmented,
    "within_10pp":   within_10pp,
    "feature_importances": {k: round(float(v), 4) for k, v in feature_importances.items()},
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nStep 8 — metrics.json updated.")
