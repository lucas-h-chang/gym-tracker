#!/usr/bin/env python3
"""
Experiment: does dropping 2021 data improve the Random Forest?

2020 is already gone (people_count > 5 filter removes COVID closures).
So this tests: all data (2021-01-05+) vs 2022-01-01+ only.

Methodology: the test set is fixed at the same absolute rows (last 20% of
the full chronological range) for both models.  Different training windows,
identical evaluation — so the numbers are directly comparable.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(__file__))
from train import engineer_features, parse_supabase_timestamps

MAX_CAPACITY = 150
CUT_2022     = pd.Timestamp("2022-01-01")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all():
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
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
    df = pd.DataFrame(rows)
    df["timestamp"]    = parse_supabase_timestamps(df["timestamp"])
    df["people_count"] = df["people_count"].astype(float)
    df = df[df["people_count"] > 5].dropna()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["percent_full"] = (df["people_count"] / MAX_CAPACITY) * 100
    print(f"  {len(df):,} rows  ({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")
    return df


# ── Training ───────────────────────────────────────────────────────────────────

def fit_rf(df_train):
    X, _ = engineer_features(df_train.reset_index(drop=True))
    y    = df_train["percent_full"].values
    rf   = RandomForestRegressor(n_estimators=20, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf


# ── Segmented evaluation ───────────────────────────────────────────────────────

def evaluate(label, rf, X_test, y_test):
    preds = rf.predict(X_test)

    h  = X_test["hour_numeric"].values
    wd = X_test["is_weekend"].values
    br = X_test["is_break"].values

    eve_peak = (h >= 17) & (h < 21) & (wd == 0) & (br == 0)
    off_peak = ~eve_peak & (wd == 0) & (br == 0)
    wknd     = wd == 1
    brk      = br == 1

    segs = {
        "evening_peak_weekday": eve_peak,
        "weekday_off_peak":     off_peak,
        "weekend":              wknd,
        "break_periods":        brk,
    }

    overall   = mean_absolute_error(y_test, preds)
    within_10 = (np.abs(preds - y_test) <= 10).mean() * 100

    print(f"\n── {label} ────────────────────────────────────────────────────")
    print(f"   Overall MAE: {overall:.2f}%   |   Within ±10pp: {within_10:.1f}%")
    print(f"   {'Segment':<28} {'N':>6}  {'MAE':>6}  {'Bias':>7}")
    print(f"   {'-'*50}")
    for name, mask in segs.items():
        if mask.sum() == 0:
            continue
        a, p = y_test[mask], preds[mask]
        print(f"   {name:<28} {mask.sum():>6,}  {mean_absolute_error(a, p):>5.2f}%  {(p - a).mean():>+6.2f}%")

    return preds


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_all()

    # Pin the test boundary at 80% of the full dataset so both models
    # are scored on exactly the same rows.
    split     = int(len(df) * 0.8)
    df_before = df.iloc[:split]   # everything before the test period
    df_test   = df.iloc[split:].reset_index(drop=True)

    X_test, _ = engineer_features(df_test)
    y_test    = df_test["percent_full"].values

    print(f"\nTest set: {df_test['timestamp'].iloc[0].date()} → "
          f"{df_test['timestamp'].iloc[-1].date()}  ({len(df_test):,} rows)")

    # ── Model A: full history (current) ───────────────────────────────────────
    print(f"\nTraining Model A — full (2021+): "
          f"{df_before['timestamp'].iloc[0].date()} → "
          f"{df_before['timestamp'].iloc[-1].date()}  ({len(df_before):,} rows)")
    rf_full = fit_rf(df_before)

    # ── Model B: 2022+ only ───────────────────────────────────────────────────
    df_recent = df_before[df_before["timestamp"] >= CUT_2022]
    print(f"Training Model B — 2022+:       "
          f"{df_recent['timestamp'].iloc[0].date()} → "
          f"{df_recent['timestamp'].iloc[-1].date()}  ({len(df_recent):,} rows)")
    rf_recent = fit_rf(df_recent)

    # ── Side-by-side results ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS — same test set, different training windows")
    print("bias = mean(pred − actual);  negative → under-predicting")
    print("=" * 60)

    preds_full   = evaluate("Model A — full history (2021+)", rf_full,   X_test, y_test)
    preds_recent = evaluate("Model B — 2022+ only",           rf_recent, X_test, y_test)

    # ── Delta summary ─────────────────────────────────────────────────────────
    print(f"\n── Delta (B − A, negative = 2022+ is better) ──────────────────")
    delta_overall = mean_absolute_error(y_test, preds_recent) - mean_absolute_error(y_test, preds_full)
    print(f"   Overall MAE:          {delta_overall:+.2f}%")

    h  = X_test["hour_numeric"].values
    wd = X_test["is_weekend"].values
    br = X_test["is_break"].values
    eve_mask = (h >= 17) & (h < 21) & (wd == 0) & (br == 0)
    if eve_mask.sum():
        d_eve = (mean_absolute_error(y_test[eve_mask], preds_recent[eve_mask])
               - mean_absolute_error(y_test[eve_mask], preds_full[eve_mask]))
        print(f"   Evening peak MAE:     {d_eve:+.2f}%")

    d_w10 = ((np.abs(preds_recent - y_test) <= 10).mean()
            - (np.abs(preds_full   - y_test) <= 10).mean()) * 100
    print(f"   Within ±10pp (Δ):    {d_w10:+.1f}pp")
