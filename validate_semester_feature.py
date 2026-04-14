"""
Validates lag features vs baseline for near-term predictions (March 1–20).

Lag features: for each prediction, look up what actually happened at the
same day-of-week + hour slot in recent history — capturing current-semester
patterns without needing an explicit semester flag.

Run with: python3 validate_semester_feature.py
"""
import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from train import engineer_features, MAX_CAPACITY

# --- Config ---
CURRENT_SEMESTER_START = date(2026, 1, 21)
SETTLING_WEEKS = 3
TEST_START = date(2026, 3, 1)
TEST_END   = date(2026, 3, 20)

# ------------------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------------------

conn = sqlite3.connect("gym_history.db")
df = pd.read_sql_query("SELECT timestamp, people_count FROM capacity_log", conn)
conn.close()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[df['people_count'] > 5].dropna()
df = df.sort_values('timestamp').reset_index(drop=True)
df['percent_full'] = (df['people_count'] / MAX_CAPACITY) * 100

train_df = df[df['timestamp'].dt.date < TEST_START].copy()
test_df  = df[(df['timestamp'].dt.date >= TEST_START) & (df['timestamp'].dt.date <= TEST_END)].copy()

print(f"Train: {len(train_df):,} rows  ({train_df['timestamp'].min().date()} → {train_df['timestamp'].max().date()})")
print(f"Test:  {len(test_df):,} rows   ({test_df['timestamp'].min().date()} → {test_df['timestamp'].max().date()})\n")

y_train = train_df['percent_full'].values
y_test  = test_df['percent_full'].values

# ------------------------------------------------------------------------------
# LAG FEATURE COMPUTATION
# Lag features are computed using only train_df, so there's no data leakage.
# For each test row at (day_of_week D, hour H), we look up:
#   lag_7d:      avg capacity at same (D, H) slot exactly 7 days prior (±1 day window)
#   rolling_2wk: avg capacity at same (D, H) slot over the past 2 weeks
# ------------------------------------------------------------------------------

def add_lag_features(target_df, history_df, fallback_mean):
    """
    For each row in target_df, compute lag features from history_df.
    Uses same (day_of_week, hour) slot — capturing weekly recurring patterns.
    """
    # Precompute daily slot averages from history: (date, dow, hour) → mean capacity
    hist = history_df.copy()
    hist['dow']  = hist['timestamp'].dt.dayofweek
    hist['hour'] = hist['timestamp'].dt.hour
    hist['date'] = hist['timestamp'].dt.date
    slot_daily = hist.groupby(['date', 'dow', 'hour'])['percent_full'].mean().reset_index()

    target = target_df.copy()
    target['dow']  = target['timestamp'].dt.dayofweek
    target['hour'] = target['timestamp'].dt.hour
    target['date'] = target['timestamp'].dt.date

    lag_7d_vals, rolling_2wk_vals = [], []

    for _, row in target.iterrows():
        d, dow, hr = row['date'], row['dow'], row['hour']

        # Only compute lag for rows within the current semester.
        # Lag from previous semesters reflects different student schedules
        # and is noise, not signal — leave it as NaN so it falls back to mean.
        if d < CURRENT_SEMESTER_START:
            lag_7d_vals.append(np.nan)
            rolling_2wk_vals.append(np.nan)
            continue

        same_slot = slot_daily[(slot_daily['dow'] == dow) & (slot_daily['hour'] == hr)]

        # 7-day lag: same slot last week (exact date)
        w7 = same_slot[same_slot['date'] == d - timedelta(days=7)]
        lag_7d_vals.append(w7['percent_full'].mean() if len(w7) > 0 else np.nan)

        # Rolling 2-week: same slot any day over the past 14 days
        w2 = same_slot[
            (same_slot['date'] >= d - timedelta(days=14)) &
            (same_slot['date'] <  d)
        ]
        rolling_2wk_vals.append(w2['percent_full'].mean() if len(w2) > 0 else np.nan)

    target['lag_7d']       = lag_7d_vals
    target['rolling_2wk']  = rolling_2wk_vals

    # Fill missing slots with training mean (e.g. rare hours with no history)
    target['lag_7d'].fillna(fallback_mean, inplace=True)
    target['rolling_2wk'].fillna(fallback_mean, inplace=True)

    return target


fallback = train_df['percent_full'].mean()
train_with_lags = add_lag_features(train_df, train_df, fallback)
test_with_lags  = add_lag_features(test_df,  train_df, fallback)

def add_lag_cols(X, source_df):
    X = X.copy()
    X['lag_7d']      = source_df['lag_7d'].values
    X['rolling_2wk'] = source_df['rolling_2wk'].values
    return X

def add_semester_cols(X, df):
    settling_end = CURRENT_SEMESTER_START + timedelta(weeks=SETTLING_WEEKS)
    ts_date = df['timestamp'].dt.date
    X = X.copy()
    X['is_current_semester'] = (ts_date >= settling_end).astype(int).values
    X['is_semester_settling'] = (
        (ts_date >= CURRENT_SEMESTER_START) & (ts_date < settling_end)
    ).astype(int).values
    return X

# ------------------------------------------------------------------------------
# TRAIN & EVALUATE MODELS
# ------------------------------------------------------------------------------

def train_eval(X_tr, y_tr, X_te, y_te):
    rf = RandomForestRegressor(n_estimators=20, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    preds = rf.predict(X_te)
    return mean_absolute_error(y_te, preds), rf, preds

# 1. Baseline
X_tr_base, feat_base = engineer_features(train_df)
X_te_base, _         = engineer_features(test_df)
mae_base, _, _ = train_eval(X_tr_base, y_train, X_te_base, y_test)

# 2. Semester flags only
X_tr_sem = add_semester_cols(X_tr_base, train_df)
X_te_sem = add_semester_cols(X_te_base, test_df)
mae_sem, _, _ = train_eval(X_tr_sem, y_train, X_te_sem, y_test)

# 3. Lag features only
X_tr_lag = add_lag_cols(X_tr_base, train_with_lags)
X_te_lag = add_lag_cols(X_te_base, test_with_lags)
mae_lag, rf_lag, _ = train_eval(X_tr_lag, y_train, X_te_lag, y_test)

# 4. Lag + semester flags
X_tr_both = add_semester_cols(X_tr_lag, train_df)
X_te_both = add_semester_cols(X_te_lag, test_df)
mae_both, _, _ = train_eval(X_tr_both, y_train, X_te_both, y_test)

# ------------------------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------------------------

print(f"{'Model':<40} {'MAE':>8}   {'vs baseline':>12}")
print("-" * 62)
models = [
    ("Baseline",                    mae_base),
    ("+ semester flags only",       mae_sem),
    ("+ lag features only",         mae_lag),
    ("+ lag + semester flags",      mae_both),
]
for name, mae in models:
    delta = mae - mae_base
    sign = "+" if delta > 0 else ""
    print(f"  {name:<38} {mae:>7.2f}%   {sign}{delta:>+.2f}%")

print(f"\nFeature importances (lag model):")
lag_feat_names = feat_base + ['lag_7d', 'rolling_2wk']
importances = sorted(zip(lag_feat_names, rf_lag.feature_importances_), key=lambda x: -x[1])
for feat, imp in importances[:10]:
    marker = " ◄" if feat in ('lag_7d', 'rolling_2wk') else ""
    print(f"  {feat:<30} {imp:.4f}{marker}")
