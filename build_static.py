#!/usr/bin/env python3
"""
build_static.py — pre-compute all data and write docs/data.json

Run after every scrape (and after retraining models).
The static site (docs/index.html) fetches this file on load.
"""
import sqlite3
import pickle
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import pytz
from datetime import datetime, timedelta

from train import GymMLP, engineer_features

california_tz = pytz.timezone('America/Los_Angeles')
now = datetime.now(california_tz)

SPRING_BREAKS = [
    ('2021-03-20', '2021-03-28'),
    ('2022-03-19', '2022-03-27'),
    ('2023-03-25', '2023-04-02'),
    ('2024-03-23', '2024-03-31'),
    ('2025-03-22', '2025-03-30'),
    ('2026-03-21', '2026-03-29'),
    ('2027-03-20', '2027-03-28'),
    ('2028-03-25', '2028-04-02'),
]


def get_open_hours(day_name):
    if day_name == 'Saturday':
        return 8, 18
    elif day_name == 'Sunday':
        return 8, 23
    else:
        return 7, 23


def is_semester_day(d):
    """True if date d falls during an active semester (not summer, winter break, or spring recess)."""
    month, dom = d.month, d.day
    is_summer = month in [6, 7, 8]
    is_winter = (month == 12 and dom >= 16) or (month == 1 and dom <= 12)
    is_sb = any(
        pd.Timestamp(s).date() <= d <= pd.Timestamp(e).date()
        for s, e in SPRING_BREAKS
    )
    return not (is_summer or is_winter or is_sb)


def load_models():
    with open('models/rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    model = GymMLP(config['n_features'])
    model.load_state_dict(torch.load('models/pytorch_model.pt', weights_only=True))
    model.eval()
    return rf, scaler, model, feature_names


PREDICTIONS_CACHE = 'models/predictions_cache.json'
WEEKLY_CACHE      = 'models/weekly_cache.json'
WEEKLY_MAX_AGE    = 24 * 3600  # seconds

MODEL_FILES = [
    'models/rf_model.pkl',
    'models/pytorch_model.pt',
    'models/scaler.pkl',
    'models/model_config.pkl',
]

def needs_predictions_rebuild():
    if not os.path.exists(PREDICTIONS_CACHE):
        return True
    cache_mtime = os.path.getmtime(PREDICTIONS_CACHE)
    return any(os.path.getmtime(f) > cache_mtime for f in MODEL_FILES)

def needs_weekly_rebuild():
    if not os.path.exists(WEEKLY_CACHE):
        return True
    age = time.time() - os.path.getmtime(WEEKLY_CACHE)
    return age > WEEKLY_MAX_AGE


def compute_predictions(rf, scaler, mlp_model, days=180):
    """Pre-compute RF and MLP predictions for every 15-min slot over the next N days."""
    timestamps = []
    keys = []

    for offset in range(days):
        d = now.date() + timedelta(days=offset)
        day_name = pd.Timestamp(d).day_name()
        open_h, close_h = get_open_hours(day_name)
        for h in range(open_h, close_h):
            for m in (0, 15, 30, 45):
                keys.append(f'{d}_{h:02d}:{m:02d}')
                timestamps.append(pd.Timestamp(f'{d} {h:02d}:{m:02d}'))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'people_count': [100] * len(timestamps),
        'percent_full': [66.7] * len(timestamps),
    })

    print(f'  Engineering features for {len(df):,} prediction slots...')
    X, _ = engineer_features(df)

    rf_preds = rf.predict(X)
    X_scaled = scaler.transform(X)
    with torch.no_grad():
        mlp_preds = mlp_model(
            torch.tensor(X_scaled, dtype=torch.float32)
        ).numpy().flatten()

    return {
        key: [round(min(float(rf_p), 100), 1), round(min(float(mlp_p), 100), 1)]
        for key, rf_p, mlp_p in zip(keys, rf_preds, mlp_preds)
    }


def compute_weekly_averages():
    """Pre-compute weekly pattern averages for all 70 filter combos (7 days x 5 ranges x 2 semester flags)."""
    conn = sqlite3.connect('gym_history.db')
    df = pd.read_sql_query('SELECT * FROM capacity_log', conn)
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['hour_label'] = (
        df['timestamp'].dt.round('15min').dt.strftime('%I:%M %p').str.lstrip('0')
    )
    df['hour_numeric'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60

    DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    TIME_RANGES = {
        'Last week':     timedelta(days=7),
        'Last month':    timedelta(days=30),
        'Last 6 months': timedelta(days=182),
        'Last year':     timedelta(days=365),
        'All time':      None,
    }

    result = {}

    for range_name, delta in TIME_RANGES.items():
        if delta is not None:
            cutoff = (now - delta).replace(hour=0, minute=0, second=0, microsecond=0)
            range_df = df[df['timestamp'] >= cutoff.replace(tzinfo=None)].copy()
        else:
            range_df = df.copy()

        for semester_only in [True, False]:
            if semester_only:
                month = range_df['timestamp'].dt.month
                dom = range_df['timestamp'].dt.day
                date_only = range_df['timestamp'].dt.date

                is_summer = month.isin([6, 7, 8])
                is_winter = ((month == 12) & (dom >= 16)) | ((month == 1) & (dom <= 12))
                is_sb = np.zeros(len(range_df), dtype=bool)
                for start, end in SPRING_BREAKS:
                    is_sb |= (
                        (date_only >= pd.Timestamp(start).date()) &
                        (date_only <= pd.Timestamp(end).date())
                    )
                filtered = range_df[~(is_summer | is_winter | is_sb)]
            else:
                filtered = range_df

            for day in DAYS:
                open_h, close_h = get_open_hours(day)
                day_data = filtered[
                    (filtered['day_of_week'] == day) &
                    (filtered['hour_numeric'] >= open_h) &
                    (filtered['hour_numeric'] <= close_h)
                ].copy()

                day_data['hour_numeric'] = (day_data['hour_numeric'] * 4).round() / 4

                avg = day_data.groupby('hour_numeric').agg(
                    percent_full=('percent_full', 'mean'),
                    hour_label=('hour_label', 'first'),
                ).reset_index()

                close_label = f"{close_h % 12 or 12}:00 {'AM' if close_h < 12 else 'PM'}"
                closing_row = pd.DataFrame([{
                    'hour_numeric': float(close_h),
                    'percent_full': 0.0,
                    'hour_label': close_label,
                }])
                avg = avg[avg['hour_numeric'] < close_h]
                avg = pd.concat([avg, closing_row], ignore_index=True)
                avg = avg.sort_values('hour_numeric')

                key = f'{day}|{range_name}|{str(semester_only).lower()}'
                result[key] = [
                    {'x': row['hour_numeric'], 'y': round(row['percent_full'], 1), 'label': row['hour_label']}
                    for _, row in avg.iterrows()
                ]

    return result


def compute_similarity_predictions():
    """
    Find the K=5 historical days most similar to today's observed pattern so far
    (same day-of-week, same semester status, closest morning fingerprint by RMSE).
    Returns distance-weighted average trajectories for remaining slots today,
    plus a blend weight that grows as more of today is observed (max 0.9).
    """
    K              = 5
    BLEND_CEIL     = 0.9
    OVERLAP_THRESH = 0.70
    EXCLUDE_WINDOW = 7   # days to exclude around today to avoid leakage
    MIN_SLOTS      = 4   # need at least 1 hour of data (4 × 15-min slots)

    conn = sqlite3.connect('gym_history.db')
    df = pd.read_sql_query('SELECT timestamp, percent_full FROM capacity_log', conn)
    conn.close()

    df['timestamp']    = pd.to_datetime(df['timestamp'])
    df['date']         = df['timestamp'].dt.date
    df['hour_numeric'] = ((df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60) * 4).round() / 4
    df['day_name']     = df['timestamp'].dt.day_name()
    df['is_semester']  = df['date'].apply(is_semester_day)

    today         = now.date()
    today_name    = pd.Timestamp(today).day_name()
    today_is_sem  = is_semester_day(today)
    now_hour      = now.hour + now.minute / 60
    open_h, close_h = get_open_hours(today_name)

    # Today's fingerprint: observed slots up to now
    today_rows   = df[df['date'] == today]
    today_finger = (
        today_rows[today_rows['hour_numeric'] <= now_hour]
        .groupby('hour_numeric')['percent_full'].mean()
    )
    finger_slots = sorted(today_finger.index.tolist())

    if len(finger_slots) < MIN_SLOTS:
        return [], 0.0

    # Candidate pool: same day-of-week, same semester status, outside exclusion window
    target_ts  = pd.Timestamp(today)
    candidates_df = df[
        (df['date'] != today) &
        (abs((pd.to_datetime(df['date']) - target_ts).dt.days) > EXCLUDE_WINDOW) &
        (df['day_name'] == today_name) &
        (df['is_semester'] == today_is_sem)
    ]

    candidates = []
    for _, group in candidates_df.groupby('date'):
        cf        = group.groupby('hour_numeric')['percent_full'].mean()
        available = [s for s in finger_slots if s in cf.index]
        if len(available) < len(finger_slots) * OVERLAP_THRESH:
            continue
        hist_vec  = np.array([cf[s]               for s in available])
        today_sub = np.array([today_finger[s]      for s in available])
        dist      = float(np.sqrt(np.mean((hist_vec - today_sub) ** 2)))
        candidates.append((dist, group))

    if not candidates:
        return [], 0.0

    candidates.sort(key=lambda x: x[0])
    top_k = candidates[:K]

    # Distance-weighted averaging
    dists   = np.array([c[0] for c in top_k])
    weights = 1.0 / (dists + 1e-6)
    weights /= weights.sum()

    # Future slots: now_hour → close_h
    future_slots = [
        h + m / 60
        for h in range(open_h, close_h)
        for m in (0, 15, 30, 45)
        if h + m / 60 >= now_hour
    ]

    similarity_preds = []
    for slot in future_slots:
        vals, wts = [], []
        for (_, grp), w in zip(top_k, weights):
            gf = grp.groupby('hour_numeric')['percent_full'].mean()
            if slot in gf.index:
                vals.append(gf[slot])
                wts.append(w)
        if vals:
            wa    = np.array(wts); wa /= wa.sum()
            h_int = int(slot)
            m_int = round((slot - h_int) * 60)
            label_h   = h_int % 12 or 12
            label_sfx = 'AM' if h_int < 12 else 'PM'
            similarity_preds.append({
                'x':     slot,
                'y':     round(float(np.dot(wa, vals)), 1),
                'label': f'{label_h}:{m_int:02d} {label_sfx}',
            })

    # Closing zero
    ch_label = f"{close_h % 12 or 12}:00 {'AM' if close_h < 12 else 'PM'}"
    similarity_preds.append({'x': float(close_h), 'y': 0.0, 'label': ch_label})

    blend_weight = round(min((now_hour - open_h) / 6.0, BLEND_CEIL), 3)
    return similarity_preds, blend_weight


def compute_today_actuals():
    """Return today's real capacity readings as [{x, y, label}], quantized to 15-min bins."""
    conn = sqlite3.connect('gym_history.db')
    df = pd.read_sql_query('SELECT * FROM capacity_log', conn)
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    today_str = now.strftime('%Y-%m-%d')
    df = df[df['timestamp'].dt.strftime('%Y-%m-%d') == today_str].copy()

    if df.empty:
        return []

    df['hour_numeric'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
    df['hour_numeric'] = (df['hour_numeric'] * 4).round() / 4
    df['hour_label'] = (
        df['timestamp'].dt.round('15min').dt.strftime('%I:%M %p').str.lstrip('0')
    )

    avg = df.groupby('hour_numeric').agg(
        percent_full=('percent_full', 'mean'),
        hour_label=('hour_label', 'first'),
    ).reset_index()

    return [
        {'x': row['hour_numeric'], 'y': round(row['percent_full'], 1), 'label': row['hour_label']}
        for _, row in avg.iterrows()
    ]


def main():
    os.makedirs('docs', exist_ok=True)

    # ── Predictions (rebuild only when models change) ──────────
    if needs_predictions_rebuild():
        print('Computing predictions (models changed or no cache)...')
        rf, scaler, mlp_model, _ = load_models()
        predictions = compute_predictions(rf, scaler, mlp_model, days=180)
        with open(PREDICTIONS_CACHE, 'w') as f:
            json.dump(predictions, f, separators=(',', ':'))
    else:
        print('Loading cached predictions...')
        with open(PREDICTIONS_CACHE) as f:
            predictions = json.load(f)

    # ── Weekly averages (rebuild at most once per 24 hours) ────
    if needs_weekly_rebuild():
        print('Computing weekly averages...')
        weekly = compute_weekly_averages()
        with open(WEEKLY_CACHE, 'w') as f:
            json.dump(weekly, f, separators=(',', ':'))
    else:
        print('Loading cached weekly averages...')
        with open(WEEKLY_CACHE) as f:
            weekly = json.load(f)

    # ── Always recompute (live data) ───────────────────────────
    print("Computing today's actuals...")
    today_actuals = compute_today_actuals()

    print('Computing similarity predictions...')
    similarity_preds, blend_weight = compute_similarity_predictions()

    with open('models/metrics.json') as f:
        metrics = json.load(f)

    data = {
        'built_at': now.strftime('%Y-%m-%d %H:%M PT'),
        'today_date': now.strftime('%Y-%m-%d'),
        'today_actuals': today_actuals,
        'today_similarity_preds': similarity_preds,
        'today_blend_weight': blend_weight,
        'predictions': predictions,
        'weekly': weekly,
        'metrics': {
            'trained_at': metrics['trained_at'][:10],
            'training_rows': metrics['training_rows'],
            'rf_mae': metrics['rf']['mae'],
            'mlp_mae': metrics['mlp']['mae'],
        },
    }

    out = 'docs/data.json'
    with open(out, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

    size_kb = os.path.getsize(out) / 1024
    print(f'Written → {out} ({size_kb:.0f} KB)')


if __name__ == '__main__':
    main()
