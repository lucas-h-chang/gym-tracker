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

                key = f'{day}|{range_name}|{semester_only}'
                result[key] = [
                    {'x': row['hour_numeric'], 'y': round(row['percent_full'], 1), 'label': row['hour_label']}
                    for _, row in avg.iterrows()
                ]

    return result


def main():
    os.makedirs('docs', exist_ok=True)

    print('Loading models...')
    rf, scaler, mlp_model, feature_names = load_models()

    print('Computing predictions for next 180 days...')
    predictions = compute_predictions(rf, scaler, mlp_model, days=180)

    print('Computing weekly averages...')
    weekly = compute_weekly_averages()

    with open('models/metrics.json') as f:
        metrics = json.load(f)

    importances = sorted(
        metrics['feature_importances'].items(), key=lambda x: -x[1]
    )[:10]

    data = {
        'built_at': now.strftime('%Y-%m-%d %H:%M PT'),
        'predictions': predictions,
        'weekly': weekly,
        'metrics': {
            'trained_at': metrics['trained_at'][:10],
            'training_rows': metrics['training_rows'],
            'rf_mae': metrics['rf']['mae'],
            'mlp_mae': metrics['mlp']['mae'],
        },
        'feature_importances': [{'name': k, 'value': v} for k, v in importances],
    }

    out = 'docs/data.json'
    with open(out, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

    size_kb = os.path.getsize(out) / 1024
    print(f'Written → {out} ({size_kb:.0f} KB)')


if __name__ == '__main__':
    main()
