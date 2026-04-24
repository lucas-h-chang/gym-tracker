#!/usr/bin/env python3
"""
backtest_tune.py — Parameter sweep to find the best similarity configuration.

Tests three independent improvements:
  1. Distance-weighted averaging (closer matches count more)
  2. Blend weight ceiling (how much we trust similarity vs ML)
  3. Number of neighbors K

Runs on the same 150-day sample as backtest_similarity.py for fair comparison.
"""
import sqlite3
import pickle
import numpy as np
import pandas as pd
import torch
from datetime import date, timedelta

from train import GymMLP, engineer_features

SPRING_BREAKS = [
    ('2021-03-20', '2021-03-28'), ('2022-03-19', '2022-03-27'),
    ('2023-03-25', '2023-04-02'), ('2024-03-23', '2024-03-31'),
    ('2025-03-22', '2025-03-30'), ('2026-03-21', '2026-03-29'),
    ('2027-03-20', '2027-03-28'), ('2028-03-25', '2028-04-02'),
]
SNAPSHOT_HOURS  = [9, 11, 13]
EXCLUDE_WINDOW  = 7
OVERLAP_THRESH  = 0.70
SAMPLE_SIZE     = 150


def get_open_hours(day_name):
    if day_name == 'Saturday':  return 8, 18
    elif day_name == 'Sunday':  return 8, 23
    else:                        return 7, 23


def is_semester_day(d):
    month, dom = d.month, d.day
    is_summer = month in [6, 7, 8]
    is_winter = (month == 12 and dom >= 16) or (month == 1 and dom <= 12)
    is_sb = any(pd.Timestamp(s).date() <= d <= pd.Timestamp(e).date() for s, e in SPRING_BREAKS)
    return not (is_summer or is_winter or is_sb)


def load_models():
    with open('models/rf_model.pkl', 'rb') as f:  rf = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:    scaler = pickle.load(f)
    with open('models/model_config.pkl', 'rb') as f: config = pickle.load(f)
    model = GymMLP(config['n_features'])
    model.load_state_dict(torch.load('models/pytorch_model.pt', weights_only=True))
    model.eval()
    return rf, scaler, model


def get_ml_predictions(rf, scaler, mlp_model, target_date, future_slots):
    if not future_slots:
        return {}
    timestamps = [pd.Timestamp(f'{target_date} {int(s):02d}:{round((s % 1)*60):02d}') for s in future_slots]
    df = pd.DataFrame({'timestamp': timestamps, 'people_count': [100]*len(timestamps), 'percent_full': [66.7]*len(timestamps)})
    X, _ = engineer_features(df)
    rf_preds = rf.predict(X)
    X_scaled = scaler.transform(X)
    with torch.no_grad():
        mlp_preds = mlp_model(torch.tensor(X_scaled, dtype=torch.float32)).numpy().flatten()
    return {slot: float(min((rf_p + mlp_p) / 2, 100)) for slot, rf_p, mlp_p in zip(future_slots, rf_preds, mlp_preds)}


def get_candidates(df_all, target_date, now_hour):
    """Pre-compute candidate pool and distances — shared across all configs."""
    day_name = pd.Timestamp(target_date).day_name()
    target_is_sem = is_semester_day(target_date)
    open_h, _ = get_open_hours(day_name)

    today_rows = df_all[df_all['date'] == target_date]
    today_finger = today_rows[today_rows['hour_numeric'] <= now_hour].groupby('hour_numeric')['percent_full'].mean()
    finger_slots = sorted(today_finger.index.tolist())

    if len(finger_slots) < 4:
        return [], finger_slots, open_h, _

    today_vec = np.array([today_finger[s] for s in finger_slots])

    target_ts = pd.Timestamp(target_date)
    candidates_df = df_all[
        (df_all['date'] != target_date) &
        (abs((pd.to_datetime(df_all['date']) - target_ts).dt.days) > EXCLUDE_WINDOW) &
        (df_all['day_name'] == day_name) &
        (df_all['is_semester'] == target_is_sem)
    ]

    candidates = []
    for cdate, group in candidates_df.groupby('date'):
        cday_finger = group.groupby('hour_numeric')['percent_full'].mean()
        available = [s for s in finger_slots if s in cday_finger.index]
        if len(available) < len(finger_slots) * OVERLAP_THRESH:
            continue
        hist_vec   = np.array([cday_finger[s] for s in available])
        today_sub  = np.array([today_finger[s] for s in available])
        dist = float(np.sqrt(np.mean((hist_vec - today_sub) ** 2)))
        candidates.append((cdate, dist, group))

    candidates.sort(key=lambda x: x[1])
    return candidates, finger_slots, open_h, _


def sim_preds_from_candidates(candidates, top_k_count, future_slots, distance_weighted):
    """Average top-K candidates, optionally weighted by inverse distance."""
    top_k = candidates[:top_k_count]
    if not top_k:
        return {}

    if distance_weighted:
        # Inverse-distance weights; add small epsilon to avoid div-by-zero on perfect matches
        dists   = np.array([c[1] for c in top_k])
        weights = 1.0 / (dists + 1e-6)
        weights /= weights.sum()
    else:
        weights = np.ones(len(top_k)) / len(top_k)

    sim_preds = {}
    for slot in future_slots:
        vals, wts = [], []
        for (_, _, grp), w in zip(top_k, weights):
            grp_data = grp.groupby('hour_numeric')['percent_full'].mean()
            if slot in grp_data.index:
                vals.append(grp_data[slot])
                wts.append(w)
        if vals:
            wts_arr = np.array(wts)
            wts_arr /= wts_arr.sum()
            sim_preds[slot] = float(np.dot(wts_arr, vals))

    return sim_preds


def run_sweep(df, rf, scaler, mlp_model, sampled_dates):
    """Full parameter sweep — returns nested results dict."""

    K_VALUES        = [5, 10, 15]
    BLEND_CEILINGS  = [0.8, 0.9, 1.0]
    DIST_WEIGHTED   = [False, True]

    # results[k][ceiling][dist_w][snap_h] = list of blend errors
    from collections import defaultdict
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    ml_errors_by_snap = defaultdict(list)  # baseline, same across all configs

    total = len(sampled_dates)
    for i, target_date in enumerate(sampled_dates):
        day_name = pd.Timestamp(target_date).day_name()
        open_h, close_h = get_open_hours(day_name)

        actual_rows = df[df['date'] == target_date]
        actual_map  = actual_rows.groupby('hour_numeric')['percent_full'].mean().to_dict()

        for snap_h in SNAPSHOT_HOURS:
            if snap_h < open_h or snap_h >= close_h:
                continue

            future_slots = [h + m/60 for h in range(open_h, close_h) for m in (0,15,30,45) if h + m/60 >= snap_h]
            actuals = {s: actual_map[s] for s in future_slots if s in actual_map}
            if not actuals:
                continue

            ml_preds = get_ml_predictions(rf, scaler, mlp_model, target_date, list(actuals.keys()))
            for s, a in actuals.items():
                if s in ml_preds:
                    ml_errors_by_snap[snap_h].append(abs(a - ml_preds[s]))

            candidates, _, oh, _ = get_candidates(df, target_date, snap_h)
            if not candidates:
                continue

            hours_observed = snap_h - open_h

            for K in K_VALUES:
                for ceiling in BLEND_CEILINGS:
                    for dw in DIST_WEIGHTED:
                        blend_w = round(min(hours_observed / 6.0, ceiling), 3)
                        sim_p   = sim_preds_from_candidates(candidates, K, list(actuals.keys()), dw)

                        for s, a in actuals.items():
                            if s not in ml_preds or s not in sim_p:
                                continue
                            blended = (1 - blend_w) * ml_preds[s] + blend_w * sim_p[s]
                            results[K][ceiling][dw][snap_h].append(abs(a - blended))

        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{total}...')

    return results, ml_errors_by_snap


def main():
    import warnings
    warnings.filterwarnings('ignore')

    print('Loading data...')
    conn = sqlite3.connect('gym_history.db')
    df = pd.read_sql_query('SELECT timestamp, percent_full FROM capacity_log', conn)
    conn.close()

    df['timestamp']  = pd.to_datetime(df['timestamp'])
    df['date']       = df['timestamp'].dt.date
    df['hour_numeric'] = ((df['timestamp'].dt.hour + df['timestamp'].dt.minute/60)*4).round()/4
    df['day_name']   = df['timestamp'].dt.day_name()
    df['is_semester'] = df['date'].apply(is_semester_day)

    print('Loading models...')
    rf, scaler, mlp_model = load_models()

    cutoff_start = date(2022, 1, 1)
    cutoff_end   = date.today() - timedelta(days=14)
    day_counts   = df.groupby('date')['hour_numeric'].count()
    eligible     = [d for d, cnt in day_counts.items() if cnt >= 40 and cutoff_start <= d <= cutoff_end]

    rng = np.random.default_rng(42)
    sampled_dates = sorted(rng.choice(eligible, size=min(SAMPLE_SIZE, len(eligible)), replace=False).tolist())
    print(f'Sweeping {len(sampled_dates)} days × 3 snapshots × 3K × 3 ceilings × 2 distance modes...\n')

    results, ml_errors = run_sweep(df, rf, scaler, mlp_model, sampled_dates)

    # ── Report ────────────────────────────────────────────────────────────────
    print('\n' + '='*72)
    print('PARAMETER SWEEP — avg blend MAE across all 3 snapshots')
    print('(lower is better; baseline ML MAEs shown at top)')
    print('='*72)

    for snap_h in SNAPSHOT_HOURS:
        ml_mae = np.mean(ml_errors[snap_h])
        label  = f"{snap_h % 12 or 12} {'AM' if snap_h < 12 else 'PM'}"
        print(f'  ML baseline @ {label}: {ml_mae:.2f}%')
    print()

    # Print table header
    print(f'{"K":>4}  {"Ceil":>5}  {"Dist-W":>6}  ', end='')
    for h in SNAPSHOT_HOURS:
        label = f"{h%12 or 12}{'AM' if h<12 else 'PM'}"
        print(f'{label:>8}', end='')
    print(f'  {"Avg":>8}')
    print('-'*72)

    rows = []
    for K in [5, 10, 15]:
        for ceil in [0.8, 0.9, 1.0]:
            for dw in [False, True]:
                maes = []
                for snap_h in SNAPSHOT_HOURS:
                    errs = results[K][ceil][dw][snap_h]
                    maes.append(np.mean(errs) if errs else float('nan'))
                avg_mae = np.nanmean(maes)
                rows.append((avg_mae, K, ceil, dw, maes))

    rows.sort(key=lambda x: x[0])

    for avg_mae, K, ceil, dw, maes in rows:
        dw_str = 'yes' if dw else 'no'
        print(f'{K:>4}  {ceil:>5.1f}  {dw_str:>6}  ', end='')
        for mae in maes:
            print(f'{mae:>7.2f}%', end='')
        print(f'  {avg_mae:>7.2f}%')

    print('='*72)
    best = rows[0]
    print(f'\nBest config: K={best[1]}, blend_ceiling={best[2]}, distance_weighted={best[3]}')
    ml_avg = np.mean([np.mean(ml_errors[h]) for h in SNAPSHOT_HOURS])
    print(f'Best avg MAE: {best[0]:.2f}%  vs  ML baseline: {ml_avg:.2f}%  (Δ={ml_avg-best[0]:+.2f}pp)\n')


if __name__ == '__main__':
    main()
