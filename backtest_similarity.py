#!/usr/bin/env python3
"""
backtest_similarity.py — Validate the similarity-based prediction approach.

Runs three configs on the same 150-day sample:
  - Base ML model (no similarity)
  - Original similarity  (K=10, equal weights, blend ceiling=0.8)
  - Improved similarity  (K=5,  dist-weighted, blend ceiling=0.9)

Usage:
    python3 backtest_similarity.py
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

SNAPSHOT_HOURS = [9, 11, 13, 15, 19]
EXCLUDE_WINDOW = 7
OVERLAP_THRESH = 0.70
SAMPLE_SIZE    = 150
MIN_SLOTS      = 4

CONFIGS = {
    'Original  (K=10, equal,  ceil=0.8)': dict(K=10, dist_weighted=False, ceiling=0.8, anchor=False),
    'Improved  (K=5,  dist-w, ceil=0.9)': dict(K=5,  dist_weighted=True,  ceiling=0.9, anchor=False),
    'Anchored  (K=5,  dist-w, ceil=0.9)': dict(K=5,  dist_weighted=True,  ceiling=0.9, anchor=True),
}


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
    with open('models/rf_model.pkl', 'rb') as f:     rf = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:       scaler = pickle.load(f)
    with open('models/model_config.pkl', 'rb') as f: config = pickle.load(f)
    model = GymMLP(config['n_features'])
    model.load_state_dict(torch.load('models/pytorch_model.pt', weights_only=True))
    model.eval()
    return rf, scaler, model


def get_ml_predictions(rf, scaler, mlp_model, target_date, slots):
    if not slots:
        return {}
    timestamps = [pd.Timestamp(f'{target_date} {int(s):02d}:{round((s%1)*60):02d}') for s in slots]
    df = pd.DataFrame({'timestamp': timestamps, 'people_count': [100]*len(timestamps), 'percent_full': [66.7]*len(timestamps)})
    X, _ = engineer_features(df)
    rf_p = rf.predict(X)
    X_sc = scaler.transform(X)
    with torch.no_grad():
        mlp_p = mlp_model(torch.tensor(X_sc, dtype=torch.float32)).numpy().flatten()
    return {s: float(min((r+m)/2, 100)) for s, r, m in zip(slots, rf_p, mlp_p)}


def get_candidates(df_all, target_date, now_hour):
    """Compute sorted candidate list once; reused by all configs for a given (day, hour)."""
    day_name      = pd.Timestamp(target_date).day_name()
    target_is_sem = is_semester_day(target_date)
    open_h, _     = get_open_hours(day_name)

    today_rows   = df_all[df_all['date'] == target_date]
    today_finger = (
        today_rows[today_rows['hour_numeric'] <= now_hour]
        .groupby('hour_numeric')['percent_full'].mean()
    )
    finger_slots = sorted(today_finger.index.tolist())

    if len(finger_slots) < MIN_SLOTS:
        return [], open_h, today_finger

    target_ts = pd.Timestamp(target_date)
    pool = df_all[
        (df_all['date'] != target_date) &
        (abs((pd.to_datetime(df_all['date']) - target_ts).dt.days) > EXCLUDE_WINDOW) &
        (df_all['day_name'] == day_name) &
        (df_all['is_semester'] == target_is_sem)
    ]

    candidates = []
    for _, group in pool.groupby('date'):
        cf = group.groupby('hour_numeric')['percent_full'].mean()
        available = [s for s in finger_slots if s in cf.index]
        if len(available) < len(finger_slots) * OVERLAP_THRESH:
            continue
        hist_vec  = np.array([cf[s]               for s in available])
        today_sub = np.array([today_finger[s]      for s in available])
        dist = float(np.sqrt(np.mean((hist_vec - today_sub) ** 2)))
        candidates.append((dist, group))

    candidates.sort(key=lambda x: x[0])
    return candidates, open_h, today_finger


def predict_with_config(candidates, slots, ml_preds, now_hour, open_h, K, dist_weighted, ceiling,
                        anchor=False, today_finger=None):
    """Blend ML + similarity for the given config; return per-slot predictions."""
    top_k = candidates[:K]
    if not top_k:
        return None

    dists = np.array([c[0] for c in top_k])
    if dist_weighted:
        weights = 1.0 / (dists + 1e-6)
        weights /= weights.sum()
    else:
        weights = np.ones(len(top_k)) / len(top_k)

    def sim_for_slot(slot):
        vals, wts = [], []
        for (_, grp), w in zip(top_k, weights):
            gf = grp.groupby('hour_numeric')['percent_full'].mean()
            if slot in gf.index:
                vals.append(gf[slot])
                wts.append(w)
        if not vals:
            return None
        wa = np.array(wts); wa /= wa.sum()
        return float(np.dot(wa, vals))

    sim_preds = {slot: v for slot in slots if (v := sim_for_slot(slot)) is not None}

    # Anchor correction: shift sim predictions so they start from today's actual level
    if anchor and today_finger is not None and len(today_finger) > 0:
        last_slot = max(today_finger.index)
        sim_at_last = sim_for_slot(last_slot)
        if sim_at_last is not None:
            offset = float(today_finger[last_slot]) - sim_at_last
            sim_preds = {
                slot: max(0.0, min(100.0, y + offset * max(0.0, 1 - (slot - last_slot) / 4)))
                for slot, y in sim_preds.items()
            }

    blend_w = round(min((now_hour - open_h) / 6.0, ceiling), 3)

    blended = {}
    for slot in slots:
        if slot in ml_preds and slot in sim_preds:
            blended[slot] = (1 - blend_w) * ml_preds[slot] + blend_w * sim_preds[slot]

    return blended


SEEDS = [42, 0, 123]


def run_one_seed(df, rf, scaler, mlp_model, eligible, seed):
    rng           = np.random.default_rng(seed)
    sampled_dates = sorted(rng.choice(eligible, size=min(SAMPLE_SIZE, len(eligible)), replace=False).tolist())

    errors = {name: {h: [] for h in SNAPSHOT_HOURS} for name in CONFIGS}
    ml_err = {h: [] for h in SNAPSHOT_HOURS}

    for i, target_date in enumerate(sampled_dates):
        day_name        = pd.Timestamp(target_date).day_name()
        open_h, close_h = get_open_hours(day_name)
        actual_map      = df[df['date'] == target_date].groupby('hour_numeric')['percent_full'].mean().to_dict()

        for snap_h in SNAPSHOT_HOURS:
            if snap_h < open_h or snap_h >= close_h:
                continue

            future_slots = [h + m/60 for h in range(open_h, close_h) for m in (0,15,30,45) if h+m/60 >= snap_h]
            actuals      = {s: actual_map[s] for s in future_slots if s in actual_map}
            if not actuals:
                continue

            ml_preds = get_ml_predictions(rf, scaler, mlp_model, target_date, list(actuals.keys()))
            for s, a in actuals.items():
                if s in ml_preds:
                    ml_err[snap_h].append(abs(a - ml_preds[s]))

            candidates, oh, today_finger = get_candidates(df, target_date, snap_h)
            if not candidates:
                continue

            for name, cfg in CONFIGS.items():
                blended = predict_with_config(
                    candidates, list(actuals.keys()), ml_preds,
                    snap_h, oh, cfg['K'], cfg['dist_weighted'], cfg['ceiling'],
                    anchor=cfg.get('anchor', False), today_finger=today_finger,
                )
                if blended:
                    for s, a in actuals.items():
                        if s in blended:
                            errors[name][snap_h].append(abs(a - blended[s]))

        if (i + 1) % 50 == 0:
            print(f'  seed={seed}: {i+1}/{len(sampled_dates)} days...')

    # Per-snapshot-hour MAE (unweighted mean of errors within each hour bucket)
    ml_maes  = {h: np.mean(ml_err[h]) if ml_err[h] else float('nan') for h in SNAPSHOT_HOURS}
    cfg_maes = {
        name: {h: np.mean(errors[name][h]) if errors[name][h] else float('nan') for h in SNAPSHOT_HOURS}
        for name in CONFIGS
    }
    return ml_maes, cfg_maes


def main():
    import warnings; warnings.filterwarnings('ignore')

    print('Loading data...')
    conn = sqlite3.connect('gym_history.db')
    df   = pd.read_sql_query('SELECT timestamp, percent_full FROM capacity_log', conn)
    conn.close()

    df['timestamp']   = pd.to_datetime(df['timestamp'])
    df['date']        = df['timestamp'].dt.date
    df['hour_numeric']= ((df['timestamp'].dt.hour + df['timestamp'].dt.minute/60)*4).round()/4
    df['day_name']    = df['timestamp'].dt.day_name()
    df['is_semester'] = df['date'].apply(is_semester_day)

    print('Loading models...')
    rf, scaler, mlp_model = load_models()

    cutoff_end = date.today() - timedelta(days=14)
    day_counts = df.groupby('date')['hour_numeric'].count()
    eligible   = [d for d, n in day_counts.items() if n >= 40 and date(2022,1,1) <= d <= cutoff_end]
    print(f'Eligible days in pool: {len(eligible)}\n')

    # Accumulate MAEs across seeds: all_ml[h] = [mae_seed1, mae_seed2, ...]
    all_ml  = {h: [] for h in SNAPSHOT_HOURS}
    all_cfg = {name: {h: [] for h in SNAPSHOT_HOURS} for name in CONFIGS}

    col_headers = ''.join(f'  {f"{h}:00":>7}' for h in SNAPSHOT_HOURS)
    width = 42 + len(col_headers) + 9

    for seed in SEEDS:
        print(f'── Seed {seed} ({SAMPLE_SIZE} days) ──────────────────────────────────')
        ml_maes, cfg_maes = run_one_seed(df, rf, scaler, mlp_model, eligible, seed)

        # Print per-seed table
        print(f'\n  {"Config":<40}{col_headers}  {"Avg":>7}')
        print('  ' + '-'*(width-2))
        for label, maes_dict in [('Base ML', ml_maes)] + [(n, cfg_maes[n]) for n in CONFIGS]:
            maes = [maes_dict[h] for h in SNAPSHOT_HOURS]
            avg  = np.nanmean(maes)
            print(f'  {label:<40}' + ''.join(f'  {v:>6.2f}%' for v in maes) + f'  {avg:>6.2f}%')
        print()

        for h in SNAPSHOT_HOURS:
            all_ml[h].append(ml_maes[h])
        for name in CONFIGS:
            for h in SNAPSHOT_HOURS:
                all_cfg[name][h].append(cfg_maes[name][h])

    # ── Aggregate across all seeds ────────────────────────────────────────────
    print('=' * width)
    print(f'AGGREGATE ({len(SEEDS)} seeds × {SAMPLE_SIZE} days = {len(SEEDS)*SAMPLE_SIZE} total)')
    print('=' * width)
    print(f'  {"Config":<40}{col_headers}  {"Avg":>7}')
    print('  ' + '-'*(width-2))

    def agg_maes(per_h): return [np.nanmean(per_h[h]) for h in SNAPSHOT_HOURS]

    rows = [('Base ML (no similarity)', agg_maes(all_ml))]
    for name in CONFIGS:
        rows.append((name, agg_maes(all_cfg[name])))

    avgs = []
    for label, maes in rows:
        avg = np.nanmean(maes)
        avgs.append(avg)
        print(f'  {label:<40}' + ''.join(f'  {v:>6.2f}%' for v in maes) + f'  {avg:>6.2f}%')

    print('=' * width)
    print(f'\nNote: each snapshot-hour MAE is the mean error across all (day × future slot) pairs')
    print(f'at that observation time. Early snapshots (9 AM) evaluate ~14 hrs of future slots;')
    print(f'late snapshots (7 PM) evaluate ~4 hrs — so each hour bucket has different sample counts.\n')
    print('Improvement over Base ML (avg across hours & seeds):')
    for (label, _), avg in zip(rows[1:], avgs[1:]):
        print(f'  {label.strip()}: {avgs[0] - avg:+.2f}pp ({(avgs[0]-avg)/avgs[0]*100:.1f}% relative)')
    print()


if __name__ == '__main__':
    main()
