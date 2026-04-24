#!/usr/bin/env python3
"""
backtest.py — Compare pure ML predictions vs blended ML+similarity predictions.

For a sample of historical days, simulates what each approach would have predicted
at snapshot hours throughout the day, then measures MAE against actuals.

Usage:
    python3 backtest.py
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
from datetime import date, timedelta
from zoneinfo import ZoneInfo

from supabase import create_client
from train import GymMLP, engineer_features

warnings.filterwarnings('ignore')

PT = ZoneInfo("America/Los_Angeles")

SPRING_BREAKS = [
    ('2021-03-20', '2021-03-28'), ('2022-03-19', '2022-03-27'),
    ('2023-03-25', '2023-04-02'), ('2024-03-23', '2024-03-31'),
    ('2025-03-22', '2025-03-30'), ('2026-03-21', '2026-03-29'),
    ('2027-03-20', '2027-03-28'), ('2028-03-25', '2028-04-02'),
]

# Production config (matches today_builder.py exactly)
K              = 5
BLEND_CEIL     = 0.9
OVERLAP_THRESH = 0.70
EXCLUDE_WINDOW = 7
MIN_SLOTS      = 4

SNAPSHOT_HOURS = [9, 11, 13, 15, 19]
SAMPLE_SIZE    = 150
SEEDS          = [42, 0, 123]
# Only look at days within the current model's training window to keep ML fair
LOOKBACK_DAYS  = 365


def get_open_hours(day_name):
    if day_name == 'Saturday': return 8, 18
    elif day_name == 'Sunday':  return 8, 23
    else:                        return 7, 23


def is_semester_day(d):
    month, dom = d.month, d.day
    is_summer  = month in [6, 7, 8]
    is_winter  = (month == 12 and dom >= 16) or (month == 1 and dom <= 12)
    is_sb      = any(pd.Timestamp(s).date() <= d <= pd.Timestamp(e).date() for s, e in SPRING_BREAKS)
    return not (is_summer or is_winter or is_sb)


def fetch_history(sb):
    BATCH, offset, rows = 9000, 0, []
    while True:
        batch = (
            sb.table("capacity_log")
            .select("timestamp,percent_full")
            .range(offset, offset + BATCH - 1)
            .order("timestamp")
            .execute()
            .data
        )
        rows.extend(batch)
        if len(batch) < BATCH:
            break
        offset += BATCH
    return rows


def load_models():
    with open('models/rf_model.pkl', 'rb') as f:     rf = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:       scaler = pickle.load(f)
    with open('models/model_config.pkl', 'rb') as f: config = pickle.load(f)
    model = GymMLP(config['n_features'])
    model.load_state_dict(torch.load('models/pytorch_model.pt', weights_only=True))
    model.eval()
    return rf, scaler, model


def get_ml_preds(rf, scaler, mlp_model, target_date, slots):
    if not slots:
        return {}
    timestamps = [pd.Timestamp(f'{target_date} {int(s):02d}:{round((s % 1) * 60):02d}') for s in slots]
    df = pd.DataFrame({'timestamp': timestamps, 'people_count': [100]*len(timestamps), 'percent_full': [66.7]*len(timestamps)})
    X, _ = engineer_features(df)
    rf_p  = rf.predict(X)
    X_sc  = scaler.transform(X)
    with torch.no_grad():
        mlp_p = mlp_model(torch.tensor(X_sc, dtype=torch.float32)).numpy().flatten()
    return {s: float(min((r + m) / 2, 100)) for s, r, m in zip(slots, rf_p, mlp_p)}


def get_blended_preds(df_all, target_date, now_hour, ml_preds, future_slots):
    """
    Runs the production similarity algorithm (matches today_builder.py exactly)
    and blends with ML. Returns blended dict or None if similarity couldn't run.
    """
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
        return None

    target_ts  = pd.Timestamp(target_date)
    pool = df_all[
        (df_all['date'] != target_date) &
        (abs((pd.to_datetime(df_all['date']) - target_ts).dt.days) > EXCLUDE_WINDOW) &
        (df_all['day_name'] == day_name) &
        (df_all['is_semester'] == target_is_sem)
    ]

    candidates = []
    for _, group in pool.groupby('date'):
        cf        = group.groupby('hour_numeric')['percent_full'].mean()
        available = [s for s in finger_slots if s in cf.index]
        if len(available) < len(finger_slots) * OVERLAP_THRESH:
            continue
        dist = float(np.sqrt(np.mean(
            (np.array([cf[s] for s in available]) - np.array([today_finger[s] for s in available])) ** 2
        )))
        candidates.append((dist, group))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    top_k   = candidates[:K]
    dists   = np.array([c[0] for c in top_k])
    weights = 1.0 / (dists + 1e-6)
    weights /= weights.sum()

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

    sim_preds = {s: v for s in future_slots if (v := sim_for_slot(s)) is not None}

    # Anchor correction (same as today_builder.py)
    last_slot     = max(finger_slots)
    sim_at_last   = sim_for_slot(last_slot)
    actual_at_last = float(today_finger[last_slot])
    if sim_at_last is not None:
        offset = actual_at_last - sim_at_last
        sim_preds = {
            s: max(0.0, min(100.0, y + offset * max(0.0, 1 - (s - last_slot) / 4.0)))
            for s, y in sim_preds.items()
        }

    blend_w = min((now_hour - open_h) / 6.0, BLEND_CEIL)

    return {
        s: (1 - blend_w) * ml_preds[s] + blend_w * sim_preds[s]
        for s in future_slots
        if s in ml_preds and s in sim_preds
    }


def run_one_seed(df, rf, scaler, mlp_model, eligible, seed):
    rng           = np.random.default_rng(seed)
    sampled_dates = sorted(rng.choice(eligible, size=min(SAMPLE_SIZE, len(eligible)), replace=False).tolist())

    ml_err      = {h: [] for h in SNAPSHOT_HOURS}
    blended_err = {h: [] for h in SNAPSHOT_HOURS}

    for i, target_date in enumerate(sampled_dates):
        day_name        = pd.Timestamp(target_date).day_name()
        open_h, close_h = get_open_hours(day_name)
        actual_map      = df[df['date'] == target_date].groupby('hour_numeric')['percent_full'].mean().to_dict()

        for snap_h in SNAPSHOT_HOURS:
            if snap_h < open_h or snap_h >= close_h:
                continue

            future_slots = [h + m/60 for h in range(open_h, close_h) for m in (0,15,30,45) if h + m/60 >= snap_h]
            actuals      = {s: actual_map[s] for s in future_slots if s in actual_map}
            if not actuals:
                continue

            ml_preds = get_ml_preds(rf, scaler, mlp_model, target_date, list(actuals.keys()))
            for s, a in actuals.items():
                if s in ml_preds:
                    ml_err[snap_h].append(abs(a - ml_preds[s]))

            blended = get_blended_preds(df, target_date, snap_h, ml_preds, list(actuals.keys()))
            if blended:
                for s, a in actuals.items():
                    if s in blended:
                        blended_err[snap_h].append(abs(a - blended[s]))

        if (i + 1) % 50 == 0:
            print(f'  seed={seed}: {i+1}/{len(sampled_dates)} days...')

    ml_maes      = {h: np.mean(ml_err[h])      if ml_err[h]      else float('nan') for h in SNAPSHOT_HOURS}
    blended_maes = {h: np.mean(blended_err[h]) if blended_err[h] else float('nan') for h in SNAPSHOT_HOURS}
    return ml_maes, blended_maes


def print_table(label, rows, snapshot_hours):
    col_w    = 9
    name_w   = 12
    headers  = ''.join(f'{f"{h}:00":>{col_w}}' for h in snapshot_hours)
    sep      = '-' * (name_w + len(headers) + col_w)
    print(f'\n{label}')
    print(f'  {"":>{name_w}}{headers}  {"Avg":>{col_w}}')
    print('  ' + sep)
    for name, maes in rows:
        vals = [maes[h] for h in snapshot_hours]
        avg  = np.nanmean(vals)
        print(f'  {name:>{name_w}}' + ''.join(f'{v:>{col_w}.2f}%' for v in vals) + f'  {avg:>{col_w}.2f}%')


def main():
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    print('Fetching capacity_log from Supabase...')
    rows = fetch_history(sb)
    print(f'  {len(rows):,} rows loaded')

    df = pd.DataFrame(rows)
    df['timestamp']    = pd.to_datetime(df['timestamp'], format='ISO8601').dt.tz_convert(PT).dt.tz_localize(None)
    df['date']         = df['timestamp'].dt.date
    df['hour_numeric'] = ((df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60) * 4).round() / 4
    df['day_name']     = df['timestamp'].dt.day_name()
    df['is_semester']  = df['date'].apply(is_semester_day)
    df = df[df['percent_full'] > 5]  # same filter as training

    print('Loading models...')
    rf, scaler, mlp_model = load_models()

    cutoff_end   = date.today() - timedelta(days=14)
    cutoff_start = date.today() - timedelta(days=LOOKBACK_DAYS)
    day_counts   = df.groupby('date')['hour_numeric'].count()
    eligible     = [
        d for d, n in day_counts.items()
        if n >= 40 and cutoff_start <= d <= cutoff_end
    ]
    print(f'Eligible days (last {LOOKBACK_DAYS}d, ≥40 readings, excluding last 14d): {len(eligible)}\n')

    all_ml      = {h: [] for h in SNAPSHOT_HOURS}
    all_blended = {h: [] for h in SNAPSHOT_HOURS}

    for seed in SEEDS:
        print(f'── Seed {seed} ({min(SAMPLE_SIZE, len(eligible))} days) ──────────────────────')
        ml_maes, blended_maes = run_one_seed(df, rf, scaler, mlp_model, eligible, seed)

        print_table(f'Seed {seed}:', [('Pure ML', ml_maes), ('Blended', blended_maes)], SNAPSHOT_HOURS)
        print()

        for h in SNAPSHOT_HOURS:
            all_ml[h].append(ml_maes[h])
            all_blended[h].append(blended_maes[h])

    # Aggregate across seeds
    def agg(per_h): return {h: np.nanmean(per_h[h]) for h in SNAPSHOT_HOURS}

    agg_ml      = agg(all_ml)
    agg_blended = agg(all_blended)

    print('=' * 70)
    print(f'AGGREGATE ({len(SEEDS)} seeds × up to {SAMPLE_SIZE} days each)')
    print('=' * 70)
    print_table('', [('Pure ML', agg_ml), ('Blended', agg_blended)], SNAPSHOT_HOURS)

    ml_avg      = np.nanmean(list(agg_ml.values()))
    blended_avg = np.nanmean(list(agg_blended.values()))
    delta       = ml_avg - blended_avg
    print(f'\n  Blended vs Pure ML: {delta:+.2f}pp avg MAE  ({delta/ml_avg*100:.1f}% relative)')
    if delta > 0:
        print('  → Blending HELPS (lower MAE than pure ML)')
    elif delta < 0:
        print('  → Blending HURTS (higher MAE than pure ML)')
    else:
        print('  → No difference')

    print(f'\n  Note: each snapshot-hour evaluates all remaining slots that day.')
    print(f'  9:00 covers ~14h of future; 19:00 covers ~4h — sample counts differ.')
    print(f'  Lookback: last {LOOKBACK_DAYS} days (aligns with current model checkpoint).\n')


if __name__ == '__main__':
    main()
