"""
predictions_builder.py — compute 180-day RF+MLP predictions → Supabase predictions table.
Runs daily at midnight PT via daily.yml.
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client

from train import GymMLP, engineer_features

PT  = ZoneInfo("America/Los_Angeles")
now = datetime.now(PT)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

BATCH_SIZE = 500


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
    model = GymMLP(config['n_features'])
    model.load_state_dict(torch.load('models/pytorch_model.pt', weights_only=True))
    model.eval()
    return rf, scaler, model


def compute_predictions(rf, scaler, mlp_model, days=180):
    """Build (slot_ts ISO string, rf_pct, mlp_pct) for every open 15-min slot over the next N days."""
    timestamps = []
    slot_ts    = []

    for offset in range(days):
        d        = now.date() + timedelta(days=offset)
        day_name = pd.Timestamp(d).day_name()
        open_h, close_h = get_open_hours(day_name)
        for h in range(open_h, close_h):
            for m in (0, 15, 30, 45):
                # Store as PT-aware ISO timestamp for Supabase TIMESTAMPTZ
                dt = datetime(d.year, d.month, d.day, h, m, tzinfo=PT)
                slot_ts.append(dt.isoformat())
                timestamps.append(pd.Timestamp(f'{d} {h:02d}:{m:02d}'))

    df = pd.DataFrame({
        'timestamp':    timestamps,
        'people_count': [100] * len(timestamps),
        'percent_full': [66.7] * len(timestamps),
    })

    print(f"  Engineering features for {len(df):,} slots...")
    X, _ = engineer_features(df)

    rf_preds = rf.predict(X)
    X_scaled = scaler.transform(X)
    with torch.no_grad():
        mlp_preds = mlp_model(
            torch.tensor(X_scaled, dtype=torch.float32)
        ).numpy().flatten()

    return [
        {
            "slot_ts": ts,
            "rf_pct":  round(min(float(rf_p), 100.0), 1),
            "mlp_pct": round(min(float(mlp_p), 100.0), 1),
        }
        for ts, rf_p, mlp_p in zip(slot_ts, rf_preds, mlp_preds)
    ]


def main():
    print("Loading models...")
    rf, scaler, mlp_model = load_models()

    print("Computing 180-day predictions...")
    records = compute_predictions(rf, scaler, mlp_model, days=180)
    print(f"  {len(records):,} slots computed")

    print("Upserting to Supabase predictions table...")
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        sb.table("predictions").upsert(batch, on_conflict="slot_ts").execute()
        print(f"  Upserted {min(i + BATCH_SIZE, len(records))}/{len(records)}")

    print(f"[{now.isoformat()}] predictions table updated: {len(records)} rows")


if __name__ == "__main__":
    main()
