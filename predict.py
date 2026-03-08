"""
Compare model predictions against actual recorded data.
Run with: python3 predict.py
"""
import pickle, sqlite3, torch, pandas as pd
from train import GymMLP, engineer_features

# Load models
rf     = pickle.load(open("models/rf_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
config = pickle.load(open("models/model_config.pkl", "rb"))
model  = GymMLP(config['n_features'])
model.load_state_dict(torch.load("models/pytorch_model.pt", weights_only=True))
model.eval()

# Load actual data from DB
conn = sqlite3.connect("gym_history.db")
actual_df = pd.read_sql_query("SELECT timestamp, percent_full FROM capacity_log", conn)
conn.close()
actual_df['timestamp'] = pd.to_datetime(actual_df['timestamp'])
actual_df['hour'] = actual_df['timestamp'].dt.hour
actual_df['day_of_week'] = actual_df['timestamp'].dt.day_name()
actual_df['month'] = actual_df['timestamp'].dt.month
actual_df['year'] = actual_df['timestamp'].dt.year


def predict(ts_str):
    df = pd.DataFrame({"timestamp": [pd.Timestamp(ts_str)], "people_count": [100], "percent_full": [66.7]})
    X, _ = engineer_features(df)
    rf_p = rf.predict(X)[0]
    with torch.no_grad():
        mlp_p = model(torch.tensor(scaler.transform(X), dtype=torch.float32)).item()
    return rf_p, mlp_p


def actual_avg(day_name, hour, month=None, year=None):
    """Average actual capacity for a given day+hour, optionally filtered by month/year."""
    mask = (actual_df['day_of_week'] == day_name) & (actual_df['hour'] == hour)
    if month:
        mask &= actual_df['month'] == month
    if year:
        mask &= actual_df['year'] == year
    rows = actual_df[mask]
    return rows['percent_full'].mean() if len(rows) > 0 else None


# ── SCENARIO TABLE ──────────────────────────────────────────────────────────
# For each scenario we show: RF pred, MLP pred, and the historical average
# for that same day-of-week + hour slot across all data we have.

scenarios = [
    # (label,                        ts_str,              day,       hr, month, year)
    ("Mon 9 AM  (normal morning)",   "2026-03-09 09:00",  "Monday",   9, None,  None),
    ("Mon 5 PM  (peak hour)",        "2026-03-09 17:00",  "Monday",  17, None,  None),
    ("Mon 10 PM (late night)",       "2026-03-09 22:00",  "Monday",  22, None,  None),
    ("Sat 12 PM (weekend)",          "2026-03-14 12:00",  "Saturday",12, None,  None),
    ("Spring break Tue 2 PM",        "2026-03-17 14:00",  "Tuesday", 14,    3,  None),
    ("Finals Mon 10 AM",             "2026-05-04 10:00",  "Monday",  10,    5,  None),
    ("Summer Tue 5 PM",              "2025-07-15 17:00",  "Tuesday", 17,    7,  None),
    ("Winter break Wed 2 PM",        "2025-12-24 14:00",  "Wednesday",14,  12,  None),
    ("First week Mon 5 PM",          "2025-08-25 17:00",  "Monday",  17,    8,  None),
    ("Dead week Mon 5 PM",           "2026-04-27 17:00",  "Monday",  17,    4,  None),
]

print(f"\n{'Scenario':<30} {'RF':>6}  {'MLP':>6}  {'Actual avg':>10}  {'RF err':>7}  {'MLP err':>8}")
print("-" * 75)
for label, ts, day, hr, mo, yr in scenarios:
    rf_p, mlp_p = predict(ts)
    avg = actual_avg(day, hr, mo, yr)
    if avg is not None:
        rf_err  = rf_p  - avg
        mlp_err = mlp_p - avg
        print(f"{label:<30} {rf_p:>5.1f}%  {mlp_p:>5.1f}%  {avg:>9.1f}%  {rf_err:>+6.1f}%  {mlp_err:>+7.1f}%")
    else:
        print(f"{label:<30} {rf_p:>5.1f}%  {mlp_p:>5.1f}%  {'no data':>10}")

# ── BROADER BACKTEST ─────────────────────────────────────────────────────────
# Take 1000 evenly-spaced rows from the DB and measure average error
print("\n── Backtest on 1,000 historical data points ──")
conn = sqlite3.connect("gym_history.db")
sample = pd.read_sql_query("""
    SELECT timestamp, people_count, percent_full FROM capacity_log
    WHERE people_count > 5
    ORDER BY timestamp
""", conn)
conn.close()

sample = sample.iloc[::len(sample)//1000].head(1000).copy()
sample['timestamp'] = pd.to_datetime(sample['timestamp'])
X_all, _ = engineer_features(sample)
rf_preds = rf.predict(X_all)
X_scaled = scaler.transform(X_all)
with torch.no_grad():
    mlp_preds = model(torch.tensor(X_scaled, dtype=torch.float32)).squeeze().numpy()

actual = sample['percent_full'].values
rf_mae  = abs(rf_preds  - actual).mean()
mlp_mae = abs(mlp_preds - actual).mean()
print(f"  RF  mean absolute error: {rf_mae:.1f}%")
print(f"  MLP mean absolute error: {mlp_mae:.1f}%")
print()
