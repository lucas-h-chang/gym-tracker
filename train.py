import os
import json
import pickle
from datetime import datetime, date

import pandas as pd

# scikit-learn: classical ML library. We use it for:
#   - RandomForestRegressor: our baseline model
#   - StandardScaler: normalizes feature values for PyTorch
#   - mean_squared_error / mean_absolute_error: measuring prediction accuracy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# PyTorch: deep learning framework. We use it to build and train a neural network by hand.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MAX_CAPACITY = 150


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
# Raw timestamps are useless to a model. We extract meaningful numbers from them.
# Each row gets transformed into a set of "features" — the inputs the model learns from.
#
# Think of features like the information you'd use yourself:
#   "Is it Monday? Is it 5 PM? Is it finals week?"
# We just encode that as numbers so the model can use it.
#
# This function is defined at module level (outside main) so test files can
# import and test it independently without triggering training.
# ==============================================================================

def engineer_features(df):
    """
    Takes a DataFrame with a 'timestamp' column and returns a feature matrix X.
    Also returns a list of column names (useful for the feature importance chart).
    """
    df = df.copy()  # don't modify the original DataFrame

    # --- Hour as a float ---
    # 3:30 PM becomes 15.5. This preserves the ordering (3:30 is between 3:00 and 4:00).
    df['hour_numeric'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60

    # --- Week of year ---
    # Week 1–52, a rough proxy for semester phase.
    # The model learns "week 35 of the year is move-in week → very busy".
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)

    # --- Is weekend ---
    # A single binary flag. Weekends have very different patterns from weekdays.
    df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)

    # --- Berkeley Academic Calendar flags ---
    # These capture events the model can't infer from the date alone.
    # Hardcoded based on Berkeley's official academic calendar.
    #
    # Why hardcode instead of compute? Because "finals week" isn't derivable from
    # the calendar date — it's specific to Berkeley's schedule each year.

    # ── BERKELEY ACADEMIC CALENDAR ─────────────────────────────────────────────
    # All dates sourced from official UCB academic calendars (2020-21 through
    # 2027-28). Parsed from PDFs provided by the user.
    #
    # Rules applied:
    #   • "Instruction Begins" = first day of class (not semester begins)
    #   • "Formal Classes End" = last day before dead week; dead week starts
    #     the following Monday (Reading/Review/Recitation Week)
    #   • Spring Recess = official dates only (no extended surrounding weekends)
    #   • Winter break = Fall Semester Ends → day before Spring Instruction Begins
    #   • Summer break = Spring Semester Ends → day before Fall Instruction Begins
    # ──────────────────────────────────────────────────────────────────────────

    # Academic holidays: Labor Day, Veterans Day, Thanksgiving + day before,
    # Christmas, New Year's, MLK Day, Presidents Day, Good Friday, etc.
    # Used with a set for O(1) per-row lookup.
    HOLIDAYS = {
        # ── Fall 2020 ──
        date(2020, 9,  7), date(2020, 11, 11), date(2020, 11, 25),
        date(2020, 11, 26), date(2020, 11, 27),
        date(2020, 12, 24), date(2020, 12, 25), date(2020, 12, 31),
        # ── Spring 2021 ──
        date(2021, 1,  1), date(2021, 1, 18), date(2021, 2, 15),
        date(2021, 3, 26), date(2021, 5, 31),
        # ── Fall 2021 ──
        date(2021, 9,  6), date(2021, 11, 11), date(2021, 11, 24),
        date(2021, 11, 25), date(2021, 11, 26),
        date(2021, 12, 23), date(2021, 12, 24), date(2021, 12, 30), date(2021, 12, 31),
        # ── Spring 2022 ──
        date(2022, 1, 17), date(2022, 2, 21), date(2022, 3, 25), date(2022, 5, 30),
        # ── Fall 2022 ──
        date(2022, 9,  5), date(2022, 11, 11), date(2022, 11, 23),
        date(2022, 11, 24), date(2022, 11, 25),
        date(2022, 12, 23), date(2022, 12, 26), date(2022, 12, 30),
        date(2023, 1,  2),
        # ── Spring 2023 ──
        date(2023, 1, 16), date(2023, 2, 20), date(2023, 3, 31),
        # ── Fall 2023 ──
        date(2023, 9,  4), date(2023, 11, 10), date(2023, 11, 22),
        date(2023, 11, 23), date(2023, 11, 24),
        date(2023, 12, 25), date(2023, 12, 26),
        date(2024, 1,  1), date(2024, 1,  2),
        # ── Spring 2024 ──
        date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
        # ── Fall 2024 ──
        date(2024, 9,  2), date(2024, 11, 11), date(2024, 11, 27),
        date(2024, 11, 28), date(2024, 11, 29),
        date(2024, 12, 24), date(2024, 12, 25), date(2024, 12, 31),
        date(2025, 1,  1),
        # ── Spring 2025 ──
        date(2025, 1, 20), date(2025, 2, 17), date(2025, 3, 28),
        # ── Fall 2025 ──
        date(2025, 9,  1), date(2025, 11, 11), date(2025, 11, 26),
        date(2025, 11, 27), date(2025, 11, 28),
        date(2025, 12, 24), date(2025, 12, 25), date(2025, 12, 31),
        date(2026, 1,  1),
        # ── Spring 2026 ──
        date(2026, 1, 19), date(2026, 2, 16), date(2026, 3, 27),
        # ── Fall 2026 ──
        date(2026, 9,  7), date(2026, 11, 11), date(2026, 11, 25),
        date(2026, 11, 26), date(2026, 11, 27),
        date(2026, 12, 24), date(2026, 12, 25), date(2026, 12, 31),
        date(2027, 1,  1),
        # ── Spring 2027 ──
        date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
        # ── Fall 2027 ──
        date(2027, 9,  6), date(2027, 11, 11), date(2027, 11, 24),
        date(2027, 11, 25), date(2027, 11, 26),
        date(2027, 12, 24), date(2027, 12, 27), date(2027, 12, 31),
        date(2028, 1,  3),
        # ── Spring 2028 ──
        date(2028, 1, 17), date(2028, 2, 21), date(2028, 3, 31),
    }

    def get_calendar_flags(ts):
        d = ts.date()

        # Finals week — exact dates from official PDFs
        finals_ranges = [
            # Spring finals
            (date(2021, 5, 10), date(2021, 5, 14)),
            (date(2022, 5,  9), date(2022, 5, 13)),
            (date(2023, 5,  8), date(2023, 5, 12)),
            (date(2024, 5,  6), date(2024, 5, 10)),
            (date(2025, 5, 12), date(2025, 5, 16)),
            (date(2026, 5, 11), date(2026, 5, 15)),
            (date(2027, 5, 10), date(2027, 5, 14)),
            (date(2028, 5,  8), date(2028, 5, 12)),
            # Fall finals
            (date(2020, 12, 14), date(2020, 12, 18)),
            (date(2021, 12, 13), date(2021, 12, 17)),
            (date(2022, 12, 12), date(2022, 12, 16)),
            (date(2023, 12, 11), date(2023, 12, 15)),
            (date(2024, 12, 16), date(2024, 12, 20)),
            (date(2025, 12, 15), date(2025, 12, 19)),
            (date(2026, 12, 14), date(2026, 12, 18)),
            (date(2027, 12, 13), date(2027, 12, 17)),
        ]
        is_finals = int(any(start <= d <= end for start, end in finals_ranges))

        # Dead week = Reading/Review/Recitation Week (Mon–Fri after Formal Classes End)
        dead_week_ranges = [
            # Spring dead weeks
            (date(2021, 5,  3), date(2021, 5,  7)),
            (date(2022, 5,  2), date(2022, 5,  6)),
            (date(2023, 5,  1), date(2023, 5,  5)),
            (date(2024, 4, 29), date(2024, 5,  3)),
            (date(2025, 5,  5), date(2025, 5,  9)),
            (date(2026, 5,  4), date(2026, 5,  8)),
            (date(2027, 5,  3), date(2027, 5,  7)),
            (date(2028, 5,  1), date(2028, 5,  5)),
            # Fall dead weeks
            (date(2020, 12,  7), date(2020, 12, 11)),
            (date(2021, 12,  6), date(2021, 12, 10)),
            (date(2022, 12,  5), date(2022, 12,  9)),
            (date(2023, 12,  4), date(2023, 12,  8)),
            (date(2024, 12,  9), date(2024, 12, 13)),
            (date(2025, 12,  8), date(2025, 12, 12)),
            (date(2026, 12,  7), date(2026, 12, 11)),
            (date(2027, 12,  6), date(2027, 12, 10)),
        ]
        is_dead_week = int(any(start <= d <= end for start, end in dead_week_ranges))

        # First week = Instruction Begins through +6 days (not "Semester Begins")
        first_week_ranges = [
            (date(2020, 8, 26), date(2020, 9,  1)),  # Fall 2020
            (date(2021, 1, 19), date(2021, 1, 25)),  # Spring 2021
            (date(2021, 8, 25), date(2021, 8, 31)),  # Fall 2021
            (date(2022, 1, 18), date(2022, 1, 24)),  # Spring 2022
            (date(2022, 8, 24), date(2022, 8, 30)),  # Fall 2022
            (date(2023, 1, 17), date(2023, 1, 23)),  # Spring 2023
            (date(2023, 8, 23), date(2023, 8, 29)),  # Fall 2023
            (date(2024, 1, 16), date(2024, 1, 22)),  # Spring 2024
            (date(2024, 8, 28), date(2024, 9,  3)),  # Fall 2024
            (date(2025, 1, 21), date(2025, 1, 27)),  # Spring 2025
            (date(2025, 8, 27), date(2025, 9,  2)),  # Fall 2025
            (date(2026, 1, 20), date(2026, 1, 26)),  # Spring 2026
            (date(2026, 8, 26), date(2026, 9,  1)),  # Fall 2026
            (date(2027, 1, 19), date(2027, 1, 25)),  # Spring 2027
            (date(2027, 8, 25), date(2027, 8, 31)),  # Fall 2027
            (date(2028, 1, 18), date(2028, 1, 24)),  # Spring 2028
        ]
        is_first_week = int(any(start <= d <= end for start, end in first_week_ranges))

        # Break ranges — official Spring Recess dates + winter and summer gaps
        # Winter: Fall Semester Ends → day before Spring Instruction Begins
        # Summer: Spring Semester Ends → day before Fall Instruction Begins
        # Spring Recess: exact Mon–Fri (+ Good Friday) from official calendar
        break_ranges = [
            # ── Winter breaks ──
            (date(2020, 12, 18), date(2021, 1, 18)),  # winter 20-21
            (date(2021, 12, 17), date(2022, 1, 17)),  # winter 21-22
            (date(2022, 12, 16), date(2023, 1, 16)),  # winter 22-23
            (date(2023, 12, 15), date(2024, 1, 15)),  # winter 23-24
            (date(2024, 12, 20), date(2025, 1, 20)),  # winter 24-25
            (date(2025, 12, 19), date(2026, 1, 19)),  # winter 25-26
            (date(2026, 12, 18), date(2027, 1, 18)),  # winter 26-27
            (date(2027, 12, 17), date(2028, 1, 17)),  # winter 27-28
            # ── Spring Recesses (official Mon–Fri + surrounding Sat/Sun) ──
            (date(2021, 3, 20), date(2021, 3, 28)),   # spring recess 2021
            (date(2022, 3, 19), date(2022, 3, 27)),   # spring recess 2022
            (date(2023, 3, 25), date(2023, 4,  2)),   # spring recess 2023
            (date(2024, 3, 23), date(2024, 3, 31)),   # spring recess 2024
            (date(2025, 3, 22), date(2025, 3, 30)),   # spring recess 2025
            (date(2026, 3, 21), date(2026, 3, 29)),   # spring recess 2026
            (date(2027, 3, 20), date(2027, 3, 28)),   # spring recess 2027
            (date(2028, 3, 25), date(2028, 4,  2)),   # spring recess 2028
            # ── Summer breaks ──
            (date(2021, 5, 14), date(2021, 8, 24)),   # summer 2021
            (date(2022, 5, 13), date(2022, 8, 23)),   # summer 2022
            (date(2023, 5, 12), date(2023, 8, 22)),   # summer 2023
            (date(2024, 5, 10), date(2024, 8, 27)),   # summer 2024
            (date(2025, 5, 16), date(2025, 8, 26)),   # summer 2025
            (date(2026, 5, 15), date(2026, 8, 25)),   # summer 2026
            (date(2027, 5, 14), date(2027, 8, 24)),   # summer 2027
        ]
        is_break = int(any(start <= d <= end for start, end in break_ranges))

        # Only flag holidays that fall outside break periods — a holiday during
        # summer/winter/spring break is redundant (is_break already captures it)
        # and would dilute the signal. This makes is_holiday mean "semester holiday".
        is_holiday = int(d in HOLIDAYS and not is_break)

        return pd.Series([is_finals, is_dead_week, is_first_week, is_break, is_holiday])

    df[['is_finals', 'is_dead_week', 'is_first_week', 'is_break', 'is_holiday']] = \
        df['timestamp'].apply(get_calendar_flags)

    # --- Day of week: one-hot encoding ---
    # Why not just use 0=Mon, 1=Tue, ..., 6=Sun?
    # Because that implies Tuesday (1) is "more" than Monday (0), which is meaningless.
    # One-hot creates 7 separate binary columns: is_monday, is_tuesday, etc.
    # Only one is 1 at a time. The model treats each day independently.
    day_dummies = pd.get_dummies(df['timestamp'].dt.day_name(), prefix='day')

    # Ensure all 7 day columns always exist (in case some days are missing in small datasets)
    for day in ['day_Monday', 'day_Tuesday', 'day_Wednesday', 'day_Thursday',
                'day_Friday', 'day_Saturday', 'day_Sunday']:
        if day not in day_dummies.columns:
            day_dummies[day] = 0

    day_dummies = day_dummies[['day_Monday', 'day_Tuesday', 'day_Wednesday',
                                'day_Thursday', 'day_Friday', 'day_Saturday', 'day_Sunday']]

    feature_cols = ['hour_numeric', 'week_of_year', 'is_weekend', 'is_finals',
                    'is_dead_week', 'is_first_week', 'is_break', 'is_holiday']
    X = pd.concat([df[feature_cols], day_dummies], axis=1)

    return X, list(X.columns)


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
# Defined at module level so it can be imported by app.py and test files
# without running the full training script.
# ==============================================================================

class GymMLP(nn.Module):
    """
    A 3-layer feed-forward neural network for predicting gym capacity.

    Architecture:
      Input (n_features)
        → Linear → ReLU → Dropout(20%)
        → Linear → ReLU → Dropout(20%)
        → Linear → single output (percent_full)

    ReLU: sets negative values to 0. Without activation functions, stacking
    linear layers is mathematically equivalent to one linear layer — you can't
    learn curves. ReLU is what gives the network its expressive power.

    Dropout: randomly zeros 20% of neurons during training to prevent overfitting
    (memorizing training data). Disabled automatically during model.eval().
    """
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # forward() is called automatically when you do model(x)
        return self.network(x)


# ==============================================================================
# MAIN TRAINING SCRIPT
# ==============================================================================
# Everything below only runs when you execute: python3 train.py
# It does NOT run when another file does: from train import engineer_features
#
# This is the standard Python pattern for making a script both runnable
# and importable. `__name__` is "__main__" when run directly, but the
# module's name (e.g. "train") when imported.
# ==============================================================================

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    # --------------------------------------------------------------------------
    # STEP 1: LOAD & CLEAN DATA
    # --------------------------------------------------------------------------
    # We pull everything from the DB and do two filters:
    #   - people_count > 5: removes COVID zeros, sensor noise, and 2021 partial-
    #     reopening readings where 1-2 staff were the only ones in the building
    #   - dropna(): removes the 4 null rows we found in our data audit
    # --------------------------------------------------------------------------

    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    print("Loading data from Supabase capacity_log...")
    BATCH  = 10000
    offset = 0
    rows   = []
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
        print(f"  Fetched {len(rows):,} rows...")

    df = pd.DataFrame(rows)
    df['timestamp']    = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
    df['people_count'] = df['people_count'].astype(float)
    df = df[df['people_count'] > 5].dropna()
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Recalculate percent_full from raw people_count (no cap at 100).
    # We intentionally allow values above 100% so the model learns the full signal —
    # overcapacity conditions are real and meaningful (gym is MORE crowded than usual).
    # We'll cap the display in app.py, but train on the real numbers.
    df['percent_full'] = (df['people_count'] / MAX_CAPACITY) * 100

    print(f"  {len(df):,} rows after cleaning (removed noise/zeros)")
    print(f"  Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    # --------------------------------------------------------------------------
    # STEP 2: FEATURE ENGINEERING
    # --------------------------------------------------------------------------

    print("  Engineering calendar features (this takes a moment)...")
    X, feature_names = engineer_features(df)
    y = df['percent_full'].values

    print(f"  Feature matrix shape: {X.shape}  ({X.shape[1]} features per row)")
    print(f"  Features: {feature_names}")

    # --------------------------------------------------------------------------
    # STEP 3: TRAIN / TEST SPLIT
    # --------------------------------------------------------------------------
    # We hold back the most recent 20% of data as a "test set" — data the model
    # never sees during training.
    #
    # IMPORTANT: We split chronologically, not randomly.
    # If we shuffled randomly, future data could leak into training
    # (e.g. the model trains on March 5 data and "predicts" March 3).
    # That would make accuracy look great but fail in real life.
    # Chronological split simulates the real-world scenario: predict the future
    # using only the past.
    # --------------------------------------------------------------------------

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nTrain/test split:")
    print(f"  Training:  {len(X_train):,} rows  ({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[split_idx-1].date()})")
    print(f"  Testing:   {len(X_test):,} rows   ({df['timestamp'].iloc[split_idx].date()} → {df['timestamp'].iloc[-1].date()})")

    # --------------------------------------------------------------------------
    # STEP 4: RANDOM FOREST (scikit-learn)
    # --------------------------------------------------------------------------
    # A Random Forest trains 100 decision trees, each on a random subset of data.
    # Each tree asks a series of yes/no questions:
    #   "Is it after 5 PM? → Yes. Is it Monday? → No. Is it finals? → No → predict 45%"
    # The final prediction is the average of all 100 trees' outputs.
    #
    # Why is this good?
    #   - No feature scaling needed (trees use comparisons, not distances)
    #   - Naturally handles non-linear patterns (5 PM on Monday ≠ 5 PM on Saturday)
    #   - Gives us feature importance: which inputs matter most?
    #   - Works well even with structured/tabular data like ours
    #
    # random_state=42: makes results reproducible (same random seed every run)
    # n_estimators=20: 20 trees keeps the saved model small (<100 MB) for GitHub
    # max_depth=15: limits how deep each tree grows, further reducing file size
    # --------------------------------------------------------------------------

    print("\n--- Training Random Forest (scikit-learn) ---")
    rf = RandomForestRegressor(n_estimators=20, max_depth=15, random_state=42, n_jobs=-1)
    # .fit() is where all the learning happens. scikit-learn handles everything internally.
    rf.fit(X_train, y_train)

    rf_preds = rf.predict(X_test)
    rf_rmse = mean_squared_error(y_test, rf_preds) ** 0.5
    rf_mae  = mean_absolute_error(y_test, rf_preds)

    print(f"  RMSE: {rf_rmse:.2f}%   ← average prediction is off by this much")
    print(f"  MAE:  {rf_mae:.2f}%    ← simpler average error (less sensitive to big mistakes)")

    # Feature importance: how much did each feature reduce prediction error?
    importances = dict(zip(feature_names, rf.feature_importances_))
    print("  Top features by importance:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:5]:
        print(f"    {feat}: {imp:.3f}")

    # Save the trained Random Forest to disk as a .pkl (pickle) file.
    # Pickle serializes a Python object to bytes so it can be reloaded later.
    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    print("  Saved → models/rf_model.pkl")

    # --------------------------------------------------------------------------
    # STEP 5: PYTORCH MLP (Neural Network)
    # --------------------------------------------------------------------------

    print("\n--- Training PyTorch MLP ---")

    # Feature scaling: neural networks ARE sensitive to feature scale.
    # hour_numeric ranges 7–23, week_of_year ranges 1–52, is_finals is 0 or 1.
    # Without scaling, the large numbers dominate and the model trains poorly.
    #
    # StandardScaler transforms each feature to have mean=0 and std=1.
    # We fit ONLY on training data — using test data stats would be "cheating"
    # (peeking at future data to inform preprocessing).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit + transform training data
    X_test_scaled  = scaler.transform(X_test)        # transform only (use training stats)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Convert to PyTorch Tensors (PyTorch's version of numpy arrays)
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train,        dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,         dtype=torch.float32).unsqueeze(1)

    # DataLoader batches the data for training.
    # batch_size=64: process 64 rows at a time
    # shuffle=True: randomize order each epoch so the model doesn't memorize sequence
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    n_features = X_train_scaled.shape[1]
    model = GymMLP(n_features)

    # Loss function: MSELoss = average of (predicted - actual)²
    # Squaring penalizes big mistakes more than small ones.
    criterion = nn.MSELoss()

    # Optimizer: Adam adjusts the learning rate automatically per parameter.
    # lr=0.001: how big a step to take each update
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop — the heart of PyTorch.
    # One "epoch" = one full pass through all training data.
    # Each step: forward pass → compute loss → backpropagate → update weights.
    #
    # Backpropagation: calculates how much each weight contributed to the loss,
    # then nudges each weight in the direction that reduces the loss.
    EPOCHS  = 200
    PATIENCE = 20

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_weights = None

    print(f"  Training for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...")

    for epoch in range(EPOCHS):
        model.train()  # enables Dropout
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()                       # clear gradients from previous step
            predictions = model(X_batch)                # forward pass
            loss = criterion(predictions, y_batch)      # measure error
            loss.backward()                             # backprop: compute gradients
            optimizer.step()                            # update weights

        model.eval()  # disables Dropout for evaluation
        with torch.no_grad():
            val_loss = criterion(model(X_test_t), y_test_t).item()

        # Early stopping: stop if no improvement for PATIENCE epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d} — val loss: {val_loss:.2f}")

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        mlp_preds = model(X_test_t).numpy().flatten()

    mlp_rmse = mean_squared_error(y_test, mlp_preds) ** 0.5
    mlp_mae  = mean_absolute_error(y_test, mlp_preds)

    print(f"  RMSE: {mlp_rmse:.2f}%")
    print(f"  MAE:  {mlp_mae:.2f}%")

    torch.save(best_weights, "models/pytorch_model.pt")
    with open("models/model_config.pkl", "wb") as f:
        pickle.dump({'n_features': n_features}, f)
    print("  Saved → models/pytorch_model.pt")

    # --------------------------------------------------------------------------
    # STEP 6: SAVE METRICS
    # --------------------------------------------------------------------------

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "rf":  {"rmse": round(rf_rmse, 2),  "mae": round(rf_mae, 2)},
        "mlp": {"rmse": round(mlp_rmse, 2), "mae": round(mlp_mae, 2)},
        "feature_importances": {k: round(float(v), 4) for k, v in importances.items()}
    }

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Training Complete ===")
    print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8}   (lower is better)")
    print(f"{'-'*40}")
    print(f"{'Random Forest':<20} {rf_rmse:>7.2f}% {rf_mae:>7.2f}%")
    print(f"{'PyTorch MLP':<20} {mlp_rmse:>7.2f}% {mlp_mae:>7.2f}%")
    print(f"\nMetrics saved → models/metrics.json")
