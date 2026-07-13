import os
import json
import pickle
from datetime import datetime, date

import pandas as pd

# scikit-learn: classical ML library. We use it for:
#   - RandomForestRegressor: our single occupancy model
#   - mean_squared_error / mean_absolute_error: measuring prediction accuracy
#
# We used to also train a PyTorch MLP and average the two. That was removed:
# measured honestly (early-stop on a validation split, report on an untouched
# test split), the MLP scored worse than the RF and dragged the blend down,
# while dragging in torch + a scaler. XGBoost was also evaluated and lost to the
# RF on every hyperparameter config — this task is a calendar lookup table, which
# bagged deep trees fit better than boosting. So: one Random Forest, no torch.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

MAX_CAPACITY = 150


def parse_supabase_timestamps(series):
    # Supabase returns TIMESTAMPTZ as UTC. Convert to PT wall-clock, then drop tz so
    # engineer_features() sees the same naive-PT timestamps that predictions_builder
    # and predict.py feed at inference time.
    return (
        pd.to_datetime(series, utc=True, format='ISO8601')
          .dt.tz_convert('America/Los_Angeles')
          .dt.tz_localize(None)
    )


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
    BATCH  = 9000
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

    if len(rows) < 50000:
        raise RuntimeError(f"Supabase returned only {len(rows):,} rows — aborting to protect existing models.")

    df = pd.DataFrame(rows)
    df['timestamp']    = parse_supabase_timestamps(df['timestamp'])
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
    # STEP 3: TRAIN / VAL / TEST SPLIT  (honest evaluation)
    # --------------------------------------------------------------------------
    # Chronological 70 / 10 / 20 split — never random (random would leak future
    # rows into training and flatter the score). The Random Forest has no early
    # stopping, so it trains on the first 80% (train + val) and every reported
    # number is computed on the untouched final 20% test set only. The 10% val
    # band is reserved so this protocol drops in unchanged if a model that DOES
    # early-stop is ever reintroduced.
    #
    # (History: the old MLP early-stopped on the *test* set and then reported on
    # it — an optimistically biased number. That model is gone; the honest
    # protocol stays.)
    # --------------------------------------------------------------------------

    split_train = int(len(X) * 0.8)   # RF trains on first 80% (train + val)
    X_fit, X_test = X.iloc[:split_train], X.iloc[split_train:]
    y_fit, y_test = y[:split_train], y[split_train:]

    print(f"\nChronological 70/10/20 split (report on untouched test only):")
    print(f"  Fit (train+val): {len(X_fit):,} rows  ({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[split_train-1].date()})")
    print(f"  Test:            {len(X_test):,} rows  ({df['timestamp'].iloc[split_train].date()} → {df['timestamp'].iloc[-1].date()})")

    # --------------------------------------------------------------------------
    # STEP 4: BASELINE — historical mean per (weekday, 15-min slot, is_break)
    # --------------------------------------------------------------------------
    # Every feature here is calendar-derived, so any model is structurally a
    # smart lookup table of "typical occupancy at (time, day, semester-phase)".
    # This dumb groupby lookup is that table with no smoothing. If the Random
    # Forest can't beat it on the same test set, the ML isn't earning its keep.
    # --------------------------------------------------------------------------

    print("\n--- Baseline (groupby mean) ---")
    key_cols = ['dow_num', 'slot', 'is_break']
    base_df = pd.DataFrame({
        'dow_num':      df['timestamp'].dt.dayofweek.values,
        'slot':         (df['timestamp'].dt.hour * 4 + df['timestamp'].dt.minute // 15).values,
        'is_break':     X['is_break'].values,
        'percent_full': y,
    })
    base_train, base_test = base_df.iloc[:split_train], base_df.iloc[split_train:]
    lookup = base_train.groupby(key_cols)['percent_full'].mean()
    base_preds = base_test.set_index(key_cols).index.map(lookup)
    base_preds = pd.Series(base_preds).fillna(base_train['percent_full'].mean()).to_numpy(dtype=float)
    base_rmse = mean_squared_error(y_test, base_preds) ** 0.5
    base_mae  = mean_absolute_error(y_test, base_preds)
    print(f"  MAE:  {base_mae:.2f}%   ← the lookup-table score the model must beat")

    # --------------------------------------------------------------------------
    # STEP 5: RANDOM FOREST (scikit-learn) — the one and only model
    # --------------------------------------------------------------------------
    # A Random Forest trains many decision trees, each on a random subset of data:
    #   "Is it after 5 PM? → Yes. Is it Monday? → No. Is it finals? → No → 45%"
    # The prediction is the average of all trees. It needs no feature scaling,
    # handles non-linear calendar patterns, and gives feature importances.
    #
    # n_estimators=20, max_depth=12: measured to match depth-15 accuracy
    # (~9.2% MAE) while cutting the pickle from ~24 MB to ~7 MB. XGBoost and a
    # PyTorch MLP were both evaluated and lost to this on the same test set.
    # --------------------------------------------------------------------------

    print("\n--- Training Random Forest (scikit-learn) ---")
    rf = RandomForestRegressor(n_estimators=20, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_fit, y_fit)

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
    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    print("  Saved → models/rf_model.pkl")

    # --------------------------------------------------------------------------
    # STEP 6: SAVE METRICS
    # --------------------------------------------------------------------------
    # baseline lives next to rf so every retrain answers "is the ML still beating
    # a plain lookup table?"

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date_range": f"{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}",
        "training_rows": len(X_fit),
        "test_rows": len(X_test),
        "rf":       {"rmse": round(rf_rmse, 2),   "mae": round(rf_mae, 2)},
        "baseline": {"rmse": round(base_rmse, 2), "mae": round(base_mae, 2)},
        "feature_importances": {k: round(float(v), 4) for k, v in importances.items()}
    }

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Training Complete (test-set MAE, lower is better) ===")
    print(f"{'Model':<24} {'RMSE':>8} {'MAE':>8}")
    print(f"{'-'*42}")
    print(f"{'Baseline (groupby mean)':<24} {base_rmse:>7.2f}% {base_mae:>7.2f}%")
    print(f"{'Random Forest':<24} {rf_rmse:>7.2f}% {rf_mae:>7.2f}%")
    print(f"\nMetrics saved → models/metrics.json")
