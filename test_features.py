"""
test_features.py — Unit tests for feature engineering logic.

These tests check the LOGIC of engineer_features(), not the model itself.
They're fast (no DB, no training) and give exact pass/fail answers.

Run with:  python3 -m pytest test_features.py -v
"""

import pandas as pd
import pytest
from train import engineer_features


def make_df(timestamp_str, people_count=80):
    """Helper: create a one-row DataFrame for a given timestamp."""
    return pd.DataFrame({
        'timestamp': [pd.Timestamp(timestamp_str)],
        'people_count': [people_count],
        'percent_full': [people_count / 150 * 100]
    })


# ==============================================================================
# CALENDAR FLAG TESTS
# These check that the hardcoded Berkeley calendar dates are correct.
# If any of these fail, the model is learning wrong context for those periods.
# ==============================================================================

def test_spring_break_2026_is_flagged():
    X, _ = engineer_features(make_df('2026-03-25 14:00:00'))  # within Mar 23-27
    assert X['is_break'].iloc[0] == 1, "March 25 2026 should be spring recess"

def test_day_before_spring_break_is_not_flagged():
    X, _ = engineer_features(make_df('2026-03-20 14:00:00'))  # Friday before surrounding Sat (Mar 21)
    assert X['is_break'].iloc[0] == 0, "March 20 2026 should NOT be spring recess"

def test_spring_finals_2026_is_flagged():
    X, _ = engineer_features(make_df('2026-05-12 10:00:00'))  # within May 11-15
    assert X['is_finals'].iloc[0] == 1

def test_dead_week_2026_is_flagged():
    X, _ = engineer_features(make_df('2026-05-05 10:00:00'))  # within May 4-8
    assert X['is_dead_week'].iloc[0] == 1

def test_first_week_fall_2025_is_flagged():
    X, _ = engineer_features(make_df('2025-08-28 15:00:00'))
    assert X['is_first_week'].iloc[0] == 1

def test_normal_tuesday_has_no_special_flags():
    X, _ = engineer_features(make_df('2026-02-10 15:00:00'))
    assert X['is_break'].iloc[0] == 0
    assert X['is_finals'].iloc[0] == 0
    assert X['is_dead_week'].iloc[0] == 0
    assert X['is_first_week'].iloc[0] == 0
    assert X['is_holiday'].iloc[0] == 0

def test_summer_is_flagged_as_break():
    X, _ = engineer_features(make_df('2025-07-15 14:00:00'))
    assert X['is_break'].iloc[0] == 1

def test_winter_break_is_flagged():
    X, _ = engineer_features(make_df('2025-12-25 14:00:00'))
    assert X['is_break'].iloc[0] == 1


# ==============================================================================
# HOUR / TIME FEATURE TESTS
# ==============================================================================

def test_hour_numeric_whole_hour():
    X, _ = engineer_features(make_df('2026-02-10 15:00:00'))
    assert X['hour_numeric'].iloc[0] == 15.0

def test_hour_numeric_half_hour():
    X, _ = engineer_features(make_df('2026-02-10 15:30:00'))
    assert X['hour_numeric'].iloc[0] == 15.5

def test_hour_numeric_quarter_hour():
    X, _ = engineer_features(make_df('2026-02-10 17:15:00'))
    assert X['hour_numeric'].iloc[0] == pytest.approx(17.25)


# ==============================================================================
# DAY OF WEEK / ONE-HOT ENCODING TESTS
# ==============================================================================

def test_monday_is_correctly_encoded():
    X, _ = engineer_features(make_df('2026-02-09 15:00:00'))  # Monday
    assert X['day_Monday'].iloc[0] == 1
    assert X['day_Tuesday'].iloc[0] == 0

def test_saturday_is_weekend():
    X, _ = engineer_features(make_df('2026-02-07 15:00:00'))  # Saturday
    assert X['is_weekend'].iloc[0] == 1
    assert X['day_Saturday'].iloc[0] == 1

def test_tuesday_is_not_weekend():
    X, _ = engineer_features(make_df('2026-02-10 15:00:00'))  # Tuesday
    assert X['is_weekend'].iloc[0] == 0

def test_exactly_one_day_column_is_active():
    """One-hot encoding must have exactly 1 active day column per row."""
    X, _ = engineer_features(make_df('2026-02-10 15:00:00'))
    day_cols = [c for c in X.columns if c.startswith('day_')]
    assert X[day_cols].sum(axis=1).iloc[0] == 1, "Exactly one day column should be 1"


# ==============================================================================
# FEATURE MATRIX STRUCTURE TESTS
# ==============================================================================

def test_feature_matrix_has_correct_columns():
    X, feature_names = engineer_features(make_df('2026-02-10 15:00:00'))
    expected = ['hour_numeric', 'week_of_year', 'is_weekend', 'is_finals',
                'is_dead_week', 'is_first_week', 'is_break', 'is_holiday',
                'day_Monday', 'day_Tuesday', 'day_Wednesday', 'day_Thursday',
                'day_Friday', 'day_Saturday', 'day_Sunday']
    assert list(X.columns) == expected
    assert feature_names == expected

def test_feature_matrix_has_15_features():
    X, _ = engineer_features(make_df('2026-02-10 15:00:00'))
    assert X.shape[1] == 15

def test_all_features_are_numeric():
    X, _ = engineer_features(make_df('2026-02-10 15:00:00'))
    for col in X.columns:
        assert X[col].dtype in ['float64', 'int64', 'uint8', 'bool'], \
            f"Column {col} has unexpected dtype {X[col].dtype}"
