"""
test_model_sanity.py — Behavioral tests for trained model predictions.

Unlike unit tests, these don't assert exact numbers — ML outputs vary slightly
each run. Instead, they assert that predictions make logical sense:
  "Summer should predict lower than semester"
  "Spring break should predict lower than a normal day"
  "Late night should predict lower than 5 PM"

These are called "behavioral tests" or "invariance tests" in ML.
If any of these fail, something is seriously wrong with the model or features.

Run with:  python3 -m pytest test_model_sanity.py -v
Requires:  models/rf_model.pkl (run train.py first)
"""

import pickle
import pytest
import pandas as pd
from train import engineer_features


# ==============================================================================
# LOAD MODEL ONCE (shared across all tests)
# ==============================================================================

def load_model():
    with open("models/rf_model.pkl", "rb") as f:
        return pickle.load(f)

def load_feature_names():
    # The deployed pickle may predate feature additions to engineer_features
    # (e.g. days_to_sem_start/end) — feature_names.pkl records what it was
    # actually fit on, so we can reindex engineer_features()'s current output
    # to match instead of sklearn raising "unseen at fit time". Mirrors
    # backtest.py::load_rf()/rf_predict_grid()'s handling of the same drift.
    try:
        with open("models/feature_names.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@pytest.fixture(scope="module")
def model():
    """Load the Random Forest once and share across all tests in this file."""
    return load_model()


def predict(model, timestamp_str):
    """Run a prediction for a given timestamp. Returns predicted percent_full as a float."""
    df = pd.DataFrame({
        'timestamp': [pd.Timestamp(timestamp_str)],
        'people_count': [100],
        'percent_full': [100 / 150 * 100]
    })
    X, _ = engineer_features(df)
    feature_names = load_feature_names()
    if feature_names is not None and list(X.columns) != list(feature_names):
        X = X[feature_names]
    return float(model.predict(X)[0])


# ==============================================================================
# RANGE TESTS — predictions must be physically plausible
# ==============================================================================

@pytest.mark.parametrize("timestamp", [
    '2026-02-09 09:00:00',  # Monday morning
    '2026-02-10 17:00:00',  # Tuesday peak
    '2026-03-17 14:00:00',  # Spring break
    '2025-07-15 11:00:00',  # Summer
    '2026-05-05 10:00:00',  # Finals
])
def test_predictions_are_non_negative(model, timestamp):
    p = predict(model, timestamp)
    assert p >= 0, f"Predicted negative capacity: {p:.1f}%"

@pytest.mark.parametrize("timestamp", [
    '2026-02-09 09:00:00',
    '2026-02-10 17:00:00',
    '2026-03-17 14:00:00',
])
def test_predictions_dont_exceed_200(model, timestamp):
    """Predictions can exceed 100% (overcapacity is real) but 200% would be absurd."""
    p = predict(model, timestamp)
    assert p <= 200, f"Prediction absurdly high: {p:.1f}%"


# ==============================================================================
# BEHAVIORAL TESTS — predictions must make real-world sense
# These are the most valuable tests: they catch model logic errors, not just crashes.
# ==============================================================================

def test_summer_less_busy_than_semester(model):
    """
    A Tuesday afternoon in summer should be less busy than the same time in semester.
    The model must have learned semester context from week_of_year and is_break.
    """
    summer   = predict(model, '2025-07-15 17:00:00')   # Tuesday 5 PM, summer
    semester = predict(model, '2025-10-14 17:00:00')   # Tuesday 5 PM, mid-semester
    assert summer < semester, \
        f"Summer ({summer:.1f}%) should predict lower than semester ({semester:.1f}%)"

def test_break_less_busy_than_normal(model):
    """Spring break should predict much lower than a normal Tuesday."""
    brk    = predict(model, '2026-03-17 15:00:00')  # spring break Tuesday
    normal = predict(model, '2026-02-10 15:00:00')  # normal Tuesday
    assert brk < normal, \
        f"Spring break ({brk:.1f}%) should be less busy than normal ({normal:.1f}%)"

def test_late_night_less_busy_than_peak(model):
    """10 PM should predict lower than 5 PM on the same day."""
    late = predict(model, '2026-02-10 22:00:00')   # 10 PM Tuesday
    peak = predict(model, '2026-02-10 17:00:00')   # 5 PM Tuesday
    assert late < peak, \
        f"10 PM ({late:.1f}%) should be less busy than 5 PM ({peak:.1f}%)"

def test_morning_less_busy_than_afternoon(model):
    """8 AM should predict lower than 5 PM on a weekday."""
    morning   = predict(model, '2026-02-10 08:00:00')
    afternoon = predict(model, '2026-02-10 17:00:00')
    assert morning < afternoon, \
        f"8 AM ({morning:.1f}%) should predict lower than 5 PM ({afternoon:.1f}%)"
