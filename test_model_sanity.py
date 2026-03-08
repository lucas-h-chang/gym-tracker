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
Requires:  models/ directory with trained artifacts (run train.py first)
"""

import pickle
import pytest
import torch
import pandas as pd
from train import GymMLP, engineer_features


# ==============================================================================
# LOAD MODELS ONCE (shared across all tests)
# ==============================================================================

def load_models():
    with open("models/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/model_config.pkl", "rb") as f:
        config = pickle.load(f)
    model = GymMLP(config['n_features'])
    model.load_state_dict(torch.load("models/pytorch_model.pt", weights_only=True))
    model.eval()
    return rf, scaler, model

@pytest.fixture(scope="module")
def models():
    """Load models once and share across all tests in this file."""
    return load_models()


def predict(models, timestamp_str):
    """
    Run a prediction for a given timestamp.
    Returns (rf_prediction, mlp_prediction) as floats.
    """
    rf, scaler, model = models
    df = pd.DataFrame({
        'timestamp': [pd.Timestamp(timestamp_str)],
        'people_count': [100],
        'percent_full': [100 / 150 * 100]
    })
    X, _ = engineer_features(df)
    rf_pred = rf.predict(X)[0]
    X_scaled = scaler.transform(X)
    with torch.no_grad():
        mlp_pred = model(torch.tensor(X_scaled, dtype=torch.float32)).item()
    return rf_pred, mlp_pred


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
def test_predictions_are_non_negative(models, timestamp):
    rf_p, mlp_p = predict(models, timestamp)
    assert rf_p >= 0,  f"RF predicted negative capacity: {rf_p:.1f}%"
    assert mlp_p >= 0, f"MLP predicted negative capacity: {mlp_p:.1f}%"

@pytest.mark.parametrize("timestamp", [
    '2026-02-09 09:00:00',
    '2026-02-10 17:00:00',
    '2026-03-17 14:00:00',
])
def test_predictions_dont_exceed_200(models, timestamp):
    """Predictions can exceed 100% (overcapacity is real) but 200% would be absurd."""
    rf_p, mlp_p = predict(models, timestamp)
    assert rf_p  <= 200, f"RF prediction absurdly high: {rf_p:.1f}%"
    assert mlp_p <= 200, f"MLP prediction absurdly high: {mlp_p:.1f}%"


# ==============================================================================
# BEHAVIORAL TESTS — predictions must make real-world sense
# These are the most valuable tests: they catch model logic errors, not just crashes.
# ==============================================================================

def test_summer_less_busy_than_semester(models):
    """
    A Tuesday afternoon in summer should be less busy than the same time in semester.
    The model must have learned semester context from week_of_year and is_break.
    """
    _, summer   = predict(models, '2025-07-15 17:00:00')   # Tuesday 5 PM, summer
    _, semester = predict(models, '2025-10-14 17:00:00')   # Tuesday 5 PM, mid-semester
    assert summer < semester, \
        f"Summer ({summer:.1f}%) should predict lower than semester ({semester:.1f}%)"

def test_break_less_busy_than_normal(models):
    """Spring break should predict much lower than a normal Tuesday."""
    rf_break,  _ = predict(models, '2026-03-17 15:00:00')  # spring break Tuesday
    rf_normal, _ = predict(models, '2026-02-10 15:00:00')  # normal Tuesday
    assert rf_break < rf_normal, \
        f"Spring break ({rf_break:.1f}%) should be less busy than normal ({rf_normal:.1f}%)"

def test_late_night_less_busy_than_peak(models):
    """10 PM should predict lower than 5 PM on the same day."""
    _, late = predict(models, '2026-02-10 22:00:00')   # 10 PM Tuesday
    _, peak = predict(models, '2026-02-10 17:00:00')   # 5 PM Tuesday
    assert late < peak, \
        f"10 PM ({late:.1f}%) should be less busy than 5 PM ({peak:.1f}%)"

def test_morning_less_busy_than_afternoon(models):
    """8 AM should predict lower than 5 PM on a weekday."""
    _, morning   = predict(models, '2026-02-10 08:00:00')
    _, afternoon = predict(models, '2026-02-10 17:00:00')
    assert morning < afternoon, \
        f"8 AM ({morning:.1f}%) should predict lower than 5 PM ({afternoon:.1f}%)"

def test_both_models_give_similar_predictions(models):
    """
    RF and MLP should broadly agree. If they differ by more than 30%,
    something is likely wrong with one of the models or feature pipelines.
    """
    for ts in ['2026-02-10 17:00:00', '2026-03-17 14:00:00', '2025-07-15 11:00:00']:
        rf_p, mlp_p = predict(models, ts)
        diff = abs(rf_p - mlp_p)
        assert diff < 35, \
            f"RF ({rf_p:.1f}%) and MLP ({mlp_p:.1f}%) disagree by {diff:.1f}% on {ts}"
