"""Frozen prospective policies. Only called AFTER production publication.

These forecasts live inside backend-only snapshot metadata. No serving table,
model artifact, client, notification, or production accuracy view reads them.
Changing a policy requires a new experiment ID and a new future holdout.
"""
import hashlib
import json
import math
from datetime import date

import carry_model as km
from academic_calendar import get_open_hours

EXPERIMENT_ID = "trail-carry-2026-11-v1"
HOLDOUT_START = date(2026, 11, 7)
HOLDOUT_OPEN_DAYS = 30
ARMS = ("production", "stronger_trail", "weaker_carry", "both")
POLICY = {
    "id": EXPERIMENT_ID,
    "holdout_start": HOLDOUT_START.isoformat(),
    "holdout_open_days": HOLDOUT_OPEN_DAYS,
    "trail_multiplier": 1.25,
    "carry_full_through_h": 1.0,
    "carry_half_from_h": 3.0,
    "carry_min_weight": 0.5,
    "baseline_clamp": [0.0, 100.0],
    "forecast_clamp": [0.0, 110.0],
    "carry_horizon_origin": "last_observed_slot",
    "fitting": "same production carry artifact; recompute gaps per baseline",
}
POLICY_HASH = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def carry_weight(horizon):
    fraction = (horizon - POLICY["carry_full_through_h"]) / (
        POLICY["carry_half_from_h"] - POLICY["carry_full_through_h"])
    return 1.0 - (1.0 - POLICY["carry_min_weight"]) * min(1.0, max(0.0, fraction))


def _forecast(base, actuals, carry, lo, hi, weak=False):
    obs = sorted(s for s in actuals if lo <= s < hi and s in base)
    gaps = km.compute_gaps(obs, [actuals[s] for s in obs], [base[s] for s in obs]) if obs else None
    if not carry or gaps is None:
        return dict(base)
    day, recent, last, last_slot, n_obs = gaps
    out = {}
    for slot, baseline in base.items():
        if slot <= max(obs):
            continue
        horizon = (slot - last_slot) / 4.0
        correction = km.correction_at(carry, max(obs) // 4, horizon, day, recent, last)
        correction *= km.shrink_factor(n_obs, horizon)
        if weak:
            correction *= carry_weight(horizon)
        out[slot] = round(min(110.0, max(0.0, baseline + correction)), 1)
    return out


def build_shadow(now, preds, base, curve, actuals, carry):
    """Use the exact production inputs, with no later reads or fitting.

    Store only strictly future open slots shared by all four arms. Empty carry
    predictions mean production served the baseline. Closing display zeros are
    excluded. The production arm is the actual published output, not a replay.
    """
    lo_h, hi_h = get_open_hours(now.strftime("%A"), now.date())
    lo, hi = lo_h * 4, hi_h * 4
    published = {round(p["x"] * 4): float(p["y"]) for p in preds} if preds else dict(base)
    wall_slot = (now.hour + now.minute / 60 + now.second / 3600
                 + now.microsecond / 3_600_000_000) * 4
    future = {s: v for s, v in published.items() if lo <= s < hi and s > wall_slot}
    if not future:
        return {"id": EXPERIMENT_ID, "policy_hash": POLICY_HASH, "status": "no_future_slots"}
    if not set(base).issubset(curve) or any(s not in base for s in future):
        raise ValueError("missing bare curve/baseline; cannot pair all four arms")
    if any(s > wall_slot + 0.5 for s in actuals):
        raise ValueError("actuals contain a future slot")
    if not all(math.isfinite(v) for mapping in (base, curve, actuals) for v in mapping.values()):
        raise ValueError("non-finite shadow input")
    # Match the existing published baseline's 0–100 limits and 0.1pp precision.
    stronger = {
        s: round(min(100.0, max(0.0, curve[s] + POLICY["trail_multiplier"] * (v - curve[s]))), 1)
        for s, v in base.items() if s in curve
    }
    reproduced = _forecast(base, actuals, carry, lo, hi)
    if any(s not in reproduced or abs(reproduced[s] - v) > 0.051 for s, v in future.items()):
        raise ValueError("production reproduction mismatch; refusing an unfair comparison")
    forecasts = {
        "production": future,
        "stronger_trail": _forecast(stronger, actuals, carry, lo, hi),
        "weaker_carry": _forecast(base, actuals, carry, lo, hi, weak=True),
        "both": _forecast(stronger, actuals, carry, lo, hi, weak=True),
    }
    if any(s not in forecasts[arm] for arm in ARMS for s in future):
        raise ValueError("candidate is missing a paired future slot")
    return {
        "id": EXPERIMENT_ID, "policy_hash": POLICY_HASH, "policy": POLICY,
        "status": "ok", "period": "pilot" if now.date() < HOLDOUT_START else "prospective",
        "forecasts": {arm: {str(s): forecasts[arm][s] for s in sorted(future)} for arm in ARMS},
    }
