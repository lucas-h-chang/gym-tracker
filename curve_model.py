"""
curve_model.py — recency-weighted, shrunken, smoothed day-curve model.

Replaces the Random Forest with a lookup table of one empirical occupancy
curve per (academic phase, day-of-week), built by build_curves.py and read
by predictions_builder.py. See SPEC_CURVE_MODEL.md §4 for the design and
HANDOFF_MODEL_REDESIGN.md for why: this task is conditional-mean estimation
on a small discrete calendar grid, not forecasting, so a regularized
empirical mean beats trees on shape, stability, and size.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from academic_calendar import classify_date, days_to_sem_start, days_to_sem_end

MAX_CAPACITY = 150

DEFAULT_PARAMS = {
    "halflife_days": 365,
    "shrink_k": 3,
    "smooth_window": 3,
    "blend_window_days": 5,
}
# Tuned via backtest.py grid search over halflife_days [120,180,365] x
# shrink_k [3,7,15] x smooth_window [1,3,5] x blend_window_days [5,7,10] on
# TUNE_ORIGINS (2024-01 -> 2025-06), selecting for good-day rate (the
# product north-star per HANDOFF_MODEL_REDESIGN.md §1) with MAE as a
# secondary check.
#   - smooth_window=1 ("off") edged out smooth_window=3 by a small margin,
#     but 3 was kept because it roughly halves the count of >8pp
#     slot-to-slot jumps (66 -> 26 on the full-history table) for ~0.2pp
#     MAE cost — see test_curve_sanity.py item 3 / SPEC §5's max-jump gate.
#   - shrink_k=3 (least shrinkage in the grid) beat 7 and 15 on every
#     holdout metric, most visibly on the "holiday" segment (MAE 16.3 vs
#     17.5 vs 18.3 for k=3/7/15) — holidays are the thinnest phase (1-2
#     distinct days/year), and pulling them harder toward the phase-wide
#     curve washes out exactly the character that makes a holiday different
#     from a regular day.
# See backtest_report.json for the confirmation run on HOLDOUT_ORIGINS
# (2025-07 -> 2026-06) — as of this tuning pass it does NOT clear the §5
# Step 3 gate (good-day rate and the "holiday" segment both trail the
# deployed RF baseline on properly held-out, leakage-free origins); see
# HANDOFF_MODEL_REDESIGN.md follow-up notes for the honest comparison.


def _as_date(d):
    # Accept a date, datetime, or pandas Timestamp.
    return d.date() if hasattr(d, "date") else d


def prepare_slots(df):
    """
    df: raw capacity_log rows with 'timestamp' (naive PT, tz already stripped —
    see train.py::parse_supabase_timestamps) and 'people_count'.

    Cleans (people_count > 5, dropna) and collapses to one row per (date, slot)
    via mean, per SPEC_CURVE_MODEL.md §3. Shared by build_curves.py and
    backtest.py so both eval and production build off identical prep.
    """
    df = df[['timestamp', 'people_count']].dropna()
    df = df[df['people_count'] > 5].copy()
    df['percent_full'] = df['people_count'].astype(float) / MAX_CAPACITY * 100
    df['date'] = df['timestamp'].dt.normalize()
    df['slot'] = df['timestamp'].dt.hour * 4 + df['timestamp'].dt.minute // 15

    collapsed = df.groupby(['date', 'slot'], as_index=False)['percent_full'].mean()
    collapsed['dow'] = collapsed['date'].dt.dayofweek
    collapsed['is_weekend'] = (collapsed['dow'] >= 5).astype(int)
    collapsed['phase'] = collapsed['date'].apply(lambda d: classify_date(d.date()))
    return collapsed


def _weighted_agg(df, keys):
    """Vectorized weighted mean/std/n_eff per group (no python-level apply)."""
    tmp = df[keys].copy()
    tmp['_w']   = df['w']
    tmp['_wx']  = df['w'] * df['percent_full']
    tmp['_wx2'] = df['w'] * df['percent_full'] ** 2
    g = tmp.groupby(keys, as_index=False)[['_w', '_wx', '_wx2']].sum()
    g['mean'] = g['_wx'] / g['_w']
    var = g['_wx2'] / g['_w'] - g['mean'] ** 2
    g['std'] = np.sqrt(np.clip(var, 0, None))
    g['n_eff'] = g['_w']
    return g[keys + ['mean', 'std', 'n_eff']]


def _shrink(n, m, s, parent_m, parent_s, k):
    """m̂ = (n·m + k·parent_m) / (n + k); same formula for s. n=0 → parent value."""
    w = n / (n + k)
    return w * m + (1 - w) * parent_m, w * s + (1 - w) * parent_s


def build_table(df, params=None, build_date=None, built_at=None):
    """
    df: output of prepare_slots (one row per date/slot/dow/is_weekend/phase).
    params: overrides over DEFAULT_PARAMS.
    build_date: cutoff — only rows with date < build_date are used (honest
      backtest requires this; for live weekly builds it's a no-op since df
      never contains future rows).

    Returns the full table dict (version, built_at, params, curves), ready
    to json.dump to models/curves.json.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    if build_date is None:
        build_date = df['date'].max() + pd.Timedelta(days=1)
    build_ts = pd.Timestamp(build_date)
    df = df[df['date'] < build_ts].copy()
    if df.empty:
        raise ValueError(f"No data before build_date={build_date}")

    age_days = (build_ts - df['date']).dt.days.clip(lower=0)
    df['w'] = 0.5 ** (age_days / p['halflife_days'])

    # ── Backoff chain, broadest → narrowest: (slot) → (phase,slot) →
    #    (phase,is_weekend,slot) → (phase,dow,slot). Each level is shrunk
    #    toward the *already-shrunk* parent level (cascading top-down).
    l0 = _weighted_agg(df, ['slot']).set_index('slot')
    l1 = _weighted_agg(df, ['phase', 'slot'])
    l2 = _weighted_agg(df, ['phase', 'is_weekend', 'slot'])
    l3 = _weighted_agg(df, ['phase', 'dow', 'slot'])

    k = p['shrink_k']

    # L1 shrunk toward L0.
    #
    # A variant of this was tried where non-'regular' phases shrink toward
    # 'regular's own curve instead of the global cross-phase (slot) curve —
    # motivated by 'break' being ~36% of rows and dragging the global curve
    # well below a typical in-session day, a seemingly poor backoff target
    # for thin in-session phases like holiday. A controlled A/B on identical
    # holdout data showed it did NOT help (holiday segment MAE 16.6 vs 16.3,
    # overall roughly flat) — holiday's own raw signal already carries ~50%
    # of the L1 shrinkage weight at the tuned shrink_k, so the parent choice
    # matters less than the theory predicted, and raw holiday means weren't
    # uniformly closer to 'regular' than to the global curve across slots
    # (checked directly: 2 of 3 sampled slots favored 'regular', 1 favored
    # the global curve). Reverted in favor of the simpler design.
    l1 = l1.merge(l0[['mean', 'std']].rename(columns={'mean': 'p_mean', 'std': 'p_std'}),
                   left_on='slot', right_index=True, how='left')
    l1['m_hat'], l1['s_hat'] = _shrink(l1['n_eff'], l1['mean'], l1['std'], l1['p_mean'], l1['p_std'], k)
    l1_shrunk = l1.set_index(['phase', 'slot'])[['m_hat', 's_hat']]

    # L2 shrunk toward L1_shrunk. Unlike L1->L0, we do NOT invent an
    # (is_weekend=1) row for every slot observed anywhere in the phase:
    # weekday and weekend open at different hours (e.g. RSF opens 7am
    # weekdays, 8am weekends), so extrapolating a weekday-only slot like
    # 7:45am onto Saturday would fabricate occupancy for an hour the gym is
    # provably closed. Only slots with real (phase, is_weekend, slot)
    # evidence get a row here.
    l2_full = l2.merge(
        l1_shrunk.rename(columns={'m_hat': 'p_mean', 's_hat': 'p_std'}),
        left_on=['phase', 'slot'], right_index=True, how='left')
    l2_full['m_hat'], l2_full['s_hat'] = _shrink(
        l2_full['n_eff'], l2_full['mean'], l2_full['std'], l2_full['p_mean'], l2_full['p_std'], k)
    l2_shrunk = l2_full.set_index(['phase', 'is_weekend', 'slot'])[['m_hat', 's_hat']]

    # Universe of (phase, is_weekend, slot) triples = what L2 actually
    # observed. L3 (phase, dow, slot) then expands each triple to its two or
    # five real dows — this is where backoff earns its keep (e.g. a holiday
    # that only ever fell on a Monday still gets a Tuesday estimate, backed
    # off to the weekday-wide curve) — without inventing hours nobody was
    # ever open.
    weekday_dows = pd.DataFrame({'dow': [0, 1, 2, 3, 4], 'is_weekend': 0})
    weekend_dows = pd.DataFrame({'dow': [5, 6], 'is_weekend': 1})
    dow_map = pd.concat([weekday_dows, weekend_dows], ignore_index=True)

    l2_universe = l2[['phase', 'is_weekend', 'slot']].drop_duplicates()
    l3_full = l2_universe.merge(dow_map, on='is_weekend')
    l3_full = l3_full.merge(l3, on=['phase', 'dow', 'slot'], how='left')
    l3_full[['mean', 'std', 'n_eff']] = l3_full[['mean', 'std', 'n_eff']].fillna(0)
    l3_full = l3_full.merge(
        l2_shrunk.rename(columns={'m_hat': 'p_mean', 's_hat': 'p_std'}),
        left_on=['phase', 'is_weekend', 'slot'], right_index=True, how='left')
    l3_full['m_hat'], l3_full['s_hat'] = _shrink(
        l3_full['n_eff'], l3_full['mean'], l3_full['std'], l3_full['p_mean'], l3_full['p_std'], k)

    # ── Smoothing: centered rolling mean over smooth_window slots, per
    #    (phase, dow) curve, applied to m_hat only, after shrinkage.
    window = p['smooth_window']
    curves = {}
    for (phase, dow), grp in l3_full.sort_values('slot').groupby(['phase', 'dow']):
        grp = grp.sort_values('slot')
        smoothed = grp['m_hat'].rolling(window=window, center=True, min_periods=1).mean() if window > 1 else grp['m_hat']
        curves[f"{phase}|{dow}"] = {
            "slot_index": grp['slot'].astype(int).tolist(),
            "mean":       smoothed.round(3).tolist(),
            "std":        grp['s_hat'].round(3).tolist(),
            "n_eff":      grp['n_eff'].round(2).tolist(),
        }

    return {
        "version": 1,
        "built_at": built_at or datetime.now().isoformat(),
        "params": p,
        "curves": curves,
    }


def phase_weights(d, blend_window_days=None):
    """
    Soft phase assignment for a prediction date, per SPEC_CURVE_MODEL.md §4.
    Default: hard classify_date. Within blend_window_days of a semester
    boundary while classified 'break', linearly blend toward the adjacent
    phase (first_week on the way in, finals on the way out).
    """
    W = blend_window_days if blend_window_days is not None else DEFAULT_PARAMS['blend_window_days']
    d = _as_date(d)
    phase = classify_date(d)

    if phase == "break":
        dts = days_to_sem_start(d)
        if 1 <= dts <= W:
            alpha = (W + 1 - dts) / (W + 1)
            return [("break", 1 - alpha), ("first_week", alpha)]

        k = -days_to_sem_end(d)
        if 1 <= k <= W:
            alpha = (W + 1 - k) / (W + 1)  # weight on "finals" (boundary-adjacent regime)
            return [("finals", alpha), ("break", 1 - alpha)]

    return [(phase, 1.0)]


def _lookup_slot(curve, slot):
    """Exact slot if present, else the nearest available slot in this curve."""
    idx = curve["slot_index"]
    if not idx:
        return None, None
    if slot in idx:
        i = idx.index(slot)
    else:
        i = int(np.argmin([abs(s - slot) for s in idx]))
    return curve["mean"][i], curve["std"][i]


def predict_one(table, d, slot, blend_window_days=None):
    """Returns (mean, std) blended across soft phase weights, or (None, None) if no curve matches."""
    d = _as_date(d)
    dow = d.weekday()
    if blend_window_days is None:
        blend_window_days = table.get("params", {}).get("blend_window_days")
    weights = phase_weights(d, blend_window_days)
    curves = table["curves"]

    total_w, mean_acc, std_acc = 0.0, 0.0, 0.0
    for phase, w in weights:
        curve = curves.get(f"{phase}|{dow}")
        if curve is None:
            continue
        m, s = _lookup_slot(curve, slot)
        if m is None:
            continue
        mean_acc += w * m
        std_acc += w * s
        total_w += w

    if total_w == 0:
        return None, None
    return mean_acc / total_w, std_acc / total_w


def predict(table, dates_slots, blend_window_days=None):
    """
    dates_slots: iterable of (date, slot) pairs.
    Returns np.array of predicted means (NaN where no curve matched).
    """
    out = np.empty(len(dates_slots))
    for i, (d, slot) in enumerate(dates_slots):
        m, _ = predict_one(table, d, slot, blend_window_days)
        out[i] = m if m is not None else np.nan
    return out


def predict_with_std(table, dates_slots, blend_window_days=None):
    means = np.empty(len(dates_slots))
    stds = np.empty(len(dates_slots))
    for i, (d, slot) in enumerate(dates_slots):
        m, s = predict_one(table, d, slot, blend_window_days)
        means[i] = m if m is not None else np.nan
        stds[i] = s if s is not None else np.nan
    return means, stds
