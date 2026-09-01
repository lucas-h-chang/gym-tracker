"""
carry_data.py — shared data prep for the within-day level correction.

Used by `build_carry.py` (produces the shipped models/carry.json) and
`replay_day.py` (the acceptance harness). Both need the same thing: for every
historical day, the actuals and **the baseline production would actually have
served** on that day.

That second part is the whole reason this module exists rather than a one-line
`cm.predict`. The deployed baseline is two layers:

    curve_model's recency-weighted curve      (rebuilt weekly)
  + predictions_builder's 28-day trailing correction   (rebuilt nightly)

Fitting the within-day correction against only the first layer would
double-count the multi-day drift the second layer has already removed, on every
day where it fires. So both are reproduced here.

Deliberately does NOT import backtest.py: that pulls in train.py -> sklearn for
the retired RF, which is not installed in a plain checkout and has nothing to do
with the curve model that replaced it.
"""
import os
import sys
import pickle
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

import curve_model as cm
from academic_calendar import (
    classify_date, days_to_sem_start, days_to_sem_end, get_open_hours,
)

SLOTS_PER_DAY = 96
CACHE = os.environ.get("CARRY_CACHE", "/tmp/rsf_capacity_cache.pkl")
RAW_CACHE = os.environ.get("CARRY_BASE_CACHE", "/tmp/rsf_raw_base_matrix.pkl")

# Match build_curves.py exactly, so the baseline here is the curve production serves.
PARAMS = {**cm.DEFAULT_PARAMS, "week_levels": True}

ORIGINS = [d for d in (date(y, m, 1) for y in (2023, 2024, 2025, 2026) for m in range(1, 13))
           if date(2023, 1, 1) <= d <= date(2026, 12, 1)]

# predictions_builder.py constants, mirrored here so drift between them is loud.
CORRECTION_DAYS  = 28
CORRECTION_MIN_N = 3


def parse_supabase_timestamps(series):
    """TIMESTAMPTZ (UTC) -> naive PT wall-clock. Mirrors train.parse_supabase_timestamps."""
    return (
        pd.to_datetime(series, utc=True, format='ISO8601')
          .dt.tz_convert('America/Los_Angeles')
          .dt.tz_localize(None)
    )


def fetch_capacity_log():
    from supabase import create_client
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_ANON_KEY"]
    sb = create_client(os.environ["SUPABASE_URL"], key)

    print("Loading data from Supabase capacity_log...")
    BATCH, offset, rows = 9000, 0, []
    while True:
        batch = (
            sb.table("capacity_log")
            .select("timestamp,people_count,sensor_ok")
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
    # Mirror build_curves.py: drop readings taken while the counter was dead.
    if 'sensor_ok' in df.columns:
        df = df[df['sensor_ok'] != False].drop(columns=['sensor_ok'])
    df['timestamp'] = parse_supabase_timestamps(df['timestamp'])
    df['people_count'] = df['people_count'].astype(float)
    return df


def segment_for_date(d, ramp_days=10):
    """regular / first_week / ramp / break_deep / finals_dead / holiday. Reporting only."""
    phase = classify_date(d)
    if phase in ("winter_break", "spring_break") or phase.startswith("summer_break_"):
        dts, dte = days_to_sem_start(d), days_to_sem_end(d)
        if (1 <= dts <= ramp_days) or (1 <= -dte <= ramp_days):
            return "ramp"
        return "break_deep"
    if phase in ("finals", "dead_week"):
        return "finals_dead"
    return phase


def correction_segment(phase):
    """Mirrors predictions_builder._correction_segment (break sub-phases pooled)."""
    if phase in ("winter_break", "spring_break") or phase.startswith("summer_break_"):
        return "break"
    return phase


def open_slot_range(d):
    open_h, close_h = get_open_hours(pd.Timestamp(d).day_name(), d)
    return open_h * 4, close_h * 4


def load_slots(use_cache=True):
    if use_cache and os.path.exists(CACHE):
        print(f"Using cached capacity_log -> {CACHE}")
        raw = pickle.load(open(CACHE, "rb"))
    else:
        raw = fetch_capacity_log()
        pickle.dump(raw, open(CACHE, "wb"))
        print(f"Cached capacity_log -> {CACHE}")

    slots = cm.prepare_slots(raw)
    n_days = slots['date'].nunique()
    print(f"Prepared {len(slots):,} (date, slot) rows, {n_days:,} distinct days, "
          f"{slots['date'].min().date()} -> {slots['date'].max().date()}")
    if n_days < 1000:
        # capacity_log's RLS policy caps the anon role at ~3 days.
        os.path.exists(CACHE) and os.remove(CACHE)
        sys.exit(
            f"\nABORT: only {n_days} distinct days returned. capacity_log is "
            "RLS-limited to ~3 days for the anon role.\nSet SUPABASE_SERVICE_KEY "
            "(Supabase dashboard -> Project Settings -> API -> service_role) in "
            "gym-tracker/.env and re-run.\n(The bad cache has been removed.)"
        )
    return slots


def build_day_matrix(slots):
    """(dates, M) where M[i, s] is day i's actual percent_full at slot s, NaN if unobserved."""
    dates = np.array(sorted(slots['date'].dt.date.unique()))
    index = {d: i for i, d in enumerate(dates)}
    M = np.full((len(dates), SLOTS_PER_DAY), np.nan)
    for d, s, pct in zip(slots['date'].dt.date, slots['slot'], slots['percent_full']):
        M[index[d], int(s)] = pct
    return dates, M


def build_base_matrix(slots, dates, origins=None, verbose=True):
    """Raw curve predictions, one honest table per monthly origin.

    A day in month M is predicted by a table built from data strictly before M,
    so no day is ever scored against a curve that contains it. This matters more
    than usual here: thin phases like first_week carry n_eff ~= 2.3, so a single
    leaked day is a large share of its own curve.
    """
    origins = origins or ORIGINS
    B = np.full((len(dates), SLOTS_PER_DAY), np.nan)
    index = {d: i for i, d in enumerate(dates)}
    scored_origin = {}

    for origin in origins:
        nxt = (pd.Timestamp(origin) + pd.DateOffset(months=1)).date()
        train = slots[slots['date'] < pd.Timestamp(origin)]
        if train.empty or train['date'].nunique() < 200:
            continue
        table = cm.build_table(train, PARAMS, build_date=origin)

        month_days = [d for d in dates if origin <= d < nxt]
        pairs, targets = [], []
        for d in month_days:
            lo, hi = open_slot_range(d)
            if lo >= hi:                      # full-facility closure day
                continue
            for s in range(lo, hi):
                pairs.append((d, s))
                targets.append((index[d], s))
        if not pairs:
            continue

        preds = cm.predict(table, pairs, blend_window_days=PARAMS.get('blend_window_days'))
        for (i, s), p in zip(targets, preds):
            B[i, s] = p
        for d in month_days:
            scored_origin[d] = origin
        if verbose:
            print(f"  origin {origin}: {len(month_days)} days, {len(pairs):,} slots")

    return B, scored_origin


def apply_28day_correction(dates, actual_M, raw_M, verbose=True):
    """Reproduce predictions_builder.py's trailing-residual layer.

    Its key is (segment, regime, dow, hour, minute) over a 28-day window, and
    `dow` is IN the key — so within any 28-day window exactly four days can
    contribute to a cell: D-7, D-14, D-21, D-28. That turns what reads like a
    rolling aggregation into a four-row lookup.

    The horizon decay (0.5^(days/7)) is 1.0 for the current day, so it is
    omitted: everything downstream of this module evaluates day-of predictions
    only. Note the correction is NOT future-only — it applies to the current day
    at full strength, which is exactly why it must be reproduced here.
    """
    from academic_calendar import is_summer_day
    index  = {d: i for i, d in enumerate(dates)}
    seg    = {d: correction_segment(classify_date(d)) for d in dates}
    regime = {d: is_summer_day(d) for d in dates}

    out = raw_M.copy()
    resid = actual_M - raw_M
    n_fired = 0

    for i, d in enumerate(dates):
        rows = [
            index[p] for p in (d - timedelta(days=k) for k in (7, 14, 21, 28))
            if p in index and seg[p] == seg[d] and regime[p] == regime[d]
        ]
        if len(rows) < CORRECTION_MIN_N:
            continue
        block  = resid[rows]
        counts = np.isfinite(block).sum(axis=0)
        # Slots nobody recorded on any of the prior days give an all-NaN column.
        # They are dropped a line below by the `counts >= MIN_N` mask, so the
        # empty-slice warning nanmean raises for them is noise, not a signal.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cell = np.nanmean(np.where(np.isfinite(block), block, np.nan), axis=0)
        usable = (counts >= CORRECTION_MIN_N) & np.isfinite(cell)
        if usable.any():
            out[i, usable] = raw_M[i, usable] + cell[usable]
            n_fired += 1

    if verbose:
        print(f"  28-day correction fired on {n_fired:,}/{len(dates):,} days")
    return out


def load_matrices(use_cache=True, verbose=True):
    """(slots, dates, actuals, deployed_base, scored_origin)."""
    slots = load_slots(use_cache)
    dates, M = build_day_matrix(slots)

    if use_cache and os.path.exists(RAW_CACHE):
        raw_M, scored_origin = pickle.load(open(RAW_CACHE, "rb"))
        print(f"Using cached raw base matrix -> {RAW_CACHE}")
    else:
        print("Building rolling-origin curve predictions (slow, cached after this)...")
        raw_M, scored_origin = build_base_matrix(slots, dates, verbose=verbose)
        pickle.dump((raw_M, scored_origin), open(RAW_CACHE, "wb"))

    print("Applying the 28-day trailing correction (matching predictions_builder)...")
    base_M = apply_28day_correction(dates, M, raw_M, verbose=verbose)
    return slots, dates, M, base_M, scored_origin
