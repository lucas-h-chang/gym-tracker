"""
nowcast_carry.py — measure how far today's evidence actually travels.

SUPERSEDED (2026-08-31). This was the exploratory measurement that established
the carry structure; its conclusions are written up in
handoffs/SPEC_TODAY_BUILDER_REWRITE.md and its raw output is kept in
nowcast_carry_report.json.

Do not extend it. The production path is now:
    carry_data.py    data prep (curve + the 28-day layer = the deployed baseline)
    carry_model.py   the model
    build_carry.py   fits models/carry.json
    replay_day.py    scores it against the deployed model, at production cadence

Two things this script gets wrong that replay_day.py gets right, so its numbers
should not be quoted in preference to a replay run:
  - it scores against the RAW curve, not the deployed baseline, so it
    double-counts the drift predictions_builder's 28-day layer already removes
  - it takes hourly cuts, which hid two regressions that only appear at the
    real 15-minute cadence (see the spec's "Comprehensive backtest" section)

Kept only so the figures quoted in the spec remain traceable to the code that
produced them.

WHY THIS EXISTS
---------------
The live today-curve blends two models on a stopwatch: `today_builder.py`
gives its similarity nowcast weight 0.9 at the last observed slot and ramps
that to 0 over BLEND_HORIZON = 2.0 hours. That horizon is a hand-picked
number, and picking it has a cost that is invisible on the chart: whenever
the two models disagree by D, the *act of handing off* adds a slope of
D * (0.9 / BLEND_HORIZON) per hour to the drawn line. On 2026-08-31 the two
models sat ~21pp apart at 10:15 AM, so the curve climbed ~2.4pp per 15 min
for reasons that had nothing to do with anyone arriving at the gym.

The fix is to stop blending two curves and instead correct ONE curve's level:

    forecast(d, x) = base(d, x) + b_day(h) * r_day + b_recent(h) * r_recent

  base      the deployed curve-model prediction (curve_model / build_curves)
  r_day     today's mean residual (actual - base) over every slot observed so far
  r_recent  today's mean residual over the last hour only
  h         horizon, in hours, from the last observation to the target slot x
  b_*       COEFFICIENTS, fitted from history — not a schedule

Splitting the residual in two is the point. "Today is quiet" is ambiguous: a
day that has run low since open is making a claim about the day, while a day
that dipped in the last 30 minutes is making a claim about the last 30
minutes. One number cannot tell those apart. Two can, and the fitted b_day /
b_recent say how far each kind of evidence carries.

This script measures b_day(h) and b_recent(h), then checks on held-out
origins whether applying them actually beats (a) leaving the base curve
alone and (b) the similarity nowcast that is deployed today.

If b_day comes back near zero at long horizons, the honest conclusion is
that a quiet morning genuinely says nothing about the evening, and the
evening should be left to the base curve. That is a real possible outcome
and the report says so rather than burying it.

HONESTY RULES OBSERVED HERE
---------------------------
- Rolling origin: the base curve for a day in month M is built only from
  data strictly before month M, so a day is never scored against a curve
  that already contains it. This matters more than usual — thin phases like
  first_week carry n_eff ~= 2.3, so one leaked day is a large share of its
  own curve.
- Each day is scored exactly once (monthly origin, 1-month scoring window),
  so no day is double-counted in the regression.
- Coefficients are fitted on FIT_ORIGINS and evaluated on HOLDOUT_ORIGINS.
  The headline comparison numbers come only from the holdout.
- The nowcast replica restricts candidates to `date <= d - 8`, mirroring
  today_builder's EXCLUDE_WINDOW rule, so it is not allowed to see the day
  it is predicting.

Run:  python3 nowcast_carry.py             (full run, a few minutes)
      python3 nowcast_carry.py --fast      (every 3rd day, for a smoke test)

Loads .env itself; the key is read into os.environ and never printed.
Writes nowcast_carry_report.json next to this script.
"""
import os
import sys
import json
import pickle
import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ── Load .env without echoing values (same pattern as blend_sweep.py) ───────
_envpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_envpath):
    for line in open(_envpath):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
if "SUPABASE_SERVICE_KEY" not in os.environ and "SUPABASE_KEY" in os.environ:
    os.environ["SUPABASE_SERVICE_KEY"] = os.environ["SUPABASE_KEY"]

import curve_model as cm
from academic_calendar import (
    classify_date, days_to_sem_start, days_to_sem_end, get_open_hours, is_semester_day,
)

# fetch_capacity_log / segment_for_date are deliberately re-implemented here
# rather than imported from backtest.py. backtest.py imports train.py for the
# retired RF baseline, which pulls in sklearn — a heavy dependency this script
# has no use for, and one that is not installed in a plain checkout. These two
# helpers are small and stable; carrying copies is cheaper than making a
# measurement script depend on a model that is no longer in the inference path.


def parse_supabase_timestamps(series):
    """TIMESTAMPTZ (UTC) -> naive PT wall-clock. Mirrors train.parse_supabase_timestamps."""
    return (
        pd.to_datetime(series, utc=True, format='ISO8601')
          .dt.tz_convert('America/Los_Angeles')
          .dt.tz_localize(None)
    )


def fetch_capacity_log():
    """Full capacity_log as naive-PT rows. Mirrors backtest.fetch_capacity_log."""
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
    # Mirror build_curves.py: drop readings taken while the counter was dead, so
    # this preps off the same rows production does.
    if 'sensor_ok' in df.columns:
        df = df[df['sensor_ok'] != False].drop(columns=['sensor_ok'])
    df['timestamp'] = parse_supabase_timestamps(df['timestamp'])
    df['people_count'] = df['people_count'].astype(float)
    return df


def segment_for_date(d, ramp_days=10):
    """regular / first_week / ramp / break_deep / finals_dead / holiday.

    Mirrors backtest.segment_for_date — report readability only, no model effect.
    """
    phase = classify_date(d)
    if phase in ("winter_break", "spring_break") or phase.startswith("summer_break_"):
        dts = days_to_sem_start(d)
        dte = days_to_sem_end(d)
        if (1 <= dts <= ramp_days) or (1 <= -dte <= ramp_days):
            return "ramp"
        return "break_deep"
    if phase in ("finals", "dead_week"):
        return "finals_dead"
    return phase

CACHE = os.environ.get("SWEEP_CACHE", "/tmp/rsf_capacity_cache.pkl")
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nowcast_carry_report.json")

# Match build_curves.py exactly, so `base` here is the curve that production
# actually serves rather than a differently-parameterised lookalike.
PARAMS = {**cm.DEFAULT_PARAMS, "week_levels": True}

ORIGINS = [d for d in (date(y, m, 1) for y in (2023, 2024, 2025, 2026) for m in range(1, 13))
           if date(2023, 1, 1) <= d <= date(2026, 8, 1)]
FIT_ORIGINS = [d for d in ORIGINS if d < date(2025, 7, 1)]
HOLDOUT_ORIGINS = [d for d in ORIGINS if d >= date(2025, 7, 1)]

SLOTS_PER_HOUR = 4
RECENT_SLOTS = 4          # "the last hour" = 4 quarter-hour slots
MIN_OBSERVED = 4          # need >= 1h of readings before a cut is usable

# Horizon buckets, in hours from the last observation to the target slot.
# Upper bound is exclusive; the last bucket catches everything beyond.
HORIZON_EDGES = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 99.0]

# ── today_builder.py's constants, mirrored so the replica drifts loudly ─────
NC_K = 5
NC_BLEND_CEIL = 0.9
NC_BLEND_HORIZON = 2.0
NC_OVERLAP_THRESH = 0.70
NC_MIN_SLOTS = 4
NC_ANCHOR_DECAY_H = 4.0
NC_EXCLUDE_DAYS = 8
NC_DATA_CUTOFF = date(2022, 1, 1)


# ───────────────────────────────────────────────────────────────────────────
# Data loading
# ───────────────────────────────────────────────────────────────────────────

def load_slots():
    if os.path.exists(CACHE):
        print(f"Using cached capacity_log -> {CACHE}")
        with open(CACHE, "rb") as f:
            raw = pickle.load(f)
    else:
        raw = fetch_capacity_log()
        with open(CACHE, "wb") as f:
            pickle.dump(raw, f)
        print(f"Cached capacity_log -> {CACHE}")

    slots = cm.prepare_slots(raw)
    n_days = slots['date'].nunique()
    print(f"Prepared {len(slots):,} (date, slot) rows, {n_days:,} distinct days, "
          f"{slots['date'].min().date()} -> {slots['date'].max().date()}")
    if n_days < 1000:
        # capacity_log's RLS policy caps the anon/publishable role at ~3 days.
        os.path.exists(CACHE) and os.remove(CACHE)
        sys.exit(
            f"\nABORT: only {n_days} distinct days returned. capacity_log is "
            "RLS-limited to ~3 days for the anon role.\nSet SUPABASE_SERVICE_KEY "
            "(Supabase dashboard -> Project Settings -> API -> service_role) in "
            "gym-tracker/.env and re-run.\n(The bad cache has been removed.)"
        )
    return slots


def build_day_matrix(slots):
    """(dates, M) where M[i, s] is day i's actual percent_full at slot s, NaN if unobserved.

    A dense 96-wide row per day is what makes the nowcast replica affordable:
    the neighbour search becomes one vectorised distance over a (n_days, 96)
    array instead of a Python loop over candidate days.
    """
    dates = np.array(sorted(slots['date'].dt.date.unique()))
    index = {d: i for i, d in enumerate(dates)}
    M = np.full((len(dates), 96), np.nan)
    for d, s, pct in zip(slots['date'].dt.date, slots['slot'], slots['percent_full']):
        M[index[d], int(s)] = pct
    return dates, M


def open_slot_range(d):
    open_h, close_h = get_open_hours(pd.Timestamp(d).day_name(), d)
    return open_h * SLOTS_PER_HOUR, close_h * SLOTS_PER_HOUR


# ───────────────────────────────────────────────────────────────────────────
# Base curve, one honest table per origin
# ───────────────────────────────────────────────────────────────────────────

def build_base_matrix(slots, dates, origins, verbose=True):
    """B[i, s] = the base-curve prediction for day i at slot s, NaN outside open hours.

    The table for a day in month M is built from data strictly before M, so no
    day is ever scored against a curve that contains it.
    """
    B = np.full((len(dates), 96), np.nan)
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
            if lo >= hi:          # full-facility closure day
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
            print(f"  origin {origin}: base curve for {len(month_days)} days, {len(pairs):,} slots")

    return B, scored_origin


# ───────────────────────────────────────────────────────────────────────────
# Replica of the deployed similarity nowcast (today_builder.py)
# ───────────────────────────────────────────────────────────────────────────

def nowcast_for_cut(day_i, cut_slot, dates, M, base_row, obs_idx, cand_mask_cache):
    """today_builder.compute_similarity_predictions for one (day, cut), as an array.

    Returns P[96] = the blended (1-w)*base + w*sim value the live site would
    have drawn for each future slot, NaN where the nowcast produced nothing.
    Mirrors the deployed algorithm: K=5 neighbours by RMSE over the observed
    slots, inverse-distance weights, an anchor offset that decays over
    NC_ANCHOR_DECAY_H hours, and a per-slot weight that starts at
    NC_BLEND_CEIL and decays to 0 over NC_BLEND_HORIZON hours.
    """
    d = dates[day_i]
    if len(obs_idx) < NC_MIN_SLOTS:
        return None

    key = (pd.Timestamp(d).dayofweek, is_semester_day(d))
    base_pool = cand_mask_cache.get(key)
    if base_pool is None:
        pool = np.array([
            (pd.Timestamp(c).dayofweek == key[0]) and (is_semester_day(c) == key[1])
            for c in dates
        ])
        cand_mask_cache[key] = pool
        base_pool = pool
    # today_builder's server-side window: DATA_CUTOFF <= date <= today - 8.
    mask = base_pool & (dates <= d - timedelta(days=NC_EXCLUDE_DAYS)) & (dates >= NC_DATA_CUTOFF)
    if not mask.any():
        return None

    sub = M[mask]                                   # (n_cand, 96)
    obs = sub[:, obs_idx]                           # (n_cand, n_obs)
    overlap = (~np.isnan(obs)).sum(axis=1)
    keep = overlap >= NC_OVERLAP_THRESH * len(obs_idx)
    if not keep.any():
        return None
    sub, obs = sub[keep], obs[keep]

    today_obs = M[day_i, obs_idx]
    diff = obs - today_obs[None, :]
    dist = np.sqrt(np.nanmean(diff ** 2, axis=1))
    order = np.argsort(dist)[:NC_K]
    top, dtop = sub[order], dist[order]

    wts = 1.0 / (dtop + 1e-6)
    wts /= wts.sum()

    # Weighted mean per slot over whichever neighbours have that slot.
    valid = ~np.isnan(top)
    wmat = valid * wts[:, None]
    denom = wmat.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        sim = np.nansum(np.where(valid, top, 0.0) * wts[:, None], axis=0) / denom
    sim[denom == 0] = np.nan

    last = obs_idx[-1]
    if np.isnan(sim[last]):
        return None
    offset = M[day_i, last] - sim[last]

    out = np.full(96, np.nan)
    lo, hi = open_slot_range(d)
    for s in range(max(cut_slot + 1, lo), hi):
        if np.isnan(sim[s]) or np.isnan(base_row[s]):
            continue
        gap_h = (s - last) / SLOTS_PER_HOUR
        anchored = sim[s] + offset * max(0.0, 1.0 - gap_h / NC_ANCHOR_DECAY_H)
        anchored = min(100.0, max(0.0, anchored))
        w = NC_BLEND_CEIL * max(0.0, 1.0 - gap_h / NC_BLEND_HORIZON)
        out[s] = (1 - w) * base_row[s] + w * anchored
    return out


# ───────────────────────────────────────────────────────────────────────────
# Sample construction
# ───────────────────────────────────────────────────────────────────────────

def build_samples(dates, M, B, scored_origin, fast=False, with_nowcast=True):
    """One row per (day, cut, future slot).

    r_day     mean residual over every observed slot so far
    r_recent  mean residual over the last RECENT_SLOTS observed slots
    y         the residual at the future slot (what we are trying to predict)
    nowcast   what the deployed blend would have drawn there, for comparison
    """
    rows = []
    cand_cache = {}
    day_list = [d for d in dates if d in scored_origin]
    if fast:
        day_list = day_list[::3]

    for n, d in enumerate(day_list):
        i = int(np.searchsorted(dates, d))
        lo, hi = open_slot_range(d)
        if lo >= hi:
            continue
        actual, base = M[i], B[i]
        resid = actual - base
        day_slots = np.arange(lo, hi)
        observed_all = day_slots[~np.isnan(resid[day_slots])]
        if len(observed_all) < MIN_OBSERVED + SLOTS_PER_HOUR:
            continue

        # Cuts on the hour, from one hour after open to one hour before close.
        for cut in range(lo + SLOTS_PER_HOUR, hi - SLOTS_PER_HOUR, SLOTS_PER_HOUR):
            obs_idx = observed_all[observed_all <= cut]
            if len(obs_idx) < MIN_OBSERVED:
                continue
            r_day = float(np.mean(resid[obs_idx]))
            r_recent = float(np.mean(resid[obs_idx[-RECENT_SLOTS:]]))
            last = int(obs_idx[-1])

            nc = None
            if with_nowcast:
                nc = nowcast_for_cut(i, cut, dates, M, base, obs_idx, cand_cache)

            future = day_slots[(day_slots > cut) & (~np.isnan(resid[day_slots]))]
            for s in future:
                rows.append((
                    d, scored_origin[d], cut, cut // SLOTS_PER_HOUR, int(s),
                    (int(s) - last) / SLOTS_PER_HOUR, len(obs_idx),
                    r_day, r_recent, float(resid[s]), float(base[s]), float(actual[s]),
                    (float(nc[s]) if (nc is not None and not np.isnan(nc[s])) else np.nan),
                ))
        if n and n % 200 == 0:
            print(f"  built samples through {d} ({n}/{len(day_list)} days, {len(rows):,} rows)")

    df = pd.DataFrame(rows, columns=[
        "date", "origin", "cut", "cut_hour", "slot", "horizon", "n_obs",
        "r_day", "r_recent", "y", "base", "actual", "nowcast",
    ])
    df["segment"] = df["date"].map(segment_for_date)
    df["hbucket"] = pd.cut(df["horizon"], bins=[0] + HORIZON_EDGES,
                           labels=[f"<={e}h" for e in HORIZON_EDGES], right=True)
    return df


# ───────────────────────────────────────────────────────────────────────────
# Fitting
# ───────────────────────────────────────────────────────────────────────────

def fit_ols(sub):
    """y ~ intercept + b_day*r_day + b_recent*r_recent. Returns coefficients + fit stats."""
    if len(sub) < 50:
        return None
    X = np.column_stack([np.ones(len(sub)), sub["r_day"].values, sub["r_recent"].values])
    y = sub["y"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(len(sub)),
        "intercept": round(float(coef[0]), 4),
        "b_day": round(float(coef[1]), 4),
        "b_recent": round(float(coef[2]), 4),
        # r_day and r_recent both contain the day's persistent level, so they are
        # collinear and OLS's split between them is not meaningful on its own —
        # it just tracks which is the less noisy proxy at this horizon. The SUM
        # is the interpretable quantity: "of a gap that is present both all day
        # and right now, how much is still there at this horizon". 1.0 = carries
        # fully, 0.0 = says nothing, leave the base curve alone.
        # NOTE: total_carry is the OPTIMAL SHRINKAGE TO APPLY, not raw persistence.
        # r_day/r_recent are noisy estimates of today's true level, so OLS
        # correctly attenuates toward 0 — the noisier the readings, the more it
        # shrinks. Verified on synthetic data: with noise-free inputs and full
        # carry planted, this returns exactly 1.0000 across all 108 cells; adding
        # an AR(1) transient of sd 6 pulls it to 0.906, sd 12 to 0.751. So a
        # value below 1.0 is not evidence the signal decays — it is the estimator
        # doing the right thing with imperfect inputs, and it is the number to
        # deploy verbatim rather than to "correct" upward.
        "total_carry": round(float(coef[1] + coef[2]), 4),
        "r2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
    }


def fit_by_horizon(df, extra_key=None):
    keys = ["hbucket"] + ([extra_key] if extra_key else [])
    out = {}
    for k, sub in df.groupby(keys, observed=True):
        f = fit_ols(sub)
        if f:
            out["|".join(str(x) for x in (k if isinstance(k, tuple) else (k,)))] = f
    return out


def fit_by_cut_and_horizon(df):
    """Coefficients keyed on (cut hour, horizon bucket) — the form production uses.

    WHY NOT HORIZON ALONE. A long horizon can only ever occur at an early cut,
    so a horizon-only fit confounds two different things: how far today's signal
    genuinely carries, and how noisy the day-so-far estimate is when only an hour
    of readings exists. Verified on synthetic data with a KNOWN full-carry
    structure, where the horizon-only total_carry decayed 0.96 -> 0.79 purely
    from that confound, with no decay planted in the data at all.

    Keying on both is also what inference actually has available: at prediction
    time the cut hour and the target slot are both known, so there is no reason
    to marginalise over one of them.
    """
    out = {}
    for (ch, hb), sub in df.groupby(["cut_hour", "hbucket"], observed=True):
        f = fit_ols(sub)
        if f:
            out[f"cut{int(ch):02d}|{hb}"] = f
    return out


def apply_coefs(df, coefs, fallback=None):
    """Correction predicted by the fitted coefficients, per row.

    Looks up the (cut hour, horizon) cell first; falls back to the horizon-only
    fit, then to no correction at all, so a thin cell can never blow up a score.
    """
    out = np.zeros(len(df))
    it = zip(df["cut_hour"].astype(int), df["hbucket"].astype(str),
             df["r_day"].values, df["r_recent"].values)
    for i, (ch, hb, rd, rr) in enumerate(it):
        c = coefs.get(f"cut{ch:02d}|{hb}")
        if c is None and fallback is not None:
            c = fallback.get(hb)
        if c is not None:
            out[i] = c["intercept"] + c["b_day"] * rd + c["b_recent"] * rr
    return out


def score(df, coefs, fallback=None):
    """MAE of: base alone, base + fitted correction, and the deployed nowcast blend."""
    corr = apply_coefs(df, coefs, fallback)
    corrected = np.clip(df["base"].values + corr, 0, 110)
    actual = df["actual"].values

    res = {
        "n": int(len(df)),
        "mae_base_only": round(float(np.abs(df["base"].values - actual).mean()), 3),
        "mae_corrected": round(float(np.abs(corrected - actual).mean()), 3),
    }
    have_nc = df["nowcast"].notna().values
    if have_nc.any():
        res["n_with_nowcast"] = int(have_nc.sum())
        res["mae_base_only_on_nc_rows"] = round(
            float(np.abs(df["base"].values[have_nc] - actual[have_nc]).mean()), 3)
        res["mae_nowcast_deployed"] = round(
            float(np.abs(df["nowcast"].values[have_nc] - actual[have_nc]).mean()), 3)
        res["mae_corrected_on_nc_rows"] = round(
            float(np.abs(corrected[have_nc] - actual[have_nc]).mean()), 3)
    return res


# ───────────────────────────────────────────────────────────────────────────
# Report
# ───────────────────────────────────────────────────────────────────────────

HB_LABELS = [f"<={e}h" for e in HORIZON_EDGES]


def print_carry_table(title, coefs):
    print(f"\n{title}")
    print(f"  {'horizon':<10} {'n':>9} {'carry':>7}  {'b_day':>7} {'b_recent':>9} {'R2':>7}")
    for hb in HB_LABELS:
        c = coefs.get(hb)
        if not c:
            continue
        r2 = c['r2'] if c['r2'] is not None else float('nan')
        print(f"  {hb:<10} {c['n']:>9,} {c['total_carry']:>7.3f}  "
              f"{c['b_day']:>7.3f} {c['b_recent']:>9.3f} {r2:>7.3f}")


def print_carry_matrix(coefs, cut_hours):
    """total_carry as a (cut hour) x (horizon) grid — the headline result.

    Read a row as: "having watched the day through this hour, how much of
    today's gap is still there N hours from now."
    """
    print("\nCARRY MATRIX — how much of today's gap survives")
    print("  rows = observed through this hour (PT), cols = hours ahead\n")
    print("  " + "obs thru".ljust(10) + "".join(h.rjust(9) for h in HB_LABELS))
    for ch in cut_hours:
        cells = []
        for hb in HB_LABELS:
            c = coefs.get(f"cut{ch:02d}|{hb}")
            cells.append(f"{c['total_carry']:9.2f}" if c else "        -")
        print(f"  {str(ch) + ':00':<10}" + "".join(cells))
    print("\n  1.00 = today's gap carries fully   0.00 = says nothing, use the base curve")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="every 3rd day, for a smoke test")
    ap.add_argument("--no-nowcast", action="store_true",
                    help="skip the deployed-nowcast replica (much faster)")
    args = ap.parse_args()

    slots = load_slots()
    dates, M = build_day_matrix(slots)
    print(f"Day matrix: {M.shape[0]:,} days x {M.shape[1]} slots")

    print("\nBuilding rolling-origin base curves (no day sees a table containing itself)...")
    B, scored_origin = build_base_matrix(slots, dates, ORIGINS)

    print("\nBuilding samples...")
    df = build_samples(dates, M, B, scored_origin,
                       fast=args.fast, with_nowcast=not args.no_nowcast)
    df = df.dropna(subset=["y", "base", "actual"])
    print(f"  {len(df):,} rows over {df['date'].nunique():,} days")

    fit = df[df["origin"].isin(FIT_ORIGINS)]
    hold = df[df["origin"].isin(HOLDOUT_ORIGINS)]
    print(f"  fit: {len(fit):,} rows ({fit['date'].nunique():,} days)   "
          f"holdout: {len(hold):,} rows ({hold['date'].nunique():,} days)")

    horizon_only = fit_by_horizon(fit)
    coefs = fit_by_cut_and_horizon(fit)
    print_carry_table("CARRY BY HORIZON (summary; confounds horizon with cut time)", horizon_only)
    print("\n  carry    = b_day + b_recent, the interpretable total (see fit_ols)")
    print("  b_day    = loading on the whole-day-so-far gap")
    print("  b_recent = loading on the last-hour gap")
    print("  the split between the two is arbitrary (they are collinear); the sum is not")

    cut_hours = sorted(fit["cut_hour"].unique().tolist())
    print_carry_matrix(coefs, cut_hours)

    seg_coefs = fit_by_horizon(fit, extra_key="segment")

    # The question that started this: observed through mid-morning, predict evening.
    mm = fit[(fit["cut"] // SLOTS_PER_HOUR).between(10, 11) &
             (fit["slot"] // SLOTS_PER_HOUR).between(17, 21)]
    morning_to_evening = fit_ols(mm)
    mm_fw = mm[mm["segment"] == "first_week"]
    morning_to_evening_first_week = fit_ols(mm_fw)

    print("\nMID-MORNING (10-11 AM observed) -> EVENING (5-9 PM predicted)")
    print("  ** this is the exact question that started this: does a quiet morning")
    print("     tell you anything about tonight? **")
    for label, f in (("all days", morning_to_evening),
                     ("first_week only", morning_to_evening_first_week)):
        if f:
            print(f"  {label:<18} n={f['n']:>7,}  carry={f['total_carry']:>6.3f}  "
                  f"(b_day={f['b_day']:>6.3f} b_recent={f['b_recent']:>6.3f})  R2={f['r2']:.3f}")
        else:
            print(f"  {label:<18} too few rows to fit")

    holdout = score(hold, coefs, horizon_only)
    print("\nHOLDOUT (2025-07 onward, coefficients never saw these days)")
    print(f"  rows scored                    {holdout['n']:>10,}")
    print(f"  MAE, base curve alone          {holdout['mae_base_only']:>10.3f} pp")
    print(f"  MAE, base + fitted correction  {holdout['mae_corrected']:>10.3f} pp")
    if "mae_nowcast_deployed" in holdout:
        print(f"\n  on the {holdout['n_with_nowcast']:,} rows where the deployed nowcast produced a value:")
        print(f"  MAE, base curve alone          {holdout['mae_base_only_on_nc_rows']:>10.3f} pp")
        print(f"  MAE, deployed nowcast blend    {holdout['mae_nowcast_deployed']:>10.3f} pp")
        print(f"  MAE, base + fitted correction  {holdout['mae_corrected_on_nc_rows']:>10.3f} pp")

    by_h = {}
    for hb, sub in hold.groupby("hbucket", observed=True):
        by_h[str(hb)] = score(sub, coefs, horizon_only)

    print("\nHOLDOUT BY HORIZON")
    print(f"  {'horizon':<10} {'n':>9} {'base':>8} {'corrected':>10} {'nowcast':>9}")
    for hb in HB_LABELS:
        r = by_h.get(hb)
        if not r:
            continue
        ncv = f"{r['mae_nowcast_deployed']:9.3f}" if "mae_nowcast_deployed" in r else "        -"
        print(f"  {hb:<10} {r['n']:>9,} {r['mae_base_only']:>8.3f} "
              f"{r['mae_corrected']:>10.3f}{ncv}")

    report = {
        "params": PARAMS,
        "fit_origins": [str(d) for d in FIT_ORIGINS],
        "holdout_origins": [str(d) for d in HOLDOUT_ORIGINS],
        "n_rows": int(len(df)),
        "n_days": int(df["date"].nunique()),
        "coefficients_by_cut_and_horizon": coefs,
        "coefficients_by_horizon": horizon_only,
        "coefficients_by_horizon_and_segment": seg_coefs,
        "morning_to_evening": morning_to_evening,
        "morning_to_evening_first_week": morning_to_evening_first_week,
        "holdout": holdout,
        "holdout_by_horizon": by_h,
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
