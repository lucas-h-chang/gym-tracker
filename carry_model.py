"""
carry_model.py — within-day level correction for the today curve.

Pure logic, no I/O — same role `curve_model.py` plays for the base curve.
`build_carry.py` fits it, `today_builder.py` applies it, `replay_day.py`
scores it, `test_carry.py` guards it.

WHAT THIS REPLACES
------------------
`today_builder.py` used to blend two competing curves on a stopwatch: a K=5
similarity nowcast weighted 0.9 at the last observed slot, ramping to 0 over
2 hours. That ramp is the bug. Whenever the nowcast and the base curve
disagree by D, the *act of handing off* injects a slope of D * (0.9 / 2.0)
per hour into the drawn line. On 2026-08-31 the two sat 21pp apart and the
chart climbed ~2.4pp per 15 min for reasons unrelated to anyone arriving —
freezing the nowcast completely flat still produced a 16-point climb.

THE MODEL
---------
One curve, corrected in level:

    forecast(slot) = base(slot) + correction
    correction     = intercept + a*gap_day + b*gap_recent + c*gap_last

The base curve keeps the *shape* of the day (it is built on years of data and
smoothed); today's readings only move the *level*. There is no second curve to
slide toward, so there is nothing left to manufacture a slope.

WHY THREE GAP TERMS
-------------------
"Today is quiet" is ambiguous. A day that has run low since open is making a
claim about the day; a day that dipped in the last 30 minutes is making a claim
about the last 30 minutes. One number cannot separate them. On 2026-08-31 at
the 10:15 cut the three read -4.3 / -16.2 / -20.7 pp — a mildly quiet day that
was also in a momentary trough, and the model correctly landed near the middle
rather than either extreme.

`gap_last` is not redundant with `gap_recent`. Without it the correction loses
to the old KNN at short range (5.142 vs 4.517 MAE at +15 min) purely because
averaging the last hour smooths away the detail that matters 15 minutes out.
With it, the correction wins at every horizon (3.973 vs 4.517).

COEFFICIENTS ARE FITTED, NOT SCHEDULED
--------------------------------------
That is the whole difference from the old design. `BLEND_HORIZON = 2.0` was a
hand-picked number; these come out of a regression on years of history, keyed
on (cut hour, horizon). Whatever slope they produce is a slope the data
supports.

Keying on BOTH matters: a long horizon can only occur at an early cut, so a
horizon-only fit confounds "how far the signal carries" with "how noisy an
hour-old estimate is". Verified on synthetic data with a known full-carry
structure, where a horizon-only fit decayed 0.96 -> 0.79 from that confound
alone with no decay present in the data.

Note that coefficients below 1.0 are NOT evidence the signal decays. They are
the optimal shrinkage for noisy inputs: on synthetic data with noise-free
inputs and full carry planted, the fit returns exactly 1.0000 across all 108
cells; an AR(1) transient of sd 6 pulls it to 0.906 and sd 12 to 0.751. Deploy
them verbatim rather than "correcting" them upward.
"""
import numpy as np
import pandas as pd

SLOTS_PER_HOUR = 4
RECENT_SLOTS   = 4      # "the last hour"
# Minimum readings before the correction fires at all. Fall back to the bare
# base curve below this.
#
# 6 (90 minutes) is not a round number, it is the smallest evidence level
# MEASURED to beat the base curve. Scored over 30 replayed days:
#
#     readings   = hrs        n      BASE       NEW      gain
#     4            1.0      490     9.295     9.536    -0.241   <- loses
#     6-7          1.5      788    12.955     9.791    +3.164
#     8-11         2.0    1,182    11.957     9.275    +2.682
#     20+          5.0    3,038    12.758     9.745    +3.013
#
# Exactly four readings is the only losing case, and the reason is structural:
# an hour after opening, every reading still sits on the steep opening ramp
# (the curve climbs ~46 -> 77% between 7:15 and 7:45), where a scrape landing a
# few minutes early or late produces a large residual that says nothing about
# the day's level. 5 was not chosen because hourly replay cuts never produce
# exactly 5 readings, so it would be interpolation rather than measurement.
MIN_OBSERVED   = 6

# Slots whose BASE prediction is below this are excluded from the gap inputs
# (not from the targets — every open slot still gets corrected).
#
# At 7:00 the base curve sits near 1.8%. A reading of 30% there is a timing
# artifact of the opening ramp, not evidence the day is running 28 points hot,
# and letting it into gap_day poisons the level estimate for the whole morning.
# Every measured number in SPEC_TODAY_BUILDER_REWRITE.md was produced with this
# filter in place.
MIN_BASE = 20.0

# Horizon buckets, in hours from the last observation to the target slot.
# Upper bound inclusive; the last bucket catches everything beyond.
HORIZON_EDGES = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 99.0]
HB_LABELS     = [f"<={e}h" for e in HORIZON_EDGES]

MIN_CELL_N = 50         # below this a (cut hour, bucket) cell is not fitted

# Shrinkage: the fitted correction is scaled by
#
#     lambda = n_obs / (n_obs + SHRINK_PER_HOUR * horizon_hours)
#
# so it is applied nearly in full 15 minutes out and pulled hard toward the bare
# base curve when extrapolating hours ahead on thin evidence. Same
# shrink-toward-the-parent form curve_model._shrink uses, with the strength
# scaled by how far the extrapolation reaches.
#
# WHY IT IS NEEDED. Scored over 357 days at true 15-minute production cadence
# (423,793 predictions), the UNSHRUNK correction was worse than doing nothing
# exactly where it extrapolated hardest:
#
#                       BASE    unshrunk   constant k=5   n/(n+1.5h)
#     overall         10.549      9.405          9.320        9.286
#     +15 min          9.568      4.511          4.791        4.516
#     horizon >8h     11.704     11.777         11.490       11.420
#     cut hour 8       9.282      9.829          9.172        9.021
#
# The cause is the year-over-year drift in the spec: the fitted coefficients
# encode a morning-to-evening relationship stronger than the one that currently
# holds, and OLS cannot know that.
#
# WHY IT SCALES WITH HORIZON AND NOT JUST EVIDENCE. A constant k fixes the
# long-range regressions but damages short range — see the +15 min column, where
# constant k=5 goes 4.511 -> 4.791 and falls behind the deployed model's 4.704.
# Fifteen minutes out the correction is almost entirely gap_last, which is
# reliable (fit R^2 0.88); eight hours out it is mostly gap_day extrapolation
# (R^2 0.29). Shrinking both by the same amount over-trusts the far end and
# under-trusts the near end. With k0 = 0 the near end is left alone entirely.
#
# 1.5 was selected on 2025-09..2026-02 and confirmed on 2026-03..2026-08, and
# the objective is flat across 1.0-2.0, so it is not a knife-edge.
SHRINK_PER_HOUR = 1.5


def shrink_factor(n_obs, horizon):
    """How much of the fitted correction to apply, given evidence and reach."""
    denom = float(n_obs) + SHRINK_PER_HOUR * float(horizon)
    return 1.0 if denom <= 0 else float(n_obs) / denom


def horizon_bucket(h):
    """Label for a horizon in hours. Mirrors the pd.cut used at fit time."""
    for e, lbl in zip(HORIZON_EDGES, HB_LABELS):
        if h <= e:
            return lbl
    return HB_LABELS[-1]


# ---------------------------------------------------------------------------
# Gap computation — the three numbers that describe "how today is going"
# ---------------------------------------------------------------------------

def compute_gaps(slots, actual, base):
    """
    slots/actual/base: equal-length sequences for the slots observed SO FAR,
    in ascending slot order.

    Returns (gap_day, gap_recent, gap_last, last_slot, n_obs), or None when
    there is not yet enough to say anything.
    """
    s = np.asarray(slots, dtype=float)
    a = np.asarray(actual, dtype=float)
    b = np.asarray(base, dtype=float)

    ok = np.isfinite(a) & np.isfinite(b) & (b >= MIN_BASE)
    if ok.sum() < MIN_OBSERVED:
        return None

    s, gaps = s[ok], (a[ok] - b[ok])
    order = np.argsort(s)
    s, gaps = s[order], gaps[order]

    return (
        float(gaps.mean()),
        float(gaps[-RECENT_SLOTS:].mean()),
        float(gaps[-1]),
        int(s[-1]),
        int(len(gaps)),
    )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def make_samples(dates, actual_M, base_M, open_range, day_filter=None, cut_step=SLOTS_PER_HOUR):
    """
    Build the regression frame: one row per (day, cut, future slot).

    dates:      array of date objects, ascending
    actual_M:   (n_days, 96) actuals, NaN where unobserved
    base_M:     (n_days, 96) base-curve predictions, NaN outside open hours
    open_range: callable(date) -> (lo_slot, hi_slot)
    day_filter: optional callable(date) -> bool

    Cuts are taken every `cut_step` slots from open+1h to close-1h, mirroring
    what the runtime sees. `y` is the residual at the future slot — the thing
    the correction is trying to predict.
    """
    rows = []
    for i, d in enumerate(dates):
        if day_filter is not None and not day_filter(d):
            continue
        lo, hi = open_range(d)
        if lo >= hi:                       # full-facility closure day
            continue

        day_slots = np.arange(lo, hi)
        actual, base = actual_M[i], base_M[i]
        resid = actual - base
        usable = day_slots[np.isfinite(resid[day_slots])]
        if len(usable) < MIN_OBSERVED + SLOTS_PER_HOUR:
            continue

        for cut in range(lo + SLOTS_PER_HOUR, hi - SLOTS_PER_HOUR, cut_step):
            obs = usable[usable <= cut]
            g = compute_gaps(obs, actual[obs], base[obs])
            if g is None:
                continue
            gap_day, gap_recent, gap_last, last_slot, _ = g

            future = usable[usable > cut]
            for s in future:
                rows.append((
                    d, cut // SLOTS_PER_HOUR, int(s),
                    (int(s) - last_slot) / SLOTS_PER_HOUR,
                    gap_day, gap_recent, gap_last,
                    float(resid[s]), float(base[s]), float(actual[s]),
                ))

    df = pd.DataFrame(rows, columns=[
        "date", "cut_hour", "slot", "horizon",
        "gap_day", "gap_recent", "gap_last", "y", "base", "actual",
    ])
    if not df.empty:
        df["hbucket"] = df["horizon"].map(horizon_bucket)
    return df


def _ols(sub):
    """y ~ 1 + gap_day + gap_recent + gap_last. Returns (coef4, n, h_mean, r2)."""
    X = np.column_stack([
        np.ones(len(sub)),
        sub["gap_day"].values,
        sub["gap_recent"].values,
        sub["gap_last"].values,
    ])
    y = sub["y"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_res, ss_tot = float((resid ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return (
        [round(float(c), 6) for c in coef],
        int(len(sub)),
        float(sub["horizon"].mean()),
        round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
    )


def _anchors_from(df, keys):
    """Fit one anchor per horizon bucket within each group, sorted by mean horizon.

    The anchor's `h` is the MEAN horizon of the rows that produced it, not the
    bucket's nominal edge — the coefficient estimates an average over the
    bucket, so that average is where it belongs on the horizon axis. This is
    what makes the interpolation in correction_at() self-calibrating.
    """
    out = {}
    for key, grp in df.groupby(keys, observed=True):
        anchors = []
        for _, sub in grp.groupby("hbucket", observed=True):
            if len(sub) < MIN_CELL_N:
                continue
            coef, n, h_mean, r2 = _ols(sub)
            anchors.append({"h": round(h_mean, 4), "n": n, "coef": coef, "r2": r2})
        if anchors:
            anchors.sort(key=lambda a: a["h"])
            k = key if not isinstance(key, tuple) else key[0]
            out[str(int(k))] = anchors
    return out


def build_table(samples, params=None, built_at=None):
    """Fit the full artifact from a make_samples() frame."""
    if samples.empty:
        raise ValueError("no samples to fit")

    pooled = []
    for _, sub in samples.groupby("hbucket", observed=True):
        if len(sub) < MIN_CELL_N:
            continue
        coef, n, h_mean, r2 = _ols(sub)
        pooled.append({"h": round(h_mean, 4), "n": n, "coef": coef, "r2": r2})
    pooled.sort(key=lambda a: a["h"])

    return {
        "version":  1,
        "built_at": built_at or pd.Timestamp.now().isoformat(),
        "params": {
            "recent_slots":  RECENT_SLOTS,
            "min_observed":  MIN_OBSERVED,
            "min_base":      MIN_BASE,
            "min_cell_n":    MIN_CELL_N,
            "horizon_edges": HORIZON_EDGES,
            **(params or {}),
        },
        "n_rows":  int(len(samples)),
        "n_days":  int(samples["date"].nunique()),
        "by_cut":  _anchors_from(samples, ["cut_hour"]),
        "pooled":  pooled,
    }


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def _interp_coef(anchors, horizon):
    """Linearly interpolate the four coefficients between bracketing anchors.

    WHY NOT JUST USE THE BUCKET'S COEFFICIENTS. Discrete buckets put a step in
    the drawn line wherever the horizon crosses a boundary. Measured on
    2026-08-31: the correction jumped from -9.5 to -5.6 pp between 6:15 and
    6:30 PM as the horizon crossed 8 hours, a +5.1pp step in the published
    curve. That is the same class of artifact as the blend ramp this model
    exists to remove, so it is interpolated away rather than tolerated.

    Outside the anchor range the endpoint is held flat — never extrapolated,
    which could run a coefficient off to an arbitrary value at horizons the
    fit never saw.
    """
    if not anchors:
        return None
    if len(anchors) == 1 or horizon <= anchors[0]["h"]:
        return anchors[0]["coef"]
    if horizon >= anchors[-1]["h"]:
        return anchors[-1]["coef"]

    for lo, hi in zip(anchors, anchors[1:]):
        if lo["h"] <= horizon <= hi["h"]:
            span = hi["h"] - lo["h"]
            t = 0.0 if span <= 0 else (horizon - lo["h"]) / span
            return [l + t * (h - l) for l, h in zip(lo["coef"], hi["coef"])]
    return anchors[-1]["coef"]


def _anchors_for_cut(table, cut_hour):
    """Anchors for this cut hour, else the nearest fitted cut hour, else pooled."""
    by_cut = table.get("by_cut") or {}
    if not by_cut:
        return table.get("pooled") or []
    key = str(int(cut_hour))
    if key in by_cut:
        return by_cut[key]
    hours = sorted(int(k) for k in by_cut)
    nearest = min(hours, key=lambda h: abs(h - cut_hour))
    return by_cut[str(nearest)]


def correction_at(table, cut_hour, horizon, gap_day, gap_recent, gap_last):
    """The level correction, in percentage points, for one future slot."""
    coef = _interp_coef(_anchors_for_cut(table, cut_hour), horizon)
    if coef is None:
        return 0.0
    i, a, b, c = coef
    return float(i + a * gap_day + b * gap_recent + c * gap_last)


def apply_to_day(table, cut_slot, last_slot, gaps, base_by_slot, lo, hi,
                 n_obs=None, clamp=(0.0, 110.0)):
    """
    Correct every remaining open slot of a day.

    gaps:         (gap_day, gap_recent, gap_last) from compute_gaps
    base_by_slot: {slot: base pct}
    n_obs:        readings behind the cut; drives shrink_factor. Passing None
                  applies the correction unshrunk, which is only appropriate
                  for unit tests that want the raw coefficients.
    Returns {slot: corrected pct} for slots after cut_slot.

    With all three gaps at zero this returns the base curve EXACTLY (up to the
    fitted intercept, which is ~0 by construction), which test_carry.py pins.
    """
    gap_day, gap_recent, gap_last = gaps
    out = {}
    for s in range(cut_slot + 1, hi):
        base = base_by_slot.get(s)
        if base is None or not np.isfinite(base):
            continue
        horizon = (s - last_slot) / SLOTS_PER_HOUR
        corr = correction_at(
            table, cut_slot // SLOTS_PER_HOUR, horizon,
            gap_day, gap_recent, gap_last,
        )
        lam = 1.0 if n_obs is None else shrink_factor(n_obs, horizon)
        out[s] = float(np.clip(base + lam * corr, *clamp))
    return out
