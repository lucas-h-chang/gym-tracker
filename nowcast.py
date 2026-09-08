"""
nowcast.py: the trailing-residual correction, as a shrinkage backoff ladder.

THE LAYER THIS IS
-----------------
The base curve (curve_model.py) is deliberately slow: halflife_days=365, so it
takes most of a year to notice that the gym has changed. This is the fast layer
on top of it. It looks at the last WINDOW_DAYS of readings, measures how far the
curve was off, and adds that back:

    residual = actual - curve_prediction
    served   = curve_prediction + shrunk_mean(residuals from similar recent days)

Curve gives the shape of a day, this gives the level it is running at.

WHY THIS MODULE EXISTS (it used to live in predictions_builder.py)
------------------------------------------------------------------
Two callers need the exact same arithmetic:

  - predictions_builder.py, which serves it, and
  - carry_data.py, which must reproduce the baseline production actually
    served in order to fit the within-day correction on top of it without
    double-counting the drift this layer already removed.

carry_data.py used to hand-mirror the logic with a comment asking future editors
to keep the copies in sync, and nothing enforced it. One implementation, two
importers, no mirror to drift.

WHY A LADDER AND NOT A QUORUM (the 2026-09 rewrite)
---------------------------------------------------
The original keyed every cell on (segment, regime, dow, hour, minute) and
required CORRECTION_MIN_N=3 rows before it would emit anything at all. Because
`dow` was in the key, only four days in a 28-day window could ever land in a
cell (D-7, D-14, D-21, D-28), so "3 of 28" was really "3 of 4", i.e. 75% of the
theoretical maximum, and a single holiday or closure knocked a cell below the
line. Measured over the calendar, that left the whole layer dark on ~38% of open
days, every year.

Worse, it was dark in a pattern. A phase shorter than 21 days can never place 3
of D-7/14/21/28 inside itself, so `first_week` (7 days), `dead_week` (5),
`finals` (5) and `holiday` (1-3) fired 0% of the time, structurally, forever.
That put a ~28-day blackout at the start of every semester, the highest-traffic,
fastest-changing stretch on the calendar, and precisely when a "running hot"
correction earns its keep. Fall 2026 was diagnosed from the outside: the 7:15am
forecast sat ~19pp under actuals for five straight weekdays with the correction
contributing exactly 0.00pp.

The fix is not new machinery. It is the machinery curve_model.build_table
already uses one layer down: a cascading shrinkage ladder, broadest to narrowest,

    R1  (regime, slot)               shrunk toward 0.0
    R2  (regime, is_weekend, slot)   shrunk toward R1
    R3  (regime, segment, dow, slot) shrunk toward R2

with weight n/(n+k) at every rung. A thin cell is believed a little, an empty
cell is believed not at all and simply inherits its parent. Nothing ever falls
off a cliff to zero, and no level needs a special case for "no data": n=0 gives
weight 0 on its own.

TWO KEY CHOICES
---------------
`segment` is demoted to the narrowest rung rather than gating the whole ladder.
The base curve has *already* accounted for phase, so a residual is the part the
curve got wrong, and there is no strong prior that drift-from-baseline is
phase-specific. Keying the broad rungs on segment is exactly what starved the
short phases, for very little in return.

`regime` (summer vs academic hours) stays in the key at every rung. Summer closes
at 8pm and the academic year at 11pm, and the pre-close emptying-out is real,
regime-specific signal the halflife-365 base curve misses. Without regime in the
key, a trailing window that is entirely summer stamps summer's ~8pm closing crash
onto an academic-hours target that is open until 11pm.

The ladder is rooted at literal 0.0. "No correction" is the correct prior for a
residual, so a stretch genuinely running on baseline collapses to ~0 by itself.
That preserves the conservatism the old quorum was clumsily buying, without the
blackouts. A whole-regime rung above R1 was built and measured and is OFF; see
correction() for why it lost.

MEASURED (2026-09-08)
---------------------
Scored as BASE MAE, i.e. raw curve + this layer, against every observed slot.
Parameters were selected on days before 2025-07-01 and confirmed on 2025-07-01
onward, matching the tune/holdout discipline curve_model uses.

    raw curve, no correction        holdout  9.0247
    legacy 4-row quorum (shipped)   holdout  9.3324
    this ladder                     holdout  8.3388     -10.6% vs the quorum

Note the middle row. On held-out data the shipped correction is WORSE than
applying no correction at all, which is what a layer that is dark 38% of the
time and unshrunk the rest of it looks like from the outside.

Held out, by segment of the day: mornings 7:15-8:15 go 6.7988 -> 5.8060 and
evenings 21:00-22:45 go 12.1062 -> 11.0389.
"""
from datetime import timedelta

import numpy as np

SLOTS_PER_DAY = 96

# ── Tunables ────────────────────────────────────────────────────────────────
# All swept together on the tune split and confirmed on the holdout. The surface
# is a broad shallow bowl rather than a spike: everything in k 1.0-3.0 by
# halflife 1.5-3.5 lands within ~1% of the optimum, so none of these is
# knife-edge.
#
# Window length barely matters once HALFLIFE_DAYS does the tapering (14, 21, 28
# and 42 all agree to the third decimal), so it stays at the 28 the quorum used.
WINDOW_DAYS = 28

# SHRINK_K is the ladder's "how much evidence before I believe a cell over its
# parent" knob: weight = n/(n+k), so the parent effectively arrives with k free
# observations. Swept here rather than inherited from curve_model's shrink_k=3,
# because that 3 was tuned on occupancy *levels* and this layer models
# residuals, which are a different and noisier quantity.
#
# The first guess here was 6.0, reasoning that noisier signal wants more
# shrinkage. That was wrong by a factor of four, and wrong in a way worth
# recording: with HALFLIFE_DAYS tapering the window, a slot carries only ~2-3
# effective observations, so k=6 meant the prior outvoted the data more than 2:1
# and the layer produced almost nothing. Reasoning about the noise without also
# reasoning about the sample size it is divided by gets the answer backwards.
#
# Selection rule, fixed before looking at the holdout: take the tune optimum,
# then move to the most conservative setting still within 1% of it, because the
# fast sweep only scores horizon 0 and a longer memory is more robust at the
# horizons it cannot see. That gives 1.5 rather than the tune-optimal 1.0.
SHRINK_K = 1.5

# Recency weighting inside the window: 0.5 ** (age_days / HALFLIFE_DAYS), the
# same idiom curve_model uses over its multi-year history. The old code weighted
# all four candidate days equally and then cut hard at 28 days; an exponential
# taper is smoother and is what makes WINDOW_DAYS a soft parameter.
#
# 2.5 days is much shorter than it looks like it should be, and it was swept, not
# chosen: monotonically better than 5, 7, 10 and 14 on both splits. That is the
# division of labour working as intended. The base curve carries a 365-day
# halflife and owns everything slow; this layer exists only to answer "what is
# different about right now", and the honest answer is that the useful signal is
# about a week old at most.
HALFLIFE_DAYS = 2.5

# Centered rolling mean over the slot axis. OFF (1 = no smoothing), and this was
# the second wrong guess: curve_model smooths its curves, so smoothing seemed
# obviously right here too. It lost in every single cell of the sweep.
#
# The reason is that curve_model smooths occupancy *levels*, which really are
# smooth in time, whereas residuals are not. The opening ramp is the clean
# counterexample: the gap between curve and reality across 7:00, 7:15, 7:30,
# 7:45 ran about -0.5, +24, +11, +7 pp in early fall 2026. A width-3 mean turns
# that +24 into roughly +11 by averaging it with the near-zero opening slot next
# door, cutting the correction at the one slot the whole rewrite was for by more
# than half. Variance control is what the shrinkage rungs are for.
SMOOTH_WINDOW = 1

# Weight applied at forecast time, by how many days out the target is:
# 0.5 ** (days / HORIZON_HALFLIFE). Was 7.0, and had to come down alongside
# HALFLIFE_DAYS: a correction whose evidence is ~2.5 days old cannot still be
# worth half its value a week later, and leaving it there measured worse.
#
# Swept against real forecast horizons rather than the horizon-0 case the fast
# sweep scores (window cut at D-h, keyed on D, decayed by h), on the holdout,
# scoring the mean over h in {0,1,2,3,5,7,10,14}:
#
#     hh=1.0   8.7530      hh=2.5   8.6888      hh=5.0   8.6952
#     hh=1.75  8.7082      hh=3.5   8.6832      hh=7.0   8.7241
#
# 3.5 is an interior optimum of that bracket. Worth stating plainly: at every
# horizon out to 14 days the corrected baseline still beats the uncorrected
# curve's 9.0247, so this layer never turns harmful with distance, it just stops
# helping much.
HORIZON_HALFLIFE = 3.5

# Whether to root the ladder in a single whole-regime figure (R0) rather than in
# 0.0. See correction() for why this is off.
USE_ROOT = False

DEFAULT_PARAMS = {
    "window_days":   WINDOW_DAYS,
    "shrink_k":      SHRINK_K,
    "halflife_days": HALFLIFE_DAYS,
    "smooth_window": SMOOTH_WINDOW,
    "use_root":      USE_ROOT,
}


def correction_segment(phase):
    """
    Coarser-than-baseline segment key, used only at the ladder's narrowest rung.

    The baseline curve keys on the fine-grained phase (summer_break_7 etc.) so
    June and July get distinct shapes. A trailing window that fine holds ~2
    same-weekday days, well under anything useful, so break sub-phases pool back
    into one bucket here. In-session phases (regular / first_week / dead_week /
    finals / holiday) keep their identity: under the ladder a short phase no
    longer starves, it just backs off to R2.
    """
    if phase in ("winter_break", "spring_break") or phase.startswith("summer_break_"):
        return "break"
    return phase


def _wstats(resid, weights, rows):
    """
    Per-slot weighted mean and effective n over the given row indices.

    NaN marks "not observed at this slot" (the gym was shut, or no reading
    landed), and contributes nothing to either the mean or n, so a slot's n is
    the count of days that actually recorded it rather than the count of days in
    the window.
    """
    n_slots = resid.shape[1]
    if len(rows) == 0:
        return np.zeros(n_slots), np.zeros(n_slots)

    block = resid[rows]
    seen  = np.isfinite(block)
    w     = np.where(seen, weights[rows][:, None], 0.0)
    n     = w.sum(axis=0)
    total = (np.where(seen, block, 0.0) * w).sum(axis=0)
    return np.divide(total, n, out=np.zeros(n_slots), where=n > 0), n


def _shrink(child_m, child_n, parent_m, k):
    """m̂ = (n·m + k·parent) / (n + k), per slot. n=0 collapses to the parent."""
    w = child_n / (child_n + k)
    return w * child_m + (1.0 - w) * parent_m


def _smooth(values, evidenced, window):
    """
    Centered rolling mean across slots, over the evidenced slots only.

    Two details matter. Closed hours must not bleed in: a day's 7:00 correction
    should never be averaged with the 6:45 slot the building was shut for, so the
    smoothing runs over the compacted evidenced series rather than the raw
    96-vector. And the first and last evidenced slots are left untouched, because
    a centered window cannot be symmetric there ,  the same reasoning as
    curve_model._smooth_slots, which measured a one-sided mean at the opening
    slot running +36pp.
    """
    if window <= 1:
        return values

    idx = np.flatnonzero(evidenced)
    half = window // 2
    if idx.size < 2 * half + 1:
        return values

    out = values.copy()
    compact = values[idx]
    smoothed = compact.copy()
    for j in range(half, len(compact) - half):
        smoothed[j] = compact[j - half:j + half + 1].mean()
    out[idx] = smoothed
    return out


class Trailing:
    """
    A residual history, sliced to a trailing window per target day.

    resid is a (n_days, SLOTS_PER_DAY) matrix of `actual - curve_prediction` in
    percentage points, NaN where a day/slot was not observed. `dates`, `segment`,
    `regime` and `dow` describe its rows.

    One instance holds all available history; `correction()` picks the window and
    fits the ladder for a single target. That is what lets carry_data walk
    thousands of historical days with honest rolling origins while
    predictions_builder makes a handful of lookups off one window, with no second
    implementation.
    """

    def __init__(self, dates, resid, segment, regime, dow, params=None):
        p = {**DEFAULT_PARAMS, **(params or {})}
        self.p = p
        self.dates   = np.asarray(dates)
        self.resid   = np.asarray(resid, dtype=float)
        self.segment = np.asarray(segment)
        self.regime  = np.asarray(regime, dtype=bool)
        self.dow     = np.asarray(dow, dtype=int)
        self.is_weekend = self.dow >= 5

        if not (len(self.dates) == len(self.resid) == len(self.segment)
                == len(self.regime) == len(self.dow)):
            raise ValueError("nowcast.Trailing: row counts disagree across inputs")

        # Day index, so a target date can find its window without a linear scan
        # per call (carry_data makes one call per historical day).
        self._order = np.argsort(self.dates)
        self._sorted_dates = self.dates[self._order]

    def _window_rows(self, target_date):
        """Rows strictly before target_date and within window_days of it.

        Strictly-before matters and is not incidental. predictions_builder runs
        just after midnight PT, so its window is in practice everything up to
        (not including) the day it is forecasting; carry_data replays days the
        same way. Letting the target day into its own window would leak the
        answer into the fit.
        """
        lo = target_date - timedelta(days=int(self.p["window_days"]))
        left  = np.searchsorted(self._sorted_dates, lo, side='left')
        right = np.searchsorted(self._sorted_dates, target_date, side='left')
        return self._order[left:right]

    def _weights(self, rows, target_date):
        """0.5 ** (age_days / halflife) for each row, relative to the target."""
        ages = np.array([(target_date - d).days for d in self.dates[rows]], dtype=float)
        return 0.5 ** (ages / self.p["halflife_days"])

    def correction(self, target_date, segment, regime, dow, as_of=None):
        """
        The (SLOTS_PER_DAY,) correction in pp for a day with this key.

        `as_of` is the day the window is cut at, defaulting to target_date. They
        differ whenever a forecast is made ahead of the day it is for: production
        rebuilds this layer nightly and then serves it for the next 90 days, so a
        target 6 days out is corrected by a window that ended 6 days before it.
        Passing as_of lets a backtest reproduce that instead of only ever scoring
        the horizon-0 case, which is the one this layer flatters itself on.

        Zero everywhere the window carries no evidence, which is the honest
        answer: nothing recent says the gym is off its curve at that slot.
        Callers apply their own horizon decay on top (see decay()).
        """
        n_slots = SLOTS_PER_DAY
        cut = target_date if as_of is None else as_of
        rows = self._window_rows(cut)
        if len(rows) == 0:
            return np.zeros(n_slots)

        k = self.p["shrink_k"]
        w = np.zeros(len(self.dates))
        w[rows] = self._weights(rows, cut)

        in_regime = rows[self.regime[rows] == bool(regime)]
        if len(in_regime) == 0:
            return np.zeros(n_slots)

        # ── R0: one number for the whole regime, collapsing the slot axis.
        #    OFF by default. It looks like the natural root for a ladder, and it
        #    measured worse: every open slot is observed by the same days, so R0
        #    adds no sample size at the slot level, it only imports bias from
        #    other hours. In fall 2026 the evening ran ~15pp cool while the
        #    morning ran ~20pp hot, so the pooled figure was about -5pp and
        #    dragged the morning rung down by that much for nothing. The right
        #    prior for "how far off is the curve at this time of day" is no
        #    correction at all. Kept as a switch because the sweep uses it.
        m1, n1 = _wstats(self.resid, w, in_regime)
        if self.p.get("use_root"):
            n0 = n1.sum()
            m0 = float((m1 * n1).sum() / n0) if n0 > 0 else 0.0
            current = np.full(n_slots, (n0 / (n0 + k)) * m0)
        else:
            current = np.zeros(n_slots)

        # ── R1: time-of-day shape of the drift, pooled over every day in the
        #    window. This is the rung that carries the load early in a semester,
        #    when no single weekday has repeated often enough to say anything.
        current = _shrink(m1, n1, current, k)
        parent_n = n1

        # ── R2 then R3, each shrunk toward the running answer above it.
        narrower = [
            in_regime[self.is_weekend[in_regime] == (int(dow) >= 5)],
            in_regime[(self.segment[in_regime] == segment) & (self.dow[in_regime] == int(dow))],
        ]
        for sel in narrower:
            m, n = _wstats(self.resid, w, sel)
            # A rung whose effective n at a slot equals its parent's is looking
            # at the same evidence under a narrower name ,  e.g. at 7:15am no
            # weekend day has a reading at all, because the gym opens at 8am on
            # weekends, so "everyone at 7:15" and "weekdays at 7:15" are the
            # identical five days. Shrinking again there would count the same
            # observations twice and overstate confidence, so pass through.
            same = np.isclose(n, parent_n)
            current = np.where(same, current, _shrink(m, n, current, k))
            parent_n = np.where(same, parent_n, n)

        # A slot with no time-of-day evidence anywhere in the window gets no
        # correction, even though R0 has a perfectly good number for it.
        #
        # R0 is a single figure for the whole regime, so without this it would
        # leak onto slots the window never observed, which means extrapolating a
        # drift measured at other hours onto an hour nothing recent can speak
        # for. curve_model draws the same line one layer down when it refuses to
        # expand an (is_weekend, slot) cell onto dows that were never open at
        # that hour: back off across days, never across time-of-day. In practice
        # this only bites closed hours, where no prediction is generated anyway.
        current = np.where(n1 > 0, current, 0.0)

        return _smooth(current, n1 > 0, self.p["smooth_window"])


def decay(days_out, halflife=HORIZON_HALFLIFE):
    """Weight for a correction applied `days_out` days ahead of the window."""
    return 0.5 ** (np.asarray(days_out, dtype=float) / halflife)


def day_keys(dates, classify_date, is_summer_day):
    """(segment, regime, dow) arrays for a sequence of dates.

    Takes the calendar functions as arguments rather than importing
    academic_calendar, so this module stays a pure numeric layer that tests can
    drive with a synthetic calendar.
    """
    segment = np.array([correction_segment(classify_date(d)) for d in dates])
    regime  = np.array([bool(is_summer_day(d)) for d in dates])
    dow     = np.array([d.weekday() for d in dates], dtype=int)
    return segment, regime, dow
