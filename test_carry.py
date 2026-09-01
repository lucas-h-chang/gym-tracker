"""
test_carry.py — guards on the within-day level correction (carry_model.py).

These exist because the bug this model replaces was NOT a bad forecast: it was
a *schedule* leaking into the drawn line. `today_builder.py`'s 2-hour weight
ramp turned a 21pp model disagreement into ~2.4pp per 15 min of climb that no
model predicted. The prototype of the replacement then reintroduced the same
species of artifact from a different cause — discrete horizon buckets put a
+5.1pp step at 6:30 PM on 2026-08-31 data when the horizon crossed 8 hours.

So the tests below are mostly about SHAPE, not accuracy: they assert that
nothing in the machinery can manufacture slope. Accuracy is the job of
replay_day.py, which scores against real days.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

import carry_model as km

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# A synthetic table with known coefficients — no data or network needed.
# ---------------------------------------------------------------------------

def make_table(coefs_by_h=None):
    """Anchors at several horizons with hand-set coefficients."""
    coefs_by_h = coefs_by_h or {
        0.25: [0.0, 0.10, 0.30, 0.55],
        1.0:  [0.0, 0.35, 0.40, 0.20],
        4.0:  [0.0, 0.75, 0.20, 0.05],
        9.0:  [0.0, 0.95, 0.05, 0.00],
    }
    anchors = [{"h": h, "n": 5000, "coef": c, "r2": 0.5}
               for h, c in sorted(coefs_by_h.items())]
    return {"version": 1, "built_at": "test", "params": {},
            "by_cut": {str(h): anchors for h in range(7, 23)},
            "pooled": anchors}


# ---------------------------------------------------------------------------
# 1. Zero gap must reproduce the base curve EXACTLY.
# ---------------------------------------------------------------------------

def test_zero_gap_is_identity():
    table = make_table()
    base = {s: 40 + 30 * np.sin(s / 10) for s in range(40, 92)}
    out = km.apply_to_day(table, cut_slot=41, last_slot=41, gaps=(0.0, 0.0, 0.0),
                          base_by_slot=base, lo=28, hi=92)
    assert out, "expected corrected slots"
    for s, v in out.items():
        assert v == pytest.approx(base[s], abs=1e-9), f"slot {s} moved with a zero gap"


# ---------------------------------------------------------------------------
# 2. Symmetry — a busier day must be handled exactly like a quieter one.
# ---------------------------------------------------------------------------

def test_symmetric_in_sign():
    table = make_table()
    base = {s: 70.0 for s in range(40, 92)}
    quiet = km.apply_to_day(table, 41, 41, (-6.0, -12.0, -18.0), base, 28, 92)
    busy  = km.apply_to_day(table, 41, 41, (+6.0, +12.0, +18.0), base, 28, 92)
    for s in quiet:
        down, up = base[s] - quiet[s], busy[s] - base[s]
        assert down == pytest.approx(up, abs=1e-9), f"slot {s} asymmetric"


# ---------------------------------------------------------------------------
# 3. The correction must be CONTINUOUS in horizon. This is the bucket-step guard.
# ---------------------------------------------------------------------------

def test_correction_is_continuous_at_every_anchor():
    """Rigorous continuity, with no magic threshold.

    For a continuous function, the change measured across a shrinking interval
    shrinks with it. For a step function it does not — the step is still there
    however far you zoom in. That is exactly the difference between smooth
    interpolation and the discrete horizon buckets that put a +5.1pp jump at
    6:30 PM on 2026-08-31, so it is the property worth asserting rather than
    "changes by less than N pp", which also forbids legitimately steep slopes.
    """
    table = make_table()
    gaps = (-10.0, -18.0, -22.0)
    for a in table["pooled"]:
        h = a["h"]
        wide = abs(km.correction_at(table, 10, h + 0.20, *gaps)
                   - km.correction_at(table, 10, h - 0.20, *gaps))
        narrow = abs(km.correction_at(table, 10, h + 0.0025, *gaps)
                     - km.correction_at(table, 10, h - 0.0025, *gaps))
        if wide < 1e-9:
            continue
        assert narrow <= wide / 20, (
            f"anchor {h}h: shrinking the probe 80x only took the change from "
            f"{wide:.4f} to {narrow:.4f} — that is a step, not a slope")


def test_bucketed_implementation_would_fail_continuity():
    """The continuity test above must actually be able to fail.

    Builds a deliberately bucketed lookup — what the prototype did — and
    confirms the same probe catches it. Without this, a continuity test that
    silently always passes would give false assurance.
    """
    table = make_table()
    gaps = (-10.0, -18.0, -22.0)

    def bucketed(h):
        anchors = table["pooled"]
        for a in anchors:
            if h <= a["h"]:
                i, b, c, d = a["coef"]
                return i + b * gaps[0] + c * gaps[1] + d * gaps[2]
        i, b, c, d = anchors[-1]["coef"]
        return i + b * gaps[0] + c * gaps[1] + d * gaps[2]

    caught = False
    for a in table["pooled"][1:]:
        h = a["h"]
        wide   = abs(bucketed(h + 0.20)   - bucketed(h - 0.20))
        narrow = abs(bucketed(h + 0.0025) - bucketed(h - 0.0025))
        if wide > 1e-9 and narrow > wide / 20:
            caught = True
    assert caught, "the continuity probe cannot detect a bucketed lookup"


def test_correction_slope_is_bounded_with_real_gaps():
    """Sanity bound on the rate of change, separate from continuity.

    Continuity alone permits an arbitrarily steep ramp. This pins the slope to
    something a real day could produce: even with a very large gap, the
    correction should not move more than 3pp per 15-minute slot beyond the
    first hour, which is where the deployed model's 2-hour ramp sat at ~2.4pp
    per slot *sustained*.
    """
    table = make_table()
    gaps = (-10.0, -18.0, -22.0)
    prev = None
    for step in range(4, 4 * 14):          # from 1h out
        c = km.correction_at(table, 10, step / 4.0, *gaps)
        if prev is not None:
            assert abs(c - prev) <= 3.0, (
                f"correction moved {c - prev:+.2f}pp in one slot at "
                f"horizon {step / 4.0}h")
        prev = c


# ---------------------------------------------------------------------------
# 4. The published curve must not be jumpier than the base curve it corrects.
# ---------------------------------------------------------------------------

def test_published_curve_no_jumpier_than_base():
    table = make_table()
    rng = np.random.default_rng(0)
    for trial in range(200):
        lo, hi = 28, 92
        base = {s: float(np.clip(50 + 35 * np.sin((s - 28) / 12) + rng.normal(0, 1.5), 5, 99))
                for s in range(lo, hi)}
        cut = int(rng.integers(lo + 4, hi - 8))
        gaps = tuple(rng.normal(0, 12, 3))
        out = km.apply_to_day(table, cut, cut, gaps, base, lo, hi)
        slots = sorted(out)
        if len(slots) < 3:
            continue
        new_jump  = max(abs(out[b] - out[a]) for a, b in zip(slots, slots[1:]))
        base_jump = max(abs(base[b] - base[a]) for a, b in zip(slots, slots[1:]))
        assert new_jump <= base_jump + 3.0, (
            f"trial {trial}: new curve jumps {new_jump:.2f}pp vs base {base_jump:.2f}pp")


# ---------------------------------------------------------------------------
# 5. Gap computation
# ---------------------------------------------------------------------------

def test_compute_gaps_separates_the_three_signals():
    """The whole reason there are three terms: a day that is mildly low overall
    but sharply low right now must report those as different numbers. These are
    the real 2026-08-31 readings at the 10:15 cut."""
    slots  = [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    actual = [73.3, 89.0, 98.0, 84.3, 79.0, 74.3, 51.7, 51.3, 64.0, 72.7, 70.7, 66.3, 67.3]
    base   = [46.3, 71.9, 77.0, 77.3, 76.9, 77.7, 77.4, 78.0, 79.6, 82.1, 84.1, 86.1, 88.2]
    g = km.compute_gaps(slots, actual, base)
    assert g is not None
    gap_day, gap_recent, gap_last, last_slot, n_obs = g
    assert last_slot == 41
    assert gap_last == pytest.approx(67.3 - 88.2, abs=0.05)
    assert gap_recent == pytest.approx(np.mean([64.0 - 79.6, 72.7 - 82.1,
                                                70.7 - 84.1, 66.3 - 86.1,
                                                67.3 - 88.2][-4:]), abs=0.05)
    # The point of the test: these must NOT collapse to one number.
    assert gap_last < gap_recent < gap_day, "three terms carry the same signal"


def test_too_few_readings_returns_none():
    """One hour of readings is deliberately not enough — see MIN_OBSERVED."""
    n = km.MIN_OBSERVED - 1
    slots = list(range(32, 32 + n))
    vals  = [70.0] * n
    assert km.compute_gaps(slots, vals, vals) is None
    slots.append(32 + n)
    vals.append(70.0)
    assert km.compute_gaps(slots, vals, vals) is not None


def test_low_base_slots_excluded_from_gaps():
    """A 7:00 reading against a ~2% base is an opening-ramp artifact, not
    evidence the day is running 30pp hot. It must not reach gap_day."""
    slots  = [28, 29, 30, 31, 32, 33, 34]
    actual = [32.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0]
    base   = [1.8, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0]   # slot 28 below MIN_BASE
    g = km.compute_gaps(slots, actual, base)
    assert g is not None
    assert g[0] == pytest.approx(0.0, abs=1e-9), "opening-ramp slot leaked into gap_day"


# ---------------------------------------------------------------------------
# 6. Fitting round-trip: plant a known structure, recover it.
# ---------------------------------------------------------------------------

def test_fit_recovers_planted_carry():
    """Residual = a persistent per-day offset. A correctly specified fit must
    recover a total carry near 1.0, since nothing decays in this world."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2025-01-06", periods=260, freq="D").date
    n = len(dates)
    actual = np.full((n, 96), np.nan)
    base   = np.full((n, 96), np.nan)
    for i in range(n):
        offset = rng.normal(0, 9)
        for s in range(28, 92):
            base[i, s] = 55.0 + 20.0 * np.sin((s - 28) / 14)
            actual[i, s] = base[i, s] + offset
    samples = km.make_samples(dates, actual, base, lambda d: (28, 92))
    assert not samples.empty
    table = km.build_table(samples)
    for a in table["pooled"]:
        carry = a["coef"][1] + a["coef"][2] + a["coef"][3]
        assert carry == pytest.approx(1.0, abs=0.05), (
            f"planted full carry, recovered {carry:.3f} at horizon {a['h']}h")


# ---------------------------------------------------------------------------
# 7. The shipped artifact, when present.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(os.path.join(HERE, "models/carry.json")),
                    reason="models/carry.json not built yet")
def test_shipped_table_covers_open_hours_and_is_smooth():
    table = json.load(open(os.path.join(HERE, "models/carry.json")))
    assert table.get("by_cut"), "no per-cut anchors"
    for cut_hour in range(8, 22):
        anchors = km._anchors_for_cut(table, cut_hour)
        assert anchors, f"no anchors reachable for cut hour {cut_hour}"
    gaps = (-10.0, -18.0, -22.0)
    for cut_hour in range(8, 22):
        anchors = km._anchors_for_cut(table, cut_hour)
        for a in anchors:
            h = a["h"]
            wide = abs(km.correction_at(table, cut_hour, h + 0.20, *gaps)
                       - km.correction_at(table, cut_hour, h - 0.20, *gaps))
            narrow = abs(km.correction_at(table, cut_hour, h + 0.0025, *gaps)
                         - km.correction_at(table, cut_hour, h - 0.0025, *gaps))
            if wide < 1e-9:
                continue
            assert narrow <= wide / 20, (
                f"shipped table steps at cut {cut_hour}, anchor {h}h")
        prev = None
        for step in range(4, 60):
            c = km.correction_at(table, cut_hour, step / 4.0, *gaps)
            if prev is not None:
                assert abs(c - prev) <= 3.0, (
                    f"shipped table moves {c - prev:+.2f}pp in one slot "
                    f"at cut {cut_hour}, horizon {step / 4.0}h")
            prev = c


# ---------------------------------------------------------------------------
# 8. Evidence-based shrinkage
# ---------------------------------------------------------------------------

def test_shrink_factor_grows_with_evidence():
    prev = -1.0
    for n in range(1, 60):
        f = km.shrink_factor(n, horizon=4.0)
        assert 0.0 < f < 1.0
        assert f > prev, "shrinkage must relax monotonically as readings accumulate"
        prev = f
    assert km.shrink_factor(200, horizon=4.0) > 0.95


def test_shrink_factor_tightens_with_reach():
    """Shrinkage scales with how far ahead you are extrapolating, not just with
    evidence. A constant factor damaged short-range accuracy (+15 min went
    4.511 -> 4.791 over 357 days, falling behind the deployed model's 4.704)."""
    prev = 2.0
    for h in [0.25, 0.5, 1, 2, 4, 8, 12]:
        f = km.shrink_factor(20, horizon=h)
        assert f < prev, "must shrink harder the further out it reaches"
        prev = f


def test_near_term_correction_is_essentially_untouched():
    """15 minutes out the correction is almost all gap_last, which is reliable
    (fit R^2 0.88). It must survive shrinkage nearly intact."""
    assert km.shrink_factor(12, horizon=0.25) > 0.95
    assert km.shrink_factor(30, horizon=0.25) > 0.98


def test_less_evidence_moves_the_curve_less():
    """The measured failure this guards: with only a couple of hours behind it,
    the unshrunk correction was worse than doing nothing at cut hour 8
    (9.829 vs 9.282 BASE over 357 days). Shrinkage is what fixed it."""
    table = make_table()
    base = {s: 70.0 for s in range(40, 92)}
    gaps = (-8.0, -14.0, -20.0)
    thin = km.apply_to_day(table, 41, 41, gaps, base, 28, 92, n_obs=6)
    thick = km.apply_to_day(table, 41, 41, gaps, base, 28, 92, n_obs=60)
    assert len(thin) > 5
    for s in thin:
        moved_thin = abs(base[s] - thin[s])
        moved_thick = abs(base[s] - thick[s])
        assert moved_thin < moved_thick, f"slot {s} moved as much on thin evidence"


def test_shrinkage_preserves_zero_gap_identity():
    table = make_table()
    base = {s: 55.0 + s % 7 for s in range(40, 92)}
    for n in (6, 12, 40):
        out = km.apply_to_day(table, 41, 41, (0.0, 0.0, 0.0), base, 28, 92, n_obs=n)
        for s, v in out.items():
            assert v == pytest.approx(base[s], abs=1e-9)


def test_shrinkage_is_symmetric():
    table = make_table()
    base = {s: 70.0 for s in range(40, 92)}
    quiet = km.apply_to_day(table, 41, 41, (-6.0, -12.0, -18.0), base, 28, 92, n_obs=9)
    busy = km.apply_to_day(table, 41, 41, (6.0, 12.0, 18.0), base, 28, 92, n_obs=9)
    for s in quiet:
        assert (base[s] - quiet[s]) == pytest.approx(busy[s] - base[s], abs=1e-9)
