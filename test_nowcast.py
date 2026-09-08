"""
test_nowcast.py: the trailing-residual ladder.

Every test here drives nowcast.Trailing with a synthetic residual matrix, so
none of them need Supabase, the curve table, or the real calendar. The point is
to pin the arithmetic the fall-2026 rewrite depends on, especially the two
properties the old quorum implementation did not have: an empty cell inherits
its parent instead of collapsing to zero, and evidence is never counted twice.
"""
from datetime import date, timedelta

import numpy as np
import pytest

import nowcast as nc

SLOT = 40  # 10:00, an arbitrary interior slot
NO_DECAY = {"halflife_days": 1e9}  # weights all ~1.0, so results are hand-computable


def make(days, k=6.0, extra=None):
    """days: list of (date, segment, regime, dow, {slot: residual})."""
    resid = np.full((len(days), nc.SLOTS_PER_DAY), np.nan)
    for i, (_, _, _, _, obs) in enumerate(days):
        for s, v in obs.items():
            resid[i, s] = v
    params = {"shrink_k": k, **NO_DECAY, **(extra or {})}
    return nc.Trailing(
        dates=[d[0] for d in days],
        resid=resid,
        segment=[d[1] for d in days],
        regime=[d[2] for d in days],
        dow=[d[3] for d in days],
        params=params,
    )


def weekdays_before(target, n):
    """The n most recent weekdays strictly before `target`."""
    out, d = [], target - timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


# ── The core property: thin and empty cells back off, they do not vanish ────

def test_empty_narrow_cell_inherits_its_parent():
    """
    The Mon Sep 14 9:00pm case. The only Monday in the window sits in a
    different segment, so the narrowest rung is completely empty. Under the old
    quorum that produced exactly 0.00pp. It must now inherit the pooled level.
    """
    target = date(2026, 9, 14)  # a Monday
    days = [
        (date(2026, 8, 31), "first_week", False, 0, {SLOT: +8.0}),   # Mon, wrong segment
        (date(2026, 9, 1),  "first_week", False, 1, {SLOT: -36.0}),
        (date(2026, 9, 2),  "regular",    False, 2, {SLOT: -17.5}),
        (date(2026, 9, 3),  "regular",    False, 3, {SLOT: -5.2}),
        (date(2026, 9, 4),  "regular",    False, 4, {SLOT: -14.8}),
    ]
    t = make(days)
    got = t.correction(target, "regular", False, 0)[SLOT]

    # No source row is both segment='regular' and dow=0, so R3 contributes
    # nothing and the answer is R2's.
    assert got < -3.0, f"empty narrow cell collapsed to {got:.2f}, expected the parent's level"
    assert got > -20.0, "inherited value should sit inside the evidence, not beyond it"


def test_no_evidence_at_all_gives_zero():
    """An empty window is the one case where 0.00 is the honest answer."""
    t = make([(date(2026, 9, 1), "regular", False, 1, {SLOT: +20.0})])
    # Target is 60 days later, so the single source day falls outside the window.
    assert t.correction(date(2026, 11, 1), "regular", False, 0)[SLOT] == 0.0


def test_slot_with_no_readings_stays_zero():
    """A slot nobody recorded gets no correction, even when other slots do."""
    t = make([(date(2026, 9, i), "regular", False, i - 1, {SLOT: +20.0}) for i in (1, 2, 3)])
    out = t.correction(date(2026, 9, 4), "regular", False, 4)
    assert out[SLOT] != 0.0
    assert out[SLOT + 20] == 0.0


# ── Shrinkage arithmetic ───────────────────────────────────────────────────

def test_shrinkage_weights_match_the_formula():
    """
    Five weekdays, all residual +10 at one slot, k=6, target dow matches exactly
    one of them. The ladder is rooted at 0.0 (use_root is off, see nowcast), so
    by hand:

        R1 = (5/11)*10                     = 4.5455
        R2 = pass-through (same five rows) = 4.5455
        R3 = (1/7)*10 + (6/7)*4.5455       = 5.3247
    """
    target = date(2026, 9, 14)
    src = weekdays_before(target, 5)
    days = [(d, "regular", False, d.weekday(), {SLOT: +10.0}) for d in src]
    t = make(days, k=6.0)
    got = t.correction(target, "regular", False, target.weekday())[SLOT]
    assert got == pytest.approx(5.3247, abs=0.01)


def test_larger_k_shrinks_harder():
    target = date(2026, 9, 14)
    days = [(d, "regular", False, d.weekday(), {SLOT: +10.0})
            for d in weekdays_before(target, 5)]
    vals = [make(days, k=k).correction(target, "regular", False, target.weekday())[SLOT]
            for k in (1.0, 3.0, 6.0, 20.0)]
    assert vals == sorted(vals, reverse=True), f"correction should fall as k rises: {vals}"
    assert vals[0] < 10.0, "even the weakest shrinkage stays inside the evidence"


def test_result_never_escapes_the_evidence():
    """Shrinkage is a weighted average, so it cannot overshoot the raw signal."""
    target = date(2026, 9, 14)
    days = [(d, "regular", False, d.weekday(), {SLOT: r})
            for d, r in zip(weekdays_before(target, 5), [+30, +10, +20, +25, +15])]
    got = make(days).correction(target, "regular", False, target.weekday())[SLOT]
    assert 0.0 < got < 30.0


def test_identical_membership_is_not_shrunk_twice():
    """
    The 7:15am case. No weekend day has a reading at that slot because the gym
    opens at 8am on weekends, so "everyone at 7:15" and "weekdays at 7:15" are
    the identical rows. Shrinking again there would count the same evidence
    twice and overstate confidence.

    Five weekdays at +20, k=6, target dow matches one of them:

        R1 = (5/11)*20                     = 9.0909
        R2 = pass-through (same five rows) = 9.0909
        R3 = (1/7)*20 + (6/7)*9.0909       = 10.6494

    Without the pass-through R2 would re-shrink to 14.049 and R3 would land at
    14.899, so the two behaviours are cleanly distinguishable.
    """
    target = date(2026, 9, 9)  # Wednesday
    days = [(d, "regular", False, d.weekday(), {SLOT: +20.0})
            for d in weekdays_before(target, 5)]
    got = make(days, k=6.0).correction(target, "regular", False, 2)[SLOT]

    assert got == pytest.approx(10.6494, abs=0.01), (
        f"got {got:.3f}; 14.90 means the R2 rung re-shrank evidence it had already used"
    )


# ── Window semantics ────────────────────────────────────────────────────────

def test_target_day_is_never_in_its_own_window():
    """Letting the target into the fit would leak the answer into the forecast."""
    target = date(2026, 9, 9)
    t = make([(target, "regular", False, 2, {SLOT: +40.0})])
    assert t.correction(target, "regular", False, 2)[SLOT] == 0.0


def test_window_length_is_respected():
    target = date(2026, 9, 30)
    old = target - timedelta(days=40)
    t = make([(old, "regular", False, old.weekday(), {SLOT: +40.0})])
    assert t.correction(target, "regular", False, 2)[SLOT] == 0.0

    recent = target - timedelta(days=5)
    t2 = make([(recent, "regular", False, recent.weekday(), {SLOT: +40.0})])
    assert t2.correction(target, "regular", False, 2)[SLOT] > 0.0


def test_recent_days_outweigh_old_ones():
    """With the real halflife, a day 1 day back counts more than one 27 days back."""
    target = date(2026, 9, 30)
    near = [(target - timedelta(days=1), "regular", False, 0, {SLOT: +30.0}),
            (target - timedelta(days=27), "regular", False, 1, {SLOT: -30.0})]
    far = [(target - timedelta(days=27), "regular", False, 0, {SLOT: +30.0}),
           (target - timedelta(days=1), "regular", False, 1, {SLOT: -30.0})]
    a = nc.Trailing([d[0] for d in near],
                    np.array([[np.nan] * SLOT + [d[4][SLOT]] + [np.nan] * (95 - SLOT) for d in near]),
                    [d[1] for d in near], [d[2] for d in near], [d[3] for d in near])
    b = nc.Trailing([d[0] for d in far],
                    np.array([[np.nan] * SLOT + [d[4][SLOT]] + [np.nan] * (95 - SLOT) for d in far]),
                    [d[1] for d in far], [d[2] for d in far], [d[3] for d in far])
    assert a.correction(target, "regular", False, 3)[SLOT] > 0
    assert b.correction(target, "regular", False, 3)[SLOT] < 0


# ── Key isolation ───────────────────────────────────────────────────────────

def test_summer_days_never_correct_an_academic_target():
    """
    regime is in the key at every rung precisely so summer's 8pm closing crash
    cannot be stamped onto a target that stays open until 11pm.
    """
    target = date(2026, 8, 27)
    days = [(target - timedelta(days=i), "break", True, (target - timedelta(days=i)).weekday(),
             {SLOT: -40.0}) for i in range(1, 15)]
    t = make(days)
    assert t.correction(target, "first_week", False, 3)[SLOT] == 0.0
    # ...but a summer target reads them fine.
    assert t.correction(target, "break", True, 3)[SLOT] < 0.0


def test_short_phases_are_corrected_now():
    """
    first_week / dead_week / finals / holiday could never satisfy the old quorum
    (each phase is shorter than 21 days, so 3 of D-7/14/21/28 was arithmetically
    impossible). Under the ladder they back off to R1/R2 and get a real number.
    """
    target = date(2026, 8, 31)  # a first_week Monday
    days = [(target - timedelta(days=i), "regular", False,
             (target - timedelta(days=i)).weekday(), {SLOT: +18.0})
            for i in range(1, 12)]
    got = make(days).correction(target, "first_week", False, 0)[SLOT]
    assert got > 5.0, f"short phase still starved: got {got:.2f}"


# ── Smoothing ───────────────────────────────────────────────────────────────

def test_smoothing_averages_the_interior_and_spares_the_ends():
    vals = np.zeros(nc.SLOTS_PER_DAY)
    ev = np.zeros(nc.SLOTS_PER_DAY, dtype=bool)
    vals[28:33] = [10.0, 0.0, 0.0, 0.0, 10.0]
    ev[28:33] = True
    out = nc._smooth(vals, ev, 3)
    assert out[28] == 10.0 and out[32] == 10.0, "endpoints must be untouched"
    assert out[29] == pytest.approx(10.0 / 3)
    assert out[30] == pytest.approx(0.0)


def test_smoothing_does_not_bleed_across_closed_hours():
    """A closed-hours gap must not drag its zeros into the open slots beside it."""
    vals = np.zeros(nc.SLOTS_PER_DAY)
    ev = np.zeros(nc.SLOTS_PER_DAY, dtype=bool)
    for s in (28, 29, 30, 60, 61, 62):
        ev[s] = True
        vals[s] = 12.0
    out = nc._smooth(vals, ev, 3)
    assert out[29] == pytest.approx(12.0)
    assert out[61] == pytest.approx(12.0)


# ── The regression this rewrite exists for ──────────────────────────────────

def test_fall_2026_morning_ramp_is_no_longer_silent():
    """
    Real residuals at 7:15am, Aug 31 to Sep 4 2026, against the baseline actually
    served. The deployed quorum produced 0.00pp for all of them because no
    (segment, dow) cell had 3 of its 4 candidate days. The ladder must recover a
    clearly positive correction.
    """
    target = date(2026, 9, 9)  # Wednesday, phase 'regular'
    observed = {
        date(2026, 8, 31): ("first_week", 0, +27.0),
        date(2026, 9, 1):  ("first_week", 1, +17.0),
        date(2026, 9, 2):  ("regular",    2, +19.7),
        date(2026, 9, 3):  ("regular",    3, +23.3),
        date(2026, 9, 4):  ("regular",    4, +11.1),
    }
    slot = 29  # 07:15
    days = [(d, seg, False, dow, {slot: r}) for d, (seg, dow, r) in observed.items()]
    got = make(days, k=nc.SHRINK_K).correction(target, "regular", False, 2)[slot]

    assert got > 8.0, f"morning ramp still under-corrected: {got:.2f}pp"
    assert got < 19.6, "must stay shrunk below the raw pooled mean"


def test_decay_fades_with_horizon():
    """Stated as the property, not as constants, so retuning the halflife does
    not require editing an assertion that was never about the number."""
    hl = nc.HORIZON_HALFLIFE
    assert nc.decay(0) == pytest.approx(1.0)
    assert nc.decay(hl) == pytest.approx(0.5)
    assert nc.decay(2 * hl) == pytest.approx(0.25)
    assert nc.decay(1, halflife=4.0) == pytest.approx(0.5 ** 0.25)


def test_correction_segment_pools_breaks_only():
    assert nc.correction_segment("summer_break_7") == "break"
    assert nc.correction_segment("winter_break") == "break"
    assert nc.correction_segment("spring_break") == "break"
    assert nc.correction_segment("first_week") == "first_week"
    assert nc.correction_segment("regular") == "regular"
    assert nc.correction_segment("finals") == "finals"
