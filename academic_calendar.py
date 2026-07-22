"""
academic_calendar.py — Berkeley academic calendar as a single source of truth.

Extracted from train.py::engineer_features()'s calendar flags (2020-21 through
2027-28, sourced from official UCB PDFs) so build_curves.py (training) and
curve_model.py (prediction) can never drift apart — both call classify_date()
from here instead of each keeping their own copy of the date ranges.

train.py keeps its own copy of these ranges (frozen, until the RF is retired
per SPEC_CURVE_MODEL.md Step 5) — this module does not replace engineer_features.
"""
from datetime import date

# ── Semester boundaries (for days_to_sem_start / days_to_sem_end) ────────────
SEM_STARTS = [
    date(2020, 8, 26), date(2021, 1, 19), date(2021, 8, 25), date(2022, 1, 18),
    date(2022, 8, 24), date(2023, 1, 17), date(2023, 8, 23), date(2024, 1, 16),
    date(2024, 8, 28), date(2025, 1, 21), date(2025, 8, 27), date(2026, 1, 20),
    date(2026, 8, 26), date(2027, 1, 19), date(2027, 8, 25), date(2028, 1, 18),
]
SEM_ENDS = [
    date(2021, 5, 14), date(2022, 5, 13), date(2023, 5, 12), date(2024, 5, 10),
    date(2025, 5, 16), date(2026, 5, 15), date(2027, 5, 14), date(2028, 5, 12),
    date(2020, 12, 18), date(2021, 12, 17), date(2022, 12, 16), date(2023, 12, 15),
    date(2024, 12, 20), date(2025, 12, 19), date(2026, 12, 18), date(2027, 12, 17),
]
BOUNDARY_CLIP = 30  # days; ramps saturate beyond ~4 weeks from a boundary

# ── Academic holidays (semester-time only; a holiday inside a break range is
#    redundant with "break" and is not double-counted — see classify_date) ───
HOLIDAYS = {
    # ── Fall 2020 ──
    date(2020, 9,  7), date(2020, 11, 11), date(2020, 11, 25),
    date(2020, 11, 26), date(2020, 11, 27),
    date(2020, 12, 24), date(2020, 12, 25), date(2020, 12, 31),
    # ── Spring 2021 ──
    date(2021, 1,  1), date(2021, 1, 18), date(2021, 2, 15),
    date(2021, 3, 26), date(2021, 5, 31),
    # ── Fall 2021 ──
    date(2021, 9,  6), date(2021, 11, 11), date(2021, 11, 24),
    date(2021, 11, 25), date(2021, 11, 26),
    date(2021, 12, 23), date(2021, 12, 24), date(2021, 12, 30), date(2021, 12, 31),
    # ── Spring 2022 ──
    date(2022, 1, 17), date(2022, 2, 21), date(2022, 3, 25), date(2022, 5, 30),
    # ── Fall 2022 ──
    date(2022, 9,  5), date(2022, 11, 11), date(2022, 11, 23),
    date(2022, 11, 24), date(2022, 11, 25),
    date(2022, 12, 23), date(2022, 12, 26), date(2022, 12, 30),
    date(2023, 1,  2),
    # ── Spring 2023 ──
    date(2023, 1, 16), date(2023, 2, 20), date(2023, 3, 31),
    # ── Fall 2023 ──
    date(2023, 9,  4), date(2023, 11, 10), date(2023, 11, 22),
    date(2023, 11, 23), date(2023, 11, 24),
    date(2023, 12, 25), date(2023, 12, 26),
    date(2024, 1,  1), date(2024, 1,  2),
    # ── Spring 2024 ──
    date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    # ── Fall 2024 ──
    date(2024, 9,  2), date(2024, 11, 11), date(2024, 11, 27),
    date(2024, 11, 28), date(2024, 11, 29),
    date(2024, 12, 24), date(2024, 12, 25), date(2024, 12, 31),
    date(2025, 1,  1),
    # ── Spring 2025 ──
    date(2025, 1, 20), date(2025, 2, 17), date(2025, 3, 28),
    # ── Fall 2025 ──
    date(2025, 9,  1), date(2025, 11, 11), date(2025, 11, 26),
    date(2025, 11, 27), date(2025, 11, 28),
    date(2025, 12, 24), date(2025, 12, 25), date(2025, 12, 31),
    date(2026, 1,  1),
    # ── Spring 2026 ──
    date(2026, 1, 19), date(2026, 2, 16), date(2026, 3, 27),
    # ── Fall 2026 ──
    date(2026, 9,  7), date(2026, 11, 11), date(2026, 11, 25),
    date(2026, 11, 26), date(2026, 11, 27),
    date(2026, 12, 24), date(2026, 12, 25), date(2026, 12, 31),
    date(2027, 1,  1),
    # ── Spring 2027 ──
    date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    # ── Fall 2027 ──
    date(2027, 9,  6), date(2027, 11, 11), date(2027, 11, 24),
    date(2027, 11, 25), date(2027, 11, 26),
    date(2027, 12, 24), date(2027, 12, 27), date(2027, 12, 31),
    date(2028, 1,  3),
    # ── Spring 2028 ──
    date(2028, 1, 17), date(2028, 2, 21), date(2028, 3, 31),
}

FINALS_RANGES = [
    # Spring finals
    (date(2021, 5, 10), date(2021, 5, 14)),
    (date(2022, 5,  9), date(2022, 5, 13)),
    (date(2023, 5,  8), date(2023, 5, 12)),
    (date(2024, 5,  6), date(2024, 5, 10)),
    (date(2025, 5, 12), date(2025, 5, 16)),
    (date(2026, 5, 11), date(2026, 5, 15)),
    (date(2027, 5, 10), date(2027, 5, 14)),
    (date(2028, 5,  8), date(2028, 5, 12)),
    # Fall finals
    (date(2020, 12, 14), date(2020, 12, 18)),
    (date(2021, 12, 13), date(2021, 12, 17)),
    (date(2022, 12, 12), date(2022, 12, 16)),
    (date(2023, 12, 11), date(2023, 12, 15)),
    (date(2024, 12, 16), date(2024, 12, 20)),
    (date(2025, 12, 15), date(2025, 12, 19)),
    (date(2026, 12, 14), date(2026, 12, 18)),
    (date(2027, 12, 13), date(2027, 12, 17)),
]

DEAD_WEEK_RANGES = [
    # Spring dead weeks
    (date(2021, 5,  3), date(2021, 5,  7)),
    (date(2022, 5,  2), date(2022, 5,  6)),
    (date(2023, 5,  1), date(2023, 5,  5)),
    (date(2024, 4, 29), date(2024, 5,  3)),
    (date(2025, 5,  5), date(2025, 5,  9)),
    (date(2026, 5,  4), date(2026, 5,  8)),
    (date(2027, 5,  3), date(2027, 5,  7)),
    (date(2028, 5,  1), date(2028, 5,  5)),
    # Fall dead weeks
    (date(2020, 12,  7), date(2020, 12, 11)),
    (date(2021, 12,  6), date(2021, 12, 10)),
    (date(2022, 12,  5), date(2022, 12,  9)),
    (date(2023, 12,  4), date(2023, 12,  8)),
    (date(2024, 12,  9), date(2024, 12, 13)),
    (date(2025, 12,  8), date(2025, 12, 12)),
    (date(2026, 12,  7), date(2026, 12, 11)),
    (date(2027, 12,  6), date(2027, 12, 10)),
]

FIRST_WEEK_RANGES = [
    (date(2020, 8, 26), date(2020, 9,  1)),  # Fall 2020
    (date(2021, 1, 19), date(2021, 1, 25)),  # Spring 2021
    (date(2021, 8, 25), date(2021, 8, 31)),  # Fall 2021
    (date(2022, 1, 18), date(2022, 1, 24)),  # Spring 2022
    (date(2022, 8, 24), date(2022, 8, 30)),  # Fall 2022
    (date(2023, 1, 17), date(2023, 1, 23)),  # Spring 2023
    (date(2023, 8, 23), date(2023, 8, 29)),  # Fall 2023
    (date(2024, 1, 16), date(2024, 1, 22)),  # Spring 2024
    (date(2024, 8, 28), date(2024, 9,  3)),  # Fall 2024
    (date(2025, 1, 21), date(2025, 1, 27)),  # Spring 2025
    (date(2025, 8, 27), date(2025, 9,  2)),  # Fall 2025
    (date(2026, 1, 20), date(2026, 1, 26)),  # Spring 2026
    (date(2026, 8, 26), date(2026, 9,  1)),  # Fall 2026
    (date(2027, 1, 19), date(2027, 1, 25)),  # Spring 2027
    (date(2027, 8, 25), date(2027, 8, 31)),  # Fall 2027
    (date(2028, 1, 18), date(2028, 1, 24)),  # Spring 2028
]

WINTER_BREAK_RANGES = [
    (date(2020, 12, 18), date(2021, 1, 18)),
    (date(2021, 12, 17), date(2022, 1, 17)),
    (date(2022, 12, 16), date(2023, 1, 16)),
    (date(2023, 12, 15), date(2024, 1, 15)),
    (date(2024, 12, 20), date(2025, 1, 20)),
    (date(2025, 12, 19), date(2026, 1, 19)),
    (date(2026, 12, 18), date(2027, 1, 18)),
    (date(2027, 12, 17), date(2028, 1, 17)),
]

SPRING_BREAK_RANGES = [
    (date(2021, 3, 20), date(2021, 3, 28)),
    (date(2022, 3, 19), date(2022, 3, 27)),
    (date(2023, 3, 25), date(2023, 4,  2)),
    (date(2024, 3, 23), date(2024, 3, 31)),
    (date(2025, 3, 22), date(2025, 3, 30)),
    (date(2026, 3, 21), date(2026, 3, 29)),
    (date(2027, 3, 20), date(2027, 3, 28)),
    (date(2028, 3, 25), date(2028, 4,  2)),
]

SUMMER_BREAK_RANGES = [
    (date(2021, 5, 14), date(2021, 8, 24)),
    (date(2022, 5, 13), date(2022, 8, 23)),
    (date(2023, 5, 12), date(2023, 8, 22)),
    (date(2024, 5, 10), date(2024, 8, 27)),
    (date(2025, 5, 16), date(2025, 8, 26)),
    (date(2026, 5, 15), date(2026, 8, 25)),
    (date(2027, 5, 14), date(2027, 8, 24)),
]

# Kept for callers that only need "is this any kind of break" (e.g. is_holiday
# suppression below). classify_date() below returns the season-specific label,
# not this generic one — pooling all three into one phase was the original
# design (per SPEC_CURVE_MODEL.md) and it's a real bug: winter break Tuesday
# 7pm runs ~6% occupancy, summer break Tuesday 7pm runs ~78% — averaging them
# into one curve produces a prediction that matches neither.
BREAK_RANGES = WINTER_BREAK_RANGES + SPRING_BREAK_RANGES + SUMMER_BREAK_RANGES


def _in_any(d, ranges):
    return any(start <= d <= end for start, end in ranges)


def _as_date(d):
    # Accept date, datetime, or pandas Timestamp without importing pandas here.
    return d.date() if hasattr(d, "date") and not isinstance(d, date) else d


def classify_date(d):
    """
    Priority order: finals > dead_week > first_week > holiday > break > regular,
    where "break" is season-specific (winter_break / spring_break / summer_break)
    rather than one pooled phase — winter break Tuesday 7pm runs ~6% occupancy,
    summer break Tuesday 7pm runs ~78%; a single "break" curve averaged the two
    and predicted neither correctly (see project memory / SPEC_CURVE_MODEL.md
    follow-up notes for the incident this fixed, 2026-07-20).

    Mirrors train.py::engineer_features()'s is_finals/is_dead_week/is_first_week/
    is_break/is_holiday flags collapsed into one label (with is_break further
    split by season here) — see test_curve_sanity.py's cross-check against
    those flags on all historical dates.
    """
    d = _as_date(d)
    if _in_any(d, FINALS_RANGES):
        return "finals"
    if _in_any(d, DEAD_WEEK_RANGES):
        return "dead_week"
    if _in_any(d, FIRST_WEEK_RANGES):
        return "first_week"
    is_break = _in_any(d, BREAK_RANGES)
    if d in HOLIDAYS and not is_break:
        return "holiday"
    if _in_any(d, WINTER_BREAK_RANGES):
        return "winter_break"
    if _in_any(d, SPRING_BREAK_RANGES):
        return "spring_break"
    if _in_any(d, SUMMER_BREAK_RANGES):
        # Summer break itself has a large intra-phase swing that a single
        # pooled curve washes out: raw Tuesday-7pm means by month run
        # May ~55%, June ~77%, July ~78%, August ~65% (shoulder months are
        # genuinely quieter than peak summer, not just noise) — split by
        # month so "which part of summer" isn't lost the way "which season"
        # was before the winter/spring/summer split above.
        return f"summer_break_{d.month}"
    return "regular"


def _signed_nearest_clipped(d, boundaries):
    d_ord = d.toordinal()
    nearest = min((b.toordinal() - d_ord for b in boundaries), key=abs)
    return max(-BOUNDARY_CLIP, min(BOUNDARY_CLIP, nearest))


def days_to_sem_start(d):
    """Signed, clipped distance to the nearest instruction-begins date. + = ahead, - = passed."""
    return _signed_nearest_clipped(_as_date(d), SEM_STARTS)


def days_to_sem_end(d):
    """Signed, clipped distance to the nearest last-day-of-finals date. + = ahead, - = passed."""
    return _signed_nearest_clipped(_as_date(d), SEM_ENDS)


# ── RSF open-hours calendar (consolidated 2026-07-21) ────────────────────────
# This was copy-pasted into scraper.py, predictions_builder.py, weekly_builder.py,
# today_builder.py, backtest.py, and test_curve_sanity.py; those now import
# SUMMER_RANGES / is_summer_day / get_open_hours from here instead of keeping
# their own copy. The non-Python mirrors (docs/index.html JS, RSFApp2.0's
# TimeUtils.swift) are NOT touched by this consolidation and remain manual —
# see CLAUDE.md.
#
# Derived from SUMMER_BREAK_RANGES above with the end date shifted -3 days
# (RSF flips back to academic-year hours ~3 days before classes resume).
SUMMER_RANGES = [
    (date(2024, 5, 10), date(2024, 8, 24)),
    (date(2025, 5, 16), date(2025, 8, 23)),
    (date(2026, 5, 15), date(2026, 8, 22)),
    (date(2027, 5, 14), date(2027, 8, 21)),
]


def is_summer_day(d):
    d = _as_date(d)
    return any(s <= d <= e for s, e in SUMMER_RANGES)


def get_open_hours(day_name, d):
    summer = is_summer_day(d)
    if day_name == 'Saturday':
        return 8, 18
    if day_name == 'Sunday':
        return 8, (20 if summer else 23)
    return 7, (20 if summer else 23)


def is_semester_day(d):
    """
    "Is this an in-session day" check used by the daily builders
    (today_builder.py, weekly_builder.py, day_profiles_builder.py) to gate
    similarity-candidate pooling and semester-only aggregates.

    In-session = NOT inside a winter, spring, or summer break range
    (BREAK_RANGES). regular, first_week, finals, dead_week, and holiday days are
    all in-session; only the three break seasons are not. Using the exact
    academic-calendar ranges (not the old month cutoffs of summer = month 6-8,
    winter = Dec 16 / Jan 12) correctly labels the late-August first-week-of-fall
    and the January semester-boundary days those cutoffs got wrong.

    This is break-RANGE membership, which matches classify_date() on every day
    except the ~1-2 per year where the last finals day coincides with the first
    listed day of the following break: classify_date() calls that day "finals"
    (phase priority), this gate calls it break. That seam is immaterial to the
    candidate pooling / semester-only aggregates this gates, and keeping it as
    plain range membership lets the JS (docs/index.html) and Swift
    (TimeUtils.swift) copies mirror the same three break-range lists so all three
    layers agree on 100% of days.
    """
    d = _as_date(d)
    return not _in_any(d, BREAK_RANGES)
