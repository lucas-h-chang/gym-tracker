"""
test_calendar_mirrors.py — the RSF calendar says the same thing in every language.

academic_calendar.py is the source of truth. Five consumers cannot import it
(two JS, one inlined-Python workflow, one Swift, one SQL), so sync_calendar.py
generates their copies. This test is the enforcement: it fails the build if any
mirror has drifted, which is the failure that actually shipped — the 2026-08
closure work reached four of the five places and the iOS app spent three days
telling users the RSF was open while it was locked for Caltopia.

Two independent checks, deliberately not the same check twice:

  1. test_no_drift re-runs the generator and asserts nothing would change. This
     catches a hand edit to a generated block.
  2. test_*_dates_match parse the ISO dates back OUT of each mirror with a regex
     and compare them to academic_calendar.py directly. This catches a broken
     GENERATOR — the case check 1 is blind to, because a generator that drops a
     year agrees with the file it just wrote.

Run with:  python3 -m pytest test_calendar_mirrors.py -v
"""
import ast
import re
import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path

import pytest

import academic_calendar as cal
import sync_calendar

ROOT = Path(__file__).resolve().parent
IOS  = ROOT.parent / "RSFApp2.0" / "RSFApp2.0" / "Utilities" / "TimeUtils.swift"

# Only gym-tracker/ is pushed to GitHub, so TimeUtils.swift is not on disk in CI.
# The Swift assertions skip there and run on every local invocation, which is the
# machine where a new academic year actually gets typed in.
needs_ios = pytest.mark.skipif(
    not IOS.exists(),
    reason="RSFApp2.0/ is not present (it lives outside the gym-tracker repo)",
)

ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
PY_DATE  = re.compile(r"date\((\d{4}),\s*(\d{1,2}),\s*(\d{1,2})\)")


def block(path, name):
    """The text between one generated block's markers."""
    text = Path(path).read_text()
    begin = sync_calendar.BEGIN.format(c="//" if path != FRESHNESS else "#", name=name)
    end   = sync_calendar.END.format(c="//" if path != FRESHNESS else "#", name=name)
    assert begin in text, f"{Path(path).name}: no GENERATED marker for {name}"
    return text[text.index(begin) + len(begin): text.index(end, text.index(begin))]


FRESHNESS = ROOT / ".github" / "workflows" / "freshness.yml"
HOURS_JS  = ROOT / "api" / "_hours.js"
INDEX     = ROOT / "docs" / "index.html"


def iso_pairs(path, name):
    """Ranges parsed straight out of the file, independent of the generator."""
    found = ISO_DATE.findall(block(path, name))
    return [(found[i], found[i + 1]) for i in range(0, len(found), 2)]


def py_pairs(name):
    found = [date(int(y), int(m), int(d)) for y, m, d in PY_DATE.findall(block(FRESHNESS, name))]
    return [(found[i], found[i + 1]) for i in range(0, len(found), 2)]


def expected(ranges):
    return [(f"{s:%Y-%m-%d}", f"{e:%Y-%m-%d}") for s, e in ranges]


BREAKS = cal.WINTER_BREAK_RANGES + cal.SPRING_BREAK_RANGES + cal.SUMMER_BREAK_RANGES
CLOSURE_PAIRS = [(s, e) for s, e, _ in cal.CLOSURES]


# ── 1. Nothing has been hand-edited ─────────────────────────────────────────
def test_no_drift():
    r = subprocess.run([sys.executable, "sync_calendar.py", "--check"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ── 2. Every mirror's dates equal the Python source ─────────────────────────
@pytest.mark.parametrize("path,name,ranges", [
    (HOURS_JS, "SUMMER_RANGES",       cal.SUMMER_RANGES),
    (HOURS_JS, "CLOSURES",            CLOSURE_PAIRS),
    (INDEX,    "SUMMER_RANGES",       cal.SUMMER_RANGES),
    (INDEX,    "BREAK_RANGES",        BREAKS),
    (INDEX,    "CLOSURES",            CLOSURE_PAIRS),
    pytest.param(IOS, "summerRanges",        cal.SUMMER_RANGES, marks=needs_ios),
    pytest.param(IOS, "semesterBreakRanges", BREAKS,            marks=needs_ios),
    pytest.param(IOS, "closureRanges",       CLOSURE_PAIRS,     marks=needs_ios),
])
def test_js_and_swift_dates_match_source(path, name, ranges):
    assert iso_pairs(path, name) == expected(ranges)


@pytest.mark.parametrize("name,ranges", [
    ("SUMMER_RANGES", cal.SUMMER_RANGES),
    ("CLOSURES",      CLOSURE_PAIRS),
])
def test_freshness_dates_match_source(name, ranges):
    assert py_pairs(name) == list(ranges)


# ── 3. The closure REASON survives the trip (JS/Swift carry it, others don't) ─
@pytest.mark.parametrize("path,name", [
    (HOURS_JS, "CLOSURES"), (INDEX, "CLOSURES"),
    pytest.param(IOS, "closureRanges", marks=needs_ios),
])
def test_closure_reasons_match_source(path, name):
    # Trailing bracket differs by language: JS closes the range with `],`,
    # Swift with `),`.
    reasons = re.findall(r"['\"]([A-Za-z][A-Za-z ]*)['\"]\s*[\])]?,\s*$",
                         block(path, name), re.MULTILINE)
    assert reasons == [why for _, _, why in cal.CLOSURES]


# ── 4. The one mirror the generator does NOT cover is still correct ─────────
def test_sql_closure_migration_covers_every_closure():
    """migrations/009 is applied history, so it is not regenerated. It still has
    to list every closure, or the day_profiles view silently keeps closure-day
    readings the rest of the stack drops."""
    sql = (ROOT / "migrations" / "009_caltopia_closures.sql").read_text()
    for s, e, _ in cal.CLOSURES:
        assert f"{s:%Y-%m-%d}" in sql, f"009_caltopia_closures.sql is missing {s}"
        assert f"{e:%Y-%m-%d}" in sql, f"009_caltopia_closures.sql is missing {e}"


# ── 5. No stray second copy of the calendar outside the generated blocks ────
QUOTED_ISO = re.compile(r"['\"](\d{4}-\d{2}-\d{2})['\"]")


def _code_lines(text):
    """Lines that are not pure comments. Doc comments legitimately contain
    example dates ('e.g. "2026-04-22"'), and the season labels inside a block
    are comments too; neither is calendar data."""
    return [ln for ln in text.split("\n") if not ln.lstrip().startswith(("//", "#"))]


@pytest.mark.parametrize("path,names", [
    (HOURS_JS, ["SUMMER_RANGES", "CLOSURES"]),
    (INDEX,    ["SUMMER_RANGES", "BREAK_RANGES", "CLOSURES"]),
    pytest.param(IOS, ["summerRanges", "semesterBreakRanges", "closureRanges"], marks=needs_ios),
])
def test_no_calendar_dates_outside_generated_blocks(path, names):
    """Every calendar date in the file lives inside a generated block.

    This is the check the date comparisons above cannot make: they only read
    what is BETWEEN the markers, so an orphaned second copy of the list sitting
    just outside them passes all of them while the file no longer compiles.
    That is exactly the damage a mis-split marker leaves behind (it happened
    while building this: the Swift splitter searched for "]" and hit the one in
    the type annotation `[(String, String)]`, leaving the original literal
    stranded below the END marker).
    """
    inside = sum(len(QUOTED_ISO.findall(block(path, n))) for n in names)
    total  = len(QUOTED_ISO.findall("\n".join(_code_lines(Path(path).read_text()))))
    assert total == inside, (
        f"{Path(path).name}: {total - inside} calendar date(s) sit outside the "
        f"generated blocks — likely an orphaned copy left by a bad splice"
    )


# ── 6. The workflow's generated block is still valid Python ────────────────
def test_freshness_inline_python_still_parses():
    """freshness.yml's calendar lives inside a YAML `run: |` heredoc, so the
    generated block's indentation has to line up with hand-written code around
    it. YAML will happily accept a block scalar whose contents are broken
    Python, and the date assertions above only compare dates — neither notices.

    This caught a real bug while the generator was being written: the markers
    were inserted at 20 spaces inside a 10-space block and the builder added its
    own indent on top, producing 24-space list entries. Every other test passed
    and the workflow would have died with IndentationError on its next run.
    """
    text = FRESHNESS.read_text()
    start = text.index("python - <<'PY'")
    body  = text[text.index("\n", start) + 1: text.index("\n          PY", start)]
    src   = textwrap.dedent(body)
    assert "SUMMER_RANGES" in src and "CLOSURES" in src, "extracted the wrong block"
    ast.parse(src)  # raises IndentationError/SyntaxError on a bad splice
