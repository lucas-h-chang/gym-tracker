"""
sync_calendar.py — render academic_calendar.py's date lists into every non-Python
mirror, and verify they have not drifted.

WHY THIS EXISTS
The RSF calendar is one set of facts, but five consumers cannot import Python:
the Vercel functions (JS), the website (JS), the freshness workflow (inlined
Python), the iOS app (Swift), and Postgres (SQL). Historically each kept a
hand-typed copy, and CLAUDE.md listed "six places to touch" when adding a year.
That process failed in the obvious way: the 2026-08 closure work landed in four
of the five places, TimeUtils.swift was missed, and for three days the iOS app
advertised an open gym while the building was locked for Caltopia.

The fix is NOT a new source of truth. academic_calendar.py stays authoritative.
What changes is that the copies are GENERATED from it instead of transcribed,
and a test regenerates them in memory and fails if any file has drifted. Silent
divergence stops being possible.

Generation rather than a runtime-shared JSON file, because every consumer needs
the data before it can run: api/scrape.js gates the 15-minute reading on it (a
network fetch there adds a failure mode to the most critical path), index.html
needs open hours before first paint, iOS ships a binary, and SQL cannot fetch at
all. All of them need it at build time regardless, so generating costs nothing
at runtime.

    python3 sync_calendar.py           # rewrite the generated blocks
    python3 sync_calendar.py --check   # exit 1 if any file has drifted (CI)

Only the text BETWEEN the markers is touched. The prose around each block is
hand-written and language-specific, and is left alone.
"""
import argparse
import sys
import json
from pathlib import Path

import academic_calendar as cal

ROOT = Path(__file__).resolve().parent
IOS  = ROOT.parent / "RSFApp2.0" / "RSFApp2.0" / "Utilities" / "TimeUtils.swift"

BEGIN = "{c} >>> GENERATED {name} — from academic_calendar.py via sync_calendar.py; do not edit by hand"
END   = "{c} <<< END GENERATED {name}"


# ── Renderers: one canonical layout per language, one range per line ─────────
def _js_pairs(ranges, indent="  "):
    return "\n".join(f"{indent}['{s:%Y-%m-%d}', '{e:%Y-%m-%d}']," for s, e in ranges)


def _js_closures(indent="  "):
    # json.dumps for the reason, not an f-string in single quotes: a label
    # containing an apostrophe ("New Year's Day") would otherwise emit invalid
    # JS and take down index.html AND api/_hours.js — i.e. the website and the
    # scraper's open-hours gate — the next time this file is synced.
    return "\n".join(
        f'{indent}["{s:%Y-%m-%d}", "{e:%Y-%m-%d}", {json.dumps(why)}],'
        for s, e, why in cal.CLOSURES
    )


def _swift_pairs(ranges, indent="    "):
    return "\n".join(f'{indent}("{s:%Y-%m-%d}", "{e:%Y-%m-%d}"),' for s, e in ranges)


def _py_pairs(ranges, indent):
    return "\n".join(
        f"{indent}(date({s.year}, {s.month}, {s.day}), date({e.year}, {e.month}, {e.day})),"
        for s, e in ranges
    )


def _breaks_js(indent="  "):
    parts = []
    for label, ranges in (("winter", cal.WINTER_BREAK_RANGES),
                          ("spring", cal.SPRING_BREAK_RANGES),
                          ("summer", cal.SUMMER_BREAK_RANGES)):
        parts.append(f"{indent}// {label}")
        parts.append(_js_pairs(ranges, indent))
    return "\n".join(parts)


def _breaks_swift(indent="    "):
    parts = []
    for label, ranges in (("winter", cal.WINTER_BREAK_RANGES),
                          ("spring", cal.SPRING_BREAK_RANGES),
                          ("summer", cal.SUMMER_BREAK_RANGES)):
        parts.append(f"{indent}// {label}")
        parts.append(_swift_pairs(ranges, indent))
    return "\n".join(parts)


# ── Block table: (file, comment token, block name, body builder) ─────────────
def blocks():
    return [
        (ROOT / "api" / "_hours.js", "//", "SUMMER_RANGES",
         lambda: "const SUMMER_RANGES = [\n" + _js_pairs(cal.SUMMER_RANGES) + "\n];"),
        (ROOT / "api" / "_hours.js", "//", "CLOSURES",
         lambda: "const CLOSURES = [\n" + _js_closures() + "\n];"),

        (ROOT / "docs" / "index.html", "//", "SUMMER_RANGES",
         lambda: "const SUMMER_RANGES = [\n" + _js_pairs(cal.SUMMER_RANGES) + "\n];"),
        (ROOT / "docs" / "index.html", "//", "BREAK_RANGES",
         lambda: "const BREAK_RANGES = [\n" + _breaks_js() + "\n];"),
        (ROOT / "docs" / "index.html", "//", "CLOSURES",
         lambda: "const CLOSURES = [\n" + _js_closures() + "\n];"),

        (ROOT / ".github" / "workflows" / "freshness.yml", "#", "SUMMER_RANGES",
         lambda: "SUMMER_RANGES = [\n" + _py_pairs(cal.SUMMER_RANGES, "    ") + "\n]"),
        (ROOT / ".github" / "workflows" / "freshness.yml", "#", "CLOSURES",
         lambda: "CLOSURES = [\n" + _py_pairs([(s, e) for s, e, _ in cal.CLOSURES], "    ") + "\n]"),

        (IOS, "//", "summerRanges",
         lambda: "private let summerRanges: [(String, String)] = [\n" + _swift_pairs(cal.SUMMER_RANGES) + "\n]"),
        (IOS, "//", "semesterBreakRanges",
         lambda: "private let semesterBreakRanges: [(String, String)] = [\n" + _breaks_swift() + "\n]"),
        (IOS, "//", "closureRanges",
         lambda: "private let closureRanges: [(String, String, String)] = [\n"
                 + "\n".join(f'    ("{s:%Y-%m-%d}", "{e:%Y-%m-%d}", {json.dumps(why)}),' for s, e, why in cal.CLOSURES)
                 + "\n]"),
    ]


def render(path, comment, name, build, text):
    """Splice one generated block into `text`. The BEGIN marker's own
    indentation is reused for the body, so a block nested inside a YAML
    `run: |` scalar stays valid."""
    begin = BEGIN.format(c=comment, name=name)
    end   = END.format(c=comment, name=name)
    if begin not in text or end not in text:
        raise SystemExit(f"{path}: missing markers for {name}\n  expected: {begin}")

    b_at = text.index(begin)
    line_start = text.rindex("\n", 0, b_at) + 1
    indent = text[line_start:b_at]
    e_at = text.index(end, b_at)

    # Builders emit an UNINDENTED body; the block's real indentation comes from
    # the BEGIN marker line, which is what keeps a block nested inside a YAML
    # `run: |` scalar valid Python.
    body = "\n".join(indent + ln if ln else ln for ln in build().split("\n"))
    return text[:b_at] + begin + "\n" + body + "\n" + indent + text[e_at:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any mirror has drifted, without writing")
    args = ap.parse_args()

    by_file = {}
    for path, comment, name, build in blocks():
        by_file.setdefault(path, []).append((comment, name, build))

    drifted, wrote, absent = [], [], []
    for path, specs in by_file.items():
        # RSFApp2.0 lives outside the gym-tracker git repo (only gym-tracker is
        # pushed to GitHub), so the Swift mirror is simply not on disk in CI.
        # Skip it loudly rather than failing: it is still checked on every local
        # run, which is where a year gets added in the first place.
        if not path.exists():
            absent.append(str(path))
            continue
        original = path.read_text()
        text = original
        for comment, name, build in specs:
            text = render(path, comment, name, build, text)
        rel = path.relative_to(ROOT.parent)
        if text == original:
            continue
        if args.check:
            drifted.append(str(rel))
        else:
            path.write_text(text)
            wrote.append(str(rel))

    for a in absent:
        print(f"  SKIPPED (not on disk): {a}")

    if args.check:
        if drifted:
            print("Calendar mirrors have drifted from academic_calendar.py:")
            for d in drifted:
                print(f"  {d}")
            print("\nRun: python3 sync_calendar.py")
            return 1
        print(f"All {len(by_file) - len(absent)} calendar mirrors match academic_calendar.py")
        return 0

    print("\n".join(f"  updated {w}" for w in wrote) or "  all mirrors already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
