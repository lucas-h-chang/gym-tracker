// _hours.js — RSF open-hours gate for the Vercel serverless functions.
//
// academic_calendar.py is the source of truth for the date lists below; the
// logic (is_summer_day / get_open_hours) is a hand port of it. The date blocks
// are marked GENERATED and are written by gym-tracker/sync_calendar.py — add a
// year in the Python and re-run it, never edit them here.
//
// Same generator feeds the other consumers that cannot import Python:
//   gym-tracker/docs/index.html                 (SUMMER_RANGES, BREAK_RANGES, CLOSURES)
//   gym-tracker/.github/workflows/freshness.yml (SUMMER_RANGES, CLOSURES)
//   RSFApp2.0/.../TimeUtils.swift               (summerRanges, semesterBreakRanges, closureRanges)
// SQL is separate: migrations/001 and 009 are applied history and are not
// regenerated.
//
// Hours (per CLAUDE.md):
//   Academic year — Mon-Fri 7-23, Sat 8-18, Sun 8-23
//   Summer        — Mon-Fri 7-20, Sat 8-18 (unchanged), Sun 8-20

// Derived from academic_calendar.py's SUMMER_BREAK_RANGES with the end date
// shifted -3 days (the RSF flips back to academic-year hours ~3 days before
// classes resume). ISO date strings so plain string comparison works.
// >>> GENERATED SUMMER_RANGES — from academic_calendar.py via sync_calendar.py; do not edit by hand
const SUMMER_RANGES = [
  ['2024-05-10', '2024-08-24'],
  ['2025-05-16', '2025-08-23'],
  ['2026-05-15', '2026-08-22'],
  ['2027-05-14', '2027-08-21'],
];
// <<< END GENERATED SUMMER_RANGES

/**
 * Current Pacific wall-clock, independent of the server's own timezone.
 * Vercel functions run in UTC, so we cannot read Date methods directly.
 */
function ptNow() {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'America/Los_Angeles',
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit',
    hour12: false, weekday: 'long',
  }).formatToParts(new Date());

  const p = Object.fromEntries(parts.map((x) => [x.type, x.value]));
  return {
    date: `${p.year}-${p.month}-${p.day}`,  // 'YYYY-MM-DD' in PT
    weekday: p.weekday,                      // 'Monday' ... 'Sunday'
    // hour12:false renders midnight as "24" in some ICU builds; normalize so
    // 00:xx never reads as hour 24 and slips past a `< openH` check.
    hour: Number(p.hour) % 24,
    minute: Number(p.minute),
  };
}

// academic_calendar.py's CLOSURES. Days the RSF is shut
// entirely — Caltopia takes over the building the Sunday and Monday before
// fall instruction begins (Tuesday too in 2026). Density keeps answering on
// those days, with 0-2 people, so without this the scraper logs a full day of
// near-zero readings into the baselines and api/_sensor.js reads the flat run
// as a dead sensor.
// >>> GENERATED CLOSURES — from academic_calendar.py via sync_calendar.py; do not edit by hand
const CLOSURES = [
  ["2021-08-22", "2021-08-23", "Caltopia"],
  ["2022-08-21", "2022-08-22", "Caltopia"],
  ["2023-08-20", "2023-08-21", "Caltopia"],
  ["2024-08-25", "2024-08-26", "Caltopia"],
  ["2025-08-24", "2025-08-25", "Caltopia"],
  ["2026-08-23", "2026-08-25", "Caltopia"],
  ["2027-08-22", "2027-08-23", "Caltopia"],
  ["2022-11-24", "2022-11-24", "Thanksgiving"],
  ["2022-12-25", "2022-12-25", "Christmas"],
  ["2023-11-23", "2023-11-23", "Thanksgiving"],
  ["2023-12-25", "2023-12-25", "Christmas"],
  ["2024-11-28", "2024-11-28", "Thanksgiving"],
  ["2024-12-24", "2024-12-25", "Christmas"],
  ["2025-01-01", "2025-01-01", "New Year's Day"],
  ["2025-11-27", "2025-11-27", "Thanksgiving"],
  ["2025-12-24", "2025-12-25", "Christmas"],
  ["2026-01-01", "2026-01-01", "New Year's Day"],
  ["2026-11-26", "2026-11-26", "Thanksgiving"],
  ["2026-12-24", "2026-12-25", "Christmas"],
  ["2027-01-01", "2027-01-01", "New Year's Day"],
  ["2027-11-25", "2027-11-25", "Thanksgiving"],
  ["2027-12-24", "2027-12-25", "Christmas"],
  ["2028-01-01", "2028-01-01", "New Year's Day"],
];
// <<< END GENERATED CLOSURES

/** dateStr: 'YYYY-MM-DD' in PT. Returns the closure reason, or null. */
function closureReason(dateStr) {
  const hit = CLOSURES.find(([s, e]) => dateStr >= s && dateStr <= e);
  return hit ? hit[2] : null;
}

/** dateStr: 'YYYY-MM-DD' in PT. */
function isSummerDay(dateStr) {
  return SUMMER_RANGES.some(([start, end]) => dateStr >= start && dateStr <= end);
}

/**
 * Returns [openHour, closeHour] as integers. Mirrors get_open_hours() exactly.
 * Open is inclusive, close is EXCLUSIVE: on an academic weekday the gym is
 * "open" for 7.0 <= now < 23.0, so the last scrape of the day is 22:45.
 */
function getOpenHours(weekday, dateStr) {
  // Empty interval [0, 0) on a closure day — `nowHour >= 0 && nowHour < 0` is
  // false at every hour, so existing gates report "closed" all day with no
  // change at the call site. Mirrors academic_calendar.get_open_hours().
  if (closureReason(dateStr)) return [0, 0];
  const summer = isSummerDay(dateStr);
  if (weekday === 'Saturday') return [8, 18];
  if (weekday === 'Sunday') return [8, summer ? 20 : 23];
  return [7, summer ? 20 : 23];
}

/** Convenience: is the RSF open at the current PT moment? */
function isOpenNow(now = ptNow()) {
  const [openH, closeH] = getOpenHours(now.weekday, now.date);
  const nowHour = now.hour + now.minute / 60;
  return { open: nowHour >= openH && nowHour < closeH, openH, closeH };
}

module.exports = {
  SUMMER_RANGES, CLOSURES, ptNow, isSummerDay, closureReason, getOpenHours, isOpenNow,
};
