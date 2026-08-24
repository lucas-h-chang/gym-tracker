// _hours.js — RSF open-hours gate for the Vercel serverless functions.
//
// This is a MANUAL MIRROR of academic_calendar.py's SUMMER_RANGES /
// is_summer_day / get_open_hours. That file is the Python source of truth;
// when you add a year there, add it here too.
//
// Other mirrors of the same calendar (see CLAUDE.md):
//   gym-tracker/academic_calendar.py            <- source of truth
//   gym-tracker/docs/index.html                 (SUMMER_RANGES + BREAK_RANGES)
//   gym-tracker/migrations/001_is_semester_day.sql
//   gym-tracker/migrations/009_caltopia_closures.sql (CLOSURES, as SQL)
//   gym-tracker/.github/workflows/freshness.yml (inlined copy)
//   RSFApp2.0/.../TimeUtils.swift
//
// Hours (per CLAUDE.md):
//   Academic year — Mon-Fri 7-23, Sat 8-18, Sun 8-23
//   Summer        — Mon-Fri 7-20, Sat 8-18 (unchanged), Sun 8-20

// Derived from academic_calendar.py's SUMMER_BREAK_RANGES with the end date
// shifted -3 days (the RSF flips back to academic-year hours ~3 days before
// classes resume). ISO date strings so plain string comparison works.
const SUMMER_RANGES = [
  ['2024-05-10', '2024-08-24'],
  ['2025-05-16', '2025-08-23'],
  ['2026-05-15', '2026-08-22'],
  ['2027-05-14', '2027-08-21'],
];

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

// Manual mirror of academic_calendar.py's CLOSURES. Days the RSF is shut
// entirely — Caltopia takes over the building the Sunday and Monday before
// fall instruction begins (Tuesday too in 2026). Density keeps answering on
// those days, with 0-2 people, so without this the scraper logs a full day of
// near-zero readings into the baselines and api/_sensor.js reads the flat run
// as a dead sensor.
const CLOSURES = [
  ['2021-08-22', '2021-08-23', 'Caltopia'],
  ['2022-08-21', '2022-08-22', 'Caltopia'],
  ['2023-08-20', '2023-08-21', 'Caltopia'],
  ['2024-08-25', '2024-08-26', 'Caltopia'],
  ['2025-08-24', '2025-08-25', 'Caltopia'],
  ['2026-08-23', '2026-08-25', 'Caltopia'],
  ['2027-08-22', '2027-08-23', 'Caltopia'],
];

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
