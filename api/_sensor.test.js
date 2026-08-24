// _sensor.test.js — behavioural tests for the sensor-stall rule.
//
// Run with:  node api/_sensor.test.js   (or `npm test` from gym-tracker/)
//
// No test framework and no database: isSensorStalled's only dependency is the
// supabase query chain, which is small enough to stand in for. The date is
// pinned on every case — the closure guard reads the calendar, so an unpinned
// test would quietly change meaning depending on the day it ran.

const { isSensorStalled } = require('./_sensor');
const { closureReason } = require('./_hours');

// Minimal stand-in for the supabase query builder chain used by _sensor.js.
function fakeDb(rows) {
  const q = {
    from: () => q, select: () => q, gte: () => q, order: () => q,
    limit: () => Promise.resolve({ data: rows, error: null }),
  };
  return q;
}
// rows are returned newest-first, as .order({ascending:false}) would
const mins = (n) => new Date(Date.now() - n * 60000).toISOString();
const seq = (counts) => counts.map((c, i) => ({ timestamp: mins(15 * (i + 1)), people_count: c }));

const OPEN_SUN = '2026-09-13';  // in-session Sunday, RSF open
const OPEN_SAT = '2026-08-22';  // Saturday before Caltopia, RSF open

const cases = [
  // The failure this exists to catch: hardware dead on a day the gym is open.
  ['Sensor dead all morning on an open Sunday', OPEN_SUN, seq([1,1,2,0,0,0,0,0,0,0]), 1, true],
  ['Six floor readings — the run completes', OPEN_SUN, seq([0,0,0,0,0]), 0, true],

  // The false positive that shipped in 008 and 009 fixes. Same readings as the
  // first case, but the RSF was shut for Caltopia, so 0-2 people was accurate.
  ['2026-08-23 Caltopia closure — closed, NOT a dead sensor', '2026-08-23', seq([1,1,2,0,0,0,0,0,0,0]), 1, false],
  ['2026-08-25 Caltopia Tuesday', '2026-08-25', seq([0,0,0,0,0,0]), 0, false],

  // Legitimately quiet moments that must never trip the alarm.
  ['2026-08-22 08:15, quiet open then a real crowd', OPEN_SAT, seq([1]), 37, false],
  ['The quiet opening slot itself (only 1 row in window)', OPEN_SAT, seq([1]), 1, false],
  ['Summer weekend 8:15 AM baseline (1% -> 13%)', OPEN_SAT, seq([1]), 20, false],
  ['Two quiet slots at open, still ramping', OPEN_SAT, seq([4,2]), 3, false],
  ['Five floor readings — one short of the run', OPEN_SUN, seq([0,0,0,0]), 0, false],
  ['Recovery: long stall but the hardware is back', OPEN_SUN, seq([0,0,0,0,0,0,0,0]), 42, false],
  ['Stall broken by one real reading mid-window', OPEN_SUN, seq([0,0,9,0,0,0]), 0, false],
  ['Genuinely quiet but above the junk floor (6-8 people)', OPEN_SUN, seq([6,7,6,8,7,6]), 7, false],
];

// The closure calendar itself, asserted against the real dates.
const CLOSURE_CASES = [
  ['2026-08-23', true,  'Caltopia Sunday'],
  ['2026-08-24', true,  'Caltopia Monday'],
  ['2026-08-25', true,  'Caltopia Tuesday (2026 only)'],
  ['2026-08-26', false, 'first day of instruction'],
  ['2026-08-22', false, 'Saturday before Caltopia'],
  ['2025-08-24', true,  'Caltopia Sunday, prior year'],
  ['2025-08-26', false, 'Tuesday 2025 — open, Tuesday is not a standing closure'],
];

(async () => {
  let fail = 0;
  for (const [name, day, rows, current, want] of cases) {
    const r = await isSensorStalled(fakeDb(rows), current, day);
    const ok = r.stalled === want;
    if (!ok) fail++;
    console.log(`${ok ? 'PASS' : 'FAIL'}  stalled=${String(r.stalled).padEnd(5)} want=${String(want).padEnd(5)} ${name}`);
    console.log(`        ${day} · ${r.reason}`);
  }

  for (const [ds, wantClosed, label] of CLOSURE_CASES) {
    const got = closureReason(ds) !== null;
    const ok = got === wantClosed;
    if (!ok) fail++;
    console.log(`${ok ? 'PASS' : 'FAIL'}  closed=${String(got).padEnd(5)} want=${String(wantClosed).padEnd(5)} ${ds} ${label}`);
  }

  // fail-open check
  const errDb = { from: () => errDb, select: () => errDb, gte: () => errDb, order: () => errDb,
                  limit: () => Promise.resolve({ data: null, error: { message: 'boom' } }) };
  const e = await isSensorStalled(errDb, 0, OPEN_SUN);
  console.log(`${e.stalled === false ? 'PASS' : 'FAIL'}  DB error fails open (stalled=${e.stalled})`);
  if (e.stalled !== false) fail++;

  console.log(fail === 0 ? '\nall cases pass' : `\n${fail} FAILURES`);
  process.exit(fail ? 1 : 0);
})();
