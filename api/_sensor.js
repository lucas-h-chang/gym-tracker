// _sensor.js — shared detection for "the RSF's occupancy hardware is dead".
//
// THE FAILURE MODE (observed 2026-08-23)
// Density does not error when the RSF's counter fails. It keeps returning a
// plausible-looking small number forever. That Sunday it reported 0-2 people
// from 8:00 AM open straight through midday, while the previous Saturday had
// already hit 37 people by 8:15. Nothing upstream was "down", so every surface
// faithfully rendered the lie: a 1% live pill, a "Filling up" trend
// extrapolated off that 1%, and a "Much quieter than usual Sundays" verdict —
// which is the worst of the three, because it states the broken number as a
// confident comparison against real history.
//
// WHY A RUN, NOT A SINGLE LOW READING
// The obvious rule ("flag anything under 10% away from open/close") is both
// too loose and too tight.
//
//   Too tight: the opening slot is LEGITIMATELY at the floor. weekly_averages
//   for all_summers has Sunday 8:00 AM at 1% and Saturday 8:00 AM at 1% — but
//   8:15 AM is already 9-10%. So a real quiet boundary lasts exactly one slot,
//   while a dead sensor lasts all day. Run length separates them on its own,
//   with no open/close carve-out — which matters, because open-hours logic is
//   already mirrored by hand in six places (see CLAUDE.md) and this must not
//   become a seventh.
//
//   Too loose: 10% is 15 people, and summer weekend mornings genuinely sit at
//   9-10% one slot after open. A 10% floor would fire on a slightly-quiet real
//   Sunday.
//
// THRESHOLD PROVENANCE
// people_count <= 5 is not a new constant. curve_model.prepare_slots() already
// drops those rows from training, because the <=5 band is flat across hours
// (closures, sensor noise) while 6+ follows the real daily open/close curve.
// Reusing it keeps one definition of "implausibly empty" in the project
// instead of two that can drift apart.
//
// NOT A STALL: A CLOSED GYM
// A full-facility closure produces an identical signature — a flat run at the
// floor, all day. The first version of this file shipped without that guard
// and would have reported the 2026-08-23 Caltopia closure as a dead sensor.
// Closure days come from the calendar, not from the readings, which is the
// only way to tell the two apart.
//
// FAIL-OPEN
// Every error path returns stalled:false. A Supabase hiccup must not black out
// a live pill that is probably fine; the cost of a missed outage is a wrong
// number for one cycle, the cost of a false outage is the whole feature dark.

const { ptNow, closureReason } = require('./_hours');

const FLOOR_COUNT = 5;   // people; mirrors curve_model.prepare_slots()
const STALL_RUN   = 6;   // consecutive floor readings (~90 min at 15-min cadence)
const WINDOW_MS   = 3 * 60 * 60 * 1000;

/**
 * @param supabase     service-role client
 * @param currentCount the count just read from Density (may be null)
 * @param todayPT      'YYYY-MM-DD' in Pacific; defaults to now. Injectable so
 *                     the tests can pin a date — otherwise every assertion
 *                     changes meaning on a closure day.
 * @returns {{stalled: boolean, reason: string, since?: string}}
 */
async function isSensorStalled(supabase, currentCount, todayPT = ptNow().date) {
  // A single healthy reading clears the alarm instantly. This is also what
  // makes the check self-healing: capacity_log still holds the stalled rows
  // when the hardware recovers, but the live count is above the floor again,
  // so we never have to "expire" the outage on a timer.
  if (currentCount == null || currentCount > FLOOR_COUNT) {
    return { stalled: false, reason: 'live count above floor' };
  }

  // A shut building genuinely holds 0-2 people, and it holds them all day —
  // exactly the signature this function looks for. 2026-08-23 was a Caltopia
  // closure, not a hardware failure, and without this guard the rule calls it
  // one. Closure days are in academic_calendar.py CLOSURES / _hours.js.
  const closure = closureReason(todayPT);
  if (closure) {
    return { stalled: false, reason: `RSF closed for ${closure}` };
  }

  const since = new Date(Date.now() - WINDOW_MS).toISOString();
  const { data, error } = await supabase
    .from('capacity_log')
    .select('timestamp, people_count')
    .gte('timestamp', since)
    .order('timestamp', { ascending: false })
    .limit(STALL_RUN * 2);

  if (error) {
    console.error('[sensor] stall lookback failed:', JSON.stringify(error));
    return { stalled: false, reason: 'lookback failed' };
  }

  // The current reading is the STALL_RUN'th; we need STALL_RUN-1 before it.
  // Bounding the lookback by time rather than row count is what keeps the
  // opening ramp safe: right after doors open there are only one or two rows
  // inside the window, so the check declines to judge until ~90 minutes in.
  const prior = (data || []).slice(0, STALL_RUN - 1);
  if (prior.length < STALL_RUN - 1) {
    return { stalled: false, reason: `only ${prior.length} readings in lookback window` };
  }

  const allFloor = prior.every(
    (r) => r.people_count != null && r.people_count <= FLOOR_COUNT
  );
  if (!allFloor) {
    return { stalled: false, reason: 'a recent reading was above the floor' };
  }

  return {
    stalled: true,
    reason: `${STALL_RUN} consecutive readings <= ${FLOOR_COUNT} people`,
    since: prior[prior.length - 1].timestamp,
  };
}

module.exports = { isSensorStalled, FLOOR_COUNT, STALL_RUN };
