// Shared scraper/live-pill rule. Two consecutive quarter-hour readings <=25%
// are invalid inside [open+15min, close-15min]. Raw readings remain in history.
// A current count above 25% clears the alarm; 6 people no longer counts as recovery.
const { ptNow, getOpenHours } = require('./_hours');
const MAX_CAPACITY = 150;
const OUTAGE_PCT = 25;
const FLOOR_COUNT = MAX_CAPACITY * OUTAGE_PCT / 100;
const STALL_RUN = 2;
const WINDOW_MS = 45 * 60 * 1000;

function interior(pt) {
  const [open, close] = getOpenHours(pt.weekday, pt.date);
  const minutes = pt.hour * 60 + pt.minute;
  return close > open && minutes >= open * 60 + 15 && minutes <= close * 60 - 15;
}
function slot(pt) { return pt.hour * 4 + Math.floor(pt.minute / 15); }
function low(count) { return Number.isFinite(count) && count >= 0 && count <= FLOOR_COUNT; }

// Pure logic, also exercised against fixtures shared with the Python escape hatch.
function evaluateSensorReading(rows, currentCount, now = new Date()) {
  const pt = ptNow(now);
  if (!low(currentCount)) return { stalled: false, reason: 'count above 25% or invalid' };
  if (!interior(pt)) return { stalled: false, reason: 'outside interior open hours' };
  const currentSlot = slot(pt);
  const prior = (rows || []).filter(row => {
    const ms = Date.parse(row.timestamp);
    if (!Number.isFinite(ms) || ms >= now.getTime()) return false;
    const previousPT = ptNow(new Date(ms));
    return previousPT.date === pt.date && slot(previousPT) < currentSlot;
  }).sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp));
  const last = prior[0];
  if (!last) return { stalled: false, reason: 'no previous distinct slot' };
  const previousPT = ptNow(new Date(last.timestamp));
  if (slot(previousPT) !== currentSlot - 1 || !interior(previousPT) || !low(last.people_count)) {
    return { stalled: false, reason: 'previous slot missing, outside margins, or above 25%' };
  }
  // Include timestamp duplicates in the confirmed prior bin so none can leak
  // into training after the first observation is retroactively invalidated.
  const timestamps = prior.filter(row => slot(ptNow(new Date(row.timestamp))) === currentSlot - 1
    && low(row.people_count)).map(row => row.timestamp);
  return { stalled: true, reason: 'two consecutive readings at or below 25%',
    since: last.timestamp, timestamps: [...new Set(timestamps)] };
}

async function isSensorStalled(supabase, currentCount, now = new Date()) {
  if (!low(currentCount) || !interior(ptNow(now))) return evaluateSensorReading([], currentCount, now);
  try {
    const { data, error } = await supabase.from('capacity_log')
      .select('timestamp, people_count')
      .gte('timestamp', new Date(now.getTime() - WINDOW_MS).toISOString())
      .lte('timestamp', now.toISOString())
      .order('timestamp', { ascending: false }).limit(12);
    if (error) throw error;
    return evaluateSensorReading(data, currentCount, now);
  } catch (error) {
    console.error('[sensor] stall lookback failed:', error.message);
    return { stalled: false, reason: 'lookback failed' };
  }
}

// Called by the scraper after inserting the second invalid reading. The live
// endpoint only reads history. Retry this idempotent, timestamp-bounded update.
async function invalidatePriorReadings(supabase, stall, { sleep = ms => new Promise(r => setTimeout(r, ms)) } = {}) {
  if (!stall.stalled || !stall.timestamps?.length) return;
  let failure;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const { error } = await supabase.from('capacity_log').update({ sensor_ok: false })
        .in('timestamp', stall.timestamps).lte('people_count', FLOOR_COUNT);
      if (error) throw error;
      return;
    } catch (error) {
      failure = error;
      if (attempt < 2) await sleep(1000 * 2 ** attempt);
    }
  }
  throw failure;
}
module.exports = { isSensorStalled, evaluateSensorReading, invalidatePriorReadings, OUTAGE_PCT, FLOOR_COUNT, STALL_RUN };
