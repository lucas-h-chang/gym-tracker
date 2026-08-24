// scrape.js — takes one live RSF occupancy reading and appends it to
// capacity_log. This is the production scraper as of 2026-08-13; it replaces
// the `Scrape live count → Supabase` step that used to run scraper.py inside
// .github/workflows/scrape.yml.
//
// WHY THIS MOVED OFF GITHUB ACTIONS
// A reading is irreplaceable: Density's /count endpoint only reports *now*, so
// a scrape missed at 15:45 is gone forever. GitHub Actions has to allocate a
// VM from a shared pool before any code runs, and on 2026-08-06 that queue
// backed up 8-17 minutes and dropped several readings outright. Vercel invokes
// an already-deployed function, so there is no allocation step to get stuck in.
// The derived builders (today_builder.py, send_workout_notifications.py) stay
// on Actions: they recompute from data already in Supabase, so a delay costs
// nothing.
//
// scraper.py is intentionally left in place as a manual backfill / escape
// hatch. It is no longer on the 15-minute path.
//
// Scheduling: cron-job.org GETs this endpoint every 15 minutes with the
// x-scrape-secret header. Like scraper.py, it self-gates on open hours, so the
// schedule does not need seasonal adjustment.

const { createClient } = require('@supabase/supabase-js');
const { ptNow, getOpenHours } = require('./_hours');
const { isSensorStalled } = require('./_sensor');

const DENSITY_URL = 'https://api.density.io/v2/spaces/spc_863128347956216317/count';
const MAX_CAP = 150;

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

module.exports = async function handler(req, res) {
  // Every invocation must actually hit Density and write a row, so this must
  // never be served from the edge cache the way live-capacity.js is.
  res.setHeader('Cache-Control', 'no-store');

  // 1. Auth. Unlike live-capacity.js (which only overwrites a 30-second display
  //    cache) this appends to permanent history, so an open endpoint would let
  //    anyone inject fabricated readings into the training data.
  const provided = req.headers['x-scrape-secret'] || (req.query && req.query.key);
  if (!process.env.SCRAPE_SECRET || provided !== process.env.SCRAPE_SECRET) {
    return res.status(401).json({ error: 'unauthorized' });
  }

  // 2. Open-hours gate, same contract as scraper.py: cron fires through
  //    academic-year hours year-round, so we exit quietly when the RSF is
  //    actually closed (e.g. summer evenings) rather than logging zeros.
  const now = ptNow();
  const [openH, closeH] = getOpenHours(now.weekday, now.date);
  const nowHour = now.hour + now.minute / 60;
  if (nowHour < openH || nowHour >= closeH) {
    console.log(`[scrape] RSF closed (open ${openH}:00-${closeH}:00); skipping insert.`);
    return res.status(200).json({ skipped: 'closed', open: openH, close: closeH });
  }

  // 3. Read Density, with one quick retry. The Actions job retried 3x with 5s
  //    sleeps; a single 1s retry fits comfortably inside the function timeout
  //    and the next cron tick is only 15 minutes out either way.
  let count;
  for (const attempt of [1, 2]) {
    try {
      const resp = await fetch(DENSITY_URL, {
        headers: { Authorization: `Bearer ${process.env.DENSITY_TOKEN}` },
      });
      if (!resp.ok) throw new Error(`Density returned ${resp.status}`);
      count = (await resp.json()).count;
      break;
    } catch (err) {
      console.error(`[scrape] density attempt ${attempt} failed:`, err.message);
      if (attempt === 2) {
        return res.status(502).json({ error: 'density unavailable', details: err.message });
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  // 4. Append. Both the raw count and the derived percentage are stored, to
  //    match scraper.py and the existing capacity_log schema. (live-capacity.js
  //    keeps only a percentage, but that is a display cache, not history.)
  //    timestamp is written as UTC; scraper.py wrote a PT offset. capacity_log
  //    .timestamp is timestamptz, so both record the identical instant.
  const pct = Math.round((count / MAX_CAP) * 1000) / 10;
  const timestamp = new Date().toISOString();

  // 4a. Flag, don't drop. Density keeps serving a plausible small number when
  //     the RSF's counter dies (see api/_sensor.js), and an unflagged zero is
  //     worse than a gap: day_profiles and weekly_builder.py average raw
  //     percent_full with no floor, so a stalled day silently drags down the
  //     "vs usual <Day>s" baseline for that weekday forever. Writing the row
  //     with sensor_ok = false keeps the forensic record and keeps the stall
  //     detector fed (it reads this very table to find its run) while taking
  //     the reading out of every downstream average. See migration 008.
  const stall = await isSensorStalled(supabase, count);
  if (stall.stalled) {
    console.warn(`[scrape] SENSOR STALL: ${stall.reason} (since ${stall.since}) — logging ${count} with sensor_ok=false`);
  }

  const { error } = await supabase.from('capacity_log').insert({
    timestamp,
    people_count: count,
    percent_full: pct,
    sensor_ok: !stall.stalled,
  });

  if (error) {
    console.error('[scrape] capacity_log insert failed:', JSON.stringify(error));
    return res.status(500).json({ error: 'insert failed', details: error.message });
  }

  console.log(`[scrape] ${timestamp} Saved: ${count} people (${pct}%)${stall.stalled ? ' [sensor_ok=false]' : ''}`);
  return res.status(200).json({
    timestamp,
    people_count: count,
    percent_full: pct,
    sensor_ok: !stall.stalled,
  });
};
