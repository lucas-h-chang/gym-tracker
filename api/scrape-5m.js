// Separate cron-job.org job: */5 * * * *, x-scrape-secret header as today.
// Explicit opt-in; never writes capacity_log or publishes a forecast.
const { createClient } = require('@supabase/supabase-js');
const { ptNow, isOpenNow } = require('./_hours');
const { readDensity } = require('./_density');
const { TABLE, EXPERIMENT, tickAt, captureProduction, insertSample } = require('./_five_minute');

function createHandler({ env = process.env, now = () => new Date(), read = readDensity,
  makeClient = () => createClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY, {
    global: { fetch: (url, options) => fetch(url, { ...options, signal: AbortSignal.timeout(4000) }) },
  }), sleep = ms => new Promise(r => setTimeout(r, ms)), logger = console } = {}) {
  return async function handler(req, res) {
    res.setHeader('Cache-Control', 'no-store');
    if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' });
    if (!env.SCRAPE_SECRET || req.headers['x-scrape-secret'] !== env.SCRAPE_SECRET) {
      return res.status(401).json({ error: 'unauthorized' });
    }
    if (env.FIVE_MINUTE_EXPERIMENT_ENABLED !== 'true') {
      return res.status(200).json({ skipped: 'experiment disabled' });
    }
    const started = now();
    if (!isOpenNow(ptNow(started)).open) return res.status(200).json({ skipped: 'closed' });

    let sb;
    try { sb = makeClient(); }
    catch { return res.status(503).json({ error: 'experiment configuration unavailable' }); }

    // The cutoff precedes the sensor request, even if publication happens while
    // it is in flight. A later forecast cannot leak into this control.
    const control = captureProduction(sb, started, ptNow(started).date);
    let count;
    try {
      for (let attempt = 0; attempt < 2; attempt++) {
        try { count = await read(); break; }
        catch (error) { if (attempt === 1) throw error; await sleep(500); }
      }
      if (!Number.isSafeInteger(count) || count < 0) throw new Error('Invalid occupancy');
    } catch {
      await control;
      return res.status(502).json({ error: 'density unavailable' });
    }
    const observed = now();
    if (!isOpenNow(ptNow(observed)).open) {
      await control;
      return res.status(200).json({ skipped: 'closed during sensor read' });
    }
    const row = { tick_at: tickAt(observed), observed_at: observed.toISOString(),
      people_count: count, experiment: EXPERIMENT, production: await control };
    try { await insertSample(sb, row, { sleep }); }
    catch {
      logger.error('[scrape-5m] isolated sample insert failed');
      return res.status(503).json({ error: 'experiment storage unavailable' });
    }
    return res.status(200).json({ experiment: EXPERIMENT, table: TABLE,
      tick_at: row.tick_at, control_status: row.production.status });
  };
}

module.exports = createHandler();
module.exports.createHandler = createHandler;
