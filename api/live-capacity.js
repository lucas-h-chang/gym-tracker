const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

const { isSensorStalled } = require('./_sensor');

const DENSITY_URL = 'https://api.density.io/v2/spaces/spc_863128347956216317/count';
const MAX_CAP     = 150;
const FRESH_SECS  = 30;

module.exports = async function handler(req, res) {
  // Edge cache: ~1 origin hit per 30s window regardless of client count.
  res.setHeader('Cache-Control', 'public, s-maxage=30, stale-while-revalidate=60');

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // 1. Read cached row.
  let cached = null;
  try {
    const { data, error } = await supabase
      .from('live_capacity')
      .select('capacity_pct, recorded_at, sensor_ok')
      .eq('id', 1)
      .maybeSingle();
    if (error) {
      console.error('[live-capacity] cache READ error:', JSON.stringify(error));
    }
    cached = data;
  } catch (err) {
    console.error('[live-capacity] cache read threw:', err);
  }

  const ageSecs = cached
    ? (Date.now() - new Date(cached.recorded_at).getTime()) / 1000
    : Infinity;

  if (cached && ageSecs < FRESH_SECS) {
    return res.status(200).json({
      capacity_pct: cached.capacity_pct,
      recorded_at:  cached.recorded_at,
      // Carried through the cache rather than recomputed: the stall check
      // costs a capacity_log query, and the whole point of this row is that
      // 30 seconds of clients share one origin hit.
      sensor_ok:    cached.sensor_ok !== false,
      source:       'cache',
      age_seconds:  Math.round(ageSecs),
    });
  }

  // 2. Cache miss → fetch Density.
  try {
    const dResp = await fetch(DENSITY_URL, {
      headers: { 'Authorization': `Bearer ${process.env.DENSITY_TOKEN}` },
    });
    if (!dResp.ok) throw new Error(`Density returned ${dResp.status}`);
    const body  = await dResp.json();
    const count = body.count;
    const pct   = Math.round((count / MAX_CAP) * 1000) / 10;
    const now   = new Date().toISOString();

    // Density does not error when the RSF's hardware dies — it keeps serving a
    // small plausible number. Without this the pill renders that number as
    // fact, and the trend and comparison cards then build confident sentences
    // on top of it. See api/_sensor.js for why the rule is a run of floor
    // readings and not a percentage threshold.
    const stall = await isSensorStalled(supabase, count);
    if (stall.stalled) {
      console.warn(`[live-capacity] SENSOR STALL: ${stall.reason} (since ${stall.since})`);
    }

    // Upsert single row (id=1) — same shape live-capacity-sync.yml used.
    const { error: upsertErr } = await supabase
      .from('live_capacity')
      .upsert(
        { id: 1, capacity_pct: pct, recorded_at: now, sensor_ok: !stall.stalled },
        { onConflict: 'id' }
      );
    if (upsertErr) {
      console.error('[live-capacity] cache WRITE error:', JSON.stringify(upsertErr));
    }

    return res.status(200).json({
      capacity_pct: pct,
      recorded_at:  now,
      sensor_ok:    !stall.stalled,
      source:       'density',
      age_seconds:  0,
    });
  } catch (err) {
    console.error('[live-capacity] density fetch failed:', err);
    // Fallback: return whatever stale value we have so the UI stays alive.
    if (cached) {
      return res.status(200).json({
        capacity_pct:   cached.capacity_pct,
        recorded_at:    cached.recorded_at,
        sensor_ok:      cached.sensor_ok !== false,
        source:         'cache_stale',
        age_seconds:    Math.round(ageSecs),
        upstream_error: err.message,
      });
    }
    return res.status(502).json({ error: 'density unavailable', details: err.message });
  }
};
