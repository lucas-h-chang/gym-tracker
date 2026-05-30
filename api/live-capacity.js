const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

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
    const { data } = await supabase
      .from('live_capacity')
      .select('capacity_pct, recorded_at')
      .eq('id', 1)
      .maybeSingle();
    cached = data;
  } catch (err) {
    console.error('[live-capacity] cache read failed:', err);
  }

  const ageSecs = cached
    ? (Date.now() - new Date(cached.recorded_at).getTime()) / 1000
    : Infinity;

  if (cached && ageSecs < FRESH_SECS) {
    return res.status(200).json({
      capacity_pct: cached.capacity_pct,
      recorded_at:  cached.recorded_at,
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

    // Upsert single row (id=1) — same shape live-capacity-sync.yml used.
    await supabase
      .from('live_capacity')
      .upsert({ id: 1, capacity_pct: pct, recorded_at: now }, { onConflict: 'id' });

    return res.status(200).json({
      capacity_pct: pct,
      recorded_at:  now,
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
        source:         'cache_stale',
        age_seconds:    Math.round(ageSecs),
        upstream_error: err.message,
      });
    }
    return res.status(502).json({ error: 'density unavailable', details: err.message });
  }
};
