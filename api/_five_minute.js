// Isolated collection only. Do not pass these rows to quarter-hour model code.
const TABLE = 'capacity_samples_5m';
const EXPERIMENT = 'cadence-5m-v1';
const MAX_CONTROL_AGE_MS = 45 * 60 * 1000;

function tickAt(now) {
  return new Date(Math.floor(now.getTime() / 300000) * 300000).toISOString();
}

async function captureProduction(supabase, before, date) {
  try {
    const { data, error } = await supabase.from('prediction_snapshots')
      .select('id,computed_at,inserted_at,preds,base').eq('source', 'live').eq('model', 'carry')
      .eq('date', date).lte('computed_at', before.toISOString())
      .lte('inserted_at', before.toISOString())
      .order('computed_at', { ascending: false }).limit(1);
    if (error) throw error;
    const row = data?.[0];
    if (!row) return { status: 'missing' };
    const age = before.getTime() - Date.parse(row.computed_at);
    if (!Number.isFinite(age) || age < 0 || !Array.isArray(row.preds)
        || !row.base || typeof row.base !== 'object' || Array.isArray(row.base)) {
      return { status: 'error' };
    }
    return { status: age > MAX_CONTROL_AGE_MS ? 'stale' : 'ok', ...row };
  } catch {
    return { status: 'error' };
  }
}

// ON CONFLICT DO NOTHING keeps the FIRST reading, including after an ambiguous
// timeout or overlapping cron calls. Every retry uses the same immutable row.
async function insertSample(supabase, row, { sleep = ms => new Promise(r => setTimeout(r, ms)) } = {}) {
  let failure;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const { error } = await supabase.from(TABLE)
        .upsert(row, { onConflict: 'tick_at', ignoreDuplicates: true });
      if (!error) return;
      failure = error;
    } catch (error) { failure = error; }
    if (attempt < 2) await sleep(500 * 2 ** attempt);
  }
  throw failure;
}

module.exports = { TABLE, EXPERIMENT, tickAt, captureProduction, insertSample };
