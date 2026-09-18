const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

// ── Supabase config ────────────────────────────────────────────
const SB_URL      = 'https://njhxcwcvyorwqlfacnal.supabase.co';
const SB_ANON_KEY = 'sb_publishable_in5UWgu0ZCkriOULxBZ8GA_kKJ0qUvA';

async function sbFetch(path) {
  const res = await fetch(`${SB_URL}/rest/v1/${path}`, {
    signal: AbortSignal.timeout(10000),
    headers: { 'apikey': SB_ANON_KEY, 'Authorization': `Bearer ${SB_ANON_KEY}` }
  });
  if (!res.ok) throw new Error(`Supabase ${res.status}: ${path}`);
  return res.json();
}

// Live polling is single-flight across timers and wake events.
let liveCapacity = null;
let liveRequest = null;
let lastLiveAttempt = 0;
function usableLive() {
  return liveCapacity && BearQuality.isFresh(liveCapacity.recordedAt, 120)
    && liveCapacity.sensorOk && liveCapacity.source !== 'cache_stale' ? liveCapacity : null;
}
function correctedPoints(data) {
  return BearQuality.isFresh(data.today_computed_at, 45 * 60) ? data.today_similarity_preds || [] : [];
}
async function updateStatusBar() {
  if (document.visibilityState === 'hidden') return;
  const pt = getPTNow();
  const { open, close } = getOpenHours(pt.getDay(), pt);
  if (pt.getHours() < open || pt.getHours() >= close) { liveCapacity = null; return; }
  if (liveRequest) return liveRequest;
  if (Date.now() - lastLiveAttempt < 1000) return;
  lastLiveAttempt = Date.now();
  liveRequest = (async () => {
    try {
      const r = await fetch('/api/live-capacity', { signal: AbortSignal.timeout(8000) });
      if (!r.ok) throw new Error(`Live count ${r.status}`);
      const d = await r.json();
      const state = BearQuality.liveState(d);
      liveCapacity = state === 'unavailable' ? null : {
        pct: Math.round(d.capacity_pct), count: Math.round(d.capacity_pct * 1.5),
        recordedAt: d.recorded_at, sensorOk: d.sensor_ok !== false, source: d.source,
      };
    } catch (_) {
      liveCapacity = null;
    } finally {
      liveRequest = null;
      if (window._insightData) renderInsightCards(window._insightData);
    }
  })();
  return liveRequest;
}

// ── Today's data refresh ─────────────────────────────────────
// The 15s tick above only redraws from data already in memory, and
// updateStatusBar() only refreshes the single live-% reading. Neither pulls
// new capacity_log rows, so a tab left open draws one straight line from
// whatever bin was loaded at boot to the live point, and the real curve in
// between never fills in. Re-pull the two tables that actually change during
// the day: capacity_log (api/scrape.js, every 15 min) and today_summary
// (today_builder.py, every 15 min). predictions and weekly_averages are
// daily/nightly builds, so they stay as loaded.
let bootDate = '';
let refreshInFlight = false;
let lastRefreshAt = 0;

async function refreshTodayData() {
  if (document.visibilityState === 'hidden') return;
  const data = window._insightData;
  if (!data) return;

  // Returning to a tab can fire visibilitychange, focus and pageshow within
  // milliseconds of each other. Collapse them so one trip back is one fetch,
  // not three, and so the 5-min timer can't overlap a wake-up refresh.
  if (refreshInFlight || Date.now() - lastRefreshAt < 10000) return;

  // The date rolled over while the tab sat open. Day pills, calendar bounds
  // and the whole prediction window are built around the boot-time date, so
  // reloading is the only honest way to move to the new day. Rare by
  // definition (once per open tab, at midnight PT).
  if (todayPT() !== bootDate) {
    location.reload();
    return;
  }

  refreshInFlight = true;

  // Narrower than the boot query, which starts at yesterday 12:00Z purely as
  // a safe margin. buildTodayActuals() discards everything that isn't today
  // PT anyway, and PT is UTC-7/-8, so today's PT midnight is always >= today
  // 00:00Z. Halving the payload matters here because this repeats all day.
  let actualsRows, todaySummaryRows;
  try {
    [actualsRows, todaySummaryRows] = await Promise.all([
      sbFetch(`capacity_log?timestamp=gte.${bootDate}T00:00:00Z&order=timestamp.asc&limit=2000`),
      sbFetch(`today_summary?date=eq.${bootDate}&select=similarity_preds,blend_weight,computed_at`),
    ]);
  } catch (e) {
    // Offline, or a Supabase blip. Keep rendering what we already have and
    // let the next tick retry rather than blanking a working page.
    console.error('refreshTodayData failed:', e);
    return;
  } finally {
    refreshInFlight = false;
    lastRefreshAt = Date.now();
  }

  data.today_actuals = buildTodayActuals(actualsRows, bootDate);

  const ts = todaySummaryRows[0] || {};
  data.today_computed_at = ts.computed_at;
  data.today_similarity_preds = ts.similarity_preds || [];
  data.today_blend_weight     = ts.blend_weight || 0;

  if (predDate === bootDate) updatePredChart(data);
  renderInsightCards(data);
}
