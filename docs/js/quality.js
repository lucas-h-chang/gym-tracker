// Also loaded by Node contract tests. No DOM or network dependencies.
(function (root) {
  function isFresh(timestamp, maxSeconds, now = Date.now()) {
    const age = (now - Date.parse(timestamp)) / 1000;
    return Number.isFinite(age) && age >= 0 && age <= maxSeconds;
  }
  function liveState(body, now = Date.now()) {
    if (!body || typeof body.capacity_pct !== 'number' || !Number.isFinite(body.capacity_pct) || body.capacity_pct < 0) return 'unavailable';
    if (body.sensor_ok === false) return 'invalid';
    if (body.source === 'cache_stale' || !isFresh(body.recorded_at, 120, now)) return 'stale';
    return 'fresh';
  }
  const api = { isFresh, liveState };
  if (typeof module !== 'undefined') module.exports = api;
  else root.BearQuality = api;
})(globalThis);
