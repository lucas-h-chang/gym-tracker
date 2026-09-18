// One bounded, validated read shared by the scraper and display cache.
const DENSITY_URL = 'https://api.density.io/v2/spaces/spc_863128347956216317/count';

async function readDensity({ fetchImpl = fetch, timeoutMs = 4000 } = {}) {
  const response = await fetchImpl(DENSITY_URL, {
    headers: { Authorization: `Bearer ${process.env.DENSITY_TOKEN}` },
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) throw new Error(`Density returned ${response.status}`);
  const { count } = await response.json();
  if (typeof count !== 'number' || !Number.isFinite(count) || count < 0) {
    throw new Error('Density returned an invalid count');
  }
  // Above nominal capacity is a legitimate observation, not a sensor error.
  return count;
}

module.exports = { readDensity };
