// Bounded retry for transient capacity_log write failures.
//
// A live Density reading cannot be reconstructed later, so a brief Supabase
// API Gateway outage must not discard it after one attempt. Keep the same row
// (especially its timestamp) across attempts. If a timeout was ambiguous and
// the first request actually committed, an identical retry can at worst create
// an identical-timestamp duplicate; downstream quarter-hour aggregation then
// gives both rows the same value and does not bias the reading.

const DEFAULT_ATTEMPTS = 3;
const DEFAULT_BASE_DELAY_MS = 1000;

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function insertCapacityRow(
  supabase,
  row,
  {
    attempts = DEFAULT_ATTEMPTS,
    baseDelayMs = DEFAULT_BASE_DELAY_MS,
    sleep = wait,
    logger = console,
  } = {}
) {
  let lastError = null;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const { error } = await supabase.from('capacity_log').insert(row);
      if (!error) return { error: null, attempts: attempt };
      lastError = error;
    } catch (error) {
      // supabase-js normally returns { error }, but network failures can throw.
      lastError = error;
    }

    const message = lastError?.message || String(lastError);
    if (attempt < attempts) {
      const delayMs = baseDelayMs * (2 ** (attempt - 1));
      logger.warn(
        `[scrape] capacity_log insert attempt ${attempt}/${attempts} failed: ` +
        `${message}; retrying in ${delayMs}ms`
      );
      await sleep(delayMs);
    }
  }

  return { error: lastError, attempts };
}

module.exports = { insertCapacityRow, DEFAULT_ATTEMPTS, DEFAULT_BASE_DELAY_MS };
