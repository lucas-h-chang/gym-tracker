const assert = require('node:assert/strict');
const { insertCapacityRow } = require('./_supabase_retry');

function fakeSupabase(outcomes) {
  const calls = [];
  return {
    calls,
    from(table) {
      assert.equal(table, 'capacity_log');
      return {
        async insert(row) {
          calls.push(row);
          const outcome = outcomes.shift();
          if (outcome instanceof Error) throw outcome;
          return outcome;
        },
      };
    },
  };
}

(async () => {
  const row = {
    timestamp: '2026-09-13T19:00:14.000Z',
    people_count: 87,
    percent_full: 58,
    sensor_ok: true,
  };

  const delays = [];
  const warnings = [];
  const recovering = fakeSupabase([
    { error: { message: 'Gateway Timeout' } },
    { error: { message: 'Gateway Timeout' } },
    { error: null },
  ]);
  const recovered = await insertCapacityRow(recovering, row, {
    sleep: async (ms) => delays.push(ms),
    logger: { warn: (message) => warnings.push(message) },
  });
  assert.deepEqual(recovered, { error: null, attempts: 3 });
  assert.deepEqual(delays, [1000, 2000]);
  assert.equal(warnings.length, 2);
  assert.equal(recovering.calls.length, 3);
  assert.ok(recovering.calls.every((call) => call === row));

  const immediate = fakeSupabase([{ error: null }]);
  assert.deepEqual(
    await insertCapacityRow(immediate, row, { sleep: async () => assert.fail() }),
    { error: null, attempts: 1 }
  );

  const failed = fakeSupabase([
    { error: { message: 'one' } },
    new Error('two'),
    { error: { message: 'three' } },
  ]);
  const final = await insertCapacityRow(failed, row, {
    baseDelayMs: 0,
    sleep: async () => {},
    logger: { warn: () => {} },
  });
  assert.equal(final.attempts, 3);
  assert.equal(final.error.message, 'three');
  assert.equal(failed.calls.length, 3);

  console.log('all capacity_log retry cases pass');
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
