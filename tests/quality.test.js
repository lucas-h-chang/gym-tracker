const { test } = require('node:test');
const assert = require('node:assert/strict');
const { liveState } = require('../docs/js/quality');
const { readDensity } = require('../api/_density');
for (const fixture of require('./fixtures/live-capacity.json')) {
  test(`live contract: ${fixture.name}`, () => {
    assert.equal(liveState(fixture.body, Date.parse('2026-09-16T18:00:00Z')), fixture.state);
  });
}
test('Density permits zero and above capacity but rejects malformed counts', async () => {
  for (const count of [0, 180]) {
    assert.equal(await readDensity({fetchImpl: async () => ({ok:true,json:async()=>({count})})}), count);
  }
  for (const count of [null, '2', -1, NaN, Infinity]) {
    await assert.rejects(readDensity({fetchImpl: async () => ({ok:true,json:async()=>({count})})}), /invalid count/);
  }
});
test('Density cancels a hanging upstream request', async () => {
  // A real pending timer models an active socket while AbortSignal expires.
  await assert.rejects(readDensity({timeoutMs: 10, fetchImpl: (_, {signal}) => new Promise((resolve,reject) => {
    const timer = setTimeout(resolve, 500);
    signal.addEventListener('abort', () => { clearTimeout(timer); reject(signal.reason); });
  })}), /timeout/i);
});
