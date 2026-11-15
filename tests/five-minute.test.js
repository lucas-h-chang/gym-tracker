const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { PGlite } = require('@electric-sql/pglite');
const { createHandler } = require('../api/scrape-5m');
const { insertSample } = require('../api/_five_minute');

function fixture(overrides = {}) {
  const reads = [], writes = [];
  let at = new Date('2026-11-16T12:05:08-08:00');
  const snapshot = { id: 5, computed_at: '2026-11-16T20:00:20Z', base: { 52: 60 }, preds: [] };
  const sb = { from(table) {
    const q = {
      select() { reads.push(table); return q; }, eq() { return q; },
      lte(key, value) { assert.ok(['computed_at', 'inserted_at'].includes(key)); assert.equal(value, at.toISOString()); return q; },
      order() { return q; }, async limit() { return { data: [snapshot], error: null }; },
      async upsert(row, options) {
        assert.equal(table, 'capacity_samples_5m');
        writes.push({ row, options }); return { error: null };
      },
    };
    assert.ok(['capacity_samples_5m', 'prediction_snapshots'].includes(table));
    return q;
  } };
  const env = { SCRAPE_SECRET: 'test', FIVE_MINUTE_EXPERIMENT_ENABLED: 'true', ...overrides.env };
  let sensorCalls = 0;
  const handler = createHandler({ env, now: () => at, makeClient: () => sb,
    read: async () => { sensorCalls++; return 82; }, sleep: async () => {}, ...overrides });
  async function call(req = {}) {
    const res = { code: 200, setHeader(key, value) { assert.equal(value, 'no-store'); },
      status(code) { this.code = code; return this; }, json(body) { this.body = body; return this; } };
    await handler({ method: 'GET', headers: { 'x-scrape-secret': 'test' }, ...req }, res);
    return res;
  }
  return { call, reads, writes, snapshot, setTime(value) { at = new Date(value); }, sensorCalls: () => sensorCalls };
}

test('experiment gates auth, opt-in, closures and summer evenings before any side effects', async () => {
  const f = fixture();
  assert.equal((await f.call({ method: 'POST' })).code, 405);
  assert.equal((await f.call({ headers: {} })).code, 401);
  f.setTime('2026-11-26T12:05:00-08:00');
  assert.equal((await f.call()).body.skipped, 'closed');
  f.setTime('2026-07-20T20:05:00-07:00');
  assert.equal((await f.call()).body.skipped, 'closed');
  assert.equal(f.sensorCalls(), 0); assert.equal(f.writes.length, 0);
  const disabled = fixture({ env: { SCRAPE_SECRET: 'test' } });
  assert.equal((await disabled.call()).body.skipped, 'experiment disabled');
  assert.equal(disabled.sensorCalls(), 0);
});

test('only writes its own table and freezes production before reading Density', async () => {
  const f = fixture();
  assert.equal((await f.call()).code, 200);
  assert.deepEqual(f.reads, ['prediction_snapshots']);
  assert.equal(f.writes[0].row.tick_at, '2026-11-16T20:05:00.000Z');
  assert.equal(f.writes[0].row.observed_at, '2026-11-16T20:05:08.000Z');
  assert.equal(f.writes[0].row.people_count, 82);
  assert.equal(f.writes[0].row.production.id, 5);
  assert.deepEqual(f.writes[0].options, { onConflict: 'tick_at', ignoreDuplicates: true });
});

test('missing/stale/failing control capture never discards the new sensor reading', async () => {
  const stale = fixture(); stale.snapshot.computed_at = '2026-11-16T18:00:00Z';
  assert.equal((await stale.call()).body.control_status, 'stale');
  const missing = fixture({ makeClient: () => ({ from: table => {
    if (table === 'capacity_samples_5m') return { upsert: async () => ({ error: null }) };
    const q = { select: () => q, eq: () => q, lte: () => q, order: () => q,
      limit: async () => ({ data: [], error: null }) }; return q;
  } }) });
  assert.equal((await missing.call()).body.control_status, 'missing');
  const failed = fixture({ makeClient: () => ({ from: table => {
    if (table === 'prediction_snapshots') throw new Error('gateway');
    assert.equal(table, 'capacity_samples_5m'); return { upsert: async () => ({ error: null }) };
  } }) });
  assert.equal((await failed.call()).body.control_status, 'error');
});

test('sensor failure writes nothing; real zero and above-capacity counts are preserved', async () => {
  const failed = fixture({ read: async () => { throw new Error('unavailable'); } });
  assert.equal((await failed.call()).code, 502); assert.equal(failed.writes.length, 0);
  for (const count of [0, 180]) {
    const f = fixture({ read: async () => count });
    assert.equal((await f.call()).code, 200); assert.equal(f.writes[0].row.people_count, count);
  }
});

test('ambiguous insert retries keep the original payload and use conflict-ignore', async () => {
  const rows = [], waits = []; let attempts = 0;
  const row = { tick_at: '2026-11-16T20:05:00Z', people_count: 82 };
  await insertSample({ from(table) {
    assert.equal(table, 'capacity_samples_5m'); return { async upsert(value, options) {
      rows.push(value); assert.equal(options.ignoreDuplicates, true);
      if (++attempts < 3) throw new Error('ambiguous timeout'); return { error: null };
    } };
  } }, row, { sleep: async ms => waits.push(ms) });
  assert.equal(rows.length, 3); assert.ok(rows.every(value => value === row));
  assert.deepEqual(waits, [500, 1000]);
});

test('five-minute migration enforces isolation, first-write wins, RLS, and immutable samples', async () => {
  const db = new PGlite();
  try {
    await db.exec(`create role anon; create role authenticated; create role service_role bypassrls;
      create table schema_migrations(version text primary key, name text);
      create table capacity_log(people_count int); insert into capacity_log values(99);`);
    await db.exec(fs.readFileSync('migrations/018_five_minute_collection.sql', 'utf8'));
    assert.equal((await db.query('select people_count from capacity_log')).rows[0].people_count, 99);
    assert.equal((await db.query("select name from schema_migrations where version='018'")).rows[0].name, 'five_minute_collection');
    const insert = count => db.query(`insert into capacity_samples_5m(tick_at,observed_at,people_count,production)
      values ('2026-11-16T20:05:00Z','2026-11-16T20:05:10Z',$1,'{"status":"missing"}')
      on conflict (tick_at) do nothing`, [count]);
    for (const role of ['anon', 'authenticated']) {
      await db.exec(`set role ${role}`);
      await assert.rejects(db.query('select * from capacity_samples_5m'), /permission denied/);
      await assert.rejects(insert(99), /permission denied/);
      await db.exec('reset role');
    }
    await db.exec('set role service_role');
    await insert(82); await insert(90);
    assert.equal((await db.query('select people_count from capacity_samples_5m')).rows[0].people_count, 82);
    await assert.rejects(db.query('update capacity_samples_5m set people_count=1'), /permission denied/);
    await assert.rejects(db.query('delete from capacity_samples_5m'), /permission denied/);
    await assert.rejects(db.query(`insert into capacity_samples_5m(tick_at,observed_at,people_count,production)
      values ('2026-11-16T20:10:01Z','2026-11-16T20:10:10Z',0,'{"status":"missing"}')`), /check constraint/);
    await assert.rejects(db.query(`insert into capacity_samples_5m(tick_at,observed_at,people_count,production)
      values ('2026-11-16T20:10:00Z','2026-11-16T20:10:10Z',0,'{}')`), /check constraint/);
  } finally { await db.close(); }
});
