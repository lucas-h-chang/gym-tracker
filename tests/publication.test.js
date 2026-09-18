const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { PGlite } = require('@electric-sql/pglite');

test('publication migrations: atomic rollback, monotonic writes, roles, claims, and audit', async () => {
  const db = new PGlite();
  try {
    await db.exec(`
      create role anon; create role authenticated; create role service_role bypassrls;
      create table capacity_log(timestamp timestamptz, people_count int, sensor_ok boolean);
      create table predictions(slot_ts timestamptz primary key, pct real);
      create table today_summary(date text primary key, similarity_preds jsonb, blend_weight real, computed_at timestamptz);
      create table weekly_averages(day_of_week text, hour_slot float8, avg_pct float8, range_type text, semester_only boolean);
      grant all on all tables in schema public to service_role;
    `);
    for (const name of ['012_prediction_snapshots.sql','013_snapshot_bare_curve.sql','014_reliable_publication.sql','015_accuracy_publication_horizon.sql','016_migration_history.sql']) {
      await db.exec(fs.readFileSync(`migrations/${name}`, 'utf8'));
    }
    const days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];
    const rows = days.flatMap(day => Array.from({length:16}, (_,i)=>({day_of_week:day,hour_slot:8+i/4,avg_pct:40,range_type:'all_time',semester_only:false})));
    const publish = (records, stamp) => db.query('select publish_weekly_averages($1::jsonb,$2::timestamptz) as ok',[JSON.stringify(records),stamp]);
    await db.exec('set role service_role');
    assert.equal((await publish(rows,'2026-09-16T08:00:00Z')).rows[0].ok,true);
    assert.equal((await publish(rows,'2026-09-15T08:00:00Z')).rows[0].ok,false);
    await assert.rejects(publish([],'2026-09-17T08:00:00Z'), /Incomplete/);
    await db.exec('reset role');
    // Force a failure after DELETE, inside the INSERT. The previous data must survive.
    await db.exec(`create function fail_insert() returns trigger language plpgsql as $$begin raise exception 'injected insert failure'; end$$;
      create trigger fail_insert before insert on weekly_averages for each row execute function fail_insert();`);
    await assert.rejects(publish(rows,'2026-09-17T08:00:00Z'), /injected/);
    assert.equal((await db.query('select count(*)::int as n from weekly_averages')).rows[0].n,112);
    assert.equal((await db.query("select built_at from pipeline_status where product='weekly_averages'")).rows[0].built_at.toISOString(),'2026-09-16T08:00:00.000Z');
    await db.exec('drop trigger fail_insert on weekly_averages');
    await db.exec('set role anon');
    await assert.rejects(publish(rows,'2026-09-18T08:00:00Z'), /permission denied/);
    await assert.rejects(db.query('select * from notification_deliveries'), /permission denied/);
    await assert.rejects(db.query('select * from prediction_accuracy'), /permission denied/);
    await db.exec('reset role');
    const claim = () => db.query("select claim_notification('device','workout','2026-09-16T18:00:00Z') as ok");
    assert.equal((await claim()).rows[0].ok,true);
    assert.equal((await claim()).rows[0].ok,false);
    assert.equal((await db.query("select publish_today_summary('2026-09-16','[]','2026-09-16T18:00:00Z','{}') as ok")).rows[0].ok,true);
    assert.equal((await db.query("select publish_today_summary('2026-09-16','[]','2026-09-16T17:00:00Z','{}') as ok")).rows[0].ok,false);
    await db.exec(`insert into prediction_snapshots(date,computed_at,preds,base,curve,last_slot)
      values ('2026-09-16','2026-09-16T18:05:00Z','[]','{"44":20,"45":40}','{"44":22,"45":42}',40);
      insert into capacity_log values ('2026-09-16T18:15:00Z',60,true);`);
    const points = (await db.query('select slot,pct,horizon_h from prediction_accuracy')).rows;
    assert.equal(points.length,1); assert.equal(points[0].slot,45); assert.equal(Number(points[0].pct),40);
    assert.ok(Math.abs(Number(points[0].horizon_h)-1/6)<1e-8);
    const preds = Array.from({length:1000}, (_,i)=>({slot_ts:new Date(Date.parse('2026-09-16T14:00:00Z')+i*900000).toISOString(),pct:50,curve_pct:51,curve_version:'v1'}));
    const pubPred = stamp => db.query('select publish_predictions($1,$2,$3) as ok',[JSON.stringify(preds),stamp,'{}']);
    assert.equal((await pubPred('2026-09-16T08:00:00Z')).rows[0].ok,true);
    assert.equal((await pubPred('2026-09-15T08:00:00Z')).rows[0].ok,false);
    assert.equal((await db.query('select count(*)::int as n from predictions')).rows[0].n,1000);
  } finally { await db.close(); }
});
