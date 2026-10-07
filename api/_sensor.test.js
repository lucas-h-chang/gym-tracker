const assert = require('node:assert/strict');
const { evaluateSensorReading, isSensorStalled, invalidatePriorReadings } = require('./_sensor');
const fixtures = require('../tests/fixtures/sensor-rule.json');
(async () => {
  for (const fixture of fixtures) {
    const result = evaluateSensorReading(fixture.rows, fixture.count, new Date(fixture.now));
    assert.equal(result.stalled, fixture.stalled, fixture.name);
    if (result.stalled) assert.ok(result.timestamps.length > 0);
    console.log('PASS', fixture.name);
  }
  const now = new Date('2026-10-07T17:30:10-07:00');
  const rows = [{timestamp:'2026-10-07T17:15:10-07:00',people_count:2}];
  const query = {from:()=>query,select:()=>query,gte:()=>query,lte:()=>query,order:()=>query,
    limit:async()=>({data:rows,error:null})};
  assert.equal((await isSensorStalled(query,2,now)).stalled,true);
  query.limit=async()=>({data:null,error:{message:'boom'}});
  assert.equal((await isSensorStalled(query,2,now)).stalled,false);
  query.limit=async()=>{throw new Error('connection failed');};
  assert.equal((await isSensorStalled(query,2,now)).stalled,false);
  const stall=evaluateSensorReading(rows,2,now);
  let attempts=0;
  const update={from:()=>update,update:value=>{assert.deepEqual(value,{sensor_ok:false});return update;},
    in:(key,values)=>{assert.equal(key,'timestamp');assert.deepEqual(values,stall.timestamps);return update;},
    lte:async(key,value)=>{assert.equal(key,'people_count');assert.equal(value,37.5);
      attempts++;return {error:attempts<2?{message:'temporary'}:null};}};
  await invalidatePriorReadings(update,stall,{sleep:async()=>{}});
  assert.equal(attempts,2);
  await invalidatePriorReadings(update,{stalled:false});
  assert.equal(attempts,2);
  update.lte=async()=>({error:{message:'persistent'}});
  await assert.rejects(invalidatePriorReadings(update,stall,{sleep:async()=>{}}));
  console.log('All sensor fixtures, fail-open reads, and bounded invalidation checks pass');
})().catch(error=>{console.error(error);process.exit(1)});
