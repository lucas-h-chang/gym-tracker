const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const hours=require('../api/_hours');
const sensor=require('../api/_sensor');

test('scraper confirms a low pair, invalidates both, stays down at six people, then recovers',async()=>{
  const rows=[];
  let time=new Date('2026-10-07T16:15:10-07:00'),count=60;
  const sb={from:()=>{
    let patch,targets,floor;
    const q={select:()=>q,gte:()=>q,order:()=>q,
      lte:(key,value)=>{floor=value;return q;},
      limit:async()=>({data:rows,error:null}),
      insert:async row=>{rows.push(row);return {error:null};},
      update:value=>{patch=value;return q;},
      in:(key,values)=>{targets=values;return q;},
      then:resolve=>{for(const row of rows)if(targets.includes(row.timestamp)&&row.people_count<=floor)Object.assign(row,patch);
        return Promise.resolve({error:null}).then(resolve);},
    };return q;
  }};
  class Clock extends Date {constructor(...args){super(...(args.length?args:[time.getTime()]));}}
  const context=vm.createContext({module:{exports:{}},Date:Clock,process:{env:{SCRAPE_SECRET:'test'}},console,
    require:path=>{
      if(path==='@supabase/supabase-js')return {createClient:()=>sb};
      if(path==='./_hours')return {...hours,ptNow:()=>hours.ptNow(time)};
      if(path==='./_sensor')return sensor;
      if(path==='./_density')return {readDensity:async()=>count};
      if(path==='./_supabase_retry')return require('../api/_supabase_retry');
      throw new Error(path);
    }});
  vm.runInContext(fs.readFileSync(require.resolve('../api/scrape'),'utf8'),context);
  async function scrape(at,value){
    time=new Date(at);count=value;
    const res={code:200,setHeader(){},status(code){this.code=code;return this;},json(body){this.body=body;return this;}};
    await context.module.exports({method:'GET',headers:{'x-scrape-secret':'test'}},res);
    assert.equal(res.code,200);return res.body;
  }
  await scrape('2026-10-07T16:15:10-07:00',60);
  assert.equal((await scrape('2026-10-07T16:30:10-07:00',2)).sensor_ok,true);
  assert.equal((await scrape('2026-10-07T16:45:10-07:00',6)).sensor_ok,false);
  assert.deepEqual(rows.map(row=>row.sensor_ok),[true,false,false]);
  assert.equal((await scrape('2026-10-07T17:00:10-07:00',7)).sensor_ok,false);
  assert.equal((await scrape('2026-10-07T17:15:10-07:00',50)).sensor_ok,true);
  assert.deepEqual(rows.map(row=>row.people_count),[60,2,6,7,50]);
});
