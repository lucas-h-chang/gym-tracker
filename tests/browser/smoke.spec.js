const { test, expect } = require('@playwright/test');
const fs = require('node:fs');
const now = new Date('2026-09-16T18:00:00Z');
const predictionRows = Array.from({length:64},(_,i)=>({slot_ts:new Date(Date.parse('2026-09-16T14:00:00Z')+i*900000).toISOString(),pct:60}));
async function mockNetwork(page, {live={}, weeklyDelay=false, summary=[], readings=[{timestamp:'2026-09-16T17:45:00Z',percent_full:50,sensor_ok:true}]}={}) {
  await page.clock.install({time:now});
  await page.route('https://fonts.**', route=>route.abort());
  await page.route('**/_vercel/insights/script.js', route=>route.fulfill({body:'',contentType:'application/javascript'}));
  // Keep the chart real, but serve its pinned file locally in tests.
  await page.route('https://cdn.jsdelivr.net/**', route=>route.fulfill({path:'node_modules/chart.js/dist/chart.umd.js',contentType:'application/javascript'}));
  await page.route('**/api/live-capacity', route=>route.fulfill({json:{capacity_pct:50,recorded_at:now.toISOString(),sensor_ok:true,source:'density',...live}}));
  let releaseWeekly;
  const gate = new Promise(resolve=>{releaseWeekly=resolve;});
  await page.route('**/rest/v1/**', async route=>{
    const url=route.request().url();
    if (url.includes('/weekly_averages')) {
      if (weeklyDelay) await gate;
      return route.fulfill({json:[]});
    }
    if (url.includes('/today_summary')) return route.fulfill({json:summary});
    if (url.includes('/capacity_log')) return route.fulfill({json:readings});
    return route.fulfill({json:predictionRows});
  });
  return releaseWeekly;
}
test('first chart paint does not wait for weekly averages',async({page})=>{
  const errors=[];page.on('pageerror',err=>errors.push(err.message));
  const release=await mockNetwork(page,{weeklyDelay:true});
  await page.goto('/');
  await expect(page.locator('body')).not.toHaveClass(/loading/);
  await expect(page.locator('#insight-pct')).toHaveText('50%');
  await expect(page.locator('#pred-chart')).toBeVisible();
  expect(errors).toEqual([]);
  release();
});

for (const width of [1280, 390]) test(`sensor warning appears above today's graph at ${width}px`, async ({page}) => {
  await page.setViewportSize({width, height:900});
  await mockNetwork(page, {readings:[
    {timestamp:'2026-09-16T17:15:00Z',percent_full:2,sensor_ok:false},
    {timestamp:'2026-09-16T17:30:00Z',percent_full:3,sensor_ok:false},
  ]});
  await page.goto('/');
  await expect(page.locator('body')).not.toHaveClass(/loading/);
  const warning = page.locator('#sensor-warning');
  await expect(warning).toBeVisible();
  await expect(warning).toHaveText('Sensor error or outage may be occurring');
  const box = await warning.boundingBox();
  const chart = await page.locator('#pred-chart').boundingBox();
  expect(box.y + box.height).toBeLessThanOrEqual(chart.y);
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(width);
  expect(await warning.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
  await page.screenshot({path:`/tmp/bear-meter-sensor-warning-${width}.png`});
  await page.getByRole('button', {name:'Tomorrow', exact:true}).click();
  await expect(warning).toBeHidden();
  await page.getByRole('button', {name:'Today', exact:true}).click();
  await expect(warning).toBeVisible();
});
for (const [name,live,label] of [
  ['invalid sensor',{sensor_ok:false},'RSF sensor offline'],
  ['stale reading',{recorded_at:'2026-09-16T16:00:00Z'},'Live reading is stale'],
  ['upstream failure',{source:'cache_stale'},'Live reading is stale'],
]) test(name+' never displays a live percentage',async({page})=>{
  await mockNetwork(page,{live});await page.goto('/');
  await expect(page.locator('body')).not.toHaveClass(/loading/);
  await expect(page.locator('#insight-pct')).toHaveText('—');
  await expect(page.locator('#insight-cap-label')).toHaveText(label);
});
test('stale corrections fall back to the baseline',async({page})=>{
  await mockNetwork(page,{summary:[{computed_at:'2026-09-16T16:00:00Z',blend_weight:1,similarity_preds:[{x:12,y:1,w:1}]}]});
  await page.goto('/');
  await expect(page.locator('body')).not.toHaveClass(/loading/);
  expect(await page.evaluate(()=>correctedPoints(window._insightData))).toEqual([]);
});

test('returning to a tab updates the live card and chart together',async({page})=>{
  await mockNetwork(page);
  await page.goto('/');
  await expect(page.locator('#insight-pct')).toHaveText('50%');
  await page.clock.runFor(2000);

  await page.route('**/api/live-capacity',route=>route.fulfill({json:{
    capacity_pct:70, recorded_at:new Date(now.getTime()+2000).toISOString(),
    sensor_ok:true, source:'density',
  }}));
  await page.evaluate(()=>window.dispatchEvent(new Event('focus')));
  await expect(page.locator('#insight-pct')).toHaveText('70%');
  await expect.poll(()=>page.evaluate(()=>predChart.data.datasets[0].data.at(-1).y)).toBe(70);
});

test('a failed startup request recovers without reloading', async ({page}) => {
  await mockNetwork(page);
  let attempts = 0;
  let recovered = false;
  await page.route('**/rest/v1/today_summary?**', route => {
    attempts++;
    return !recovered
      ? route.fulfill({status:503, json:{message:'temporarily unavailable'}})
      : route.fulfill({json:[]});
  });
  await page.goto('/');
  await expect(page.locator('#data-warning')).toContainText('Could not load gym data:');
  recovered = true;
  await page.clock.runFor(5100);
  await expect(page.locator('#insight-pct')).toHaveText('50%');
  await expect(page.locator('#insight-trend-val')).toHaveText('Filling up');
  await expect(page.locator('#data-warning')).toBeHidden();
  await expect(page.locator('#pred-chart')).toBeVisible();
  expect(attempts).toBeGreaterThanOrEqual(2);
});

test('a temporary live failure preserves a fresh reading and still expires it', async ({page}) => {
  await mockNetwork(page);
  await page.goto('/');
  await expect(page.locator('#insight-pct')).toHaveText('50%');
  await page.route('**/api/live-capacity', route => route.fulfill({status:502,json:{error:'temporary failure'}}));
  await page.clock.runFor(31000);
  await expect(page.locator('#insight-pct')).toHaveText('50%');
  await page.clock.runFor(100000);
  await expect(page.locator('#insight-pct')).toHaveText('—');
});

test('a synchronous live-request failure cannot block later polls', async ({page}) => {
  await mockNetwork(page);
  let liveRequests = 0;
  page.on('request', request => {
    if (request.url().endsWith('/api/live-capacity')) liveRequests++;
  });
  await page.addInitScript(() => {
    const originalFetch = window.fetch;
    let failOnce = true;
    window.fetch = function (...args) {
      if (args[0] === '/api/live-capacity' && failOnce) {
        failOnce = false;
        throw new TypeError('Simulated failure before fetch returns a promise');
      }
      return originalFetch.apply(this, args);
    };
  });
  await page.goto('/');
  await expect(page.locator('body')).not.toHaveClass(/loading/);
  await expect(page.locator('#insight-pct')).toHaveText('—');
  expect(liveRequests).toBe(0);
  await page.clock.runFor(31000);
  await expect(page.locator('#insight-pct')).toHaveText('50%');
  expect(liveRequests).toBe(1);
});
