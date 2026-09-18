const { test, expect } = require('@playwright/test');
const fs = require('node:fs');
const now = new Date('2026-09-16T18:00:00Z');
const predictionRows = Array.from({length:64},(_,i)=>({slot_ts:new Date(Date.parse('2026-09-16T14:00:00Z')+i*900000).toISOString(),pct:60}));
async function mockNetwork(page, {live={}, weeklyDelay=false, summary=[]}={}) {
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
    if (url.includes('/capacity_log')) return route.fulfill({json:[{timestamp:'2026-09-16T17:45:00Z',percent_full:50,sensor_ok:true}]});
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
