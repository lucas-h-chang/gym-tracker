const { test, expect } = require('@playwright/test');
const fs = require('node:fs');
const vm = require('node:vm');
// Calendar execution has no DOM dependency: test the real generated module
// across DST and closure boundaries without controlling the user's browser.
test('web calendar and bins preserve DST, closure, invalid readings, and real zeros', async () => {
  const context = vm.createContext({Date, console});
  vm.runInContext(fs.readFileSync('docs/js/calendar.js','utf8'),context);
  expect(vm.runInContext("slotTsToKey('2026-03-08T14:00:00Z')",context)).toBe('2026-03-08_07:00');
  expect(vm.runInContext("slotTsToKey('2026-11-01T16:00:00Z')",context)).toBe('2026-11-01_08:00');
  expect(vm.runInContext("getOpenHours(1, new Date('2026-08-24T12:00:00')).close",context)).toBe(0);
  context.rows=[{timestamp:'2026-09-16T14:00:00Z',percent_full:0,sensor_ok:true},{timestamp:'2026-09-16T14:00:00Z',percent_full:90,sensor_ok:false}];
  expect(vm.runInContext("buildTodayActuals(rows,'2026-09-16')[0].y",context)).toBe(0);
});
