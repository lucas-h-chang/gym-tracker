const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({ Date });
vm.runInContext(fs.readFileSync(require.resolve('../docs/js/calendar.js'), 'utf8'), context);

function warning(readings, day = '2026-10-06', nowHour = 23) {
  context.rows = readings.map(([time, pct, sensorOk = true]) => ({
    timestamp: `${day}T${time}:00-07:00`, percent_full: pct, sensor_ok: sensorOk,
  }));
  context.day = day;
  context.nowHour = nowHour;
  return vm.runInContext('hasSensorWarning(rows, day, nowHour)', context);
}

test('two consecutive low readings trigger even if already flagged, and remain after recovery', () => {
  assert.equal(warning([['16:15', 2, false], ['16:30', 3, false], ['16:45', 70]]), true);
  assert.equal(warning([['16:15', 29.9], ['16:30', 0]]), true);
});
test('single, interrupted, missing, duplicate and invalid readings do not form a pair', () => {
  for (const rows of [
    [['16:15', 2]], [['16:15', 2], ['16:30', 30], ['16:45', 2]],
    [['16:15', 2], ['16:45', 2]], [['16:15', 2], ['16:15', 2]],
    [['16:15', null], ['16:30', 2]], [['16:15', -1], ['16:30', 2]],
  ]) assert.equal(warning(rows), false);
});
test('both readings must fall within the inclusive 15-minute margins', () => {
  assert.equal(warning([['07:00', 2], ['07:15', 2]]), false);
  assert.equal(warning([['07:15', 2], ['07:30', 2]]), true);
  assert.equal(warning([['22:30', 2], ['22:45', 2]]), true);
  assert.equal(warning([['22:45', 2], ['23:00', 2]]), false);
});
test('summer, weekend, closures and current time use the real calendar', () => {
  assert.equal(warning([['19:45', 2], ['20:00', 2]], '2026-07-06'), false);
  assert.equal(warning([['17:30', 2], ['17:45', 2]], '2026-10-10'), true);
  assert.equal(warning([['17:45', 2], ['18:00', 2]], '2026-10-10'), false);
  assert.equal(warning([['10:00', 2], ['10:15', 2]], '2026-08-24'), false);
  assert.equal(warning([['16:15', 2], ['16:30', 2]], '2026-10-06', 16.25), false);
});
