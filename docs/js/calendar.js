const _ptBounds = {};
function _getPTBounds(year) {
  if (_ptBounds[year]) return _ptBounds[year];
  const mar1       = Date.UTC(year, 2, 1);
  const mar1dow    = new Date(mar1).getUTCDay();
  const firstSunMar = mar1 + (mar1dow === 0 ? 0 : (7 - mar1dow)) * 86400000;
  const dstStart   = firstSunMar + 7 * 86400000 + 10 * 3600000;
  const nov1       = Date.UTC(year, 10, 1);
  const nov1dow    = new Date(nov1).getUTCDay();
  const firstSunNov = nov1 + (nov1dow === 0 ? 0 : (7 - nov1dow)) * 86400000;
  const dstEnd     = firstSunNov + 9 * 3600000;
  return (_ptBounds[year] = { dstStart, dstEnd });
}
function slotTsToKey(slotTs) {
  const ms = Date.parse(slotTs);
  const { dstStart, dstEnd } = _getPTBounds(new Date(ms).getUTCFullYear());
  const ptMs = ms + ((ms >= dstStart && ms < dstEnd) ? -7 : -8) * 3600000;
  const d = new Date(ptMs);
  return (
    d.getUTCFullYear() + '-' +
    String(d.getUTCMonth() + 1).padStart(2, '0') + '-' +
    String(d.getUTCDate()).padStart(2, '0') + '_' +
    String(d.getUTCHours()).padStart(2, '0') + ':' +
    String(d.getUTCMinutes()).padStart(2, '0')
  );
}

function slotToLabel(h) {
  const hour   = Math.floor(h);
  const min    = Math.round((h - hour) * 60);
  const period = hour >= 12 ? 'PM' : 'AM';
  const dh     = hour > 12 ? hour - 12 : (hour === 0 ? 12 : hour);
  return min === 0 ? `${dh}:00 ${period}` : `${dh}:${String(min).padStart(2,'0')} ${period}`;
}

const RANGE_DB_TO_DISPLAY = {
  'last_week': 'Last 7 days', 'last_month': 'Last month',
  'last_6_months': 'Last 6 months', 'last_year': 'Last year',
  'all_time': 'All time', 'this_semester': 'This semester',
  // All-time slices partitioned by academic period, for the "compared to
  // usual <Day>s" card (see weekly_builder.py period_type / _emit_day_records).
  'all_summers': 'All summers', 'all_semesters': 'All semesters',
  'all_breaks': 'All breaks',
};

function buildTodayActuals(rows, today) {
  const bins = {};
  rows.forEach(row => {
    if (row.sensor_ok === false || !Number.isFinite(row.percent_full) || row.percent_full < 0) return;
    const d  = new Date(row.timestamp);
    const pt = new Date(d.toLocaleString('en-US', { timeZone: 'America/Los_Angeles' }));
    const ds = `${pt.getFullYear()}-${String(pt.getMonth()+1).padStart(2,'0')}-${String(pt.getDate()).padStart(2,'0')}`;
    if (ds !== today) return;
    const total   = pt.getHours() * 60 + pt.getMinutes();
    const rounded = Math.round(total / 15) * 15;
    const h = Math.floor(rounded / 60), m = rounded % 60;
    const key = h + m / 60;
    if (!bins[key]) bins[key] = { sum: 0, n: 0, h, m };
    bins[key].sum += row.percent_full;
    bins[key].n   += 1;
  });
  return Object.values(bins)
    .sort((a, b) => (a.h + a.m/60) - (b.h + b.m/60))
    .map(b => ({ x: b.h + b.m/60, y: Math.round(b.sum/b.n*10)/10, label: formatTime(b.h, b.m) }));
}


function getPTNow() {
  return new Date(new Date().toLocaleString('en-US', { timeZone: 'America/Los_Angeles' }));
}

// academic_calendar.py's SUMMER_RANGES. Generated from academic_calendar.py by gym-tracker/sync_calendar.py. Add a
// year THERE and re-run it; a hand edit here is reverted by the next run and
// fails test_calendar_mirrors.py in the meantime.
// >>> GENERATED SUMMER_RANGES — from academic_calendar.py via sync_calendar.py; do not edit by hand
const SUMMER_RANGES = [
  ['2024-05-10', '2024-08-24'],
  ['2025-05-16', '2025-08-23'],
  ['2026-05-15', '2026-08-22'],
  ['2027-05-14', '2027-08-21'],
];
// <<< END GENERATED SUMMER_RANGES

function isSummerHours(date) {
  const ds = (date instanceof Date ? date : new Date(date)).toLocaleDateString('en-CA');
  return SUMMER_RANGES.some(([s, e]) => ds >= s && ds <= e);
}

// academic_calendar.py's BREAK_RANGES (winter + spring + summer break).
// Generated from academic_calendar.py by gym-tracker/sync_calendar.py. Used only by periodTypeOf() to classify TODAY
// for the "compared to usual <Day>s" card, so it mirrors weekly_builder.py's
// period_type() exactly (summer peeled off first via isSummerHours, then a
// non-in-session day is a "break").
// >>> GENERATED BREAK_RANGES — from academic_calendar.py via sync_calendar.py; do not edit by hand
const BREAK_RANGES = [
  // winter
  ['2020-12-18', '2021-01-18'],
  ['2021-12-17', '2022-01-17'],
  ['2022-12-16', '2023-01-16'],
  ['2023-12-15', '2024-01-15'],
  ['2024-12-20', '2025-01-20'],
  ['2025-12-19', '2026-01-19'],
  ['2026-12-18', '2027-01-18'],
  ['2027-12-17', '2028-01-17'],
  // spring
  ['2021-03-20', '2021-03-28'],
  ['2022-03-19', '2022-03-27'],
  ['2023-03-25', '2023-04-02'],
  ['2024-03-23', '2024-03-31'],
  ['2025-03-22', '2025-03-30'],
  ['2026-03-21', '2026-03-29'],
  ['2027-03-20', '2027-03-28'],
  ['2028-03-25', '2028-04-02'],
  // summer
  ['2021-05-14', '2021-08-24'],
  ['2022-05-13', '2022-08-23'],
  ['2023-05-12', '2023-08-22'],
  ['2024-05-10', '2024-08-27'],
  ['2025-05-16', '2025-08-26'],
  ['2026-05-15', '2026-08-25'],
  ['2027-05-14', '2027-08-24'],
];
// <<< END GENERATED BREAK_RANGES

function isSemesterDay(date) {
  const ds = (date instanceof Date ? date : new Date(date)).toLocaleDateString('en-CA');
  return !BREAK_RANGES.some(([s, e]) => ds >= s && ds <= e);
}

// summer / semester / break, matching weekly_builder.py period_type().
function periodTypeOf(date) {
  if (isSummerHours(date))  return 'summer';
  if (!isSemesterDay(date)) return 'break';
  return 'semester';
}

// period_type -> the weekly_averages range_type carrying that all-time slice.
const PERIOD_RANGE_TYPE = {
  summer: 'all_summers', semester: 'all_semesters', break: 'all_breaks',
};

// The word the comparison card uses to name which baseline it picked, so the
// header says WHAT it is comparing against instead of a flat "usual".
// Mirrors RSFApp2.0 TimeUtils.swift periodLabel().
const PERIOD_LABEL = { summer: 'SUMMER', semester: 'SEMESTER', break: 'BREAK' };

// academic_calendar.py's CLOSURES — days the RSF is shut entirely. Caltopia
// takes over the building the Sunday and Monday before fall instruction begins
// (Tuesday too in 2026). Generated from academic_calendar.py by gym-tracker/sync_calendar.py.
// >>> GENERATED CLOSURES — from academic_calendar.py via sync_calendar.py; do not edit by hand
const CLOSURES = [
  ["2021-08-22", "2021-08-23", "Caltopia"],
  ["2022-08-21", "2022-08-22", "Caltopia"],
  ["2023-08-20", "2023-08-21", "Caltopia"],
  ["2024-08-25", "2024-08-26", "Caltopia"],
  ["2025-08-24", "2025-08-25", "Caltopia"],
  ["2026-08-23", "2026-08-25", "Caltopia"],
  ["2027-08-22", "2027-08-23", "Caltopia"],
  ["2022-11-24", "2022-11-24", "Thanksgiving"],
  ["2022-12-25", "2022-12-25", "Christmas"],
  ["2023-11-23", "2023-11-23", "Thanksgiving"],
  ["2023-12-25", "2023-12-25", "Christmas"],
  ["2024-11-28", "2024-11-28", "Thanksgiving"],
  ["2024-12-24", "2024-12-25", "Christmas"],
  ["2025-01-01", "2025-01-01", "New Year's Day"],
  ["2025-11-27", "2025-11-27", "Thanksgiving"],
  ["2025-12-24", "2025-12-25", "Christmas"],
  ["2026-01-01", "2026-01-01", "New Year's Day"],
  ["2026-11-26", "2026-11-26", "Thanksgiving"],
  ["2026-12-24", "2026-12-25", "Christmas"],
  ["2027-01-01", "2027-01-01", "New Year's Day"],
  ["2027-11-25", "2027-11-25", "Thanksgiving"],
  ["2027-12-24", "2027-12-25", "Christmas"],
  ["2028-01-01", "2028-01-01", "New Year's Day"],
];
// <<< END GENERATED CLOSURES

function closureReason(date) {
  const ds = (date instanceof Date ? date : new Date(date)).toLocaleDateString('en-CA');
  const hit = CLOSURES.find(([s, e]) => ds >= s && ds <= e);
  return hit ? hit[2] : null;
}

function getOpenHours(jsDay, date) {
  // Empty interval on a closure day, matching academic_calendar.get_open_hours():
  // every `open <= h < close` test is false for all 24 hours, so the day reads
  // as closed everywhere without special-casing each caller.
  if (date != null && closureReason(date)) return { open: 0, close: 0 };
  const summer = date != null && isSummerHours(date);
  if (jsDay === 6) return { open: 8, close: 18 };
  if (jsDay === 0) return { open: 8, close: summer ? 20 : 23 };
  return { open: 7, close: summer ? 20 : 23 };
}

// The next date the RSF actually opens. "Opens tomorrow at 8:00 AM" is wrong
// on the Saturday night before a Caltopia closure, and wrong again on each day
// inside a multi-day one, so the reopen day is searched for rather than
// assumed to be tomorrow. Bounded so a bad CLOSURES entry can't spin.
function nextOpenDate(from) {
  const d = new Date(from.getTime());
  for (let i = 0; i < 14; i++) {
    d.setDate(d.getDate() + 1);
    const { open, close } = getOpenHours(d.getDay(), d);
    if (close > open) return { date: new Date(d.getTime()), open, isTomorrow: i === 0 };
  }
  return null;
}

function formatHour(h) {
  if (h === 0 || h === 24) return '12 AM';
  if (h === 12) return '12 PM';
  return h > 12 ? `${h - 12} PM` : `${h} AM`;
}

function formatTime(h, m) {
  const period = h >= 12 ? 'PM' : 'AM';
  const dh = h > 12 ? h - 12 : (h === 0 ? 12 : h);
  return m === 0 ? `${dh}:00 ${period}` : `${dh}:${String(m).padStart(2, '0')} ${period}`;
}

function todayPT() {
  const pt = getPTNow();
  return `${pt.getFullYear()}-${String(pt.getMonth()+1).padStart(2,'0')}-${String(pt.getDate()).padStart(2,'0')}`;
}
