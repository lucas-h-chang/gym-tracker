// api/history.js - read-only history feed for docs/history.html
//
// WHY THIS EXISTS AS A SERVERLESS FUNCTION AND NOT A DIRECT SUPABASE READ
// -----------------------------------------------------------------------
// migrations/006_lock_down_history.sql deliberately walled the raw history off
// from the publishable anon key: `capacity_log` has an RLS policy limiting
// anon to `timestamp > now() - interval '3 days'`, and the `day_profiles` view
// had its grant revoked outright. The history browser needs 2022-to-now, so it
// cannot use the key embedded in the frontend - and re-opening the tables would
// undo 006 and hand the whole training set back to anyone with the key.
//
// This endpoint keeps that lockdown intact: it reads as service_role (which
// bypasses RLS, same as scrape.js / live-capacity.js) and only ever hands the
// browser ONE MONTH of already-bucketed quarter-hour averages at a time. There
// is no "give me everything" mode, and the CDN cache below means a month costs
// roughly one origin read per day regardless of traffic. No migration needed.
//
// Routes
//   GET /api/history?meta=1            -> { first_date, last_date, today }
//   GET /api/history?month=YYYY-MM     -> { month, days: { date: {n, pts} } }
//
// `pts` is [[hour_slot, pct], ...] on the same quarter-hour grid the website's
// "actual so far today" line uses, bucketed with the same round-to-nearest rule
// as index.html's buildTodayActuals() (see the note on SLOT_MINUTES below).

const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

const PAGE = 1000;          // PostgREST page size; a month is ~2,900 readings
const DATA_START = '2022-01-01';

// Bucketing is round-to-NEAREST 15 minutes, matching index.html's
// buildTodayActuals() (`Math.round(total / 15) * 15`) rather than the floor
// rule used elsewhere in the pipeline. That is deliberate: this page renders
// the same "actual" line the live site draws, so it has to bin the same way or
// the two products would disagree about the same day. See the slot-bucketing
// note in the project memory for why the two rules coexist.
const SLOT_MINUTES = 15;

// Single formatter, reused: constructing Intl.DateTimeFormat per row dominates
// the runtime otherwise. h23 (not hour12:false) so midnight is "00", never "24".
const PT = new Intl.DateTimeFormat('en-CA', {
  timeZone: 'America/Los_Angeles',
  year: 'numeric', month: '2-digit', day: '2-digit',
  hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
});

function ptDateAndSlot(iso) {
  const p = {};
  for (const part of PT.formatToParts(new Date(iso))) p[part.type] = part.value;
  const minutes = Number(p.hour) * 60 + Number(p.minute);
  const rounded = Math.round(minutes / SLOT_MINUTES) * SLOT_MINUTES;
  return {
    date: `${p.year}-${p.month}-${p.day}`,
    slot: rounded / 60,
  };
}

// Route-level cache policy, unless the request was authenticated - a response
// gated on a secret must stay out of the shared CDN cache, so in that mode the
// `private` header set in the handler wins and this is a no-op.
function setCache(res, value) {
  if (process.env.HISTORY_KEY) return;
  res.setHeader('Cache-Control', value);
}

function todayPT() {
  const p = {};
  for (const part of PT.formatToParts(new Date())) p[part.type] = part.value;
  return `${p.year}-${p.month}-${p.day}`;
}

// Fetch every capacity_log row in [gte, lt), paging past PostgREST's row cap.
// Paging rather than one limit=10000: Supabase can be configured with a
// db-max-rows ceiling, and a silent truncation here would show up as a day
// whose curve just stops halfway through the afternoon.
async function fetchRange(gteISO, ltISO) {
  const rows = [];
  for (let from = 0; ; from += PAGE) {
    const { data, error } = await supabase
      .from('capacity_log')
      .select('timestamp, percent_full')
      .gte('timestamp', gteISO)
      .lt('timestamp', ltISO)
      .order('timestamp', { ascending: true })
      .range(from, from + PAGE - 1);
    if (error) throw new Error(`capacity_log read failed: ${error.message}`);
    rows.push(...data);
    if (data.length < PAGE) return rows;
  }
}

async function handleMeta(res) {
  const [first, last] = await Promise.all([
    supabase.from('capacity_log').select('timestamp')
      .order('timestamp', { ascending: true }).limit(1).maybeSingle(),
    supabase.from('capacity_log').select('timestamp')
      .order('timestamp', { ascending: false }).limit(1).maybeSingle(),
  ]);
  if (first.error) throw new Error(first.error.message);
  if (last.error)  throw new Error(last.error.message);
  if (!first.data || !last.data) {
    return res.status(200).json({ first_date: null, last_date: null, today: todayPT() });
  }
  // Clamp to DATA_START so a stray pre-2022 row can't open up empty calendar
  // years; day_profiles applies the same cutoff.
  const firstDate = ptDateAndSlot(first.data.timestamp).date;
  return res.status(200).json({
    first_date: firstDate < DATA_START ? DATA_START : firstDate,
    last_date:  ptDateAndSlot(last.data.timestamp).date,
    today:      todayPT(),
  });
}

async function handleMonth(res, monthStr) {
  const m = /^(\d{4})-(\d{2})$/.exec(monthStr);
  if (!m) return res.status(400).json({ error: 'month must be YYYY-MM' });
  const year  = Number(m[1]);
  const month = Number(m[2]);           // 1-based
  if (month < 1 || month > 12 || year < 2000 || year > 2100) {
    return res.status(400).json({ error: 'month out of range' });
  }

  // Pad both ends by 12h of UTC so the window covers the PT month under either
  // DST offset. Over-fetched rows fall outside the month once bucketed to a PT
  // date and are dropped below, so the padding costs ~80 rows and no accuracy.
  const gte = new Date(Date.UTC(year, month - 1, 1) - 12 * 3600e3).toISOString();
  const lt  = new Date(Date.UTC(year, month,     1) + 12 * 3600e3).toISOString();

  const rows = await fetchRange(gte, lt);

  // date -> slot -> {sum, n}
  const byDate = new Map();
  const prefix = `${m[1]}-${m[2]}-`;
  for (const row of rows) {
    if (row.percent_full == null) continue;
    const { date, slot } = ptDateAndSlot(row.timestamp);
    if (!date.startsWith(prefix)) continue;      // padding spill
    let day = byDate.get(date);
    if (!day) byDate.set(date, (day = { n: 0, slots: new Map() }));
    day.n += 1;
    const bin = day.slots.get(slot);
    if (bin) { bin.sum += row.percent_full; bin.count += 1; }
    else     { day.slots.set(slot, { sum: row.percent_full, count: 1 }); }
  }

  const days = {};
  for (const [date, day] of byDate) {
    days[date] = {
      n: day.n,
      pts: [...day.slots.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([slot, bin]) => [slot, Math.round((bin.sum / bin.count) * 10) / 10]),
    };
  }

  // A finished month can never change, so it is cached for a day and served
  // stale for a week. The in-progress month has to pick up each 15-min scrape,
  // which is what makes the page "keep updating as days go on" with no rebuild.
  const isCurrent = monthStr === todayPT().slice(0, 7);
  setCache(res, isCurrent
    ? 'public, s-maxage=300, stale-while-revalidate=900'
    : 'public, s-maxage=86400, stale-while-revalidate=604800');

  return res.status(200).json({ month: monthStr, days, generated_at: new Date().toISOString() });
}

module.exports = async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Optional access gate. This endpoint is the one hole in the wall that
  // 006_lock_down_history.sql built around the raw history, so if HISTORY_KEY
  // is set in the Vercel environment it must be presented - same shape as
  // scrape.js's SCRAPE_SECRET (header, or ?k= so a bookmarked URL works).
  //
  // Deliberately opt-in rather than required: with the variable unset the page
  // works the moment it deploys, and the history is no more exposed than the
  // 55-or-so requests it would take to walk it a month at a time. Set the
  // variable to close that off.
  if (process.env.HISTORY_KEY) {
    const provided = (req.headers && req.headers['x-history-key']) || (req.query && req.query.k);
    if (provided !== process.env.HISTORY_KEY) {
      res.setHeader('Cache-Control', 'no-store');
      return res.status(401).json({ error: 'unauthorized' });
    }
    // A per-user secret must never be cached by the shared CDN.
    res.setHeader('Cache-Control', 'private, max-age=60');
  }

  try {
    if (req.query.meta) {
      // Bounded by the scrape cadence at the top end; an hour is plenty since
      // the frontend only uses this for calendar limits.
      setCache(res, 'public, s-maxage=3600, stale-while-revalidate=86400');
      return await handleMeta(res);
    }
    if (req.query.month) {
      return await handleMonth(res, String(req.query.month));
    }
    return res.status(400).json({ error: 'pass ?meta=1 or ?month=YYYY-MM' });
  } catch (err) {
    console.error('[history]', err);
    res.setHeader('Cache-Control', 'no-store');
    return res.status(500).json({ error: 'history read failed' });
  }
};
