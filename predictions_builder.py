"""
predictions_builder.py — compute 90-day curve-model predictions → Supabase predictions table.
Runs daily at midnight PT via daily.yml.
"""
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client

import curve_model as cm
from academic_calendar import classify_date, is_summer_day, get_open_hours
from supabase_io import parse_supabase_timestamps, paginated_fetch

PT  = ZoneInfo("America/Los_Angeles")
now = datetime.now(PT)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

BATCH_SIZE = 500

# is_summer_day/get_open_hours/SUMMER_RANGES live in academic_calendar.py
# (consolidated 2026-07-21 — see CLAUDE.md).

CORRECTION_DAYS       = 28   # trailing window for residual computation
CORRECTION_MIN_N      = 3    # min observations per (segment, regime, dow, hour, minute) cell to apply correction
CORRECTION_DECAY_HALFLIFE = 7  # days; residual weight = 0.5^(horizon/halflife), so the
                               # 28-day-trailing nowcast only moves the near-term days it
                               # can actually track and fades to the base curve far out.


def load_curves():
    with open('models/curves.json') as f:
        return json.load(f)


def _correction_segment(phase):
    """
    Coarser-than-baseline segment key for the evening correction only.

    The baseline curve is keyed on the fine-grained phase (e.g.
    summer_break_7) so June and July get their own distinct shapes. But a
    CORRECTION_DAYS=28 trailing window only contains ~2 same-weekday days in
    any single calendar month -- well under CORRECTION_MIN_N, so keying the
    correction on the same fine phase leaves almost every cell empty right
    when a month boundary makes the nowcast matter most (checked directly:
    July 2026 had exactly 2 Tuesdays in the trailing window). Pooling all
    break sub-phases back into one "break" bucket here gives ~4 same-weekday
    samples instead, without touching the baseline curve's own granularity.
    """
    if phase in ("winter_break", "spring_break") or phase.startswith("summer_break_"):
        return "break"
    return phase


def build_evening_correction(table):
    """
    Fetch the last CORRECTION_DAYS days of actuals, re-predict with the curve
    table, and return a dict keyed by (segment, regime, dow, hour, minute) →
    mean residual (pp), where segment = _correction_segment(phase) and
    regime = is_summer_day(date) (summer closes at 8pm, academic-year closes
    at 11pm on weekdays/Sunday). This is the curve model's own
    trailing-residual nowcast -- same mechanism the RF pipeline used (see git
    history), ported to correct the curve model's baseline instead so we get
    both the curve model's structurally-correct shape (no closed-hours
    extrapolation, no pooled-break averaging -- see
    academic_calendar.classify_date) and RF's ability to track "this stretch
    is running hotter/cooler than the multi-year average" (halflife_days=365
    makes the raw curve far too slow to pick that up on its own). See
    _correction_segment() for why the correction uses a coarser segment than
    the baseline curve's own phase.

    regime is in the key on both sides (built here off each source row's own
    date, applied in compute_predictions off each target slot's date) so a
    summer-hours residual can never correct an academic-hours target or vice
    versa -- without this, a trailing window that's entirely summer (e.g.
    the whole of July) stamps summer's ~8pm closing-crash residual onto an
    academic-hours target day that's still open until 11pm, producing a
    sharp unnatural dip at 7:45pm followed by a jump back to baseline at 8pm
    where no summer correction data exists. When a target's regime has no
    matching trailing-window data at all (e.g. an academic-hours target
    whose entire trailing window is summer), the correction dict simply has
    no cells for that regime and compute_predictions falls back cleanly to
    the base curve for those slots.

    No closing-slot trimming: the pre-close emptying-out (e.g. summer's ~40%
    at 7:45pm before the 8pm close) is real, regime-specific signal the
    halflife-365 base curve misses (it averages in busy academic-year
    evenings), and the regime key already keeps it from reaching an
    academic-hours target -- so it is kept, not discarded.

    The magnitude fade with forecast horizon is applied at prediction time
    (see compute_predictions / CORRECTION_DECAY_HALFLIFE), not here: a
    trailing-28-day residual tracks "this stretch is running hot/cool" only
    for the next ~week (verified by backtest: the flat correction improved
    day 1-7 evenings but added noise from day ~15 out, dragging the 90-day
    average below the raw base curve).

    minute is rounded to the nearest 15-min boundary (0/15/30/45) so scraped
    timestamps (which land off-quarter) align with the prediction slots.
    """
    lo = (now - timedelta(days=CORRECTION_DAYS)).isoformat()
    hi = now.isoformat()

    rows = paginated_fetch(sb, "capacity_log", "timestamp,percent_full", gte=lo, lte=hi, order="timestamp")

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    df['timestamp']    = parse_supabase_timestamps(df['timestamp'])
    df['percent_full'] = df['percent_full'].astype(float)
    df = df[df['percent_full'] > 0].dropna().reset_index(drop=True)

    df['date']   = df['timestamp'].dt.date
    df['dow']    = df['timestamp'].dt.dayofweek
    df['hour']   = df['timestamp'].dt.hour
    df['minute'] = (df['timestamp'].dt.minute // 15) * 15  # round to 0/15/30/45
    df['slot']   = df['hour'] * 4 + df['minute'] // 15
    df['segment'] = df['date'].map(classify_date).map(_correction_segment)
    df['regime'] = df['date'].map(is_summer_day)

    df['curve_pred'] = cm.predict(table, list(zip(df['date'], df['slot'])))
    df = df.dropna(subset=['curve_pred']).reset_index(drop=True)
    df['residual'] = df['percent_full'] - df['curve_pred']

    correction = {}
    for (seg, rg, dow, hr, mn), g in df.groupby(['segment', 'regime', 'dow', 'hour', 'minute']):
        if len(g) >= CORRECTION_MIN_N:
            correction[(seg, rg, int(dow), int(hr), int(mn))] = g['residual'].mean()

    n_cells = len(correction)
    print(f"  Nowcast correction (all open hours): {len(df):,} recent rows → {n_cells} (segment, regime, dow, hour, minute) cells")
    return correction


def compute_predictions(table, correction, days=91):
    """Build (slot_ts ISO string, pct) for every open 15-min slot over the next N days."""
    slot_ts, dates_slots, segments, regimes, dows, hours, minutes, horizons = [], [], [], [], [], [], [], []

    for offset in range(days):
        d        = now.date() + timedelta(days=offset)
        day_name = pd.Timestamp(d).day_name()
        open_h, close_h = get_open_hours(day_name, d)
        dow     = d.weekday()
        segment = _correction_segment(classify_date(d))
        regime  = is_summer_day(d)
        for h in range(open_h, close_h):
            for m in (0, 15, 30, 45):
                # Store as PT-aware ISO timestamp for Supabase TIMESTAMPTZ
                dt = datetime(d.year, d.month, d.day, h, m, tzinfo=PT)
                slot_ts.append(dt.isoformat())
                dates_slots.append((d, h * 4 + m // 15))
                segments.append(segment)
                regimes.append(regime)
                dows.append(dow)
                hours.append(h)
                minutes.append(m)
                horizons.append(offset)  # days from today, for correction decay

    print(f"  Predicting {len(dates_slots):,} slots from curve table...")
    preds = cm.predict(table, dates_slots)

    records = []
    for ts, p, seg, rg, dw, hr, mn, hz in zip(slot_ts, preds, segments, regimes, dows, hours, minutes, horizons):
        if p != p:  # NaN -> no curve matched this (phase, dow, slot)
            continue
        # Apply the trailing-residual nowcast to every open slot, not just
        # evenings: a "running hot/cool" stretch is an all-day phenomenon, and
        # backtest on the week-aware base showed all-hours strictly dominates
        # evening-only (adds a morning gain, ~2.5% better on forecast days 1-7,
        # evenings unchanged). Decay toward 0 as the horizon grows -- the
        # 28-day nowcast only tracks the current stretch for ~a week.
        decay = 0.5 ** (hz / CORRECTION_DECAY_HALFLIFE)
        p += decay * correction.get((seg, rg, dw, hr, mn), 0.0)
        records.append({
            "slot_ts": ts,
            "pct":     round(min(max(float(p), 0.0), 100.0), 1),
        })
    return records


def main():
    print("Loading curve table...")
    table = load_curves()

    print("Building evening correction table...")
    correction = build_evening_correction(table)

    print("Computing predictions (today + 90 days)...")
    records = compute_predictions(table, correction, days=91)
    print(f"  {len(records):,} slots computed")

    print("Upserting to Supabase predictions table...")
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        sb.table("predictions").upsert(batch, on_conflict="slot_ts").execute()
        print(f"  Upserted {min(i + BATCH_SIZE, len(records))}/{len(records)}")

    # Purge stale in-horizon rows: upsert only ever adds/overwrites slots we generate
    # today, it never removes ones from an earlier run whose open/close hours no
    # longer match (e.g. a date that used to be generated with academic-year hours
    # and is now correctly summer-hours-only keeps its old post-close rows forever
    # otherwise). Diff today's generated slot set against what's actually in the
    # table over the same horizon and delete anything left over.
    horizon_start = datetime(now.year, now.month, now.day, 0, 0, tzinfo=PT)
    horizon_end   = horizon_start + timedelta(days=91)
    generated_instants = {datetime.fromisoformat(r["slot_ts"]) for r in records}

    existing, offset = [], 0
    while True:
        batch = (
            sb.table("predictions")
            .select("slot_ts")
            .gte("slot_ts", horizon_start.isoformat())
            .lt("slot_ts", horizon_end.isoformat())
            .range(offset, offset + 8999)
            .execute()
            .data
        )
        existing.extend(batch)
        if len(batch) < 9000:
            break
        offset += 9000

    stale = [
        r["slot_ts"] for r in existing
        if datetime.fromisoformat(r["slot_ts"]) not in generated_instants
    ]
    for i in range(0, len(stale), BATCH_SIZE):
        sb.table("predictions").delete().in_("slot_ts", stale[i:i + BATCH_SIZE]).execute()
    print(f"  Purged {len(stale)} stale in-horizon rows")

    # Purge stale far-future rows left over from when we generated 180 days, so the
    # table stays bounded to the ~90-day horizon we now compute. The +93-day margin sits
    # beyond the clients' +92-day fetch bound, so this can never delete a slot that's
    # still viewable, even accounting for PT/UTC boundary fuzz.
    purge_from = (now.date() + timedelta(days=93)).isoformat()
    sb.table("predictions").delete().gte("slot_ts", purge_from).execute()
    print(f"  Purged any predictions on/after {purge_from}")

    print(f"[{now.isoformat()}] predictions table updated: {len(records)} rows")


if __name__ == "__main__":
    main()
