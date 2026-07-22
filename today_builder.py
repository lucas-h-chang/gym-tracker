"""
today_builder.py — compute similarity-based predictions for today → Supabase today_summary.
Runs every 15 min alongside scraper.py.

Reads pre-aggregated candidate day profiles from `day_profiles` instead of re-downloading the
full capacity_log every run — ~0.3 MB/run vs ~10 MB (Finding E). Only today's own rows are
fetched live, for the fingerprint.

`day_profiles` is now a live Postgres VIEW over capacity_log (see migrations/002_day_profiles_view.sql),
not a table built once/day by day_profiles_builder.py (moved to legacy/ — kept only as the
reference the view SQL was translated from). fetch_candidates()'s server-side filters are
unchanged since the view exposes identical columns.
"""
import os
import numpy as np
import pandas as pd
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client

from academic_calendar import is_summer_day, get_open_hours, is_semester_day
from supabase_io import paginated_fetch

PT  = ZoneInfo("America/Los_Angeles")
now = datetime.now(PT)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

# Fixed data boundary: keep 2022 onward, drop COVID-era 2020-2021.
# Shared with day_profiles_builder.py (and, later, train.py via T20) — keep in sync.
DATA_CUTOFF = date(2022, 1, 1)

# SPRING_BREAKS/SUMMER_RANGES/is_summer_day/get_open_hours/is_semester_day live
# in academic_calendar.py (consolidated 2026-07-21 — see CLAUDE.md).


def _pt_iso(d, t):
    """ISO8601 for a PT wall-clock (date, time)."""
    return datetime.combine(d, t, tzinfo=PT).isoformat()


def _hour_slot(ts):
    """Quarter-hour bin for a PT timestamp series (identical formula to day_profiles_builder)."""
    return ((ts.dt.hour + ts.dt.minute / 60) * 4).round() / 4


# ---------------------------------------------------------------------------
# Fetch: today's fingerprint + candidate day profiles
# ---------------------------------------------------------------------------

def fetch_today_rows():
    """Today's capacity_log rows (a few dozen) for the live fingerprint."""
    return (
        sb.table("capacity_log")
        .select("timestamp,percent_full")
        .gte("timestamp", _pt_iso(now.date(), time.min))
        .order("timestamp")
        .limit(2000)
        .execute()
        .data
    )


def fetch_candidates():
    """Pre-aggregated candidate profiles for today's weekday + semester phase.

    Server-side filters: same weekday, same semester phase, dates in
    [DATA_CUTOFF, today-8]. The `date <= today-8` bound folds in both `date != today`
    and the EXCLUDE_WINDOW=7 rule. Paginated so it never trips the ~9998-row API cap
    as history accumulates.
    """
    today        = now.date()
    today_name   = pd.Timestamp(today).day_name()
    today_is_sem = is_semester_day(today)
    hi           = (today - timedelta(days=8)).isoformat()
    lo           = DATA_CUTOFF.isoformat()

    BATCH, offset, rows = 9000, 0, []
    while True:
        batch = (
            sb.table("day_profiles")
            .select("date,hour_slot,avg_pct")
            .eq("day_name", today_name)
            .eq("is_semester", today_is_sem)
            .lte("date", hi)
            .gte("date", lo)
            .range(offset, offset + BATCH - 1)
            .order("date")
            .execute()
            .data
        )
        rows.extend(batch)
        if len(batch) < BATCH:
            break
        offset += BATCH
    return rows


def fetch_history_fallback():
    """Full capacity_log from DATA_CUTOFF (paginated). Used only if day_profiles is
    empty (e.g. before the first backfill), so a rollout gap can't blank the nowcast.

    Effectively dead code now that day_profiles is a live view over capacity_log
    (migrations/002_day_profiles_view.sql) — it always returns rows for any date with
    data, so fetch_candidates() should never come back empty in normal operation.
    Left in place as a defensive fallback (e.g. a transient view/permissions error)
    rather than removed."""
    return paginated_fetch(
        sb, "capacity_log", "timestamp,percent_full",
        gte=_pt_iso(DATA_CUTOFF, time.min), order="timestamp",
    )


# ---------------------------------------------------------------------------
# Shape helpers: build the fingerprint + candidate profile dict
# ---------------------------------------------------------------------------

def build_today_finger(today_rows):
    """Series indexed by hour_slot: today's mean % per quarter-hour, up to now."""
    if not today_rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(today_rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601').dt.tz_convert(PT)
    df['hour_slot'] = _hour_slot(df['timestamp'])
    now_hour = now.hour + now.minute / 60
    df = df[df['hour_slot'] <= now_hour]
    return df.groupby('hour_slot')['percent_full'].mean()


def candidates_from_profiles(rows):
    """{date: Series(hour_slot -> avg_pct)} from day_profiles rows."""
    by_date = {}
    for r in rows:
        by_date.setdefault(r['date'], {})[float(r['hour_slot'])] = float(r['avg_pct'])
    return {d: pd.Series(slots) for d, slots in by_date.items()}


def candidates_from_history(rows):
    """Fallback: build the same {date: Series} from raw capacity_log rows, applying the
    same filters fetch_candidates does server-side (weekday, phase, [DATA_CUTOFF, today-8])."""
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601').dt.tz_convert(PT)
    df['date']      = df['timestamp'].dt.date
    df['hour_slot'] = _hour_slot(df['timestamp'])
    df['day_name']  = df['timestamp'].dt.day_name()
    df['is_sem']    = df['date'].apply(is_semester_day)

    today        = now.date()
    today_name   = pd.Timestamp(today).day_name()
    today_is_sem = is_semester_day(today)
    mask = (
        (df['date'] <= today - timedelta(days=8)) &
        (df['date'] >= DATA_CUTOFF) &
        (df['day_name'] == today_name) &
        (df['is_sem'] == today_is_sem)
    )
    return {
        d: g.groupby('hour_slot')['percent_full'].mean()
        for d, g in df[mask].groupby('date')
    }


# ---------------------------------------------------------------------------
# Similarity nowcast (algorithm unchanged; now consumes pre-aggregated inputs)
# ---------------------------------------------------------------------------

def compute_similarity_predictions(today_finger, candidates):
    K              = 5
    BLEND_CEIL     = 0.9
    BLEND_HORIZON  = 2.0   # hours the nowcast bridges before handing back to the base curve
    OVERLAP_THRESH = 0.70
    MIN_SLOTS      = 4

    today           = now.date()
    today_name      = pd.Timestamp(today).day_name()
    now_hour        = now.hour + now.minute / 60
    open_h, close_h = get_open_hours(today_name, today)

    finger_slots = sorted(today_finger.index.tolist())
    if len(finger_slots) < MIN_SLOTS:
        return [], 0.0

    scored = []
    for cf in candidates.values():
        available = [s for s in finger_slots if s in cf.index]
        if len(available) < len(finger_slots) * OVERLAP_THRESH:
            continue
        hist_vec  = np.array([cf[s]           for s in available])
        today_sub = np.array([today_finger[s] for s in available])
        dist      = float(np.sqrt(np.mean((hist_vec - today_sub) ** 2)))
        scored.append((dist, cf))

    if not scored:
        return [], 0.0

    scored.sort(key=lambda x: x[0])
    top_k = scored[:K]

    dists    = np.array([c[0] for c in top_k])
    weights  = 1.0 / (dists + 1e-6)
    weights /= weights.sum()

    future_slots = [
        h + m / 60
        for h in range(open_h, close_h)
        for m in (0, 15, 30, 45)
        if h + m / 60 >= now_hour
    ]

    similarity_preds = []
    for slot in future_slots:
        vals, wts = [], []
        for (_, cf), w in zip(top_k, weights):
            if slot in cf.index:
                vals.append(cf[slot])
                wts.append(w)
        if vals:
            wa    = np.array(wts); wa /= wa.sum()
            h_int = int(slot)
            m_int = round((slot - h_int) * 60)
            label_h   = h_int % 12 or 12
            label_sfx = 'AM' if h_int < 12 else 'PM'
            similarity_preds.append({
                'x':     slot,
                'y':     round(float(np.dot(wa, vals)), 1),
                'label': f'{label_h}:{m_int:02d} {label_sfx}',
            })

    # Anchor correction: shift predictions to match today's last observed level.
    last_slot = max(finger_slots)
    vals, wts = [], []
    for (_, cf), w in zip(top_k, weights):
        if last_slot in cf.index:
            vals.append(cf[last_slot])
            wts.append(w)
    if vals:
        wa             = np.array(wts); wa /= wa.sum()
        sim_at_last    = float(np.dot(wa, vals))
        actual_at_last = float(today_finger[last_slot])
        offset         = actual_at_last - sim_at_last
        for p in similarity_preds:
            decay = max(0.0, 1.0 - (p['x'] - last_slot) / 4.0)
            p['y'] = round(max(0.0, min(100.0, p['y'] + offset * decay)), 1)

    ch_label = f"{close_h % 12 or 12}:00 {'AM' if close_h < 12 else 'PM'}"
    similarity_preds.append({'x': float(close_h), 'y': 0.0, 'label': ch_label})

    # Per-slot blend weight. The nowcast's job is to smoothly bridge today's
    # live level onto the base curve — not to forecast hours out — so each
    # slot's trust in the nowcast starts at BLEND_CEIL right at the last
    # observed slot and decays linearly to 0 over BLEND_HORIZON hours, handing
    # far slots (e.g. tonight's peak, seen only through morning-shape
    # neighbors) fully back to the recency-aware base curve. Replaces the old
    # time-of-day ramp, which under-weighted the bridge in the morning and
    # over-rode the evening peak by pinning w≈0.9 across every future slot.
    for p in similarity_preds:
        decay  = max(0.0, 1.0 - (p['x'] - last_slot) / BLEND_HORIZON)
        p['w'] = round(BLEND_CEIL * decay, 3)

    # Retained as a scalar fallback for any cached today_summary row still read
    # by an older frontend build; live blending now uses each point's own 'w'.
    blend_weight = round(min((now_hour - open_h) / 6.0, BLEND_CEIL), 3)
    return similarity_preds, blend_weight


def main():
    # Skip entirely when the RSF is closed — nothing to nowcast (Finding E).
    open_h, close_h = get_open_hours(now.strftime('%A'), now.date())
    now_hour = now.hour + now.minute / 60
    if now_hour < open_h or now_hour >= close_h:
        print(f"[{now.isoformat()}] RSF closed (open {open_h}:00-{close_h}:00); skipping today_summary build.")
        return

    today_finger = build_today_finger(fetch_today_rows())

    candidate_rows = fetch_candidates()
    if candidate_rows:
        candidates = candidates_from_profiles(candidate_rows)
        print(f"Loaded {len(candidate_rows):,} profile rows across {len(candidates)} candidate days")
    else:
        print("day_profiles empty — falling back to full-history fetch")
        candidates = candidates_from_history(fetch_history_fallback())

    print("Computing similarity predictions...")
    preds, blend_weight = compute_similarity_predictions(today_finger, candidates)

    today_str = now.strftime('%Y-%m-%d')
    sb.table("today_summary").upsert({
        "date":             today_str,
        "similarity_preds": preds,
        "blend_weight":     blend_weight,
        "computed_at":      now.isoformat(),
    }).execute()

    print(f"[{now.isoformat()}] today_summary updated: {len(preds)} slots, blend={blend_weight}")


if __name__ == "__main__":
    main()
