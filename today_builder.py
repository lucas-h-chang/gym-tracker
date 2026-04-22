"""
today_builder.py — compute similarity-based predictions for today → Supabase today_summary.
Runs every 15 min alongside scraper.py.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from supabase import create_client

PT  = ZoneInfo("America/Los_Angeles")
now = datetime.now(PT)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

SPRING_BREAKS = [
    ('2021-03-20', '2021-03-28'),
    ('2022-03-19', '2022-03-27'),
    ('2023-03-25', '2023-04-02'),
    ('2024-03-23', '2024-03-31'),
    ('2025-03-22', '2025-03-30'),
    ('2026-03-21', '2026-03-29'),
    ('2027-03-20', '2027-03-28'),
    ('2028-03-25', '2028-04-02'),
]


def get_open_hours(day_name):
    if day_name == 'Saturday':
        return 8, 18
    elif day_name == 'Sunday':
        return 8, 23
    else:
        return 7, 23


def is_semester_day(d):
    month, dom = d.month, d.day
    is_summer = month in [6, 7, 8]
    is_winter = (month == 12 and dom >= 16) or (month == 1 and dom <= 12)
    is_sb = any(
        pd.Timestamp(s).date() <= d <= pd.Timestamp(e).date()
        for s, e in SPRING_BREAKS
    )
    return not (is_summer or is_winter or is_sb)


def fetch_history():
    """Fetch last 2 years of capacity_log from Supabase (paginated)."""
    cutoff = (now - timedelta(days=730)).isoformat()
    BATCH  = 10000
    offset = 0
    rows   = []
    while True:
        batch = (
            sb.table("capacity_log")
            .select("timestamp,percent_full")
            .gte("timestamp", cutoff)
            .range(offset, offset + BATCH - 1)
            .order("timestamp")
            .execute()
            .data
        )
        rows.extend(batch)
        if len(batch) < BATCH:
            break
        offset += BATCH
    return rows


def compute_similarity_predictions(df):
    K              = 5
    BLEND_CEIL     = 0.9
    OVERLAP_THRESH = 0.70
    EXCLUDE_WINDOW = 7
    MIN_SLOTS      = 4

    df['date']         = df['timestamp'].dt.date
    df['hour_numeric'] = ((df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60) * 4).round() / 4
    df['day_name']     = df['timestamp'].dt.day_name()
    df['is_semester']  = df['date'].apply(is_semester_day)

    today         = now.date()
    today_name    = pd.Timestamp(today).day_name()
    today_is_sem  = is_semester_day(today)
    now_hour      = now.hour + now.minute / 60
    open_h, close_h = get_open_hours(today_name)

    today_rows   = df[df['date'] == today]
    today_finger = (
        today_rows[today_rows['hour_numeric'] <= now_hour]
        .groupby('hour_numeric')['percent_full'].mean()
    )
    finger_slots = sorted(today_finger.index.tolist())

    if len(finger_slots) < MIN_SLOTS:
        return [], 0.0

    target_ts     = pd.Timestamp(today)
    candidates_df = df[
        (df['date'] != today) &
        (abs((pd.to_datetime(df['date']) - target_ts).dt.days) > EXCLUDE_WINDOW) &
        (df['day_name'] == today_name) &
        (df['is_semester'] == today_is_sem)
    ]

    candidates = []
    for _, group in candidates_df.groupby('date'):
        cf        = group.groupby('hour_numeric')['percent_full'].mean()
        available = [s for s in finger_slots if s in cf.index]
        if len(available) < len(finger_slots) * OVERLAP_THRESH:
            continue
        hist_vec  = np.array([cf[s]          for s in available])
        today_sub = np.array([today_finger[s] for s in available])
        dist      = float(np.sqrt(np.mean((hist_vec - today_sub) ** 2)))
        candidates.append((dist, group))

    if not candidates:
        return [], 0.0

    candidates.sort(key=lambda x: x[0])
    top_k = candidates[:K]

    dists   = np.array([c[0] for c in top_k])
    weights = 1.0 / (dists + 1e-6)
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
        for (_, grp), w in zip(top_k, weights):
            gf = grp.groupby('hour_numeric')['percent_full'].mean()
            if slot in gf.index:
                vals.append(gf[slot])
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
    for (_, grp), w in zip(top_k, weights):
        gf = grp.groupby('hour_numeric')['percent_full'].mean()
        if last_slot in gf.index:
            vals.append(gf[last_slot])
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

    blend_weight = round(min((now_hour - open_h) / 6.0, BLEND_CEIL), 3)
    return similarity_preds, blend_weight


def main():
    print("Fetching history from Supabase...")
    rows = fetch_history()
    print(f"  {len(rows):,} rows loaded")

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(PT)

    print("Computing similarity predictions...")
    preds, blend_weight = compute_similarity_predictions(df)

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
