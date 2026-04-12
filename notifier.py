"""
notifier.py — RSF SMS notification system.
Runs after scraper.py in the GitHub Action every 15 minutes.
"""

import os
import sqlite3
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from twilio.rest import Client as TwilioClient
from supabase import create_client

from predict import predict
from train import engineer_features

# ── Constants ────────────────────────────────────────────────────
PT = ZoneInfo("America/Los_Angeles")

# RSF open hours: weekday (Mon=0..Sun=6) → (open_hour, close_hour)
RSF_HOURS = {
    0: (7, 23),   # Monday
    1: (7, 23),   # Tuesday
    2: (7, 23),   # Wednesday
    3: (7, 23),   # Thursday
    4: (7, 23),   # Friday
    5: (8, 18),   # Saturday
    6: (8, 23),   # Sunday
}

# ── Clients ──────────────────────────────────────────────────────
twilio_client = TwilioClient(
    os.environ["TWILIO_ACCOUNT_SID"],
    os.environ["TWILIO_AUTH_TOKEN"],
)
TWILIO_FROM = os.environ["TWILIO_FROM_NUMBER"]

sb = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)


# ── SMS ──────────────────────────────────────────────────────────
def send_sms(to: str, body: str):
    twilio_client.messages.create(to=to, from_=TWILIO_FROM, body=body)
    print(f"  SMS → {to[:6]}****: {body[:80]}")


# ── DB helpers ───────────────────────────────────────────────────
def init_notification_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS notification_state (
            phone_number   TEXT PRIMARY KEY,
            last_quiet_alert TEXT,
            last_digest_date TEXT,
            quiet_streak   INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS window_alert_state (
            phone_number  TEXT,
            day_of_week   TEXT,
            window_start  INTEGER,
            last_alert_date TEXT,
            PRIMARY KEY (phone_number, day_of_week, window_start)
        );
    """)
    conn.commit()


def get_or_init_state(conn, phone: str) -> dict:
    row = conn.execute(
        "SELECT last_quiet_alert, last_digest_date, quiet_streak "
        "FROM notification_state WHERE phone_number = ?",
        (phone,),
    ).fetchone()
    if row:
        return {
            "last_quiet_alert": row[0],
            "last_digest_date": row[1],
            "quiet_streak": row[2] or 0,
        }
    conn.execute(
        "INSERT OR IGNORE INTO notification_state (phone_number) VALUES (?)", (phone,)
    )
    conn.commit()
    return {"last_quiet_alert": None, "last_digest_date": None, "quiet_streak": 0}


def save_state(conn, phone: str, state: dict):
    conn.execute(
        """
        INSERT INTO notification_state (phone_number, last_quiet_alert, last_digest_date, quiet_streak)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(phone_number) DO UPDATE SET
            last_quiet_alert = excluded.last_quiet_alert,
            last_digest_date = excluded.last_digest_date,
            quiet_streak     = excluded.quiet_streak
        """,
        (phone, state["last_quiet_alert"], state["last_digest_date"], state["quiet_streak"]),
    )
    conn.commit()


def get_window_alert_date(conn, phone: str, day_of_week: str, window_start: int):
    row = conn.execute(
        "SELECT last_alert_date FROM window_alert_state "
        "WHERE phone_number=? AND day_of_week=? AND window_start=?",
        (phone, day_of_week, window_start),
    ).fetchone()
    return row[0] if row else None


def save_window_alert(conn, phone: str, day_of_week: str, window_start: int, alert_date: str):
    conn.execute(
        """
        INSERT INTO window_alert_state (phone_number, day_of_week, window_start, last_alert_date)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(phone_number, day_of_week, window_start) DO UPDATE SET
            last_alert_date = excluded.last_alert_date
        """,
        (phone, day_of_week, window_start, alert_date),
    )
    conn.commit()


# ── Historical average (semester-only) ───────────────────────────
# Break ranges mirror train.py exactly — summer, winter, and spring recess periods.
# We filter these out so the "typical" baseline only reflects in-semester traffic.
_BREAK_RANGES = [
    (date(2020, 12, 19), date(2021,  1, 18)),
    (date(2021,  3, 22), date(2021,  3, 26)),
    (date(2021,  5, 14), date(2021,  8, 24)),
    (date(2021, 12, 18), date(2022,  1, 17)),
    (date(2022,  3, 21), date(2022,  3, 25)),
    (date(2022,  5, 13), date(2022,  8, 23)),
    (date(2022, 12, 17), date(2023,  1, 16)),
    (date(2023,  3, 27), date(2023,  3, 31)),
    (date(2023,  5, 12), date(2023,  8, 22)),
    (date(2023, 12, 16), date(2024,  1, 14)),
    (date(2024,  3, 25), date(2024,  3, 29)),
    (date(2024,  5, 10), date(2024,  8, 27)),
    (date(2024, 12, 21), date(2025,  1, 19)),
    (date(2025,  3, 24), date(2025,  3, 28)),
    (date(2025,  5, 16), date(2025,  8, 26)),
    (date(2025, 12, 20), date(2026,  1, 20)),
    (date(2026,  3, 23), date(2026,  3, 27)),
    (date(2026,  5, 15), date(2026,  8, 25)),
]


def historical_semester_avg(conn, day_name: str, hour: int):
    """Average percent_full for a given day+hour, semester weeks only."""
    rows = conn.execute(
        "SELECT timestamp, percent_full FROM capacity_log WHERE percent_full IS NOT NULL"
    ).fetchall()

    total, count = 0.0, 0
    for ts_str, pct in rows:
        try:
            dt = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        if dt.strftime("%A") != day_name or dt.hour != hour:
            continue
        d = dt.date()
        if not any(start <= d <= end for start, end in _BREAK_RANGES):
            total += pct
            count += 1

    return round(total / count, 1) if count > 0 else None


# ── Format helpers ────────────────────────────────────────────────
def fmt_hour(h: int) -> str:
    if h == 0 or h == 24:
        return "12 AM"
    if h == 12:
        return "12 PM"
    return f"{h - 12} PM" if h > 12 else f"{h} AM"


# ── Digest helpers ────────────────────────────────────────────────
def compute_today_peak(date_str: str, open_hour: int, close_hour: int) -> tuple[int, int]:
    """Return (peak_hour, peak_pct) using MLP predictions across today's open hours."""
    best_pct, best_hour = -1.0, open_hour
    for h in range(open_hour, close_hour):
        _, mlp = predict(f"{date_str} {h:02d}:00")
        if mlp > best_pct:
            best_pct, best_hour = mlp, h
    return best_hour, round(best_pct)


def compute_best_windows(date_str: str, open_hour: int, close_hour: int, n: int = 3) -> list:
    """Return top n 1-hour slots with lowest MLP-predicted capacity."""
    slots = []
    for h in range(open_hour, close_hour):
        _, mlp = predict(f"{date_str} {h:02d}:00")
        slots.append((mlp, h))
    slots.sort()
    result = []
    for pct, h in slots[:n]:
        result.append(f"{fmt_hour(h)}–{fmt_hour(h + 1)} (~{round(pct)}%)")
    return result


def get_special_day_line(now_pt: datetime):
    """Check academic calendar flags for today; return a descriptive string or None."""
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp(now_pt)],
        "people_count": [100],
        "percent_full": [50],
    })
    X, _ = engineer_features(df)
    row = X.iloc[0]
    if row.get("is_finals", 0):
        return "It's finals week — gym may be busier than usual."
    if row.get("is_dead_week", 0):
        return "It's dead week — gym may be emptier than usual."
    if row.get("is_first_week", 0):
        return "It's the first week of classes — gym may be busier than usual."
    return None


# ── Main ─────────────────────────────────────────────────────────
def main():
    now_pt    = datetime.now(PT)
    today_str = now_pt.strftime("%Y-%m-%d")
    day_name  = now_pt.strftime("%A")
    cur_hour  = now_pt.hour
    weekday   = now_pt.weekday()          # Mon=0..Sun=6
    open_hour, close_hour = RSF_HOURS[weekday]

    print(f"[notifier] {now_pt.strftime('%Y-%m-%d %H:%M PT')} | {day_name} {cur_hour}:00 | open={open_hour}–{close_hour}")

    # ── Load DB ──────────────────────────────────────────────────
    conn = sqlite3.connect("gym_history.db")
    init_notification_tables(conn)

    row = conn.execute(
        "SELECT percent_full FROM capacity_log ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    if not row:
        print("[notifier] No capacity data. Exiting.")
        conn.close()
        return
    current_pct = row[0]
    print(f"[notifier] Current occupancy: {current_pct}%")

    # ── Fetch active subscribers ─────────────────────────────────
    subs_resp = sb.table("subscribers").select("*").eq("is_active", True).execute()
    subscribers = subs_resp.data or []
    if not subscribers:
        print("[notifier] No active subscribers.")
        conn.close()
        return

    phones = [s["phone_number"] for s in subscribers]
    wins_resp = (
        sb.table("subscriber_windows")
        .select("*")
        .in_("phone_number", phones)
        .execute()
    )
    all_windows = wins_resp.data or []
    print(f"[notifier] {len(subscribers)} active subscribers, {len(all_windows)} windows")

    # ── Shared computed values ───────────────────────────────────
    historical_avg = historical_semester_avg(conn, day_name, cur_hour)
    is_quiet = (current_pct < historical_avg * 0.75) if historical_avg is not None else False
    print(f"[notifier] hist_avg={historical_avg}, is_quiet={is_quiet}")

    # is_break flag for today (suppresses quiet alerts during breaks)
    df_today = pd.DataFrame({
        "timestamp": [pd.Timestamp(now_pt)],
        "people_count": [100],
        "percent_full": [50],
    })
    X_today, _ = engineer_features(df_today)
    is_break = bool(X_today.iloc[0].get("is_break", 0))

    # Digest data — computed lazily only if at least one subscriber needs it
    _digest_computed = False
    today_peak_hour = today_peak_pct = None
    best_windows = []
    special_day_line = None

    def ensure_digest_data():
        nonlocal _digest_computed, today_peak_hour, today_peak_pct, best_windows, special_day_line
        if _digest_computed:
            return
        today_peak_hour, today_peak_pct = compute_today_peak(today_str, open_hour, close_hour)
        best_windows = compute_best_windows(today_str, open_hour, close_hour)
        special_day_line = get_special_day_line(now_pt)
        _digest_computed = True

    # ── Per-subscriber processing ────────────────────────────────
    for sub in subscribers:
        phone = sub["phone_number"]
        state = get_or_init_state(conn, phone)

        sub_windows = [
            w for w in all_windows
            if w["phone_number"] == phone and w["day_of_week"] == day_name
        ]

        # A. Window threshold ──────────────────────────────────────
        if sub.get("receive_window_threshold", True):
            for win in sub_windows:
                ws  = win["window_start"]
                we  = win["window_end"]
                thr = win.get("threshold_pct", 60)
                if ws <= cur_hour < we:
                    last_date = get_window_alert_date(conn, phone, day_name, ws)
                    if current_pct < thr and last_date != today_str:
                        body = (
                            f"RSF is at {current_pct}% right now — below your {thr}% target "
                            f"for this {fmt_hour(ws)}–{fmt_hour(we)} window. Good time to go!"
                        )
                        send_sms(phone, body)
                        save_window_alert(conn, phone, day_name, ws, today_str)

        # B. Quieter than usual ────────────────────────────────────
        if sub.get("receive_quiet", True):
            in_active_window = (open_hour + 1) <= cur_hour <= (close_hour - 2)
            if not is_break and in_active_window:
                if is_quiet:
                    state["quiet_streak"] += 1
                else:
                    state["quiet_streak"] = 0

                if state["quiet_streak"] >= 2:
                    can_send = True
                    last_alert = state.get("last_quiet_alert")
                    if last_alert:
                        try:
                            last_dt = datetime.fromisoformat(last_alert)
                            if last_dt.tzinfo is None:
                                last_dt = last_dt.replace(tzinfo=PT)
                            can_send = (now_pt - last_dt).total_seconds() > 3 * 3600
                        except ValueError:
                            pass
                    if can_send:
                        body = (
                            f"RSF is unusually quiet right now — {current_pct}% full vs. "
                            f"the typical {historical_avg}% for {day_name} at this hour. "
                            f"Great time to go!"
                        )
                        send_sms(phone, body)
                        state["last_quiet_alert"] = now_pt.isoformat()
                        state["quiet_streak"] = 0
            else:
                state["quiet_streak"] = 0

        # C. Daily digest ─────────────────────────────────────────
        if sub.get("receive_digest", True):
            digest_hour = sub.get("digest_hour", 6)
            if cur_hour == digest_hour and state.get("last_digest_date") != today_str:
                ensure_digest_data()
                peak_line = f"Peak: ~{fmt_hour(today_peak_hour)} ({today_peak_pct}% full)"
                windows_line = "Best times: " + ", ".join(best_windows) if best_windows else ""
                body = f"Good morning! RSF outlook for today ({day_name}):\n{peak_line}"
                if windows_line:
                    body += f"\n{windows_line}"
                if special_day_line:
                    body += f"\n{special_day_line}"
                send_sms(phone, body)
                state["last_digest_date"] = today_str

        save_state(conn, phone, state)

    conn.close()
    print("[notifier] Done.")


if __name__ == "__main__":
    main()
