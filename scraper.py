import os
import sys
import requests
from datetime import date, datetime
from zoneinfo import ZoneInfo
from supabase import create_client

URL     = "https://api.density.io/v2/spaces/spc_863128347956216317/count"
HEADERS = {"Authorization": f"Bearer {os.environ['DENSITY_TOKEN']}"}
MAX_CAP = 150
PT      = ZoneInfo("America/Los_Angeles")

# Summer-hours windows: derived from train.py summer-break ranges, end-shifted -3 days.
# Cron fires through academic hours year-round; this guard prevents inserts during
# summer evenings when the RSF is actually closed.
SUMMER_RANGES = [
    (date(2024, 5, 10), date(2024, 8, 24)),
    (date(2025, 5, 16), date(2025, 8, 23)),
    (date(2026, 5, 15), date(2026, 8, 22)),
    (date(2027, 5, 14), date(2027, 8, 21)),
]


def is_summer_day(d):
    return any(s <= d <= e for s, e in SUMMER_RANGES)


def get_open_hours(day_name, d):
    summer = is_summer_day(d)
    if day_name == 'Saturday':
        return 8, 18
    if day_name == 'Sunday':
        return 8, (20 if summer else 23)
    return 7, (20 if summer else 23)


now = datetime.now(PT)
open_h, close_h = get_open_hours(now.strftime('%A'), now.date())
now_hour = now.hour + now.minute / 60
if now_hour < open_h or now_hour >= close_h:
    print(f"[{now.isoformat()}] RSF closed (open {open_h}:00 – {close_h}:00); skipping insert.")
    sys.exit(0)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

data  = requests.get(URL, headers=HEADERS).json()
count = data["count"]
pct   = round((count / MAX_CAP) * 100, 1)
ts    = now.isoformat()

sb.table("capacity_log").insert({
    "timestamp":    ts,
    "people_count": count,
    "percent_full": pct,
}).execute()

print(f"[{ts}] Saved: {count} people ({pct}%)")
