import os
import sys
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from supabase import create_client

from academic_calendar import get_open_hours

URL     = "https://api.density.io/v2/spaces/spc_863128347956216317/count"
HEADERS = {"Authorization": f"Bearer {os.environ['DENSITY_TOKEN']}"}
MAX_CAP = 150
PT      = ZoneInfo("America/Los_Angeles")

# Open-hours gate: cron fires through academic hours year-round; this guard
# prevents inserts during summer evenings when the RSF is actually closed.
# get_open_hours/is_summer_day/SUMMER_RANGES live in academic_calendar.py
# (consolidated 2026-07-21 — see CLAUDE.md).

now = datetime.now(PT)
open_h, close_h = get_open_hours(now.strftime('%A'), now.date())
now_hour = now.hour + now.minute / 60
if now_hour < open_h or now_hour >= close_h:
    print(f"[{now.isoformat()}] RSF closed (open {open_h}:00 – {close_h}:00); skipping insert.")
    sys.exit(0)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

data  = requests.get(URL, headers=HEADERS, timeout=10).json()
count = data["count"]
pct   = round((count / MAX_CAP) * 100, 1)
ts    = now.isoformat()

sb.table("capacity_log").insert({
    "timestamp":    ts,
    "people_count": count,
    "percent_full": pct,
}).execute()

print(f"[{ts}] Saved: {count} people ({pct}%)")
