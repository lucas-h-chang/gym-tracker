import os
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from supabase import create_client

URL     = "https://api.density.io/v2/spaces/spc_863128347956216317/count"
HEADERS = {"Authorization": "Bearer shr_o69HxjQ0BYrY2FPD9HxdirhJYcFDCeRolEd744Uj88e"}
MAX_CAP = 150
PT      = ZoneInfo("America/Los_Angeles")

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

data  = requests.get(URL, headers=HEADERS).json()
count = data["count"]
pct   = round((count / MAX_CAP) * 100, 1)
ts    = datetime.now(PT).isoformat()

sb.table("capacity_log").insert({
    "timestamp":    ts,
    "people_count": count,
    "percent_full": pct,
}).execute()

print(f"[{ts}] Saved: {count} people ({pct}%)")
