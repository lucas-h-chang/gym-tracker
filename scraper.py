"""
scraper.py — takes one live RSF occupancy reading and appends it to capacity_log.

NOT ON THE 15-MINUTE PATH ANY MORE (2026-08-13). The production scraper is now
api/scrape.js, running on Vercel and poked by cron-job.org. This file is kept as
a manual backfill tool and escape hatch: run it by hand if the Vercel endpoint
ever misbehaves, and it will write an identical row.

    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... DENSITY_TOKEN=... python3 scraper.py

Why the move: a reading is irreplaceable (Density's /count only reports *now*),
and GitHub Actions must allocate a VM before any code runs. On 2026-08-06 that
queue backed up 8-17 minutes and dropped several readings. Vercel invokes an
already-deployed function, so there is no allocation step to get stuck in. The
derived builders (today_builder.py, send_workout_notifications.py) stay in
.github/workflows/scrape.yml because they recompute from Supabase, so a delay
costs nothing. See handoffs/SPEC_VERCEL_SCRAPE.md.

Keep this file's behaviour in sync with api/scrape.js.
"""
import os
import sys
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from supabase import create_client

from academic_calendar import get_open_hours

from sensor_guard import is_sensor_stalled, FLOOR_COUNT
from supabase_io import execute_read

MAX_CAP = 150
PT = ZoneInfo("America/Los_Angeles")


def main():
    now = datetime.now(PT)
    open_h, close_h = get_open_hours(now.strftime('%A'), now.date())
    if not open_h <= now.hour + now.minute / 60 < close_h:
        print(f"[{now.isoformat()}] RSF closed; skipping insert.")
        return
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    response = requests.get("https://api.density.io/v2/spaces/spc_863128347956216317/count",
        headers={"Authorization": f"Bearer {os.environ['DENSITY_TOKEN']}"}, timeout=10)
    response.raise_for_status()
    count = response.json()["count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError('Invalid Density count')
    stall = is_sensor_stalled(sb, count, now)
    pct = round(count / MAX_CAP * 100, 1)
    sb.table("capacity_log").insert({"timestamp": now.isoformat(),
        "people_count": count, "percent_full": pct, "sensor_ok": not stall['stalled']}).execute()
    if stall['stalled']:
        # Bounded/idempotent update; reuse retry handling for transient failures.
        execute_read(lambda: sb.table('capacity_log').update({'sensor_ok': False})
            .in_('timestamp', stall['timestamps']).lte('people_count', FLOOR_COUNT))
    print(f"[{now.isoformat()}] Saved: {count} people ({pct}%), sensor_ok={not stall['stalled']}")


if __name__ == '__main__':
    main()
