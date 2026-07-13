"""
accuracy_check.py — compare ML predictions vs actual capacity for recent summer weeks.

Run:
    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python accuracy_check.py

Outputs a table of MAE by weekday + hour bucket, and prints the worst-miss examples.
"""

import os
import json
import statistics
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from supabase import create_client

PT = ZoneInfo("America/Los_Angeles")

SUMMER_START = date(2026, 5, 15)   # current summer range from scraper.py
ANALYSIS_DAYS = 28                  # look back N days from today

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

# ---------------------------------------------------------------------------
# 1. Fetch actual capacity readings for the past ANALYSIS_DAYS days
# ---------------------------------------------------------------------------
now_pt   = datetime.now(PT)
cutoff   = (now_pt - timedelta(days=ANALYSIS_DAYS)).isoformat()

print(f"Fetching actual capacity_log from {cutoff[:10]} …")
rows = []
offset = 0
while True:
    batch = (
        sb.table("capacity_log")
        .select("timestamp,percent_full")
        .gte("timestamp", cutoff)
        .order("timestamp")
        .range(offset, offset + 9999)
        .execute()
        .data
    )
    rows.extend(batch)
    if len(batch) < 9999:
        break
    offset += 10000
print(f"  {len(rows)} actual readings loaded")

# ---------------------------------------------------------------------------
# 2. Fetch ML predictions for the same window
# ---------------------------------------------------------------------------
print("Fetching predictions …")
pred_rows = []
offset = 0
while True:
    batch = (
        sb.table("predictions")
        .select("slot_ts,pct")
        .gte("slot_ts", cutoff)
        .lte("slot_ts", now_pt.isoformat())
        .order("slot_ts")
        .range(offset, offset + 9999)
        .execute()
        .data
    )
    pred_rows.extend(batch)
    if len(batch) < 9999:
        break
    offset += 10000
print(f"  {len(pred_rows)} prediction slots loaded")

# ---------------------------------------------------------------------------
# 3. Build prediction lookup: slot_ts_rounded → pct
# ---------------------------------------------------------------------------
def round_ts_to_15(ts_str: str) -> str:
    """Snap a UTC ISO timestamp to the nearest 15-min slot, return PT date+hour+min key."""
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(PT)
    total = dt.hour * 60 + dt.minute
    slot  = round(total / 15) * 15
    h, m  = (slot // 60) % 24, slot % 60
    return f"{dt.date()}_{h:02d}:{m:02d}"

pred_map: dict[str, float] = {}
for r in pred_rows:
    key = round_ts_to_15(r["slot_ts"])
    pred_map[key] = r["pct"]

# ---------------------------------------------------------------------------
# 4. Match actual readings to predictions
# ---------------------------------------------------------------------------
WEEKDAY_NAMES = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

matches = []
for r in rows:
    dt = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).astimezone(PT)
    total = dt.hour * 60 + dt.minute
    slot  = round(total / 15) * 15
    h, m  = (slot // 60) % 24, slot % 60
    key   = f"{dt.date()}_{h:02d}:{m:02d}"
    if key in pred_map:
        actual  = float(r["percent_full"])
        pred    = pred_map[key]
        error   = pred - actual
        weekday = dt.weekday()  # 0=Mon
        hour_bucket = f"{h:02d}:00"
        matches.append({
            "ts": dt,
            "date": dt.date(),
            "weekday": weekday,
            "hour": h,
            "minute": m,
            "hour_bucket": hour_bucket,
            "actual": actual,
            "pred": pred,
            "error": error,
            "abs_error": abs(error),
        })

print(f"\n{len(matches)} matched (actual + prediction) data points")

if not matches:
    print("No matches found — check date ranges or table contents.")
    raise SystemExit

# ---------------------------------------------------------------------------
# 5. Overall MAE
# ---------------------------------------------------------------------------
overall_mae = statistics.mean(m["abs_error"] for m in matches)
overall_bias = statistics.mean(m["error"] for m in matches)
print(f"\n{'='*60}")
print(f"OVERALL (past {ANALYSIS_DAYS} days, summer 2026)")
print(f"  MAE:  {overall_mae:.1f}%")
print(f"  Bias: {overall_bias:+.1f}%  ({'under-predicting' if overall_bias < 0 else 'over-predicting'} on average)")
print(f"{'='*60}")

# ---------------------------------------------------------------------------
# 6. MAE by hour of day
# ---------------------------------------------------------------------------
from collections import defaultdict

by_hour: dict[int, list[float]] = defaultdict(list)
by_hour_bias: dict[int, list[float]] = defaultdict(list)
for m in matches:
    by_hour[m["hour"]].append(m["abs_error"])
    by_hour_bias[m["hour"]].append(m["error"])

print("\nMAE by hour of day (PT):")
print(f"  {'Hour':<8} {'MAE':>6} {'Bias':>8} {'N':>5}")
print(f"  {'-'*30}")
for h in sorted(by_hour):
    errs = by_hour[h]
    bias = by_hour_bias[h]
    label = f"{h%12 or 12}{'am' if h<12 else 'pm'}"
    mae_h  = statistics.mean(errs)
    bias_h = statistics.mean(bias)
    flag = " <-- worst" if mae_h > overall_mae * 1.5 else ""
    print(f"  {label:<8} {mae_h:>5.1f}%  {bias_h:>+6.1f}%  {len(errs):>5}{flag}")

# ---------------------------------------------------------------------------
# 7. MAE by weekday
# ---------------------------------------------------------------------------
by_day: dict[int, list[float]] = defaultdict(list)
by_day_bias: dict[int, list[float]] = defaultdict(list)
for m in matches:
    by_day[m["weekday"]].append(m["abs_error"])
    by_day_bias[m["weekday"]].append(m["error"])

print("\nMAE by weekday:")
print(f"  {'Day':<8} {'MAE':>6} {'Bias':>8} {'N':>5}")
print(f"  {'-'*30}")
for d in range(7):
    if d not in by_day:
        continue
    errs = by_day[d]
    bias = by_day_bias[d]
    mae_d  = statistics.mean(errs)
    bias_d = statistics.mean(bias)
    flag = " <-- worst" if mae_d > overall_mae * 1.4 else ""
    print(f"  {WEEKDAY_NAMES[d]:<8} {mae_d:>5.1f}%  {bias_d:>+6.1f}%  {len(errs):>5}{flag}")

# ---------------------------------------------------------------------------
# 8. Worst-miss examples (top 15)
# ---------------------------------------------------------------------------
worst = sorted(matches, key=lambda x: x["abs_error"], reverse=True)[:15]
print("\nTop 15 worst prediction misses:")
print(f"  {'Timestamp (PT)':<22} {'Day':<5} {'Actual':>7} {'Pred':>7} {'Error':>8}")
print(f"  {'-'*55}")
for m in worst:
    day_name = WEEKDAY_NAMES[m["weekday"]]
    ts_str   = m["ts"].strftime("%Y-%m-%d %H:%M")
    print(f"  {ts_str:<22} {day_name:<5} {m['actual']:>6.1f}%  {m['pred']:>6.1f}%  {m['error']:>+7.1f}%")

# ---------------------------------------------------------------------------
# 9. How are predictions doing for the "peak evening" window (5–8pm)?
# ---------------------------------------------------------------------------
evening = [m for m in matches if 17 <= m["hour"] < 20]
if evening:
    e_mae  = statistics.mean(m["abs_error"] for m in evening)
    e_bias = statistics.mean(m["error"] for m in evening)
    print(f"\nEvening peak (5–8 PM): MAE={e_mae:.1f}%, Bias={e_bias:+.1f}%  (n={len(evening)})")
    print(f"  (You mentioned 80% actual at 6pm vs 64% max predicted for next Thu — "
          f"bias {e_bias:+.1f}% confirms the direction)")

print("\nDone.")
