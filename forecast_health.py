"""Quick production forecast health check from live prediction snapshots.

Usage:
    python3 forecast_health.py --days 30

This scores what users actually saw. It never mixes reconstructed snapshots
into production accuracy and excludes today because its actuals are incomplete.
"""
import argparse
import os
from datetime import date, datetime, timedelta
from statistics import mean
from zoneinfo import ZoneInfo

from academic_calendar import get_open_hours


def load_env():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def recent_open_dates(count, end):
    """The last `count` open dates ending at `end`, oldest first."""
    out, current = [], end
    while len(out) < count:
        open_h, close_h = get_open_hours(current.strftime("%A"), current)
        if open_h < close_h:
            out.append(current)
        current -= timedelta(days=1)
    return list(reversed(out))


def fetch_pages(sb, columns, start, end):
    rows, offset, batch = [], 0, 1000
    while True:
        page = (
            sb.table("prediction_accuracy")
            .select(columns)
            .eq("source", "live")
            .eq("model", "carry")
            .gte("date", start.isoformat())
            .lte("date", end.isoformat())
            .order("computed_at").order("snapshot_id").order("slot")
            .range(offset, offset + batch - 1)
            .execute()
            .data
        )
        rows.extend(page)
        if not page:
            return rows
        offset += len(page)


def fetch_accuracy(sb, start, end):
    common = ("snapshot_id,slot,date,computed_at,horizon_h,pct,base_pct,actual_pct")
    try:
        return fetch_pages(sb, common + ",curve_pct", start, end), True
    except Exception as exc:
        # Migration 013 may not have been applied yet. The two already-stored
        # production layers remain useful, so degrade explicitly instead of
        # turning a missing optional column into a dead health check.
        message = str(exc).lower()
        if "curve_pct" not in message and "column" not in message and "schema" not in message:
            raise
        return fetch_pages(sb, common, start, end), False


def stats(rows, prediction):
    pairs = [
        (float(r[prediction]), float(r["actual_pct"]))
        for r in rows
        if r.get(prediction) is not None and r.get("actual_pct") is not None
    ]
    if not pairs:
        return None
    errors = [pred - actual for pred, actual in pairs]
    return {"n": len(errors), "mae": mean(abs(e) for e in errors), "bias": mean(errors)}


def horizon_label(value):
    h = float(value)
    if h < 1:
        return "0–1h"
    if h < 3:
        return "1–3h"
    if h < 6:
        return "3–6h"
    return "6h+"


def verdict(rows):
    days = len({r["date"] for r in rows})
    final, baseline = stats(rows, "pct"), stats(rows, "base_pct")
    if days < 7 or not final or not baseline:
        return "INSUFFICIENT DATA"
    carry_gain = baseline["mae"] - final["mae"]
    if carry_gain > 0.3:
        return "HEALTHY — within-day carry is helping"
    if carry_gain < -0.3:
        return "DEGRADED — within-day carry is hurting"
    return "WATCH — carry difference is inside the ~0.3pp noise band"


def fmt_stat(label, value):
    if value is None:
        return f"  {label:<26} {'N/A':>8} {'N/A':>9}"
    return f"  {label:<26} {value['mae']:>7.2f} {value['bias']:>+9.2f}"


def render(rows, requested_days, start, end, has_curve):
    final = stats(rows, "pct")
    baseline = stats(rows, "base_pct")
    # A new curve column may exist for only a fraction of this window.
    # Measure the layer gain on paired rows, never subtract unmatched averages.
    curve_rows = [r for r in rows if r.get("curve_pct") is not None
                  and r.get("base_pct") is not None and r.get("actual_pct") is not None]
    curve = stats(curve_rows, "curve_pct") if has_curve else None
    curve_baseline = stats(curve_rows, "base_pct") if has_curve else None
    scored_days = sorted({r["date"] for r in rows})
    snapshots = {r["snapshot_id"] for r in rows}

    lines = [
        f"FORECAST HEALTH — last {requested_days} complete open days",
        f"Window: {start} through {end}", "",
        "Forecast                       MAE      Bias",
        fmt_stat("Today (curve+trail+carry)", final),
        fmt_stat("Curve + trailing", baseline),
        fmt_stat("Bare curve", curve), "",
    ]
    if final and baseline:
        lines.append(f"  Carry improvement:    {baseline['mae'] - final['mae']:+.2f} pp")
    if baseline and curve:
        lines.append(f"  Trailing improvement: {curve['mae'] - curve_baseline['mae']:+.2f} pp")
    elif not has_curve:
        lines.append("  Trailing improvement: N/A until migration 013 is applied")
    elif curve is None:
        lines.append("  Trailing improvement: N/A (window predates bare-curve snapshots)")

    if curve:
        lines.append(f"  Trailing comparison uses {len(curve_rows):,} paired points")

    lines.extend(["", "By horizon:"])
    for label in ("0–1h", "1–3h", "3–6h", "6h+"):
        bucket = [r for r in rows if r.get("horizon_h") is not None
                  and horizon_label(r["horizon_h"]) == label]
        fstat, bstat = stats(bucket, "pct"), stats(bucket, "base_pct")
        cstat = stats(bucket, "curve_pct") if has_curve else None
        fmae = f"{fstat['mae']:.2f}" if fstat else "N/A"
        bmae = f"{bstat['mae']:.2f}" if bstat else "N/A"
        cmae = f"{cstat['mae']:.2f}" if cstat else "N/A"
        lines.append(f"  {label:<5} today {fmae:>5} | trail {bmae:>5} | curve {cmae:>5}")

    lines.extend([
        "", "Coverage:",
        f"  {len(scored_days)}/{requested_days} open days have scored live forecasts",
        f"  {len(rows):,} forecast points across {len(snapshots):,} publications",
        "", f"Verdict: {verdict(rows)}",
        "Note: a 30-day view is directional; model acceptance still uses a full year.",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Score recently served forecasts")
    parser.add_argument("--days", type=int, default=30,
                        help="number of complete open days to include (default: 30)")
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be at least 1")

    load_env()
    from supabase import create_client
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not key:
        raise SystemExit("SUPABASE_SERVICE_KEY is required (prediction accuracy is backend-only)")
    if "SUPABASE_URL" not in os.environ:
        raise SystemExit("SUPABASE_URL is required")

    pt_today = datetime.now(ZoneInfo("America/Los_Angeles")).date()
    dates = recent_open_dates(args.days, pt_today - timedelta(days=1))
    sb = create_client(os.environ["SUPABASE_URL"], key)
    rows, has_curve = fetch_accuracy(sb, dates[0], dates[-1])
    print(render(rows, args.days, dates[0], dates[-1], has_curve))


if __name__ == "__main__":
    main()
