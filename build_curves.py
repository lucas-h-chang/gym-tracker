"""
build_curves.py — weekly job: pull capacity_log -> curve_model.build_table ->
models/curves.json + models/curve_metrics.json.

Replaces train.py's monthly RF retrain per SPEC_CURVE_MODEL.md. Runs via
.github/workflows/build_curves.yml (Sun 03:00 PT).
"""
import os
import json
from datetime import datetime

import pandas as pd

import curve_model as cm
from supabase_io import parse_supabase_timestamps, paginated_fetch

MIN_DISTINCT_DAYS = 1500  # mirrors train.py's row guard


def fetch_capacity_log():
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    print("Loading data from Supabase capacity_log...")
    rows = paginated_fetch(sb, "capacity_log", "timestamp,people_count", order="timestamp")

    df = pd.DataFrame(rows)
    df['timestamp'] = parse_supabase_timestamps(df['timestamp'])
    df['people_count'] = df['people_count'].astype(float)
    return df


def main():
    os.makedirs("models", exist_ok=True)

    raw = fetch_capacity_log()
    slots = cm.prepare_slots(raw)

    distinct_days = slots['date'].nunique()
    print(f"  {distinct_days:,} distinct days, {len(slots):,} (date, slot) rows after cleaning")
    if distinct_days < MIN_DISTINCT_DAYS:
        raise RuntimeError(
            f"Only {distinct_days:,} distinct days (< {MIN_DISTINCT_DAYS:,}) — "
            "aborting to protect the existing curves.json"
        )

    print("Building curve table...")
    # week_levels opt-in: differentiate weeks-of-semester within a phase (esp.
    # the long 'regular' phase, where early-Sept and late-Oct were previously
    # one identical curve). Validated on holdout: overall MAE 8.98->8.69,
    # morning 7-9am 8.02->7.02, good-day rate 10.6%->12.6%. DEFAULT_PARAMS keeps
    # it off so the backtest baseline / other callers are unaffected.
    table = cm.build_table(slots, {**cm.DEFAULT_PARAMS, "week_levels": True})
    n_curves = len(table['curves'])
    print(f"  Built {n_curves} (phase, dow) curves")

    with open("models/curves.json", "w") as f:
        json.dump(table, f)
    size_kb = os.path.getsize("models/curves.json") / 1024
    print(f"  Saved -> models/curves.json ({size_kb:.1f} KB)")

    # Lightweight companion metrics: cell coverage / thinness per phase, so a
    # retrain that suddenly loses a phase or goes very thin is visible in the
    # commit diff without re-running the full backtest.
    coverage = {}
    for key, curve in table['curves'].items():
        phase, dow = key.split('|')
        coverage.setdefault(phase, []).extend(curve['n_eff'])
    curve_metrics = {
        "built_at": table['built_at'],
        "params": table['params'],
        "distinct_days": int(distinct_days),
        "date_range": f"{slots['date'].min().date()} -> {slots['date'].max().date()}",
        "n_curves": n_curves,
        "coverage_by_phase": {
            phase: {
                "cells": len(n_effs),
                "min_n_eff": round(min(n_effs), 2),
                "median_n_eff": round(sorted(n_effs)[len(n_effs) // 2], 2),
            }
            for phase, n_effs in coverage.items()
        },
    }
    with open("models/curve_metrics.json", "w") as f:
        json.dump(curve_metrics, f, indent=2)
    print("  Saved -> models/curve_metrics.json")

    print(f"\n[{datetime.now().isoformat()}] curves.json rebuilt: {n_curves} curves, "
          f"{distinct_days:,} distinct days")


if __name__ == "__main__":
    main()
