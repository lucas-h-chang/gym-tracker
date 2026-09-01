"""
build_carry.py — fit the within-day level correction -> models/carry.json.

Runs alongside build_curves.py in build_curves.yml (weekly), and commits its
artifact the same way. `today_builder.py` then just reads the file every 15
minutes: all the fitting happens here, none of it on the serving path.

WHY THIS IS A BUILD STEP AND NOT RUNTIME
----------------------------------------
`today_builder.py` runs every 15 minutes. Fitting means walking several years of
history to build ~400k regression rows — fine weekly, absurd every quarter hour.
Same split as curve_model/build_curves: the model is a committed file, the
runtime only applies it.

WHY IT NEEDS REFITTING RATHER THAN A ONE-TIME BAKE
--------------------------------------------------
The coefficients are the optimal shrinkage for a particular signal-to-noise
balance, and that balance keeps moving as the base curve improves. Measured
across the history: within-day noise has held flat (8.72 -> 8.75) while
day-to-day spread has shrunk (12.99 -> 8.38) because the curve got ~29% more
accurate. By 2026 the noise exceeds the signal. Coefficients fitted on 2023 data
would over-correct today, so this must run on a schedule, not once.

Run:  python3 build_carry.py
Requires SUPABASE_URL and SUPABASE_SERVICE_KEY.
"""
import os
import json
from datetime import datetime

import carry_model as km
from carry_data import load_matrices, open_slot_range

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "carry.json")


def main():
    slots, dates, M, base_M, scored_origin = load_matrices()

    print("Building regression samples against the deployed baseline...")
    usable = set(scored_origin)
    samples = km.make_samples(
        dates, M, base_M, open_slot_range,
        day_filter=lambda d: d in usable,
    )
    if samples.empty:
        raise SystemExit("no samples — cannot fit")
    print(f"  {len(samples):,} rows over {samples['date'].nunique():,} days")

    table = km.build_table(samples, built_at=datetime.now().isoformat())

    n_cut = len(table["by_cut"])
    n_anch = sum(len(v) for v in table["by_cut"].values())
    print(f"  fitted {n_anch} anchors across {n_cut} cut hours "
          f"(+{len(table['pooled'])} pooled fallback)")

    print("\n  carry by horizon (pooled) — b_day + b_recent + b_last:")
    print(f"    {'horizon':>9}{'n':>10}{'carry':>8}{'R2':>8}")
    for a in table["pooled"]:
        c = a["coef"]
        print(f"    {a['h']:>9.2f}{a['n']:>10,}{c[1] + c[2] + c[3]:>8.3f}"
              f"{a['r2'] if a['r2'] is not None else float('nan'):>8.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(table, f, indent=1)
    print(f"\nWrote {OUT} ({os.path.getsize(OUT):,} bytes)")


if __name__ == "__main__":
    main()
