"""
blend_sweep.py — rolling-origin sweep of curve_model's blend_window_days.

Motivated by the 2026-08-17..22 forecast audit: the week before Fall 2026
classes ran 20-30pp hotter than the summer_break_8 curve, and with
blend_window_days=5 the first_week blend only switched on 5 days out.
This asks whether widening the window helps on properly held-out history
instead of on that one week.

blend_window_days is a PREDICT-time argument (curve_model.predict_one takes
it as a kwarg; build_table only stores it in params and never reads it), so
each origin's table is built exactly once and re-scored under every window.

Reuses backtest.py's origins, grid, and segment definitions. Loads .env
itself; the key is never printed.

Run:  python3 blend_sweep.py
Writes blend_sweep_report.json next to this script.
"""
import os
import sys
import json
import pickle
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ── Load .env without echoing values ────────────────────────────────────────
_envpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_envpath):
    for line in open(_envpath):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
if "SUPABASE_SERVICE_KEY" not in os.environ and "SUPABASE_KEY" in os.environ:
    os.environ["SUPABASE_SERVICE_KEY"] = os.environ["SUPABASE_KEY"]

import curve_model as cm
from academic_calendar import classify_date, days_to_sem_start, days_to_sem_end
from backtest import open_slots_grid, fetch_capacity_log, segment_for_date, hour_bucket

CACHE = os.environ.get("SWEEP_CACHE", "/tmp/rsf_capacity_cache.pkl")
WINDOWS = [0, 5, 7, 10, 14, 21]   # 0 == blend disabled entirely
PARAMS = {**cm.DEFAULT_PARAMS, "week_levels": True}   # match build_curves.py / production

# backtest.ORIGINS stops at 2026-06-01. Add Jul/Aug 2026 so the Fall-2026
# ramp is also scored at a short horizon (those origins score only through
# the last day of actuals, which is fine — they are just thinner).
ORIGINS = [d for d in (date(y, m, 1) for y in (2023, 2024, 2025, 2026) for m in range(1, 13))
           if date(2023, 1, 1) <= d <= date(2026, 8, 1)]


def load_slots():
    if os.path.exists(CACHE):
        print(f"Using cached capacity_log -> {CACHE}")
        with open(CACHE, "rb") as f:
            raw = pickle.load(f)
    else:
        raw = fetch_capacity_log()
        with open(CACHE, "wb") as f:
            pickle.dump(raw, f)
        print(f"Cached capacity_log -> {CACHE}")
    slots = cm.prepare_slots(raw)
    n_days = slots['date'].nunique()
    print(f"Prepared {len(slots):,} (date, slot) rows, "
          f"{n_days:,} distinct days, "
          f"{slots['date'].min().date()} -> {slots['date'].max().date()}")
    if n_days < 1000:
        # capacity_log's RLS policy caps the anon/publishable role at ~3 days.
        # A rolling-origin backtest needs the full 2022+ history, so this is
        # almost always "the key in .env is the anon key, not service_role".
        os.path.exists(CACHE) and os.remove(CACHE)
        sys.exit(
            f"\nABORT: only {n_days} distinct days returned. capacity_log is "
            "RLS-limited to ~3 days for the anon role.\nSet SUPABASE_SERVICE_KEY "
            "(Supabase dashboard -> Project Settings -> API -> service_role) in "
            "gym-tracker/.env and re-run.\n(The bad cache has been removed.)"
        )
    return slots


def ramp_direction(d):
    """'into_sem' for the pre-semester ramp, 'out_of_sem' post-finals, else None."""
    phase = classify_date(d)
    if not (phase in ("winter_break", "spring_break") or phase.startswith("summer_break_")):
        return None
    if 1 <= days_to_sem_start(d) <= 21:
        return "into_sem"
    if 1 <= -days_to_sem_end(d) <= 21:
        return "out_of_sem"
    return None


def main():
    slots = load_slots()
    actual_map = {(r.date.date(), int(r.slot)): float(r.percent_full)
                  for r in slots.itertuples(index=False)}

    records = []
    for origin in ORIGINS:
        train = slots[slots['date'] < pd.Timestamp(origin)]
        if train.empty or train['date'].nunique() < 200:
            continue
        table = cm.build_table(train, PARAMS, build_date=origin)

        grid = open_slots_grid(origin, 90)
        pairs = [(d, s) for d, s in grid if (d, s) in actual_map]
        if not pairs:
            continue

        # w=0 means "never blend": phase_weights' `1 <= dts <= W` tests are all
        # false at W=0, so the hard classify_date phase is used unblended.
        preds = {w: cm.predict(table, pairs, blend_window_days=w) for w in WINDOWS}

        for i, (d, s) in enumerate(pairs):
            rec = {"origin": origin, "date": d, "slot": s, "actual": actual_map[(d, s)],
                   "segment": segment_for_date(d), "ramp_dir": ramp_direction(d),
                   "dts": days_to_sem_start(d), "hour": s // 4}
            for w in WINDOWS:
                rec[f"w{w}"] = preds[w][i]
            records.append(rec)
        print(f"  origin {origin}: {len(pairs):,} scored slots")

    df = pd.DataFrame(records).dropna(subset=["actual"])
    print(f"\n{len(df):,} scored (origin, date, slot) rows total\n")

    def stat(sub, w):
        col = sub[f"w{w}"].dropna()
        if col.empty:
            return None, None, None
        a = sub.loc[col.index, "actual"]
        return (round(float((col - a).abs().mean()), 2),
                round(float((col - a).mean()), 2),
                int(len(col)))

    def block(title, sub):
        if sub.empty:
            print(f"\n{title}: no rows"); return {}
        print(f"\n{title}  (n={len(sub):,})")
        print(f"  {'window':>8}{'MAE':>9}{'bias':>9}")
        out = {}
        best = min(WINDOWS, key=lambda w: stat(sub, w)[0] if stat(sub, w)[0] is not None else 1e9)
        for w in WINDOWS:
            mae, bias, n = stat(sub, w)
            if mae is None:
                continue
            out[w] = {"mae": mae, "bias": bias, "n": n}
            flag = "  <-- best" if w == best else ("   (deployed)" if w == 5 else "")
            label = "no blend" if w == 0 else f"W={w}"
            print(f"  {label:>8}{mae:>8.2f}{bias:>+9.2f}{flag}")
        return out

    report = {"params": PARAMS, "windows": WINDOWS,
              "origins": [str(o) for o in ORIGINS], "n_rows": int(len(df)), "slices": {}}

    report["slices"]["overall"] = block("OVERALL (all 90-day horizons, all segments)", df)
    report["slices"]["ramp_into_sem"] = block(
        "PRE-SEMESTER RAMP  (break phase, 1-21 days before a semester start)",
        df[df["ramp_dir"] == "into_sem"])
    report["slices"]["ramp_out_of_sem"] = block(
        "POST-SEMESTER RAMP (break phase, 1-21 days after a semester end)",
        df[df["ramp_dir"] == "out_of_sem"])

    for lo, hi in [(1, 5), (6, 10), (11, 14), (15, 21)]:
        sub = df[(df["ramp_dir"] == "into_sem") & (df["dts"] >= lo) & (df["dts"] <= hi)]
        report["slices"][f"dts_{lo}_{hi}"] = block(
            f"  pre-semester, {lo}-{hi} days out", sub)

    print("\n" + "=" * 62)
    print("REGRESSION CHECK: MAE by segment")
    print("=" * 62)
    segs = sorted(df["segment"].unique())
    print(f"  {'segment':<14}" + "".join(f"{('no blend' if w==0 else f'W={w}'):>10}" for w in WINDOWS))
    seg_out = {}
    for seg in segs:
        sub = df[df["segment"] == seg]
        row = {w: stat(sub, w)[0] for w in WINDOWS}
        seg_out[seg] = row
        print(f"  {seg:<14}" + "".join(f"{row[w]:>10.2f}" for w in WINDOWS))
    report["slices"]["by_segment"] = seg_out

    # Fall 2026 specifically (the week that prompted this)
    aug = df[(df["date"] >= date(2026, 8, 10)) & (df["date"] <= date(2026, 8, 25))]
    report["slices"]["aug_2026_ramp"] = block(
        "FALL 2026 PRE-SEMESTER WEEK (2026-08-10 .. 08-25, all origins)", aug)

    with open("blend_sweep_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nSaved -> blend_sweep_report.json")


if __name__ == "__main__":
    main()
