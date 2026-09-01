"""
replay_day.py — replay a real day reading-by-reading and score the new level
correction against the model currently in production.

THE QUESTION THIS ANSWERS
-------------------------
"Pretend we only have data up to and not including day D. Now walk through D
one reading at a time. At 9 AM, at 10 AM, at 11 AM — what would each model have
said about the rest of that day, and how close was it?"

Three models are scored on identical cuts:

  BASE  the deployed `predictions` baseline, uncorrected
  OLD   today_builder.py as it ships today — K=5 similarity nowcast, inverse
        distance weights, 4h anchor decay, 2h weight ramp
  NEW   carry_model.py — base curve plus a fitted level correction

WHAT MAKES THIS HONEST
----------------------
- The curve table for day D is built from data STRICTLY BEFORE D.
- The 28-day trailing correction is rebuilt from [D-28, D), reproducing
  predictions_builder.py so BASE matches what production actually serves.
  This matters: fitting or scoring against the raw curve would double-count
  the drift that layer already removes on days where it fires.
- Carry coefficients are fitted on data strictly before the FIRST OF D'S MONTH
  — deliberately staler than "before D", so the fit can never be flattered by
  data near the day it is scoring. Production refits quarterly, so this is
  still optimistic relative to nothing and conservative relative to reality.
- OLD's candidate pool is restricted to `date <= D - 8`, mirroring
  today_builder's EXCLUDE_WINDOW so it cannot see the day it is predicting.

One simplification, applied equally to all three models so it favours none:
the curve table is rebuilt as of D rather than as of the preceding Sunday, so
it is up to 7 days fresher than production's weekly rebuild. With
halflife_days=365 that difference is negligible.

Run:
  python3 replay_day.py --date 2026-08-30            # one day, curves per cut
  python3 replay_day.py --date 2026-08-30 --cut 11   # one cut, 15-min detail
  python3 replay_day.py --last 30                    # aggregate + acceptance bar

Loads .env itself; the key is never printed.
"""
import os
import sys
import json
import pickle
import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

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

import carry_model as km
from academic_calendar import is_semester_day
from carry_data import (
    load_matrices, open_slot_range, segment_for_date,
    CORRECTION_DAYS, CORRECTION_MIN_N,
)

# ---------------------------------------------------------------------------
# The model currently in production, replicated so it can be scored fairly.
# ---------------------------------------------------------------------------
# Constants lifted from today_builder.py. If that file changes and this does
# not, the comparison silently stops being a comparison — so they are named
# here rather than inlined.
NC_K               = 5
NC_BLEND_CEIL      = 0.9
NC_BLEND_HORIZON   = 2.0
NC_OVERLAP_THRESH  = 0.70
NC_MIN_SLOTS       = 4
NC_ANCHOR_DECAY_H  = 4.0
NC_EXCLUDE_DAYS    = 8
NC_DATA_CUTOFF     = date(2022, 1, 1)


def nowcast_for_cut(day_i, cut_slot, dates, M, base_row, obs_idx, cand_cache):
    """today_builder.compute_similarity_predictions for one (day, cut).

    Returns P[96] = the blended (1-w)*base + w*sim value the live site would
    have drawn, NaN where the nowcast produced nothing. K=5 neighbours by RMSE
    over the observed slots, inverse-distance weights, an anchor offset decaying
    over NC_ANCHOR_DECAY_H hours, and the per-slot weight that starts at
    NC_BLEND_CEIL and ramps to 0 over NC_BLEND_HORIZON hours — the ramp that
    this whole change exists to remove.

    Candidates are capped at `date <= D - 8`, mirroring today_builder's
    EXCLUDE_WINDOW, so it cannot see the day it is predicting.
    """
    d = dates[day_i]
    if len(obs_idx) < NC_MIN_SLOTS:
        return None

    key = (pd.Timestamp(d).dayofweek, is_semester_day(d))
    pool = cand_cache.get(key)
    if pool is None:
        pool = np.array([
            (pd.Timestamp(c).dayofweek == key[0]) and (is_semester_day(c) == key[1])
            for c in dates
        ])
        cand_cache[key] = pool
    mask = pool & (dates <= d - timedelta(days=NC_EXCLUDE_DAYS)) & (dates >= NC_DATA_CUTOFF)
    if not mask.any():
        return None

    sub = M[mask]
    obs = sub[:, obs_idx]
    keep = (~np.isnan(obs)).sum(axis=1) >= NC_OVERLAP_THRESH * len(obs_idx)
    if not keep.any():
        return None
    sub, obs = sub[keep], obs[keep]

    dist = np.sqrt(np.nanmean((obs - M[day_i, obs_idx][None, :]) ** 2, axis=1))
    order = np.argsort(dist)[:NC_K]
    top, dtop = sub[order], dist[order]

    wts = 1.0 / (dtop + 1e-6)
    wts /= wts.sum()
    valid = ~np.isnan(top)
    denom = (valid * wts[:, None]).sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        sim = np.nansum(np.where(valid, top, 0.0) * wts[:, None], axis=0) / denom
    sim[denom == 0] = np.nan

    last = obs_idx[-1]
    if np.isnan(sim[last]):
        return None
    offset = M[day_i, last] - sim[last]

    out = np.full(96, np.nan)
    lo, hi = open_slot_range(d)
    for s in range(max(cut_slot + 1, lo), hi):
        if np.isnan(sim[s]) or np.isnan(base_row[s]):
            continue
        gap_h = (s - last) / 4.0
        anchored = np.clip(sim[s] + offset * max(0.0, 1.0 - gap_h / NC_ANCHOR_DECAY_H), 0.0, 100.0)
        w = NC_BLEND_CEIL * max(0.0, 1.0 - gap_h / NC_BLEND_HORIZON)
        out[s] = (1 - w) * base_row[s] + w * anchored
    return out


# ---------------------------------------------------------------------------
# Carry coefficients, fitted strictly before the month of the replayed day
# ---------------------------------------------------------------------------

_fit_cache = {}

def carry_table_for(day, dates, M, base_M, scored_origin,
                    fit_cut_step=km.SLOTS_PER_HOUR):
    key = (day.year, day.month, fit_cut_step)
    if key in _fit_cache:
        return _fit_cache[key]

    cutoff = date(day.year, day.month, 1)
    usable = {d for d in scored_origin if d < cutoff}
    samples = km.make_samples(
        dates, M, base_M, open_slot_range,
        day_filter=lambda d: d in usable, cut_step=fit_cut_step,
    )
    if samples.empty:
        raise SystemExit(f"no training samples before {cutoff}")
    table = km.build_table(samples, params={"fit_cutoff": str(cutoff)})
    print(f"  carry fit for {key[0]}-{key[1]:02d} (cut_step={fit_cut_step}): "
          f"{table['n_rows']:,} rows / {table['n_days']:,} days before {cutoff}")
    _fit_cache[key] = table
    return table


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay(day, dates, M, base_M, scored_origin, cand_cache,
           cut_step=km.SLOTS_PER_HOUR, fit_cut_step=km.SLOTS_PER_HOUR):
    """Per-cut predictions for one day. Returns (rows, per_cut_detail)."""
    i = int(np.searchsorted(dates, day))
    if i >= len(dates) or dates[i] != day:
        return [], {}
    lo, hi = open_slot_range(day)
    if lo >= hi:
        return [], {}

    actual, base = M[i], base_M[i]
    resid = actual - base
    usable = np.arange(lo, hi)[np.isfinite(resid[np.arange(lo, hi)])]
    if len(usable) < km.MIN_OBSERVED + km.SLOTS_PER_HOUR:
        return [], {}

    table = carry_table_for(day, dates, M, base_M, scored_origin, fit_cut_step)
    rows, detail = [], {}

    for cut in range(lo + km.SLOTS_PER_HOUR, hi - km.SLOTS_PER_HOUR, cut_step):
        obs = usable[usable <= cut]
        g = km.compute_gaps(obs, actual[obs], base[obs])
        if g is None:
            continue
        gap_day, gap_recent, gap_last, last_slot, n_obs = g

        base_by_slot = {int(s): float(base[s]) for s in range(lo, hi) if np.isfinite(base[s])}
        new_pred = km.apply_to_day(table, cut, last_slot,
                                   (gap_day, gap_recent, gap_last), base_by_slot, lo, hi,
                                   n_obs=n_obs)
        old_pred = nowcast_for_cut(i, cut, dates, M, base, obs, cand_cache)

        per_cut = []
        for s in usable[usable > cut]:
            s = int(s)
            o = float(old_pred[s]) if (old_pred is not None and np.isfinite(old_pred[s])) else np.nan
            n = new_pred.get(s, np.nan)
            rec = {
                "date": day, "cut_hour": cut // km.SLOTS_PER_HOUR, "slot": s,
                "horizon": (s - last_slot) / km.SLOTS_PER_HOUR,
                "actual": float(actual[s]), "base": float(base[s]),
                "new": n, "old": o,
                "gap_day": gap_day, "gap_recent": gap_recent, "gap_last": gap_last,
                "n_obs": n_obs, "segment": segment_for_date(day),
            }
            rows.append(rec)
            per_cut.append(rec)
        detail[cut // km.SLOTS_PER_HOUR] = per_cut

    return rows, detail


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _mae(col, sub):
    v = sub[col].values.astype(float)
    a = sub["actual"].values.astype(float)
    ok = np.isfinite(v) & np.isfinite(a)
    return float(np.abs(v[ok] - a[ok]).mean()) if ok.any() else float("nan")


def print_day(day, detail):
    print(f"\n{'='*78}\n{day}  ({segment_for_date(day)}, {day.strftime('%A')})\n{'='*78}")
    for ch in sorted(detail):
        recs = detail[ch]
        if not recs:
            continue
        g = recs[0]
        print(f"\n  observed through {ch}:00   "
              f"gaps  day {g['gap_day']:+.1f}  hour {g['gap_recent']:+.1f}  "
              f"last {g['gap_last']:+.1f}   ({g['n_obs']} readings)")
        print(f"    {'time':<9}{'actual':>8}{'BASE':>8}{'OLD':>8}{'NEW':>8}"
              f"{'  |err| base/old/new':>24}")
        for r in recs:
            if r["slot"] % 4:
                continue
            h = r["slot"] // 4
            lbl = f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"
            old = f"{r['old']:8.1f}" if np.isfinite(r["old"]) else f"{'-':>8}"
            eb, en = abs(r["base"] - r["actual"]), abs(r["new"] - r["actual"])
            eo = abs(r["old"] - r["actual"]) if np.isfinite(r["old"]) else float("nan")
            print(f"    {lbl:<9}{r['actual']:>8.1f}{r['base']:>8.1f}{old}{r['new']:>8.1f}"
                  f"{eb:>9.1f}{eo:>7.1f}{en:>7.1f}")
        sub = pd.DataFrame(recs)
        print(f"    {'MAE':<9}{'':>8}{_mae('base', sub):>8.2f}"
              f"{_mae('old', sub):>8.2f}{_mae('new', sub):>8.2f}")


def print_cut_detail(day, detail, cut_hour):
    recs = detail.get(cut_hour)
    if not recs:
        print(f"no cut at {cut_hour}:00 for {day}")
        return
    g = recs[0]
    print(f"\n{day}, observed through {cut_hour}:00  "
          f"(gaps {g['gap_day']:+.1f} / {g['gap_recent']:+.1f} / {g['gap_last']:+.1f})\n")
    print(f"  {'time':<9}{'actual':>8}{'BASE':>8}{'OLD':>8}{'NEW':>8}{'NEW step':>10}")
    prev = None
    for r in recs:
        h, m = r["slot"] // 4, (r["slot"] % 4) * 15
        lbl = f"{h % 12 or 12}:{m:02d}{'am' if h < 12 else 'pm'}"
        old = f"{r['old']:8.1f}" if np.isfinite(r["old"]) else f"{'-':>8}"
        step = f"{r['new'] - prev:+10.1f}" if prev is not None else f"{'-':>10}"
        print(f"  {lbl:<9}{r['actual']:>8.1f}{r['base']:>8.1f}{old}{r['new']:>8.1f}{step}")
        prev = r["new"]


def persistence(sub, col):
    """|correction| far out as a fraction of |correction| near in.

    WHY THIS AND NOT A SMOOTHNESS TEST. The first acceptance bar compared the
    published curve's max slot-to-slot jump against the BASE curve's. That was
    wrong: the base curve is a multi-year average, so real troughs are averaged
    out of it and it is smoother than any single day can be. Requiring the
    forecast to match it forbids tracking a genuine recovery — it flagged
    2026-08-31 09:00, where the gym was 26.6pp below the curve in a momentary
    dip and the model predicted a +8.4pp bounce against an actual +8.7pp.

    The defect actually worth preventing is the correction being ramped away on
    a schedule, so the curve drifts back to the base regardless of evidence.
    That is what this measures, and it needs no tuned threshold: over 30 days
    the deployed model scores exactly 0.00 (gone by construction after its
    2-hour ramp) against 0.96 for the level correction.

    Returns NaN when the near-in correction is too small to form a stable
    ratio — there is nothing to persist if today looks like an average day.
    """
    near = sub[(sub["horizon"] >= 0.75) & (sub["horizon"] <= 1.5)]
    far  = sub[sub["horizon"] >= 4.0]
    if len(near) < 2 or len(far) < 2:
        return np.nan
    n = float(np.abs((near[col] - near["base"]).values).mean())
    f = float(np.abs((far[col] - far["base"]).values).mean())
    if not np.isfinite(n) or n < 3.0:
        return np.nan
    return f / n


def report(df):
    print(f"\n{'='*78}\nAGGREGATE — {len(df):,} predictions over {df['date'].nunique()} days"
          f"\n{'='*78}")
    print(f"\n  {'':<22}{'BASE':>9}{'OLD':>9}{'NEW':>9}")
    print(f"  {'overall MAE':<22}{_mae('base', df):>9.3f}{_mae('old', df):>9.3f}"
          f"{_mae('new', df):>9.3f}")

    print(f"\n  BY HORIZON\n  {'horizon':<12}{'n':>8}{'BASE':>9}{'OLD':>9}{'NEW':>9}"
          f"{'NEW-OLD':>10}")
    df = df.copy()
    df["hb"] = df["horizon"].map(km.horizon_bucket)
    per_h = {}
    for hb in km.HB_LABELS:
        sub = df[df["hb"] == hb]
        if len(sub) < 50:
            continue
        b, o, n = _mae("base", sub), _mae("old", sub), _mae("new", sub)
        per_h[hb] = (b, o, n)
        print(f"  {hb:<12}{len(sub):>8,}{b:>9.3f}{o:>9.3f}{n:>9.3f}{n - o:>10.3f}")

    print(f"\n  BY CUT HOUR\n  {'obs through':<12}{'n':>8}{'BASE':>9}{'OLD':>9}{'NEW':>9}")
    per_cut = {}
    for ch, sub in df.groupby("cut_hour"):
        if len(sub) < 50:
            continue
        b, o, n = _mae("base", sub), _mae("old", sub), _mae("new", sub)
        per_cut[int(ch)] = (b, o, n)
        flag = "  <-- NEW worse than BASE" if n > b else ""
        print(f"  {str(ch) + ':00':<12}{len(sub):>8,}{b:>9.3f}{o:>9.3f}{n:>9.3f}{flag}")

    persist = []
    for (d, ch), sub in df.groupby(["date", "cut_hour"]):
        row = {"date": d, "cut_hour": int(ch)}
        for col in ("new", "old"):
            row[col] = persistence(sub, col)
        persist.append(row)
    P = pd.DataFrame(persist)
    p_new = P["new"].dropna()
    p_old = P["old"].dropna()

    print("\n  CORRECTION PERSISTENCE  |corr beyond 4h| / |corr around 1h|")
    print("  1.0 = today's level still counts in the evening")
    print("  0.0 = the correction has been ramped away and the curve is back to BASE")
    for v, lbl in ((p_new, "NEW"), (p_old, "OLD")):
        if len(v):
            print(f"    {lbl:<6} n={len(v):>4}   median {v.median():5.2f}"
                  f"   p10 {v.quantile(.10):5.2f}   p90 {v.quantile(.90):5.2f}")

    print(f"\n{'='*78}\nACCEPTANCE BAR\n{'='*78}")
    ok1 = _mae("new", df) < _mae("old", df)
    bad_h = [h for h, (b, o, n) in per_h.items() if n > o + 0.05]
    ok2 = not bad_h
    bad_b = [h for h, (b, o, n) in per_h.items()
             if float(h[2:-1]) <= 8.0 and n > b + 0.05]
    ok3 = not bad_b
    med = float(p_new.median()) if len(p_new) else float("nan")
    ok4 = len(p_new) > 0 and med >= 0.5

    print(f"  1. NEW beats OLD overall                    "
          f"{'PASS' if ok1 else 'FAIL'}   "
          f"({_mae('new', df):.3f} vs {_mae('old', df):.3f})")
    print(f"  2. NEW beats OLD at every horizon           "
          f"{'PASS' if ok2 else 'FAIL'}" + (f"   worse at: {bad_h}" if bad_h else ""))
    print(f"  3. NEW beats BASE at every horizon <= 8h    "
          f"{'PASS' if ok3 else 'FAIL'}" + (f"   worse at: {bad_b}" if bad_b else ""))
    print(f"  4. Correction persists to the evening       "
          f"{'PASS' if ok4 else 'FAIL'}   "
          f"(median {med:.2f}; OLD is {p_old.median() if len(p_old) else float('nan'):.2f})")
    print("     (machinery continuity is asserted against the shipped")
    print("      models/carry.json in test_carry.py, not here)")
    verdict = ok1 and ok2 and ok3 and ok4
    print(f"\n  VERDICT: {'PASS — clear to proceed to shadow mode' if verdict else 'FAIL — do not ship'}")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="single day to replay, YYYY-MM-DD")
    ap.add_argument("--last", type=int, help="replay the last N days with data")
    ap.add_argument("--cut", type=int, help="with --date: show one cut in 15-min detail")
    ap.add_argument("--dump", help="write the scored frame to this path for analysis")
    ap.add_argument("--since", help="replay every day from this date onward, YYYY-MM-DD")
    ap.add_argument("--cut-step", type=int, default=km.SLOTS_PER_HOUR,
                    help="slots between replayed cuts. 4 = hourly (fast sweep), "
                         "1 = every 15 min, which is what production actually does")
    ap.add_argument("--fit-cut-step", type=int, default=km.SLOTS_PER_HOUR,
                    help="cadence used when FITTING coefficients (default 4)")
    args = ap.parse_args()
    if not args.date and not args.last and not args.since:
        ap.error("pass --date, --last or --since")

    slots, dates, M, base_M, scored_origin = load_matrices()
    scored = sorted(scored_origin)

    if args.date:
        targets = [date.fromisoformat(args.date)]
    elif args.since:
        lo = date.fromisoformat(args.since)
        targets = [d for d in scored if d >= lo]
    else:
        targets = scored[-args.last:]
    print(f"\nReplaying {len(targets)} days at cut_step={args.cut_step} "
          f"({'every 15 min — production cadence' if args.cut_step == 1 else 'hourly'})")

    cand_cache = {}
    all_rows = []
    for d in targets:
        rows, detail = replay(d, dates, M, base_M, scored_origin, cand_cache,
                              args.cut_step, args.fit_cut_step)
        if not rows:
            print(f"  {d}: skipped (closed, or too few readings)")
            continue
        all_rows.extend(rows)
        if args.date:
            if args.cut is not None:
                print_cut_detail(d, detail, args.cut)
            else:
                print_day(d, detail)

    if not all_rows:
        sys.exit("nothing to score")
    df = pd.DataFrame(all_rows)
    if args.dump:
        df.to_pickle(args.dump)
        print(f"wrote scored frame -> {args.dump}")
    if args.last or args.since:
        ok = report(df)
        sys.exit(0 if ok else 1)
    else:
        report(df)


if __name__ == "__main__":
    main()
