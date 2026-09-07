"""
backfill_snapshots.py — turn a replay_day.py run into `prediction_snapshots` rows.

WHY IT WORKS THIS WAY
---------------------
`prediction_snapshots` only starts filling from the moment today_builder.py is
deployed with the snapshot write. Every day before that is gone from
today_summary, and the only honest way to recover it is to re-derive it — which
is exactly what replay_day.py already does, correctly and under test.

So this script does NOT reimplement the model. It reads the scored frame that
`replay_day.py --dump` writes and reshapes it into snapshot rows. There is one
forecaster in this repo, and it stays that way.

    python3 replay_day.py --date 2026-09-03 --dump /tmp/replay.pkl
    python3 backfill_snapshots.py /tmp/replay.pkl            # preview only
    python3 backfill_snapshots.py /tmp/replay.pkl --write    # actually insert

HOW RECONSTRUCTED ROWS DIFFER FROM LIVE ONES
--------------------------------------------
They are written with source='reconstructed' and must never be averaged in with
live rows, because they are close to but not the same as what was served:

  * replay builds the curve table as of the replayed day; production rebuilds it
    weekly, on the preceding Sunday.
  * replay fits carry coefficients as of the first of the day's month;
    production refits quarterly.
  * replay cuts hourly by default; production publishes every 15 minutes, so a
    backfilled day has ~8 rows where a live day has ~40.

All three make the reconstruction slightly *fresher* than production, so treat
reconstructed accuracy as a mild upper bound on what was actually served.

Loads .env itself; the key is never printed.
"""
import os
import sys
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import snapshots

_envpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_envpath):
    for line in open(_envpath):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

PT = ZoneInfo("America/Los_Angeles")

# Columns replay_day.py's --dump frame is guaranteed to carry. Asserted rather
# than assumed so a change to the harness fails loudly here instead of quietly
# backfilling a column of NULLs.
REQUIRED = {"date", "cut_hour", "slot", "actual", "base", "new",
            "gap_day", "gap_recent", "gap_last", "n_obs"}


def rows_from_frame(df, model=snapshots.MODEL_CARRY):
    """One snapshot row per (date, cut_hour) in a replay_day --dump frame."""
    missing = REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"dump frame is missing columns: {sorted(missing)}")

    out = []
    for (day, cut_hour), g in df.groupby(["date", "cut_hour"], sort=True):
        g = g.sort_values("slot")
        cut_slot = int(cut_hour) * 4
        # replay measures its horizons from the last slot that fed the gaps.
        # Recover it from the frame rather than assuming it equals cut_slot:
        # horizon = (slot - last_slot) / 4, so last_slot = slot - 4 * horizon.
        first = g.iloc[0]
        last_slot = int(round(first["slot"] - 4 * first["horizon"])) \
            if "horizon" in g.columns else cut_slot

        preds = [
            {"x": int(r.slot) / 4, "y": round(float(r.new), 1), "w": 1.0,
             "label": snapshots.slot_label(int(r.slot))}
            for r in g.itertuples()
        ]
        base = {int(r.slot): float(r.base) for r in g.itertuples()}

        # A synthetic but deterministic publish time: the top of the cut hour in
        # PT. Deterministic matters — it is half the natural key, so re-running
        # this script updates the same rows instead of duplicating the day.
        computed_at = datetime(
            day.year, day.month, day.day, int(cut_hour), 0, tzinfo=PT
        ).isoformat()

        out.append(snapshots.build_row(
            day, computed_at, preds, base,
            source=snapshots.SOURCE_RECONSTRUCTED,
            model=model,
            cut_slot=cut_slot,
            last_slot=last_slot,
            n_obs=int(first["n_obs"]),
            gaps=(first["gap_day"], first["gap_recent"], first["gap_last"]),
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="path written by replay_day.py --dump")
    ap.add_argument("--write", action="store_true",
                    help="actually insert; without it this only prints a preview")
    args = ap.parse_args()

    df = pd.read_pickle(args.dump)
    rows = rows_from_frame(df)

    days = sorted({r["date"] for r in rows})
    print(f"{len(rows)} reconstructed snapshots across {len(days)} day(s): "
          f"{days[0]} .. {days[-1]}")
    for r in rows:
        print(f"  {r['computed_at'][:16]}  cut_slot {r['cut_slot']:>2}  "
              f"n_obs {r['n_obs']:>2}  {len(r['preds']):>2} slots  "
              f"gaps {r['gap_day']:+.1f}/{r['gap_recent']:+.1f}/{r['gap_last']:+.1f}")

    if not args.write:
        print("\npreview only — re-run with --write to insert")
        return

    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    sb.table("prediction_snapshots").upsert(
        rows, on_conflict="source,computed_at").execute()
    print(f"\ninserted/updated {len(rows)} rows (source='reconstructed')")


if __name__ == "__main__":
    main()
