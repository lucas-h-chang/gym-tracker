"""
snapshots.py — shape one row for the `prediction_snapshots` audit table.

WHY THIS TABLE EXISTS
---------------------
`today_summary` is a SINGLE ROW per day that today_builder.py overwrites every
15 minutes, and it only ever holds the slots still in the future. So the moment
4 PM arrives, there is no record anywhere of what the site was telling people at
9 AM. "How good was the forecast this morning?" could only be answered by
re-deriving it with replay_day.py, retraining the whole curve table to do it.

This table keeps every published forecast instead of the latest one. Live
accuracy tracking then becomes a join against capacity_log rather than a model
rebuild — see the `prediction_accuracy` view in
migrations/012_prediction_snapshots.sql.

WHAT A ROW IS
-------------
Exactly what a client would have rendered at `computed_at`, self-contained:

  preds  the corrected forecast today_builder published (may be [] when there
         were too few readings to correct anything — that is a real, meaningful
         state, not a missing row)
  base   the uncorrected baseline it was built from, snapshotted because
         predictions_builder rebuilds `predictions` daily and purges past
         slots, so the baseline a day was served from is otherwise lost too

Storing `base` on every row duplicates ~40 floats 40 times a day (~6 MB/year).
That is bought deliberately: a self-contained audit row can be scored years
later without reasoning about what else had been rebuilt in between.

SOURCES
-------
  live           written by today_builder.py at publish time. Ground truth for
                 "what did the site actually say".
  reconstructed  written by backfill_snapshots.py from a replay_day.py run.
                 Close to, but NOT identical to, what was served: replay builds
                 the curve table as of the day rather than the preceding Sunday
                 and fits carry coefficients as of the month. Always filter on
                 `source` before quoting a number as production accuracy.
"""

SOURCE_LIVE          = "live"
SOURCE_RECONSTRUCTED = "reconstructed"

MODEL_CARRY = "carry"


def slot_label(slot):
    """Quarter-hour slot index (0-95) -> '5:15 PM', the label clients display."""
    h, m = slot // 4, (slot % 4) * 15
    return f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def build_row(date, computed_at, preds, base, *, source=SOURCE_LIVE,
              model=MODEL_CARRY, cut_slot=None, last_slot=None, n_obs=None,
              gaps=None):
    """One `prediction_snapshots` row, ready to insert.

    date         PT calendar date being forecast ('YYYY-MM-DD' or a date)
    computed_at  ISO8601 timestamp the forecast was published
    preds        the published [{x, y, w, label}] list, exactly as today_summary
                 receives it. [] is valid and is stored as [].
    base         {slot: pct} baseline the correction was applied to
    gaps         (gap_day, gap_recent, gap_last) or None when uncorrected

    Diagnostics are nullable on purpose: below carry_model.MIN_OBSERVED readings
    there are no gaps to record, and that row still needs to exist so a gap in
    the table means "the job did not run", never "the job ran and said nothing".
    """
    gap_day, gap_recent, gap_last = gaps if gaps else (None, None, None)
    return {
        "date":        str(date),
        "computed_at": computed_at,
        "source":      source,
        "model":       model,
        "cut_slot":    None if cut_slot  is None else int(cut_slot),
        "last_slot":   None if last_slot is None else int(last_slot),
        "n_obs":       None if n_obs     is None else int(n_obs),
        "gap_day":     None if gap_day    is None else round(float(gap_day), 3),
        "gap_recent":  None if gap_recent is None else round(float(gap_recent), 3),
        "gap_last":    None if gap_last   is None else round(float(gap_last), 3),
        "preds":       preds,
        # jsonb object keys are text in Postgres either way; stringify here so
        # the round-trip is stable rather than depending on the client library.
        "base":        {str(int(s)): round(float(v), 1) for s, v in base.items()},
    }
