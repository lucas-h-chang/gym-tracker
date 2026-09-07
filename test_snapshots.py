"""
test_snapshots.py — guards on the forecast audit trail (snapshots.py,
backfill_snapshots.py, migration 012).

The value of prediction_snapshots is entirely in being trustworthy after the
fact. A row that is subtly mis-shaped is worse than no row, because it will be
averaged into an accuracy number months from now by someone who was not here
when it was written. So these tests are about FAITHFULNESS: that a snapshot
says exactly what was published, that a "we published nothing" state survives
as itself rather than as a hole, and that the natural key really is unique so a
re-run cannot double-count a minute.

No network and no Supabase — everything here is pure shaping.
"""
import numpy as np
import pandas as pd
import pytest

import snapshots
import backfill_snapshots as bf


# ---------------------------------------------------------------------------
# slot_label — the string clients display, so boundaries matter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slot,expected", [
    (0,  "12:00 AM"),
    (28, "7:00 AM"),    # open
    (47, "11:45 AM"),   # last slot before noon
    (48, "12:00 PM"),   # noon is PM, not 0 PM
    (68, "5:00 PM"),
    (92, "11:00 PM"),   # close
    (95, "11:45 PM"),
])
def test_slot_label(slot, expected):
    assert snapshots.slot_label(slot) == expected


# ---------------------------------------------------------------------------
# build_row
# ---------------------------------------------------------------------------

def test_empty_preds_is_preserved_not_dropped():
    """Below MIN_OBSERVED we publish nothing, and that is a real state.

    If this row were skipped, a gap in the table would be ambiguous between
    "the model declined to correct" and "the job never ran" — which is the one
    distinction an audit table exists to make.
    """
    row = snapshots.build_row("2026-09-03", "2026-09-03T07:30:00-07:00",
                              [], {}, cut_slot=30, n_obs=3)
    assert row["preds"] == []
    assert row["n_obs"] == 3
    assert row["gap_day"] is None


def test_base_keys_are_strings_and_values_rounded():
    """jsonb object keys are text in Postgres; stringify here so the round-trip
    does not depend on the client library's serializer."""
    row = snapshots.build_row("2026-09-03", "t", [], {28: 39.4999, 29: 40.0})
    assert row["base"] == {"28": 39.5, "29": 40.0}


def test_numpy_scalars_are_coerced_to_python_types():
    """The backfill feeds numpy scalars straight out of a DataFrame; the JSON
    encoder in the Supabase client cannot serialize those."""
    row = snapshots.build_row("2026-09-03", "t", [], {np.int64(28): np.float64(39.5)},
                              cut_slot=np.int64(36), last_slot=np.int64(35),
                              n_obs=np.int64(8),
                              gaps=(np.float64(4.3), np.float64(0.4), np.float64(1.3)))
    for k in ("cut_slot", "last_slot", "n_obs"):
        assert type(row[k]) is int, k
    for k in ("gap_day", "gap_recent", "gap_last"):
        assert type(row[k]) is float, k
    assert list(row["base"]) == ["28"]
    assert type(row["base"]["28"]) is float


def test_source_defaults_to_live():
    assert snapshots.build_row("2026-09-03", "t", [], {})["source"] == "live"


# ---------------------------------------------------------------------------
# backfill_snapshots.rows_from_frame
# ---------------------------------------------------------------------------

def make_frame(cuts=(9, 10), day=pd.Timestamp("2026-09-03").date()):
    """A minimal stand-in for replay_day.py --dump."""
    rows = []
    for cut in cuts:
        last_slot = cut * 4 - 1          # deliberately != cut_slot
        for slot in range(cut * 4 + 4, cut * 4 + 12):
            rows.append({
                "date": day, "cut_hour": cut, "slot": slot,
                "horizon": (slot - last_slot) / 4,
                "actual": 70.0, "base": 65.0, "new": 66.5, "old": 65.0,
                "gap_day": 4.3, "gap_recent": 0.4, "gap_last": 1.3,
                "n_obs": 8, "segment": "regular",
            })
    return pd.DataFrame(rows)


def test_one_row_per_cut_with_the_published_shape():
    rows = bf.rows_from_frame(make_frame())
    assert len(rows) == 2
    r = rows[0]
    assert r["source"] == "reconstructed"
    assert len(r["preds"]) == 8
    p = r["preds"][0]
    # x is the fractional hour clients plot on; slot 40 -> 10.0
    assert p == {"x": 40 / 4, "y": 66.5, "w": 1.0, "label": "10:00 AM"}


def test_last_slot_is_recovered_from_horizon_not_assumed():
    """carry_model measures horizons from the last slot that fed the gaps, which
    is not always the cut slot. Getting this wrong shifts every horizon bucket
    in prediction_accuracy by a quarter-hour."""
    rows = bf.rows_from_frame(make_frame(cuts=(9,)))
    assert rows[0]["cut_slot"] == 36
    assert rows[0]["last_slot"] == 35


def test_computed_at_is_deterministic_so_reruns_upsert():
    """computed_at is half the natural key. If it moved between runs, a second
    backfill would lay down a duplicate day and silently double-weight it."""
    a = bf.rows_from_frame(make_frame())
    b = bf.rows_from_frame(make_frame())
    assert [r["computed_at"] for r in a] == [r["computed_at"] for r in b]
    assert a[0]["computed_at"].startswith("2026-09-03T09:00:00-07:00")


def test_natural_key_is_unique_within_a_backfill():
    rows = bf.rows_from_frame(make_frame(cuts=(9, 10, 11, 12)))
    keys = {(r["source"], r["computed_at"]) for r in rows}
    assert len(keys) == len(rows)


def test_missing_columns_fail_loudly():
    """A change to replay_day's dump must break here, not quietly backfill NULLs."""
    df = make_frame().drop(columns=["gap_recent"])
    with pytest.raises(SystemExit):
        bf.rows_from_frame(df)
