"""
supabase_io.py — shared Supabase read/parse helpers.

Extracted (2026-07-21) so predictions_builder.py and build_curves.py don't need to
import from train.py just to get parse_supabase_timestamps. train.py itself is
unchanged and stays the source of truth for the frozen RF pipeline (see
gym-tracker/legacy/README.md and CLAUDE.md for why it wasn't moved there).
"""
import pandas as pd


def parse_supabase_timestamps(series):
    # Supabase returns TIMESTAMPTZ as UTC. Convert to PT wall-clock, then drop tz so
    # engineer_features() / curve_model.py see the same naive-PT timestamps that
    # predictions_builder and build_curves feed at inference/build time.
    return (
        pd.to_datetime(series, utc=True, format='ISO8601')
          .dt.tz_convert('America/Los_Angeles')
          .dt.tz_localize(None)
    )


def paginated_fetch(sb, table, select, *, gte=None, lte=None, order="timestamp", batch=9000):
    """
    Reproduces the `while True: .range(offset, offset+batch-1); ... if len(page) <
    batch: break` pagination loop duplicated across weekly_builder.py,
    day_profiles_builder.py, today_builder.py, predictions_builder.py, and
    build_curves.py. gte/lte (when given) filter on the same column used for
    `order` — true for every call site refactored onto this helper.

    Call sites with extra .eq() filters (e.g. today_builder's day_profiles query)
    or an exclusive `.lt()` bound (e.g. predictions_builder's stale-row purge
    query) don't map cleanly onto this signature and were left as their own
    loops rather than forced to fit.
    """
    offset, rows = 0, []
    while True:
        q = sb.table(table).select(select)
        if gte is not None:
            q = q.gte(order, gte)
        if lte is not None:
            q = q.lte(order, lte)
        page = q.range(offset, offset + batch - 1).order(order).execute().data
        rows.extend(page)
        if len(page) < batch:
            break
        offset += batch
    return rows
