"""
supabase_io.py — shared Supabase read/parse helpers.

Extracted (2026-07-21) so predictions_builder.py and build_curves.py didn't have to
import train.py just to get parse_supabase_timestamps.

This is now the single definition. carry_data.py and backtest.py kept private
copies until 2026-09-09; nothing enforced that the three agreed, and a silent
divergence here shifts every timestamp by an hour at a DST boundary.
"""
import os
import time
import pandas as pd


def client():
    """Create credentials/network clients only at an entry point, never on import."""
    from supabase import create_client
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


def execute_read(make_query, *, attempts=3, sleep=time.sleep):
    """Retry only transient reads. Never replay an arbitrary database write."""
    for attempt in range(attempts):
        try:
            return make_query().execute()
        except Exception as exc:
            code = str(getattr(exc, "code", ""))
            transient = code in {"502", "503", "504", "57014", "08006"}
            transient |= any(s in str(exc).lower() for s in (
                "bad gateway", "service unavailable", "gateway timeout",
                "connection", "timed out", "timeout", "temporarily unavailable"))
            if not transient or attempt + 1 == attempts:
                raise
            sleep(2 ** attempt)


def parse_supabase_timestamps(series):
    # Supabase returns TIMESTAMPTZ as UTC. Convert to PT wall-clock, then drop tz so
    # engineer_features() / curve_model.py see the same naive-PT timestamps that
    # predictions_builder and build_curves feed at inference/build time.
    return (
        pd.to_datetime(series, utc=True, format='ISO8601')
          .dt.tz_convert('America/Los_Angeles')
          .dt.tz_localize(None)
    )


def paginated_fetch(sb, table, select, *, gte=None, lte=None, order="timestamp", batch=1000):
    """Fetch a stable ordered table, respecting server-side response caps.

    Filtering uses the ordering column. Read retries recreate each request.
    1,000-row pages avoid the gateway timeout seen in the weekly curve build.
    Only an empty page ends the scan; a smaller response may be a server cap.
    """
    offset, rows = 0, []
    while True:
        def query():
            q = sb.table(table).select(select)
            if gte is not None:
                q = q.gte(order, gte)
            if lte is not None:
                q = q.lte(order, lte)
            return q.range(offset, offset + batch - 1).order(order)
        page = execute_read(query).data
        rows.extend(page)
        # A server-side response cap can be lower than the requested range.
        # Advance by what arrived and stop only on an empty page.
        if not page:
            break
        offset += len(page)
    return rows
