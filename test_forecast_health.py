from datetime import date

import forecast_health as fh


def sample_rows(days=8, carry_delta=-2.0, trailing_delta=-1.0):
    rows = []
    for i in range(days):
        actual = 50.0
        curve = actual + 10.0
        base = curve + trailing_delta
        final = base + carry_delta
        rows.append({
            "snapshot_id": i, "date": f"2026-09-{i + 1:02d}",
            "horizon_h": 0.5 if i % 2 == 0 else 7.0,
            "pct": final, "base_pct": base, "curve_pct": curve,
            "actual_pct": actual,
        })
    return rows


def test_stats_reports_mae_bias_and_count():
    result = fh.stats(sample_rows(), "pct")
    assert result == {"n": 8, "mae": 7.0, "bias": 7.0}


def test_verdict_uses_carry_gain_and_requires_seven_days():
    assert fh.verdict(sample_rows()).startswith("HEALTHY")
    assert fh.verdict(sample_rows(days=6)) == "INSUFFICIENT DATA"
    assert fh.verdict(sample_rows(carry_delta=2.0)).startswith("DEGRADED")


def test_render_distinguishes_trailing_and_carry():
    text = fh.render(sample_rows(), 8, date(2026, 9, 1), date(2026, 9, 8), True)
    assert "Carry improvement:    +2.00 pp" in text
    assert "Trailing improvement: +1.00 pp" in text
    assert "0–1h" in text and "6h+" in text


def test_render_explains_pre_migration_curve_gap():
    rows = sample_rows()
    for row in rows:
        row.pop("curve_pct")
    text = fh.render(rows, 8, date(2026, 9, 1), date(2026, 9, 8), False)
    assert "N/A until migration 013 is applied" in text


def test_recent_open_dates_skips_known_full_closure():
    dates = fh.recent_open_dates(2, date(2026, 8, 24))
    assert dates == [date(2026, 8, 21), date(2026, 8, 22)]
