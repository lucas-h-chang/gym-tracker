from datetime import date, datetime, timedelta

import pytest

from five_minute_health import PT, flag_outages, normalize, paired_points, production_at, render


def rows_at(start="2026-11-16T12:00:00-08:00", counts=(90,) * 20):
    start = datetime.fromisoformat(start)
    rows = []
    for i, count in enumerate(counts):
        tick = start + timedelta(minutes=i * 5)
        rows.append({"tick_at": tick.isoformat(), "observed_at": (tick + timedelta(seconds=10)).isoformat(),
                     "people_count": count, "experiment": "cadence-5m-v1",
                     "production": {"status": "ok", "computed_at": start.isoformat(),
                                    "base": {str(s): 55 for s in range(28, 93)}, "preds": []}})
    return rows


def test_counted_time_not_three_times_as_much_evidence_and_short_lows_stay_valid():
    rows = rows_at(counts=[90, 2, 3, 4, 5, 90, 0, 90])
    samples = flag_outages(normalize(rows))
    assert [r["valid"] for r in samples.values()] == [True, False, False, False, False, True, True, True]
    # No invalidation after just two five-minute lows.
    assert all(r["valid"] for r in flag_outages(normalize(rows_at(counts=[0, 0]))).values())


def test_missing_intervals_do_not_bridge_sensor_runs():
    rows = rows_at(counts=[0] * 4)
    del rows[1:3]
    assert all(r["valid"] for r in flag_outages(normalize(rows)).values())


def test_opening_zeros_preserved_closures_and_late_scrapes_excluded():
    rows = rows_at("2026-11-16T07:00:00-08:00", [0, 0, 0, 0])
    assert all(r["valid"] for r in flag_outages(normalize(rows)).values())
    assert not normalize(rows_at("2026-11-26T12:00:00-08:00", [1]))
    rows[0]["observed_at"] = "2026-11-16T07:02:00-08:00"
    assert len(normalize(rows)) == 3
    assert normalize(rows_at(counts=[180])).popitem()[1]["actual"] == 120


def test_control_uses_corrected_points_over_base_interpolates_and_never_extrapolates():
    issue = datetime(2026, 11, 16, 12, 5, 10, tzinfo=PT)
    control = {"status": "ok", "computed_at": "2026-11-16T12:00:00-08:00",
               "base": {"49": 50, "50": 60}, "preds": [{"x": 12.25, "y": 80}]}
    assert production_at(control, issue, issue.replace(minute=20)) == pytest.approx(80 - 20 / 3)
    assert production_at(control, issue, issue.replace(minute=35)) is None
    control["computed_at"] = "2026-11-16T12:10:00-08:00"
    assert production_at(control, issue, issue.replace(minute=20)) is None
    control["status"] = "stale"
    assert production_at(control, issue, issue.replace(minute=20)) is None


def test_same_issues_and_actuals_for_all_arms_and_no_future_reading_in_coarse_control():
    rows = rows_at(counts=list(range(80, 100)))
    samples = normalize(rows)
    points = paired_points(samples)
    p = next(p for p in points if p["lead"] == 15 and p["hold_5m"] == pytest.approx(81 / 1.5))
    assert p["hold_15m"] == pytest.approx(80 / 1.5)
    assert p["actual"] == pytest.approx(84 / 1.5)
    assert p["production"] == 55
    assert {p["lead"] for p in points} == {15, 30, 60}
    # Removing a quarter-hour anchor removes its dependent comparisons, rather
    # than substituting a later observation from the same quarter-hour bin.
    del samples[min(samples)]
    assert all(p["hold_5m"] >= 83 / 1.5 for p in paired_points(samples))


def test_report_warns_on_missing_collection_and_is_explicit_about_benchmarks():
    text, healthy = render(rows_at(), [date(2026, 11, 16)])
    assert not healthy
    assert "hold_5m" in text and "hold_15m" in text
    assert "not verified sensor ground truth" in text
    assert "WATCH" in text
    text, healthy = render([], [date(2026, 11, 16)])
    assert not healthy and "0/192" in text


def test_full_collection_is_healthy_and_reports_each_lead_and_target_period():
    rows = rows_at("2026-11-16T07:00:00-08:00", [90] * 192)
    for row in rows:
        row["production"]["computed_at"] = row["tick_at"]
    text, healthy = render(rows, [date(2026, 11, 16)])
    assert healthy
    assert "192/192" in text
    for lead in [15, 30, 60]:
        for band in ["all hours", "noon–5pm", "5–9pm", "9pm–close"]:
            assert f"+{lead}m {band}" in text
