from copy import deepcopy
from datetime import date, datetime
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

import shadow_forecast as sf
import shadow_health as sh
import today_builder as today

PT = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 11, 7, 11, 0, 5, tzinfo=PT)


def inputs():
    base = {s: 50.0 for s in range(32, 72)}
    curve = {s: 60.0 for s in base}
    actuals = {s: 40.0 for s in range(32, 45)}
    # Carry the full day-gap, before existing observation/horizon shrinkage.
    carry = {"by_cut": {}, "pooled": [{"h": 1.0, "coef": [0, 1, 0, 0]}]}
    preds, _ = today.compute_level_correction(actuals, base, carry, NOW)
    return preds, base, curve, actuals, carry


def test_shadow_is_paired_with_exact_production_and_does_not_mutate_inputs():
    args = inputs()
    before = deepcopy(args)
    result = sf.build_shadow(NOW, *args)
    assert args == before
    assert result["status"] == "ok"
    assert result["period"] == "prospective"
    points = result["forecasts"]
    assert all(set(points[a]) == set(points["production"]) for a in sf.ARMS)
    assert min(map(int, points["production"])) == 45
    assert "72" not in points["production"]  # closing zero is display only
    production = {str(round(p["x"] * 4)): p["y"] for p in args[0]}
    assert all(production[s] == v for s, v in points["production"].items())
    assert points["weaker_carry"]["45"] == points["production"]["45"]
    assert points["weaker_carry"]["60"] > points["production"]["60"]


def test_early_fallback_tests_trailing_and_does_not_invent_carry():
    _, base, curve, _, carry = inputs()
    result = sf.build_shadow(NOW, [], base, curve, {}, carry)["forecasts"]
    assert result["production"]["45"] == 50
    assert result["stronger_trail"]["45"] == 47.5
    assert result["weaker_carry"] == result["production"]
    assert result["both"] == result["stronger_trail"]


def test_combined_recomputes_gaps_against_stronger_baseline():
    _, base, curve, actuals, carry = inputs()
    result = sf.build_shadow(NOW, *inputs())["forecasts"]
    stronger = {s: 47.5 for s in base}
    # Different baseline changes the gap from -10 to -7.5; do not simply stack
    # a stale production carry correction on top of the stronger trailing one.
    expected, _ = today.compute_level_correction(actuals, stronger, carry, NOW)
    expected = {str(round(p["x"] * 4)): p["y"] for p in expected}
    assert result["stronger_trail"]["60"] == expected["60"]


def test_policy_weight_tapers_continuously_and_never_reverses():
    assert [sf.carry_weight(h) for h in (0, 1, 2, 3, 12)] == [1, 1, .75, .5, .5]
    assert sf.carry_weight(3 - 1e-8) == pytest.approx(.5)


def test_missing_curve_and_changed_production_fail_closed():
    preds, base, curve, actuals, carry = inputs()
    curve.pop(32)  # even observed slots must have a valid candidate baseline
    with pytest.raises(ValueError, match="missing bare curve"):
        sf.build_shadow(NOW, preds, base, curve, actuals, carry)
    preds, base, curve, actuals, carry = inputs()
    preds[0]["y"] += 1
    with pytest.raises(ValueError, match="reproduction mismatch"):
        sf.build_shadow(NOW, preds, base, curve, actuals, carry)


def test_shadow_failure_keeps_live_audit_and_cannot_change_publication(monkeypatch):
    sb = Mock()
    monkeypatch.setattr(sf, "build_shadow", Mock(side_effect=RuntimeError("shadow broken")))
    preds, base, curve, actuals, carry = inputs()
    metadata = {"carry_version": "original"}
    today.record_snapshot(sb, NOW, preds, base, curve, {}, metadata, actuals=actuals, carry=carry)
    sb.rpc.assert_not_called()  # audit has no permission/code path to republish
    row = sb.table.return_value.upsert.call_args.args[0]
    assert row["preds"] == preds
    assert row["source"] == "live"
    assert row["metadata"]["shadow"]["status"] == "error"
    assert metadata == {"carry_version": "original"}


def row(day="2026-11-07", forecast=40):
    return {"date": day, "computed_at": f"{day}T11:00:05-08:00", "metadata": {"shadow": {
        "id": sf.EXPERIMENT_ID, "policy_hash": sf.POLICY_HASH, "status": "ok",
        "forecasts": {arm: {"44": 999, "45": forecast} for arm in sf.ARMS}}}}


def test_scores_only_future_paired_slots_and_rejects_other_policy_and_pilot():
    good = row()
    wrong = deepcopy(good)
    wrong["metadata"]["shadow"]["policy_hash"] = "different"
    points, valid, expected = sh.paired_points(
        [good, wrong, row("2026-11-06")],
        {("2026-11-07", 44): 10, ("2026-11-07", 45): 50}, {"2026-11-07"})
    assert (len(points), valid, expected) == (1, 1, 2)
    assert points[0]["horizon"] == pytest.approx(895 / 3600)
    assert sh.arm_stats(points, "production")["mae"] == 10


def test_daily_gain_gives_each_day_equal_weight():
    first = {"date": "2026-11-07", "actual": 50, "production": 60, "both": 50}
    second = {"date": "2026-11-08", "actual": 50, "production": 50, "both": 54}
    result = sh.arm_stats([first] * 100 + [second], "both")
    assert result["day_gain"] == 3  # (10 - 4) / 2, not publication-weighted
    assert result["days_won"] == 1


def test_actuals_drop_sensor_failures_and_keep_true_zero():
    def reading(value, good=True):
        return {"timestamp": "2026-11-07T11:15:09-08:00", "people_count": value, "sensor_ok": good}
    assert sh.actual_slots([reading(0), reading(150, False)]) == {("2026-11-07", 45): 0}


@pytest.mark.parametrize("fraction", ["", ".8", ".07", ".005", ".0052", ".00529", ".005291"])
def test_postgrest_fractional_precision_in_actuals_and_publications(fraction):
    actuals = sh.actual_slots([{
        "timestamp": f"2026-11-07T19:15:09{fraction}+00:00",
        "people_count": 75, "sensor_ok": True,
    }])
    publication = row()
    publication["computed_at"] = f"2026-11-07T19:00:05{fraction}+00:00"
    points, valid, expected = sh.paired_points([publication], actuals, {"2026-11-07"})
    assert (len(points), valid, expected) == (1, 1, 1)
    assert points[0]["actual"] == 50


def test_holdout_is_30_future_open_days_and_configuration_is_frozen():
    days = sh.holdout_dates()
    assert len(days) == 30 and days[0] == date(2026, 11, 7)
    assert days == sorted(set(days))
    assert sf.POLICY_HASH == "46e74bc2a34a20eb4f765cd480da607cdb33dc60c84fa86de4bac97236899645"
