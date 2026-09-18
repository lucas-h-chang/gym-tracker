import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import carry_data
import data_quality as dq
import forecast_health as fh
import pipeline_health as ph
import send_workout_notifications as notifications
import supabase_io
import today_builder as today

PT = ZoneInfo('America/Los_Angeles')
NOW = datetime(2026, 9, 16, 11, tzinfo=PT)


def test_live_contract_shared_with_web():
    fixtures = json.loads(Path('tests/fixtures/live-capacity.json').read_text())
    for fixture in fixtures:
        assert dq.valid_live_pct(fixture['body'], NOW) == fixture['pct'], fixture['name']


def test_today_excludes_invalid_future_stale_and_keeps_real_zero():
    def row(minutes, value, good=True):
        return {'timestamp': (NOW+timedelta(minutes=minutes)).isoformat(),
                'percent_full': value, 'sensor_ok': good}
    assert today.actuals_by_slot([row(0, 0), row(0, 100, False), row(15, 90)], NOW) == {44: 0}
    assert today.actuals_by_slot([row(-46, 50)], NOW) == {}
    assert today.actuals_by_slot([row(0, 1, False)], NOW) == {}
    assert today.actuals_by_slot([row(0, float('inf')), row(0, -1)], NOW) == {}
    assert today.actuals_by_slot([row(-30, 50), row(0, 1, False)], NOW) == {42: 50}


def test_today_publication_uses_published_curve_not_current_checkout(monkeypatch):
    sb = Mock()
    sb.rpc.return_value.execute.return_value.data = True
    monkeypatch.setattr(today, 'fetch_today_predictions', lambda *a: ({44: 40}, {44: 35}, ['old-curve']))
    monkeypatch.setattr(today, 'fetch_today_rows', lambda *a: [])
    monkeypatch.setattr(today, 'load_carry', lambda: {'built_at': 'new-carry'})
    today.main(sb, NOW)
    snapshot = sb.table.return_value.upsert.call_args.args[0]
    assert snapshot['curve'] == {'44': 35.0}
    assert snapshot['metadata']['baseline_versions'] == ['old-curve']
    assert snapshot['preds'] == []


def test_older_today_run_does_not_record_a_publication(monkeypatch):
    sb = Mock(); sb.rpc.return_value.execute.return_value.data = False
    monkeypatch.setattr(today, 'fetch_today_predictions', lambda *a: ({44: 40}, {}, []))
    monkeypatch.setattr(today, 'fetch_today_rows', lambda *a: [])
    monkeypatch.setattr(today, 'load_carry', lambda: None)
    today.main(sb, NOW)
    sb.table.assert_not_called()


def test_retry_reads_transient_only():
    query = Mock()
    from postgrest.exceptions import APIError
    original_failure = APIError({'message':'JSON could not be generated','code':504,'hint':'Refer to full message','details':'Gateway Timeout'})
    query.execute.side_effect = [original_failure, SimpleNamespace(data=[1])]
    assert supabase_io.execute_read(lambda: query, sleep=lambda _: None).data == [1]
    query.execute.side_effect = RuntimeError('permission denied')
    before = query.execute.call_count
    with pytest.raises(RuntimeError):
        supabase_io.execute_read(lambda: query, sleep=lambda _: None)
    assert query.execute.call_count == before + 1


def test_pagination_survives_server_cap_lower_than_requested():
    sb = Mock(); q = sb.table.return_value.select.return_value
    q.range.return_value.order.return_value.execute.side_effect = [SimpleNamespace(data=[1,2]),SimpleNamespace(data=[3]),SimpleNamespace(data=[])]
    assert supabase_io.paginated_fetch(sb, 't', '*', batch=1000) == [1,2,3]
    assert [c.args[0] for c in q.range.call_args_list] == [0,2,3]


def test_fitting_origins_advance_into_2027_and_cache_key_invalidates(tmp_path):
    assert carry_data.origins_through(date(2027, 2, 10))[-1] == date(2027, 2, 1)
    p = tmp_path/'matrix.pkl'
    carry_data.write_cache(p, 'old-data', [1])
    assert carry_data.read_cache(p, 'old-data') == [1]
    assert carry_data.read_cache(p, 'changed-data') is None
    assert carry_data.read_cache(p, 'old-data', max_age=-1) is None


def test_partial_bare_curve_rollout_uses_paired_rows():
    rows = [dict(snapshot_id=1,date='2026-09-16',horizon_h=1,pct=50,base_pct=50,curve_pct=60,actual_pct=50),
            dict(snapshot_id=2,date='2026-09-15',horizon_h=1,pct=90,base_pct=90,curve_pct=None,actual_pct=50)]
    text = fh.render(rows, 2, date(2026,9,15),date(2026,9,16),True)
    assert 'Trailing improvement: +10.00 pp' in text
    assert '1 paired points' in text


def test_monitor_detects_each_derived_product_and_gates_closed_hours():
    stamp = NOW.isoformat()
    statuses = {p: {'built_at': stamp} for p in ('predictions','weekly_averages')}
    artifacts = {p:stamp for p in ('curves','carry')}
    assert ph.evaluate(NOW,statuses,{'computed_at':stamp},{'computed_at':stamp},artifacts) == []
    broken = ph.evaluate(NOW,{},None,None,{'curves':None,'carry':None})
    assert len(broken) == 6
    closed = datetime(2026,8,24,12,tzinfo=PT)
    issues = ph.evaluate(closed,{},None,None,{})
    assert all('today_summary' not in p and 'prediction_snapshots' not in p for p in issues)


def test_notifications_due_no_early_send_catchup_and_closure():
    row = {'prefs':{'workoutReminderEnabled':True,'workoutDays':[4],
                    'workoutTimes':[{'weekday':4,'hour':11,'minute':0}]}}
    assert notifications.due_notifications(row, NOW-timedelta(minutes=1)) == []
    assert notifications.due_notifications(row, NOW+timedelta(minutes=20)) == [('workout',NOW)]
    assert notifications.due_notifications(row, NOW+timedelta(minutes=31)) == []
    assert notifications.due_notifications({'prefs':{'dailySummaryEnabled':True,'dailySummaryHour':11}}, datetime(2026,8,24,11,tzinfo=PT)) == []


def test_duplicate_dispatch_claim_prevents_second_send():
    sb = Mock(); sender = Mock(); sender.send.return_value='sent'
    sb.rpc.return_value.execute.side_effect=[SimpleNamespace(data=True), SimpleNamespace(data=False)]
    assert notifications.deliver(sb,sender,'device','workout',NOW,'hello') == 'sent'
    assert notifications.deliver(sb,sender,'device','workout',NOW,'hello') == 'duplicate'
    sender.send.assert_called_once()


def test_no_after_closing_prediction_and_zero_is_real():
    now = NOW.replace(hour=22, minute=45)
    assert notifications.render_body('workout',now,[(now, 10)],0) == '0% now'
