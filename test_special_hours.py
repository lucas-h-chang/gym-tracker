"""Holiday hours agree across the serving calendar, UI and monitor."""
import json
import subprocess
import textwrap
from datetime import date
from pathlib import Path
import pytest
from academic_calendar import get_open_hours

CASES = [(date(y, 11, d), (10, 16)) for y, days in [(2026, (27,28,29)), (2027, (26,27,28))] for d in days]
CASES += [(date(2026,11,26),(0,0)), (date(2027,11,25),(0,0)), (date(2026,11,30),(7,23)), (date(2027,11,29),(7,23)), (date(2025,11,28),(7,23))]

@pytest.mark.parametrize('day,hours', CASES)
def test_hours_in_python_web_api_and_monitor(day,hours):
    assert get_open_hours(day.strftime('%A'),day) == hours
    script = """
const fs=require('node:fs'), vm=require('node:vm');
const context=vm.createContext({});
vm.runInContext(fs.readFileSync('docs/js/calendar.js','utf8'),context);
const ds=process.argv[1];
const web=vm.runInContext(`getOpenHours(new Date('${ds}T12:00:00').getDay(),new Date('${ds}T12:00:00'))`,context);
const api=require('./api/_hours').getOpenHours(process.argv[2],ds);
console.log(JSON.stringify({web,api}));
"""
    result=json.loads(subprocess.check_output(['node','-e',script,day.isoformat(),day.strftime('%A')],text=True))
    assert result == {'web': {'open':hours[0],'close':hours[1]}, 'api':list(hours)}
    text=Path('.github/workflows/freshness.yml').read_text()
    start=text.index("python - <<'PY'")
    src=textwrap.dedent(text[text.index('\n',start)+1:text.index('\n          PY',start)])
    # Calendar definitions precede emit; avoid the monitor's network activity.
    definitions=src[src.index('from datetime import'):src.index('def emit(')]
    namespace={"os": __import__("os")}
    exec(definitions,namespace)
    assert namespace['get_open_hours'](day.strftime('%A'),day) == hours


def test_weekly_averages_exclude_outside_special_hours():
    import pandas as pd
    from weekly_builder import _emit_day_records
    frame=pd.DataFrame({
        'timestamp':pd.to_datetime(['2026-11-27 09:45','2026-11-27 10:00','2026-11-27 15:45','2026-11-27 16:00']),
        'day_of_week':['Friday']*4,
        'hour_numeric':[9.75,10,15.75,16],
        'percent_full':[99,20,30,99],
    })
    records=[]
    _emit_day_records(frame,'last_week',False,records)
    friday=[r for r in records if r['day_of_week']=='Friday']
    assert [(r['hour_slot'],r['avg_pct']) for r in friday] == [(10,20),(15.75,30),(16,0)]
