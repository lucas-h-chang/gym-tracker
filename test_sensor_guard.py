import json
from datetime import datetime
from pathlib import Path
import pytest
from sensor_guard import evaluate_sensor_reading

CASES = json.loads((Path(__file__).parent / 'tests/fixtures/sensor-rule.json').read_text())

@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_sensor_rule_parity(case):
    result = evaluate_sensor_reading(case['rows'], case['count'], datetime.fromisoformat(case['now']))
    assert result['stalled'] is case['stalled']
    if result['stalled']:
        assert result['timestamps']
