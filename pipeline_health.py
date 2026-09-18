"""Read-only checks for every published data product, independent of the scraper."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from academic_calendar import get_open_hours
from data_quality import timestamp_age

PT = ZoneInfo('America/Los_Angeles')


def evaluate(now, statuses, summary, snapshot, artifacts):
    problems = []
    for product in ('predictions', 'weekly_averages'):
        age = timestamp_age(statuses.get(product, {}).get('built_at'), now)
        if age is None or not 0 <= age <= 36 * 3600:
            problems.append(f'{product}: publication missing or older than 36 hours')
    for product, stamp in artifacts.items():
        # Historical build artifacts used naive UTC on Actions.
        try:
            if stamp and datetime.fromisoformat(stamp.replace('Z', '+00:00')).tzinfo is None:
                stamp += '+00:00'
        except (ValueError, AttributeError):
            stamp = None
        age = timestamp_age(stamp, now)
        if age is None or not 0 <= age <= 9 * 86400:
            problems.append(f'{product}: artifact missing or older than 9 days')
    pt = now.astimezone(PT)
    open_h, close_h = get_open_hours(pt.strftime('%A'), pt.date())
    # Allow the first two ticks after opening; closures need no today products.
    hour = pt.hour + pt.minute/60
    if open_h + .5 <= hour < close_h:
        for product, row in [('today_summary', summary), ('prediction_snapshots', snapshot)]:
            age = timestamp_age((row or {}).get('computed_at'), now)
            if age is None or not 0 <= age <= 45 * 60:
                problems.append(f'{product}: missing or older than 45 minutes while open')
    return problems


def get_rows(table, **params):
    url = os.environ['SUPABASE_URL'].rstrip('/') + '/rest/v1/' + table + '?' + urlencode(params)
    key = os.environ['SUPABASE_SERVICE_KEY']
    req = Request(url, headers={'apikey': key, 'Authorization': f'Bearer {key}'})
    with urlopen(req, timeout=15) as response:
        return json.load(response)


def main():
    now = datetime.now(timezone.utc)
    today = now.astimezone(PT).date().isoformat()
    problems = []
    data = {}
    queries = {
        'statuses': ('pipeline_status', {'select': 'product,built_at'}),
        'summary': ('today_summary', {'select': 'computed_at', 'date': f'eq.{today}', 'limit': 1}),
        'snapshot': ('prediction_snapshots', {'select': 'computed_at', 'date': f'eq.{today}',
                       'source': 'eq.live', 'order': 'computed_at.desc', 'limit': 1}),
    }
    for name, (table, params) in queries.items():
        try:
            data[name] = get_rows(table, **params)
        except Exception as exc:
            # A monitor failing to read is itself actionable, not a green run.
            problems.append(f'{table}: check failed ({type(exc).__name__})')
            data[name] = []
    artifacts = {}
    for name in ('curves', 'carry'):
        try:
            artifacts[name] = json.loads(Path(__file__).with_name('models').joinpath(name+'.json').read_text()).get('built_at')
        except (OSError, ValueError):
            artifacts[name] = None
    problems += evaluate(now, {r['product']: r for r in data['statuses']},
                         next(iter(data['summary']), None), next(iter(data['snapshot']), None), artifacts)
    print('\n'.join(problems) if problems else 'All published products are fresh')
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f'unhealthy={str(bool(problems)).lower()}\n')
            f.write('detail=' + '; '.join(problems) + '\n')
    return bool(problems)


if __name__ == '__main__':
    main()  # Workflow consumes outputs and creates one deduplicated alert.
