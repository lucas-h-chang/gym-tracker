"""Scheduled APNs reminders with bounded catch-up and durable delivery claims.

One APNs attempt per (device, kind, intended time). An ambiguous network result
is recorded as unknown, not retried automatically: APNs may have accepted it.
No credentials or network work happen during import.
"""
import base64
import json
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import requests

from academic_calendar import get_open_hours
from data_quality import is_fresh, valid_live_pct
from supabase_io import client, paginated_fetch

PT = ZoneInfo('America/Los_Angeles')
CATCHUP = timedelta(minutes=30)
APNS_HOST = 'https://api.push.apple.com'
LIVE_CAPACITY_URL = 'https://rsfnow.com/api/live-capacity'


def _b64url(data):
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode()


class APNsSender:
    def __init__(self):
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        pem = os.environ['APNS_KEY_P8'].replace('\\n', '\n').encode()
        if not pem.startswith(b'-----'):
            pem = b'-----BEGIN PRIVATE KEY-----\n' + b'\n'.join(
                pem[i:i+64] for i in range(0, len(pem), 64)) + b'\n-----END PRIVATE KEY-----\n'
        self.key = load_pem_private_key(pem, password=None)
        self.http = httpx.Client(http2=True, timeout=10)
        self.jwt = None
        self.jwt_at = 0

    def token(self):
        if self.jwt and time.time() - self.jwt_at < 50 * 60:
            return self.jwt
        from cryptography.hazmat.primitives.asymmetric.ec import ECDSA
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
        from cryptography.hazmat.primitives.hashes import SHA256
        self.jwt_at = int(time.time())
        header = _b64url(json.dumps({'alg': 'ES256', 'kid': os.environ['APNS_KEY_ID']}).encode())
        payload = _b64url(json.dumps({'iss': os.environ['APNS_TEAM_ID'], 'iat': self.jwt_at}).encode())
        signing = f'{header}.{payload}'.encode()
        r, s = decode_dss_signature(self.key.sign(signing, ECDSA(SHA256())))
        self.jwt = f'{header}.{payload}.{_b64url(r.to_bytes(32,"big") + s.to_bytes(32,"big"))}'
        return self.jwt

    def send(self, token, title, body):
        try:
            response = self.http.post(f'{APNS_HOST}/3/device/{token}',
                headers={'authorization': f'bearer {self.token()}',
                         'apns-topic': 'com.lucaschang.BearMeter', 'apns-push-type': 'alert'},
                json={'aps': {'alert': {'title': title, 'body': body}, 'sound': 'default'}})
            if response.status_code == 200:
                return 'sent'
            try:
                reason = response.json().get('reason')
            except ValueError:
                reason = None
            if response.status_code == 410 or reason in ('BadDeviceToken', 'Unregistered'):
                return 'invalid'
            print(f'APNs rejected delivery: HTTP {response.status_code}, reason={reason}')
            return 'failed'
        except httpx.HTTPError as exc:
            print(f'APNs delivery outcome unknown: {type(exc).__name__}')
            return 'unknown'

    def close(self):
        self.http.close()


def parse_prefs(row):
    try:
        value = row.get('prefs', {})
        value = json.loads(value) if isinstance(value, str) else value
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def due_notifications(row, now):
    """Use intended wall time, not rounded job-start time; never send early."""
    prefs = parse_prefs(row)
    weekday = (now.weekday() + 1) % 7 + 1  # Swift: Sunday=1
    candidates = []
    if prefs.get('dailySummaryEnabled'):
        candidates.append(('summary', prefs.get('dailySummaryHour', 8), prefs.get('dailySummaryMinute', 0)))
    times = prefs.get('workoutTimes', [])
    if (prefs.get('workoutReminderEnabled') and isinstance(prefs.get('workoutDays'), list)
            and weekday in prefs['workoutDays'] and isinstance(times, list)):
        for entry in times:
            if isinstance(entry, dict) and entry.get('weekday') == weekday:
                candidates.append(('workout', entry.get('hour', 18), entry.get('minute', 0)))
                break
    due = []
    for kind, hour, minute in candidates:
        if type(hour) is not int or type(minute) is not int or not (0 <= hour < 24 and 0 <= minute < 60):
            continue
        intended = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        open_h, close_h = get_open_hours(intended.strftime('%A'), intended.date())
        if (open_h <= hour + minute/60 < close_h and timedelta(0) <= now - intended <= CATCHUP):
            due.append((kind, intended))
    return due


def fetch_live_pct(now):
    try:
        response = requests.get(LIVE_CAPACITY_URL, timeout=8)
        response.raise_for_status()
        body = response.json()
        if body.get('source') == 'cache_stale':
            return None
        return valid_live_pct(body, now)
    except (requests.RequestException, ValueError, TypeError):
        return None


def fetch_forecast(sb, now):
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = (sb.table('predictions').select('slot_ts,pct')
            .gte('slot_ts', start.isoformat()).lt('slot_ts', (start + timedelta(days=1)).isoformat())
            .order('slot_ts').execute().data)
    summaries = (sb.table('today_summary').select('similarity_preds,blend_weight,computed_at')
                 .eq('date', now.date().isoformat()).limit(1).execute().data)
    summary = summaries[0] if summaries else {}
    points = summary.get('similarity_preds') or []
    if not is_fresh(summary.get('computed_at'), now):
        points = []
    corrections = {p['x']: p for p in points}
    forecast = []
    for row in rows:
        stamp = datetime.fromisoformat(row['slot_ts'].replace('Z', '+00:00')).astimezone(PT)
        hour = stamp.hour + stamp.minute/60
        p = corrections.get(hour)
        weight = p.get('w', summary.get('blend_weight', 0)) if p else 0
        pct = (1 - weight) * row['pct'] + weight * p['y'] if p else row['pct']
        forecast.append((stamp, pct))
    return forecast


def render_body(kind, now, forecast, live):
    if kind == 'summary':
        if not forecast:
            return 'Open Bear Meter for today’s forecast.'
        peak_time, peak_pct = max(forecast, key=lambda p: p[1])
        avg = round(sum(pct for _, pct in forecast)/len(forecast))
        label = peak_time.strftime('%I:%M %p').lstrip('0')
        return f'Avg {avg}% · peak around {label} ({round(peak_pct)}%)'
    parts = [f'{live}% now'] if live is not None else []
    for minutes in (30, 60):
        target = now + timedelta(minutes=minutes)
        open_h, close_h = get_open_hours(target.strftime('%A'), target.date())
        if not open_h <= target.hour + target.minute/60 < close_h:
            continue
        best = min(forecast, key=lambda p: abs(p[0]-target), default=None)
        if best and abs(best[0] - target) <= timedelta(minutes=8):
            parts.append(f'{round(best[1])}% in {minutes} min')
    return ' · '.join(parts) if parts else 'Time to hit the gym!'


def deliver(sb, sender, token, kind, intended, body):
    key = {'p_token': token, 'p_kind': kind, 'p_scheduled_at': intended.isoformat()}
    if not sb.rpc('claim_notification', key).execute().data:
        return 'duplicate'
    # A crash here leaves a claim. Never automatically retry ambiguous sends.
    status = sender.send(token, 'Your Daily RSF Summary' if kind == 'summary' else 'Workout Reminder', body)
    (sb.table('notification_deliveries').update({'status': status})
        .eq('token', token).eq('kind', kind).eq('scheduled_at', intended.isoformat()).execute())
    if status == 'invalid':
        sb.table('device_tokens').delete().eq('token', token).execute()
    return status


def main(sb=None, now=None, sender=None):
    now = now or datetime.now(PT)
    open_h, close_h = get_open_hours(now.strftime('%A'), now.date())
    if not open_h <= now.hour + now.minute/60 < close_h:
        print('RSF closed; no reminders sent')
        return
    sb = sb or client()
    rows = paginated_fetch(sb, 'device_tokens', 'token,prefs', order='token')
    due = [(row['token'], kind, intended) for row in rows for kind, intended in due_notifications(row, now)]
    if not due:
        print('No reminders due')
        return
    forecast = fetch_forecast(sb, now)
    live = fetch_live_pct(now) if any(kind == 'workout' for _, kind, _ in due) else None
    own_sender = sender is None
    sender = sender or APNsSender()
    try:
        for token, kind, intended in due:
            result = deliver(sb, sender, token, kind, intended, render_body(kind, now, forecast, live))
            print(f'{kind}: {result}')  # Never log device tokens.
    finally:
        if own_sender:
            sender.close()


if __name__ == '__main__':
    main()
