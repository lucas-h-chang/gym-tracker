import os
import json
import time
import base64
import struct
import hashlib
import hmac
import math
import requests
import pytz
from datetime import datetime, timedelta, timezone
from supabase import create_client

SUPABASE_URL     = os.environ["SUPABASE_URL"]
SUPABASE_KEY     = os.environ["SUPABASE_SERVICE_KEY"]
APNS_KEY_P8      = os.environ["APNS_KEY_P8"]
APNS_KEY_ID      = os.environ["APNS_KEY_ID"]
APNS_TEAM_ID     = os.environ["APNS_TEAM_ID"]
APNS_BUNDLE_ID   = "com.lucaschang.BearMeter"
APNS_HOST        = "https://api.push.apple.com"
LIVE_CAPACITY_URL = "https://rsfnow.com/api/live-capacity"

PT = pytz.timezone("America/Los_Angeles")


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def now_pt() -> datetime:
    return datetime.now(PT)

def round_to_15(dt: datetime) -> tuple[int, int]:
    """Return (hour, minute) snapped to nearest 15-min slot."""
    total = dt.hour * 60 + dt.minute
    slot  = round(total / 15) * 15
    return (slot // 60) % 24, slot % 60

def slot_matches(user_hour: int, user_minute: int, slot_hour: int, slot_minute: int) -> bool:
    return user_hour == slot_hour and user_minute == slot_minute


# ---------------------------------------------------------------------------
# APNs JWT (ES256) — no external JWT library; uses cryptography package
# ---------------------------------------------------------------------------

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _make_jwt() -> str:
    from cryptography.hazmat.primitives.serialization import load_pem_private_key
    from cryptography.hazmat.primitives.asymmetric.ec import ECDSA
    from cryptography.hazmat.primitives.hashes import SHA256

    key_pem = APNS_KEY_P8.encode()
    if not key_pem.startswith(b"-----"):
        # GitHub stores secrets without newlines sometimes; fix it up
        key_pem = (
            b"-----BEGIN PRIVATE KEY-----\n" +
            b"\n".join(key_pem[i:i+64] for i in range(0, len(key_pem), 64)) +
            b"\n-----END PRIVATE KEY-----\n"
        )

    private_key = load_pem_private_key(key_pem, password=None)

    header  = _b64url(json.dumps({"alg": "ES256", "kid": APNS_KEY_ID}).encode())
    payload = _b64url(json.dumps({"iss": APNS_TEAM_ID, "iat": int(time.time())}).encode())
    signing_input = f"{header}.{payload}".encode()

    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    sig_der = private_key.sign(signing_input, ECDSA(SHA256()))
    r, s = decode_dss_signature(sig_der)
    # Convert to raw 64-byte P-256 signature (r||s, each 32 bytes big-endian)
    sig_bytes = r.to_bytes(32, "big") + s.to_bytes(32, "big")

    return f"{header}.{payload}.{_b64url(sig_bytes)}"


def send_apns_push(device_token: str, title: str, body: str) -> bool:
    import httpx
    jwt = _make_jwt()
    url = f"{APNS_HOST}/3/device/{device_token}"
    payload = json.dumps({
        "aps": {
            "alert": {"title": title, "body": body},
            "sound": "default"
        }
    })
    headers = {
        "authorization":  f"bearer {jwt}",
        "apns-topic":     APNS_BUNDLE_ID,
        "apns-push-type": "alert",
        "content-type":   "application/json",
    }
    try:
        with httpx.Client(http2=True) as client:
            resp = client.post(url, content=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            return True
        print(f"  APNs error {resp.status_code}: {resp.text}")
        return False
    except Exception as e:
        print(f"  APNs exception: {e}")
        return False


# ---------------------------------------------------------------------------
# Live capacity
# ---------------------------------------------------------------------------

def fetch_live_pct() -> int | None:
    try:
        r = requests.get(LIVE_CAPACITY_URL, timeout=8)
        r.raise_for_status()
        return round(r.json()["capacity_pct"])
    except Exception as e:
        print(f"Live capacity fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Predictions (+30 / +60 min) — blended ML + similarity
# ---------------------------------------------------------------------------

def fetch_today_summary(sb) -> tuple[list[dict], float] | tuple[None, None]:
    """Returns (similarity_preds, blend_weight) from today_summary, or (None, None)."""
    today = now_pt().strftime("%Y-%m-%d")
    try:
        rows = (
            sb.table("today_summary")
            .select("similarity_preds,blend_weight")
            .eq("date", today)
            .limit(1)
            .execute()
            .data
        )
        if not rows:
            return None, None
        row = rows[0]
        sim = row["similarity_preds"]
        if isinstance(sim, str):
            sim = json.loads(sim)
        return sim, float(row["blend_weight"])
    except Exception as e:
        print(f"today_summary fetch failed: {e}")
        return None, None


def fetch_predictions_near(sb, target_times: list[datetime]) -> dict[datetime, int]:
    """Fetch ML predictions for target times, blend with similarity if available."""
    if not target_times:
        return {}

    sim_preds, blend_weight = fetch_today_summary(sb)
    # Build sim lookup keyed by hour_numeric (e.g. 17.5 for 5:30pm)
    sim_map: dict[float, float] = {}
    if sim_preds:
        sim_map = {p["x"]: p["y"] for p in sim_preds}

    earliest = min(target_times) - timedelta(minutes=16)
    latest   = max(target_times) + timedelta(minutes=16)

    rows = (
        sb.table("predictions")
        .select("slot_ts,rf_pct,mlp_pct")
        .gte("slot_ts", earliest.astimezone(timezone.utc).isoformat())
        .lte("slot_ts", latest.astimezone(timezone.utc).isoformat())
        .execute()
        .data
    )

    results = {}
    for target in target_times:
        best = None
        best_diff = float("inf")
        for row in rows:
            slot_dt = datetime.fromisoformat(row["slot_ts"].replace("Z", "+00:00"))
            diff = abs((slot_dt - target.astimezone(timezone.utc)).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best = row

        if best:
            ml_pct = (best["rf_pct"] + best["mlp_pct"]) / 2
            # Derive hour_numeric for this slot in PT
            slot_pt   = datetime.fromisoformat(best["slot_ts"].replace("Z", "+00:00")).astimezone(PT)
            hour_num  = slot_pt.hour + slot_pt.minute / 60.0
            # Find nearest sim key
            if sim_map and blend_weight:
                nearest_x = min(sim_map, key=lambda x: abs(x - hour_num))
                if abs(nearest_x - hour_num) <= 0.25:  # within 15 min
                    sim_y  = sim_map[nearest_x]
                    ml_pct = (1 - blend_weight) * ml_pct + blend_weight * sim_y
            results[target] = round(ml_pct)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    now        = now_pt()
    slot_h, slot_m = round_to_15(now)
    # weekday: Python 0=Mon … 6=Sun → iOS 1=Sun … 7=Sat
    py_weekday = now.weekday()           # 0=Mon
    ios_weekday = (py_weekday + 2) % 7 or 7  # 2=Mon, 3=Tue … 7=Sat, 1=Sun

    print(f"[{now.strftime('%Y-%m-%d %H:%M PT')}] slot={slot_h:02d}:{slot_m:02d}, iOS weekday={ios_weekday}")

    # Fetch all device tokens
    rows = sb.table("device_tokens").select("token,prefs").execute().data

    matched_tokens = []
    for row in rows:
        try:
            prefs = json.loads(row["prefs"]) if isinstance(row["prefs"], str) else row["prefs"]
        except Exception:
            continue

        if not prefs.get("workoutReminderEnabled"):
            continue
        workout_days = prefs.get("workoutDays", [])
        if ios_weekday not in workout_days:
            continue
        workout_times = prefs.get("workoutTimes", [])
        # Find the entry for today's weekday
        today_time = next((t for t in workout_times if t.get("weekday") == ios_weekday), None)
        if not today_time:
            continue

        user_h = today_time.get("hour", 18)
        user_m = today_time.get("minute", 0)
        # Snap user time to nearest 15-min slot for matching
        user_slot_h, user_slot_m = round_to_15(
            now.replace(hour=user_h, minute=user_m, second=0, microsecond=0)
        )
        if slot_matches(user_slot_h, user_slot_m, slot_h, slot_m):
            matched_tokens.append(row["token"])

    if not matched_tokens:
        print("No tokens matched this time slot.")
        return

    print(f"{len(matched_tokens)} token(s) matched.")

    # Fetch live capacity and predictions once for all matched users
    live_pct = fetch_live_pct()

    now_utc   = now.astimezone(timezone.utc)
    t30       = now_utc + timedelta(minutes=30)
    t60       = now_utc + timedelta(minutes=60)
    pred_map  = fetch_predictions_near(sb, [t30, t60])
    p30       = pred_map.get(t30)
    p60       = pred_map.get(t60)

    parts = []
    if live_pct is not None: parts.append(f"{live_pct}% now")
    if p30       is not None: parts.append(f"{p30}% in 30 min")
    if p60       is not None: parts.append(f"{p60}% in 60 min")
    body = " · ".join(parts) if parts else "Time to hit the gym!"

    print(f"Body: {body}")

    sent = 0
    for token in matched_tokens:
        if send_apns_push(token, "Workout Reminder", body):
            sent += 1
    print(f"Sent {sent}/{len(matched_tokens)} pushes.")


if __name__ == "__main__":
    main()
