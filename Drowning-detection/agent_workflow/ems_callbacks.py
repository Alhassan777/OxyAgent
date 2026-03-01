"""
EMS callbacks for ambulance integration.

Usage:
    from agents.ems_agent import register_ems_callback
    from ems_callbacks import twilio_sms_callback, webhook_callback
    register_ems_callback("twilio", twilio_sms_callback)
    # Then set config.EMS_CALLBACK = "twilio"
"""

from __future__ import annotations


def twilio_sms_callback(payload: dict, context: dict) -> None:
    """
    Example: Send SMS via Twilio. Requires:
        pip install twilio
        TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, TWILIO_TO env vars.
    """
    import os
    try:
        from twilio.rest import Client
        sid = os.environ.get("TWILIO_ACCOUNT_SID")
        token = os.environ.get("TWILIO_AUTH_TOKEN")
        from_num = os.environ.get("TWILIO_FROM")
        to_num = os.environ.get("TWILIO_TO")
        if not all([sid, token, from_num, to_num]):
            print("EMS: Twilio not configured (missing env vars)")
            return
        client = Client(sid, token)
        msg = (
            f"DROWNING ALERT: Victim at {payload.get('victim_pool_coords')}. "
            f"p_distress={payload.get('p_distress', 0):.2f}. "
            f"Lifeguard {payload.get('dispatched_lifeguard')} ETA {payload.get('eta_seconds')}s."
        )
        client.messages.create(body=msg, from_=from_num, to=to_num)
        print("\033[91mEMS TRIGGERED (Twilio SMS sent)\033[0m")
    except ImportError:
        print("EMS: twilio package not installed")
    except Exception as e:
        print(f"EMS Twilio error: {e}")


def webhook_callback(payload: dict, context: dict) -> None:
    """
    Example: POST payload to EMS webhook. Requires EMS_WEBHOOK_URL env var.
    """
    import os
    import urllib.request
    import json as _json

    url = os.environ.get("EMS_WEBHOOK_URL")
    if not url:
        print("EMS: EMS_WEBHOOK_URL not set")
        return
    try:
        data = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass
        print(f"\033[91mEMS TRIGGERED (webhook POST to {url})\033[0m")
    except Exception as e:
        print(f"EMS webhook error: {e}")


