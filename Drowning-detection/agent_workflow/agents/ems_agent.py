"""EMS Agent: trigger ambulance on severe/unresponsive with pluggable callbacks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_AGENT_DIR = Path(__file__).resolve().parent.parent


def _log_callback(payload: dict, context: dict) -> None:
    """Default: log to alerts.log."""
    alerts_path = _AGENT_DIR / "alerts.log"
    with alerts_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    print("\033[91mEMS TRIGGERED (logged)\033[0m")


def _null_callback(payload: dict, context: dict) -> None:
    """No-op for testing."""
    print("\033[91mEMS TRIGGERED (no-op)\033[0m")


EMS_CALLBACKS: dict[str, Callable[[dict, dict], None]] = {
    "log": _log_callback,
    "null": _null_callback,
}


def register_ems_callback(name: str, fn: Callable[[dict, dict], None]) -> None:
    EMS_CALLBACKS[name] = fn


class EMSAgent:
    """Triggers EMS when severe or unresponsive; uses pluggable callback."""

    def __init__(self, callback_name: str = "log") -> None:
        self.callback_name = callback_name
        self._triggered_for_incident = False

    def _get_callback(self) -> Callable[[dict, dict], None]:
        return EMS_CALLBACKS.get(self.callback_name, _log_callback)

    def reset(self) -> None:
        self._triggered_for_incident = False

    def check(
        self,
        p_distress: float,
        p_unresponsive: float,
        time_in_risk: float,
        victim_pos: tuple[float, float] | None,
        dispatch_plan: dict | None,
    ) -> dict | None:
        """
        Check if EMS should be triggered.
        Returns payload if triggered, None otherwise.
        """
        triggered = (p_unresponsive > 0.7 and time_in_risk > 5.0) or (
            p_distress > 0.9 and time_in_risk > 15.0
        )

        if not triggered or self._triggered_for_incident:
            return None

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "victim_pool_coords": victim_pos,
            "p_distress": p_distress,
            "p_unresponsive": p_unresponsive,
            "dispatched_lifeguard": dispatch_plan.get("lifeguard") if dispatch_plan else None,
            "eta_seconds": dispatch_plan.get("eta_seconds") if dispatch_plan else None,
            "status": "EMS_TRIGGERED",
        }

        callback = self._get_callback()
        context = {"dispatch_plan": dispatch_plan}
        callback(payload, context)

        self._triggered_for_incident = True
        return payload
