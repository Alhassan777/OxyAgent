"""Decision Agent: state machine, severity, lifeguard assignment."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

try:
    from .. import config
except ImportError:
    import config  # type: ignore


class State(Enum):
    MONITOR = "MONITOR"
    ALERT = "ALERT"
    DISPATCH = "DISPATCH"
    ESCALATE = "ESCALATE"
    RESOLVED = "RESOLVED"


SeverityLevel = str  # "low" | "medium" | "high" | "critical"


@dataclass
class WorldState:
    state: State = State.MONITOR
    p_distress: float = 0.0
    p_unresponsive: float = 0.0
    threat_detected: bool = False
    time_in_risk: float = 0.0


class DecisionAgent:
    """State machine and severity assessment."""

    def __init__(self) -> None:
        self.world = WorldState()
        self._lock = threading.Lock()
        self._distress_window: deque[float] = deque(maxlen=getattr(config, "TEMPORAL_WINDOW", 10))
        self._risk_start_ts: float | None = None
        self._no_threat_start_ts: float | None = None
        self._unresponsive_start_ts: float | None = None
        self._ack_requested = False
        self.last_dispatch_plan: dict | None = None
        self.last_victim_pos: tuple[float, float] | None = None
        self.last_swimmer_positions: list[tuple[float, float]] = []

    @staticmethod
    def _extract_objects(detections: dict | list | None) -> list[dict]:
        if detections is None:
            return []
        if isinstance(detections, list):
            return detections
        if isinstance(detections, dict):
            if isinstance(detections.get("detections"), list):
                return detections["detections"]
            if isinstance(detections.get("objects"), list):
                return detections["objects"]
        return []

    @property
    def current_state(self) -> State:
        return self.world.state

    def lifeguard_acknowledged(self) -> None:
        with self._lock:
            self._ack_requested = True

    def reset(self) -> None:
        with self._lock:
            self.world = WorldState()
            self._distress_window.clear()
            self._risk_start_ts = None
            self._no_threat_start_ts = None
            self._unresponsive_start_ts = None
            self.last_dispatch_plan = None
            self.last_victim_pos = None
            self.last_swimmer_positions = []
            self._ack_requested = False
            print("Decision Agent: reset")

    def _transition(self, new_state: State) -> None:
        if self.world.state == new_state:
            return
        old = self.world.state
        self.world.state = new_state
        print(f"STATE TRANSITION: {old.value} -> {new_state.value}")

    def compute_severity(self) -> SeverityLevel:
        """Map p_distress and p_unresponsive to severity."""
        p = self.world.p_distress
        u = self.world.p_unresponsive
        if p > 0.9 or (u > 0.7 and self.world.threat_detected):
            return "critical"
        if p > 0.75 or u > 0.5:
            return "high"
        if p > 0.6:
            return "medium"
        return "low"

    def process(
        self,
        detections: dict | list | None,
        frame_count: int,
        dispatch_plan: dict | None = None,
        eta: float = 0.0,
    ) -> dict:
        """
        Run state machine. Returns actions dict with state, severity, explanation.
        """
        from .detection_agent import compute_priority

        with self._lock:
            now = datetime.now(timezone.utc).timestamp()
            objects = self._extract_objects(detections)

            p_distress_candidates = []
            p_unresponsive_candidates = []
            threat_detected = False

            for obj in objects:
                label = str(obj.get("label", "")).lower()
                p_d = float(obj.get("p_distress", obj.get("score", 0.0)))
                p_u = float(obj.get("p_unresponsive", 0.0))
                p_distress_candidates.append(p_d)
                p_unresponsive_candidates.append(p_u)
                if label == "drowning" or p_d > config.ALERT_THRESHOLD:
                    threat_detected = True

            raw_p_distress = max(p_distress_candidates) if p_distress_candidates else 0.0
            raw_p_unresponsive = max(p_unresponsive_candidates) if p_unresponsive_candidates else 0.0

            self._distress_window.append(raw_p_distress)
            p_distress = sum(self._distress_window) / max(len(self._distress_window), 1)
            p_unresponsive = raw_p_unresponsive

            self.world.p_distress = p_distress
            self.world.p_unresponsive = p_unresponsive
            self.world.threat_detected = threat_detected

            if threat_detected:
                if self._risk_start_ts is None:
                    self._risk_start_ts = now
                self.world.time_in_risk = now - self._risk_start_ts
                self._no_threat_start_ts = None
            else:
                self.world.time_in_risk = 0.0
                self._risk_start_ts = None
                if self._no_threat_start_ts is None:
                    self._no_threat_start_ts = now
                self._unresponsive_start_ts = None

            if p_unresponsive > 0.7 and threat_detected:
                if self._unresponsive_start_ts is None:
                    self._unresponsive_start_ts = now
            else:
                self._unresponsive_start_ts = None

            if self._ack_requested:
                self._transition(State.MONITOR)
                self._ack_requested = False

            if self._no_threat_start_ts is not None and (now - self._no_threat_start_ts) >= 3.0:
                self._transition(State.RESOLVED)
            else:
                if self.world.state == State.MONITOR and p_distress > config.ALERT_THRESHOLD:
                    self._transition(State.ALERT)
                if self.world.state == State.ALERT and p_distress > config.DISPATCH_THRESHOLD:
                    self._transition(State.DISPATCH)
                if self.world.state == State.DISPATCH:
                    if (
                        self._unresponsive_start_ts is not None
                        and (now - self._unresponsive_start_ts) > config.UNRESPONSIVE_SECONDS
                    ):
                        self._transition(State.ESCALATE)
                    elif p_distress > config.ESCALATE_THRESHOLD:
                        self._transition(State.ESCALATE)

            severity = self.compute_severity()
            priority = compute_priority(p_distress, p_unresponsive, self.world.time_in_risk, eta)

            explanation = (
                f"state={self.world.state.value} p_distress={p_distress:.2f} "
                f"p_unresponsive={p_unresponsive:.2f} risk={self.world.time_in_risk:.1f}s severity={severity}"
            )

            return {
                "state": self.world.state.value,
                "severity": severity,
                "p_distress": p_distress,
                "p_unresponsive": p_unresponsive,
                "time_in_risk": self.world.time_in_risk,
                "threat_detected": threat_detected,
                "explanation": explanation,
                "priority": priority,
                "frame_count": frame_count,
                "dispatched_lifeguard": dispatch_plan.get("lifeguard") if dispatch_plan else None,
            }

    def set_dispatch_plan(
        self,
        victim_pos: tuple[float, float],
        swimmer_positions: list[tuple[float, float]],
        plan: dict,
    ) -> None:
        with self._lock:
            self.last_victim_pos = (float(victim_pos[0]), float(victim_pos[1]))
            self.last_swimmer_positions = [
                (float(p[0]), float(p[1])) for p in (swimmer_positions or [])
            ]
            self.last_dispatch_plan = plan
