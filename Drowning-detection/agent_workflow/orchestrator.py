"""Orchestrator: coordinates Detection, Path, Decision, and EMS agents."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

try:
    from . import config
    from .agents import DetectionAgent, DecisionAgent, EMSAgent, PathAgent
    from .tracker import Tracker
except ImportError:
    import config  # type: ignore
    from agents import DetectionAgent, DecisionAgent, EMSAgent, PathAgent  # type: ignore
    from tracker import Tracker  # type: ignore


def pixel_to_pool(px: tuple, frame_shape: tuple) -> tuple[float, float]:
    pool_x = (px[0] / frame_shape[1]) * config.POOL_W
    pool_y = (px[1] / frame_shape[0]) * config.POOL_L
    return (pool_x, pool_y)


class Orchestrator:
    """Coordinates all agents for the lifeguard system."""

    def __init__(
        self,
        analyze_frame_fn: Callable,
        ems_callback: str = "log",
    ) -> None:
        self.analyze_frame = analyze_frame_fn
        self.detection_agent = DetectionAgent()
        self.path_agent = PathAgent(
            crowd_sigma=getattr(config, "PATH_CROWD_SIGMA", 1.5),
            crowd_amplitude=getattr(config, "PATH_CROWD_AMPLITUDE", 2.0),
            density_speed_k=getattr(config, "PATH_DENSITY_SPEED_K", 0.5),
            n_boundary_points=getattr(config, "PATH_N_BOUNDARY_POINTS", 80),
        )
        self.decision_agent = DecisionAgent()
        self.ems_agent = EMSAgent(callback_name=ems_callback)
        self.tracker = Tracker(
            iou_threshold=config.TRACK_IOU_THRESHOLD,
            max_age=config.TRACK_MAX_AGE,
            min_persist=config.TRACK_MIN_PERSIST,
        )
        self._lock = threading.Lock()
        self.last_dispatch_plan: dict | None = None
        self.last_victim_pos: tuple[float, float] | None = None
        self.last_swimmer_positions: list[tuple[float, float]] = []

    def _normalize_detections(self, result: Any) -> dict:
        if isinstance(result, dict):
            if isinstance(result.get("detections"), list):
                return result
            if isinstance(result.get("objects"), list):
                return {"detections": result["objects"]}
        if isinstance(result, list):
            return {"detections": result}
        return {"detections": []}

    @property
    def current_state(self) -> str:
        return self.decision_agent.current_state.value

    def lifeguard_acknowledged(self) -> None:
        self.decision_agent.lifeguard_acknowledged()

    def reset(self) -> None:
        self.decision_agent.reset()
        self.ems_agent.reset()
        self.tracker.reset()

    def run_inference(
        self,
        image,
        frame_count: int,
        frame_shape: tuple,
    ) -> dict:
        """
        Run full pipeline: analyze_frame -> tracker -> detection enrich -> path -> decision -> EMS.
        Returns combined result with detections, actions, dispatch_plan, etc.
        """
        raw = self.analyze_frame(image)
        if not isinstance(raw, dict):
            raw = {"detections": [], "threat_detected": False, "threat_count": 0}
        raw_detections = self._normalize_detections(raw)

        tracked_detections = self.tracker.update(
            raw_detections.get("detections", []),
            frame_count,
        )

        self.detection_agent.enrich(tracked_detections, frame_count)

        detections = {
            "detections": tracked_detections,
            "threat_detected": sum(1 for d in tracked_detections if d.get("is_threat")) > 0,
            "threat_count": sum(1 for d in tracked_detections if d.get("is_threat")),
        }

        swimmers: list[tuple[float, float]] = []
        victim: tuple[float, float] | None = None
        victim_track = self.tracker.get_victim()
        swimmer_tracks = self.tracker.get_swimmers()

        if victim_track:
            bbox = victim_track.get("bbox", [])
            if len(bbox) == 4:
                victim_px = ((int(bbox[0]) + int(bbox[2])) // 2, (int(bbox[1]) + int(bbox[3])) // 2)
                victim = pixel_to_pool(victim_px, frame_shape)

            for st in swimmer_tracks:
                bbox = st.get("bbox", [])
                if len(bbox) == 4:
                    px = ((int(bbox[0]) + int(bbox[2])) // 2, (int(bbox[1]) + int(bbox[3])) // 2)
                    swimmers.append(pixel_to_pool(px, frame_shape))

        dispatch_plan = None
        eta = 0.0
        if victim:
            dispatch_plan = self.path_agent.dispatch(victim, swimmers)
            eta = float(dispatch_plan.get("eta_seconds", 0.0))
            self.decision_agent.set_dispatch_plan(victim, swimmers, dispatch_plan)
            with self._lock:
                self.last_dispatch_plan = dispatch_plan
                self.last_victim_pos = victim
                self.last_swimmer_positions = swimmers

        actions = self.decision_agent.process(detections, frame_count, dispatch_plan, eta)

        if dispatch_plan:
            dispatch_plan["state"] = self.decision_agent.current_state.value
            dispatch_plan["dispatch_ts"] = time.time()
            dispatch_plan["explanation"] = (
                f"Dispatching Lifeguard {dispatch_plan['lifeguard']}: ETA {dispatch_plan['eta_seconds']:.1f}s "
                f"via {dispatch_plan.get('route_type', 'jump_in')} at "
                f"({dispatch_plan['jump_point'][0]:.1f},{dispatch_plan['jump_point'][1]:.1f}m). "
                f"p_distress={self.decision_agent.world.p_distress:.2f} sustained "
                f"{self.decision_agent.world.time_in_risk:.1f}s."
            )
            actions["explanation"] = dispatch_plan["explanation"]

        ems_payload = self.ems_agent.check(
            self.decision_agent.world.p_distress,
            actions.get("p_unresponsive", 0.0),
            actions.get("time_in_risk", 0.0),
            self.last_victim_pos,
            self.last_dispatch_plan,
        )
        if ems_payload:
            actions["ems_payload"] = ems_payload

        if not detections.get("threat_detected"):
            self.ems_agent.reset()

        return {
            "detections": detections,
            "actions": actions,
            "dispatch_plan": dispatch_plan,
            "swimmer_positions": swimmers,
            "victim_pos": victim,
            "tracks": self.tracker.get_all_tracks(),
        }
