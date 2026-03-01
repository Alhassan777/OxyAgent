"""Detection Agent: adds p_unresponsive heuristics to tracked detections."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

try:
    from .. import config
except ImportError:
    import config  # type: ignore

# Priority weights
PRIORITY_A, PRIORITY_B, PRIORITY_C, PRIORITY_D = 0.4, 0.3, 0.2, 0.1


def compute_priority(p_distress: float, p_unresponsive: float, time_in_risk: float, eta: float) -> float:
    return PRIORITY_A * p_distress + PRIORITY_B * p_unresponsive + PRIORITY_C * time_in_risk - PRIORITY_D * eta


@dataclass
class MotionSample:
    centroid_x: float
    centroid_y: float
    area: float
    frame_idx: int


class DetectionAgent:
    """Enriches tracked detections with p_unresponsive from bbox motion heuristics."""

    def __init__(
        self,
        motion_history_frames: int = 30,
        low_motion_threshold: float = 0.02,
        area_shrink_threshold: float = 0.15,
    ) -> None:
        self.motion_history_frames = motion_history_frames
        self.low_motion_threshold = low_motion_threshold
        self.area_shrink_threshold = area_shrink_threshold
        self._track_history: dict[int, deque[MotionSample]] = defaultdict(
            lambda: deque(maxlen=motion_history_frames)
        )

    def _bbox_to_sample(self, bbox: list | tuple, frame_idx: int) -> MotionSample:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = max(1e-6, (x2 - x1) * (y2 - y1))
        return MotionSample(centroid_x=cx, centroid_y=cy, area=area, frame_idx=frame_idx)

    def _compute_p_unresponsive(self, track_id: int, frame_idx: int) -> float:
        """Heuristic: low position variance + shrinking area suggests stationary/limp."""
        history = self._track_history[track_id]
        if len(history) < 5:
            return 0.0

        samples = list(history)
        cx_mean = sum(s.centroid_x for s in samples) / len(samples)
        cy_mean = sum(s.centroid_y for s in samples) / len(samples)
        var_x = sum((s.centroid_x - cx_mean) ** 2 for s in samples) / len(samples)
        var_y = sum((s.centroid_y - cy_mean) ** 2 for s in samples) / len(samples)
        position_variance = (var_x + var_y) ** 0.5

        areas = [s.area for s in samples]
        area_early = sum(areas[: len(areas) // 2]) / max(1, len(areas) // 2)
        area_late = sum(areas[len(areas) // 2 :]) / max(1, len(areas) - len(areas) // 2)
        area_ratio = area_late / max(1e-6, area_early)

        frame_span = samples[-1].frame_idx - samples[0].frame_idx
        if frame_span < 10:
            return 0.0

        norm_var = position_variance / max(1e-6, (area_early ** 0.5))
        low_motion = norm_var < self.low_motion_threshold
        shrinking = area_ratio < (1.0 - self.area_shrink_threshold)

        if low_motion and shrinking:
            return min(0.9, 0.5 + 0.2 * (1.0 - norm_var / self.low_motion_threshold))
        if low_motion:
            return min(0.7, 0.3 + 0.2 * (1.0 - norm_var / self.low_motion_threshold))
        return 0.0

    def enrich(
        self,
        tracked_detections: list[dict],
        frame_idx: int,
    ) -> list[dict]:
        """
        Add p_unresponsive to threat detections based on bbox motion history.
        Mutates detections in place and returns them.
        """
        track_id_to_det = {d.get("track_id"): d for d in tracked_detections if d.get("track_id") is not None}

        for det in tracked_detections:
            track_id = det.get("track_id")
            bbox = det.get("bbox") or det.get("box")
            if track_id is None or not bbox or len(bbox) != 4:
                continue

            sample = self._bbox_to_sample(bbox, frame_idx)
            self._track_history[track_id].append(sample)

            p_unresponsive = self._compute_p_unresponsive(track_id, frame_idx)
            if det.get("is_threat") or (str(det.get("label", "")).lower() == "drowning"):
                det["p_unresponsive"] = p_unresponsive

        for tid in list(self._track_history.keys()):
            if tid not in track_id_to_det:
                if len(self._track_history[tid]) > 0:
                    last_frame = self._track_history[tid][-1].frame_idx
                    if frame_idx - last_frame > 60:
                        del self._track_history[tid]

        return tracked_detections
