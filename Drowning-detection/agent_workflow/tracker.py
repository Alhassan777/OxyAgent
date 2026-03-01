"""IoU-based multi-object tracking for drowning detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-6)


def center(box: tuple | list) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class Track:
    id: int
    bbox: tuple[int, int, int, int]
    last_frame: int
    label: str = "unknown"
    label_votes: dict = field(default_factory=dict)
    label_run: int = 0
    last_label: str = "unknown"
    p_distress: float = 0.5
    is_threat: bool = False

    def update(self, detection: dict, frame_idx: int) -> None:
        bbox = detection.get("bbox", [])
        if len(bbox) == 4:
            self.bbox = tuple(bbox)
        self.last_frame = frame_idx

        label = detection.get("label", "unknown")
        self.label = label
        self.p_distress = detection.get("p_distress", 0.5)
        self.is_threat = detection.get("is_threat", False)

        if label != "unknown":
            self.label_votes[label] = self.label_votes.get(label, 0) + 1
            if label == self.last_label:
                self.label_run += 1
            else:
                self.last_label = label
                self.label_run = 1

    def majority_label(self) -> str:
        if not self.label_votes:
            return self.label
        return max(self.label_votes, key=self.label_votes.get)

    def to_detection(self) -> dict:
        return {
            "track_id": self.id,
            "bbox": list(self.bbox),
            "label": self.majority_label(),
            "p_distress": self.p_distress,
            "is_threat": self.is_threat,
            "label_votes": dict(self.label_votes),
            "label_run": self.label_run,
        }


class Tracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 20,
        min_persist: int = 1,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_persist = min_persist
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    def update(self, detections: list[dict], frame_idx: int) -> list[dict]:
        det_boxes = []
        for d in detections:
            bbox = d.get("bbox", [])
            if len(bbox) == 4:
                det_boxes.append(tuple(bbox))
            else:
                det_boxes.append(None)

        pairs = []
        for track_id, track in self.tracks.items():
            for di, box in enumerate(det_boxes):
                if box is None:
                    continue
                ov = iou(track.bbox, box)
                if ov >= self.iou_threshold:
                    pairs.append((ov, track_id, di))

        pairs.sort(reverse=True)
        used_tracks = set()
        used_dets = set()

        for _, track_id, det_idx in pairs:
            if track_id in used_tracks or det_idx in used_dets:
                continue
            used_tracks.add(track_id)
            used_dets.add(det_idx)

            self.tracks[track_id].update(detections[det_idx], frame_idx)

        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue

            new_track = Track(
                id=self.next_id,
                bbox=tuple(bbox),
                last_frame=frame_idx,
                label=det.get("label", "unknown"),
                p_distress=det.get("p_distress", 0.5),
                is_threat=det.get("is_threat", False),
            )
            if new_track.label != "unknown":
                new_track.label_votes[new_track.label] = 1
                new_track.last_label = new_track.label
                new_track.label_run = 1

            self.tracks[self.next_id] = new_track
            self.next_id += 1

        stale = [
            tid for tid, t in self.tracks.items()
            if frame_idx - t.last_frame > self.max_age
        ]
        for tid in stale:
            del self.tracks[tid]

        return self.get_stable_detections()

    def get_stable_detections(self) -> list[dict]:
        stable = []
        for track in self.tracks.values():
            if track.label_run < self.min_persist:
                continue
            stable.append(track.to_detection())
        return stable

    def get_all_tracks(self) -> list[dict]:
        return [t.to_detection() for t in self.tracks.values()]

    def get_victim(self) -> dict | None:
        victims = [
            t for t in self.tracks.values()
            if t.is_threat or t.majority_label() == "drowning"
        ]
        if not victims:
            return None
        return max(victims, key=lambda t: t.p_distress).to_detection()

    def get_swimmers(self) -> list[dict]:
        return [
            t.to_detection() for t in self.tracks.values()
            if not t.is_threat and t.majority_label() == "swimming"
        ]
