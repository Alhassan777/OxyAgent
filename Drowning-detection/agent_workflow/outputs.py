"""Output logging: ambulance log, path images (stickman), annotated detection video."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config  # type: ignore


def _pool_to_canvas(
    coord: tuple[float, float],
    pool_pixel_w: int,
    pool_pixel_h: int,
) -> tuple[int, int]:
    """Map pool coords (meters) to image coords within pool area."""
    x = int((coord[0] / max(config.POOL_W, 1e-6)) * pool_pixel_w)
    y = int((coord[1] / max(config.POOL_L, 1e-6)) * pool_pixel_h)
    return x, y


def draw_stickman(
    image: np.ndarray,
    center: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 255),
    scale: float = 1.0,
) -> None:
    """Draw a simple stickman (lifeguard) at center (x, y)."""
    cx, cy = int(center[0]), int(center[1])
    s = int(12 * scale)
    # Head
    cv2.circle(image, (cx, cy - s), s // 2, color, 2)
    # Body
    cv2.line(image, (cx, cy - s // 2), (cx, cy + s), color, 2)
    # Arms (raised, lifeguard pose)
    cv2.line(image, (cx, cy - s // 3), (cx - s, cy - s), color, 2)
    cv2.line(image, (cx, cy - s // 3), (cx + s, cy - s), color, 2)
    # Legs
    cv2.line(image, (cx, cy + s), (cx - s // 2, cy + 2 * s), color, 2)
    cv2.line(image, (cx, cy + s), (cx + s // 2, cy + 2 * s), color, 2)


def create_path_image(
    lifeguard_pos: tuple[float, float],
    victim_pos: tuple[float, float],
    jump_point: tuple[float, float],
    swimmer_positions: list[tuple[float, float]],
    dispatch_plan: dict,
    img_w: int = 600,
    img_h: int = 400,
) -> np.ndarray:
    """
    Create an image showing: pool, stickman (lifeguard), victim, swimmers,
    and the shortest path (deck segment + swim segment).
    """
    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    canvas[:] = (30, 40, 50)  # Dark blue-gray (pool water look)

    pad = 20
    pool_pixel_w = img_w - 2 * pad
    pool_pixel_h = img_h - 80
    pool_rect = (pad, 30, img_w - pad, img_h - 50)
    cv2.rectangle(canvas, pool_rect[:2], pool_rect[2:], (80, 100, 120), 2)
    cv2.putText(
        canvas, "POOL", (img_w // 2 - 30, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
    )

    def to_xy(c: tuple[float, float]) -> tuple[int, int]:
        px, py = _pool_to_canvas(c, pool_pixel_w, pool_pixel_h)
        return (pad + px, 30 + py)

    lg_xy = to_xy(lifeguard_pos)
    vic_xy = to_xy(victim_pos)
    jp_xy = to_xy(jump_point)

    # Swimmers (small blue circles)
    for sw in swimmer_positions:
        sx, sy = to_xy(sw)
        cv2.circle(canvas, (sx, sy), 4, (255, 150, 0), -1)

    # Path: deck (lifeguard -> jump point) in white dashed
    for i in range(0, 100, 20):
        t0, t1 = i / 100, min((i + 10) / 100, 1.0)
        p0 = (int(lg_xy[0] + t0 * (jp_xy[0] - lg_xy[0])), int(lg_xy[1] + t0 * (jp_xy[1] - lg_xy[1])))
        p1 = (int(lg_xy[0] + t1 * (jp_xy[0] - lg_xy[0])), int(lg_xy[1] + t1 * (jp_xy[1] - lg_xy[1])))
        cv2.line(canvas, p0, p1, (255, 255, 255), 3)

    # Path: swim (jump point -> victim) in red dashed
    for i in range(0, 100, 20):
        t0, t1 = i / 100, min((i + 10) / 100, 1.0)
        p0 = (int(jp_xy[0] + t0 * (vic_xy[0] - jp_xy[0])), int(jp_xy[1] + t0 * (vic_xy[1] - jp_xy[1])))
        p1 = (int(jp_xy[0] + t1 * (vic_xy[0] - jp_xy[0])), int(jp_xy[1] + t1 * (vic_xy[1] - jp_xy[1])))
        cv2.line(canvas, p0, p1, (0, 0, 255), 3)

    # Jump point (X marker)
    jx, jy = jp_xy
    cv2.line(canvas, (jx - 8, jy - 8), (jx + 8, jy + 8), (0, 255, 0), 2)
    cv2.line(canvas, (jx - 8, jy + 8), (jx + 8, jy - 8), (0, 255, 0), 2)

    # Stickman (lifeguard)
    draw_stickman(canvas, lg_xy, (0, 255, 255), 1.2)
    cv2.putText(canvas, "LG", (lg_xy[0] + 15, lg_xy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Victim (pulsing red circle)
    cv2.circle(canvas, vic_xy, 10, (0, 0, 255), 3)
    cv2.putText(canvas, "VICTIM", (vic_xy[0] + 12, vic_xy[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    eta = dispatch_plan.get("eta_seconds", 0)
    route = dispatch_plan.get("route_type", "jump_in")
    lg = dispatch_plan.get("lifeguard", "?")
    footer = f"Shortest path | ETA: {eta:.1f}s | Lifeguard {lg} | {route}"
    cv2.putText(canvas, footer, (pad, img_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


class OutputLogger:
    """Manages output files: ambulance log, path images, annotated video."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ambulance_log_path = self.output_dir / "ambulance_calls.log"
        self.path_images_dir = self.output_dir / "path_images"
        self.path_images_dir.mkdir(exist_ok=True)
        self._path_image_count = 0
        self._video_writer: cv2.VideoWriter | None = None
        self._video_path: Path | None = None

    def log_ambulance(self, payload: dict) -> None:
        """Append ambulance/EMS trigger to dedicated log."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "AMBULANCE_CALLED",
            **payload,
        }
        with self.ambulance_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"\033[91m[OUTPUT] Ambulance log: {self.ambulance_log_path}\033[0m")

    def save_path_image(
        self,
        lifeguard_pos: tuple[float, float],
        victim_pos: tuple[float, float],
        jump_point: tuple[float, float],
        swimmer_positions: list[tuple[float, float]],
        dispatch_plan: dict,
    ) -> Path:
        """Save path visualization with stickman. Returns path to saved file."""
        img = create_path_image(
            lifeguard_pos, victim_pos, jump_point, swimmer_positions, dispatch_plan,
        )
        self._path_image_count += 1
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        fname = f"path_{self._path_image_count:03d}_{ts}.png"
        out_path = self.path_images_dir / fname
        cv2.imwrite(str(out_path), img)
        print(f"[OUTPUT] Path image: {out_path}")
        return out_path

    def start_video_writer(
        self,
        fps: float = 15.0,
        frame_w: int = 1280,
        frame_h: int = 720,
    ) -> Path:
        """Start recording annotated video."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"detections_{ts}.mp4"
        self._video_path = self.output_dir / fname
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            str(self._video_path), fourcc, fps, (frame_w, frame_h),
        )
        print(f"[OUTPUT] Recording video: {self._video_path}")
        return self._video_path

    def write_video_frame(self, frame: np.ndarray) -> None:
        """Write a frame to the annotated video."""
        if self._video_writer is not None:
            self._video_writer.write(frame)

    def stop_video_writer(self) -> None:
        """Stop and release video writer."""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            if self._video_path:
                print(f"[OUTPUT] Video saved: {self._video_path}")

    def get_output_summary(self) -> dict:
        """Return paths of saved outputs."""
        return {
            "output_dir": str(self.output_dir),
            "ambulance_log": str(self.ambulance_log_path) if self.ambulance_log_path.exists() else None,
            "path_images_dir": str(self.path_images_dir),
            "path_image_count": self._path_image_count,
            "video": str(self._video_path) if self._video_path and self._video_path.exists() else None,
        }
