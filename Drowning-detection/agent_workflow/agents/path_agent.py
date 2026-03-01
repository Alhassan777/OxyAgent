"""Path Agent: compute optimal route considering crowd, swim speed, and route type."""

from __future__ import annotations

import math
from typing import Literal

try:
    from .. import config
except ImportError:
    import config  # type: ignore


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _sample_pool_boundary(n_points: int = 80) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    w = config.POOL_W
    l = config.POOL_L
    perimeter = 2.0 * (w + l)

    for i in range(n_points):
        s = (i / n_points) * perimeter
        if s < w:
            points.append((s, 0.0))
        elif s < w + l:
            points.append((w, s - w))
        elif s < 2 * w + l:
            points.append((w - (s - (w + l)), l))
        else:
            points.append((0.0, l - (s - (2 * w + l))))
    return points


def _point_to_segment_dist(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    if vv <= 1e-9:
        return _euclidean(p, a)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    proj = (ax + t * vx, ay + t * vy)
    return _euclidean(p, proj)


def _gaussian_penalty(distance: float, sigma: float = 1.5, amplitude: float = 2.0) -> float:
    return amplitude * math.exp(-(distance * distance) / (2.0 * sigma * sigma))


def _lifeguard_label(pos: tuple[float, float]) -> str:
    if _euclidean(pos, config.LIFEGUARD_A) <= _euclidean(pos, config.LIFEGUARD_B):
        return "A"
    return "B"


def _build_density_map(
    swimmers: list[tuple[float, float]],
    grid_nx: int = 5,
    grid_ny: int = 10,
) -> list[list[float]]:
    """Pool grid (nx x ny), each cell = swimmer count (normalized)."""
    density = [[0.0] * grid_ny for _ in range(grid_nx)]
    w, l = config.POOL_W, config.POOL_L
    if w <= 0 or l <= 0:
        return density

    for sx, sy in swimmers:
        ix = min(int(sx / w * grid_nx), grid_nx - 1)
        iy = min(int(sy / l * grid_ny), grid_ny - 1)
        ix = max(0, ix)
        iy = max(0, iy)
        density[ix][iy] += 1.0

    return density


def _density_along_segment(
    a: tuple[float, float],
    b: tuple[float, float],
    density_map: list[list[float]],
    n_steps: int = 10,
) -> float:
    """Average density along segment from a to b."""
    total = 0.0
    w, l = config.POOL_W, config.POOL_L
    grid_nx, grid_ny = len(density_map), len(density_map[0]) if density_map else 1

    for i in range(n_steps):
        t = (i + 0.5) / n_steps
        px = a[0] + t * (b[0] - a[0])
        py = a[1] + t * (b[1] - a[1])
        ix = min(int(px / max(w, 1e-6) * grid_nx), grid_nx - 1)
        iy = min(int(py / max(l, 1e-6) * grid_ny), grid_ny - 1)
        ix, iy = max(0, ix), max(0, iy)
        total += density_map[ix][iy]
    return total / n_steps


def _effective_swim_speed(density: float, k: float = 0.5) -> float:
    """Reduce swim speed in crowded zones."""
    base = getattr(config, "SWIM_SPEED", 1.0)
    return base / (1.0 + k * density)


RouteType = Literal["jump_in", "go_around"]


class PathAgent:
    """Computes optimal dispatch route with crowd density and route-type selection."""

    def __init__(
        self,
        crowd_sigma: float = 1.5,
        crowd_amplitude: float = 2.0,
        density_speed_k: float = 0.5,
        n_boundary_points: int = 80,
    ) -> None:
        self.crowd_sigma = crowd_sigma
        self.crowd_amplitude = crowd_amplitude
        self.density_speed_k = density_speed_k
        self.n_boundary_points = n_boundary_points

    def compute_route(
        self,
        lifeguard_pos: tuple[float, float],
        victim_pos: tuple[float, float],
        swimmer_positions: list[tuple[float, float]],
        route_type: RouteType = "jump_in",
    ) -> dict:
        """
        Compute best route. For jump_in: sample perimeter, pick best jump point.
        For go_around: walk deck to nearest point, then short swim.
        """
        deck_speed = getattr(config, "DECK_SPEED", 3.0)
        swim_speed = getattr(config, "SWIM_SPEED", 1.0)
        lg = _lifeguard_label(lifeguard_pos)
        density_map = _build_density_map(swimmer_positions)

        if route_type == "go_around":
            return self._compute_go_around(
                lifeguard_pos, victim_pos, swimmer_positions, density_map, deck_speed, swim_speed, lg
            )

        best: dict | None = None
        boundary = _sample_pool_boundary(self.n_boundary_points)

        for j in boundary:
            deck_time = _euclidean(lifeguard_pos, j) / max(deck_speed, 1e-6)
            swim_dist = _euclidean(j, victim_pos)

            crowd_penalty = 0.0
            for swimmer in swimmer_positions:
                d = _point_to_segment_dist(
                    (float(swimmer[0]), float(swimmer[1])),
                    j,
                    victim_pos,
                )
                crowd_penalty += _gaussian_penalty(d, sigma=self.crowd_sigma, amplitude=self.crowd_amplitude)

            seg_density = _density_along_segment(j, victim_pos, density_map)
            eff_swim_speed = _effective_swim_speed(seg_density, self.density_speed_k)
            swim_time = (swim_dist / max(eff_swim_speed, 1e-6)) * (1.0 + crowd_penalty)
            total_time = deck_time + swim_time

            candidate = {
                "route_type": "jump_in",
                "jump_point": (float(j[0]), float(j[1])),
                "eta_seconds": float(total_time),
                "deck_time": float(deck_time),
                "swim_time": float(swim_time),
                "crowd_penalty": float(crowd_penalty),
                "lifeguard": lg,
            }

            if best is None or candidate["eta_seconds"] < best["eta_seconds"]:
                best = candidate

        return best or {
            "route_type": "jump_in",
            "jump_point": (0.0, 0.0),
            "eta_seconds": float("inf"),
            "deck_time": float("inf"),
            "swim_time": float("inf"),
            "crowd_penalty": 0.0,
            "lifeguard": lg,
        }

    def _compute_go_around(
        self,
        lifeguard_pos: tuple[float, float],
        victim_pos: tuple[float, float],
        swimmer_positions: list[tuple[float, float]],
        density_map: list[list[float]],
        deck_speed: float,
        swim_speed: float,
        lg: str,
    ) -> dict:
        """Deck path along perimeter to nearest point to victim, then short swim."""
        boundary = _sample_pool_boundary(self.n_boundary_points)
        best: dict | None = None

        for j in boundary:
            deck_dist = _euclidean(lifeguard_pos, j)
            swim_dist = _euclidean(j, victim_pos)
            deck_time = deck_dist / max(deck_speed, 1e-6)

            crowd_penalty = 0.0
            for swimmer in swimmer_positions:
                d = _point_to_segment_dist(
                    (float(swimmer[0]), float(swimmer[1])),
                    j,
                    victim_pos,
                )
                crowd_penalty += _gaussian_penalty(d, sigma=self.crowd_sigma, amplitude=self.crowd_amplitude)

            seg_density = _density_along_segment(j, victim_pos, density_map)
            eff_swim_speed = _effective_swim_speed(seg_density, self.density_speed_k)
            swim_time = (swim_dist / max(eff_swim_speed, 1e-6)) * (1.0 + crowd_penalty)
            total_time = deck_time + swim_time

            candidate = {
                "route_type": "go_around",
                "jump_point": (float(j[0]), float(j[1])),
                "eta_seconds": float(total_time),
                "deck_time": float(deck_time),
                "swim_time": float(swim_time),
                "crowd_penalty": float(crowd_penalty),
                "lifeguard": lg,
            }

            if best is None or candidate["eta_seconds"] < best["eta_seconds"]:
                best = candidate

        return best or {
            "route_type": "go_around",
            "jump_point": (0.0, 0.0),
            "eta_seconds": float("inf"),
            "deck_time": float("inf"),
            "swim_time": float("inf"),
            "crowd_penalty": 0.0,
            "lifeguard": lg,
        }

    def dispatch(
        self,
        victim_pos: tuple[float, float],
        swimmer_positions: list[tuple[float, float]],
    ) -> dict:
        """Best plan across both lifeguards and route types."""
        plan_a_jump = self.compute_route(
            config.LIFEGUARD_A, victim_pos, swimmer_positions, route_type="jump_in"
        )
        plan_a_around = self.compute_route(
            config.LIFEGUARD_A, victim_pos, swimmer_positions, route_type="go_around"
        )
        plan_b_jump = self.compute_route(
            config.LIFEGUARD_B, victim_pos, swimmer_positions, route_type="jump_in"
        )
        plan_b_around = self.compute_route(
            config.LIFEGUARD_B, victim_pos, swimmer_positions, route_type="go_around"
        )

        candidates = [plan_a_jump, plan_a_around, plan_b_jump, plan_b_around]
        best = min(candidates, key=lambda p: p["eta_seconds"])
        return dict(best)
