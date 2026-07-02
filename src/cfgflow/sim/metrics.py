from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeMetrics:
    speed_mps: float
    veh_n: int
    occupancy_pct: float
    congestion: float  # 0..1


def compute_congestion(*, speed_mps: float, freeflow_speed_mps: float | None) -> float:
    if freeflow_speed_mps is None or freeflow_speed_mps <= 1e-6:
        return 0.0
    r = speed_mps / freeflow_speed_mps
    r = 0.0 if r < 0.0 else 1.0 if r > 1.0 else r
    return 1.0 - r

