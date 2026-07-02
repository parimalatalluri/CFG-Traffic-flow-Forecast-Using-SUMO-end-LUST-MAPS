from __future__ import annotations

import logging
import math
import pathlib
from dataclasses import dataclass

from cfgflow.sim.sumo_env import import_sumo_libs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EdgeGeom:
    edge_id: str
    shape: list[tuple[float, float]]
    speed_mps: float
    length_m: float


class NetCache:
    def __init__(self, net_path: pathlib.Path) -> None:
        self.net_path = net_path
        self._net = None
        self._edges: list[EdgeGeom] | None = None
        self._bbox: dict[str, float] | None = None
        self._edge_speed: dict[str, float] = {}

    def _load(self) -> None:
        if self._net is not None:
            return
        import_sumo_libs()
        import sumolib  # type: ignore

        if not self.net_path.exists():
            raise FileNotFoundError(str(self.net_path))
        logger.info("Loading SUMO net: %s", self.net_path)
        self._net = sumolib.net.readNet(str(self.net_path))

    def edges(self) -> list[EdgeGeom]:
        if self._edges is not None:
            return self._edges
        self._load()
        assert self._net is not None

        edges: list[EdgeGeom] = []
        min_x = math.inf
        min_y = math.inf
        max_x = -math.inf
        max_y = -math.inf

        for e in self._net.getEdges():
            eid = e.getID()
            if eid.startswith(":"):
                continue
            shape = [(float(x), float(y)) for x, y in e.getShape()]
            if len(shape) < 2:
                continue
            speed = float(e.getSpeed())
            length = float(e.getLength())
            edges.append(EdgeGeom(edge_id=eid, shape=shape, speed_mps=speed, length_m=length))
            self._edge_speed[eid] = speed
            for x, y in shape:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        self._edges = edges
        self._bbox = {"minX": min_x, "minY": min_y, "maxX": max_x, "maxY": max_y}
        logger.info("Loaded %d edges", len(edges))
        return edges

    def bbox(self) -> dict[str, float]:
        if self._bbox is None:
            _ = self.edges()
        assert self._bbox is not None
        return self._bbox

    def edge_speed_mps(self, edge_id: str) -> float | None:
        if not self._edge_speed:
            _ = self.edges()
        return self._edge_speed.get(edge_id)

    def as_xy_json(self) -> dict:
        edges = self.edges()
        return {
            "bbox": self.bbox(),
            "edges": [
                {"id": e.edge_id, "shape": [[x, y] for (x, y) in e.shape], "speed": e.speed_mps}
                for e in edges
            ],
        }

    def nearest_edge_id(self, *, x: float, y: float) -> str | None:
        edges = self.edges()
        best_id: str | None = None
        best_d2 = math.inf
        for e in edges:
            d2 = _polyline_min_d2(x, y, e.shape)
            if d2 < best_d2:
                best_d2 = d2
                best_id = e.edge_id
        return best_id


def _polyline_min_d2(px: float, py: float, pts: list[tuple[float, float]]) -> float:
    best = math.inf
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        best = min(best, _seg_dist2(px, py, ax, ay, bx, by))
    return best


def _seg_dist2(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy
    t = (apx * abx + apy * aby) / denom
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    cx = ax + t * abx
    cy = ay + t * aby
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy

