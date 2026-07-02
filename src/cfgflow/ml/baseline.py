from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math


@dataclass
class BaselineForecaster:
    # latest observed congestion (0..1) per edge
    last_congestion: dict[str, float] = field(default_factory=dict)
    last_veh_n: dict[str, int] = field(default_factory=dict)
    last_occupancy_pct: dict[str, float] = field(default_factory=dict)
    history_len: int = 60
    history: dict[str, deque[float]] = field(default_factory=dict)
    occupancy_history: dict[str, deque[float]] = field(default_factory=dict)
    veh_n_history: dict[str, deque[int]] = field(default_factory=dict)
    dt_s: float = 1.0  # time between updates (best-effort)

    def set_dt_s(self, dt_s: float) -> None:
        if dt_s > 1e-6:
            self.dt_s = float(dt_s)

    def update(self, edge_id: str, congestion: float, veh_n: int | None = None, occupancy_pct: float | None = None) -> None:
        self.last_congestion[edge_id] = float(congestion)
        if veh_n is not None:
            self.last_veh_n[edge_id] = int(veh_n)
        if occupancy_pct is not None:
            self.last_occupancy_pct[edge_id] = float(occupancy_pct)

        q = self.history.get(edge_id)
        if q is None:
            q = deque(maxlen=int(self.history_len))
            self.history[edge_id] = q
        q.append(float(congestion))

        if occupancy_pct is not None:
            oq = self.occupancy_history.get(edge_id)
            if oq is None:
                oq = deque(maxlen=int(self.history_len))
                self.occupancy_history[edge_id] = oq
            oq.append(float(occupancy_pct))

        if veh_n is not None:
            vq = self.veh_n_history.get(edge_id)
            if vq is None:
                vq = deque(maxlen=int(self.history_len))
                self.veh_n_history[edge_id] = vq
            vq.append(int(veh_n))

    def predict(self, *, edge_ids: list[str], horizons_s: list[int]) -> dict[int, dict[str, float]]:
        """
        Heuristic baseline using live vehicle/occupancy:
        - trend extrapolation from recent congestion history
        - slight increase when occupancy/vehicle count is high
        """
        out: dict[int, dict[str, float]] = {}
        dt = max(1e-6, float(self.dt_s))

        for h in horizons_s:
            steps_ahead = int(round(float(h) / dt))
            steps_ahead = 1 if steps_ahead < 1 else steps_ahead

            pred_h: dict[str, float] = {}
            for eid in edge_ids:
                last = float(self.last_congestion.get(eid, 0.0))
                hist = self.history.get(eid)

                slope = 0.0
                if hist and len(hist) >= 4:
                    window = list(hist)[-8:]
                    slope = (window[-1] - window[0]) / max(1, (len(window) - 1))

                occ = float(self.last_occupancy_pct.get(eid, 0.0))
                veh = int(self.last_veh_n.get(eid, 0))
                occ_boost = 0.10 * (occ / 100.0)
                veh_boost = 0.02 * math.log1p(max(0, veh))

                y = last + slope * float(steps_ahead) + occ_boost + veh_boost
                if y < 0.0:
                    y = 0.0
                elif y > 1.0:
                    y = 1.0
                pred_h[eid] = float(y)

            out[int(h)] = pred_h

        return out
