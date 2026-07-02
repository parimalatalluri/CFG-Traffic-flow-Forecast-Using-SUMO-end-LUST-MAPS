from __future__ import annotations

import pathlib
from dataclasses import dataclass

from cfgflow.ml.baseline import BaselineForecaster
from cfgflow.ml.model import build_model, build_row_normalized_adjacency, require_torch


@dataclass
class TorchForecaster(BaselineForecaster):
    model_path: pathlib.Path | None = None
    dt_s: float = 1.0
    t_in: int = 12
    t_out: int = 3
    edge_ids: list[str] | None = None
    arcs: list[tuple[int, int]] | None = None
    device: str = "cpu"

    def load(self, model_path: pathlib.Path, device: str = "cpu") -> None:
        require_torch()
        import torch

        ckpt = torch.load(str(model_path), map_location=device)
        meta = ckpt.get("meta", {})
        self.dt_s = float(meta.get("dt_s", 1.0))
        self.t_in = int(meta.get("t_in", 12))
        self.t_out = int(meta.get("t_out", 3))
        self.edge_ids = list(meta.get("edge_ids", []))
        self.arcs = [tuple(x) for x in meta.get("arcs", [])]
        cfg = meta.get("config", {})
        hidden = int(cfg.get("hidden", 48))
        kernel = int(cfg.get("kernel", 3))
        alpha = float(cfg.get("alpha", 0.7))

        n = len(self.edge_ids)
        self._a_norm = build_row_normalized_adjacency(n=n, arcs=self.arcs).to(device)
        self._model = build_model(n=n, t_out=self.t_out, hidden=hidden, kernel=kernel, alpha=alpha).to(device)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        self.model_path = model_path
        self.device = device
        self.history_len = max(self.history_len, self.t_in)

    def update(self, edge_id: str, congestion: float, veh_n: int | None = None, occupancy_pct: float | None = None) -> None:
        # keep baseline histories for fallback + model input window
        super().update(edge_id, congestion, veh_n=veh_n, occupancy_pct=occupancy_pct)

    def predict(self, *, edge_ids: list[str], horizons_s: list[int]) -> dict[int, dict[str, float]]:
        if not getattr(self, "_model", None) or not self.edge_ids:
            return super().predict(edge_ids=edge_ids, horizons_s=horizons_s)

        require_torch()
        import torch

        # Map requested horizons in seconds to model step indices
        step_idx = []
        for h in horizons_s:
            k = int(round(float(h) / max(1e-6, self.dt_s)))
            k = 1 if k < 1 else k
            k = self.t_out if k > self.t_out else k
            step_idx.append(k - 1)

        # Build input window in model edge order
        n = len(self.edge_ids)
        x = torch.zeros((1, self.t_in, n), dtype=torch.float32, device=self.device)
        for j, eid in enumerate(self.edge_ids):
            hist = self.history.get(eid)
            if not hist:
                continue
            vals = list(hist)[-self.t_in :]
            if len(vals) < self.t_in:
                vals = [0.0] * (self.t_in - len(vals)) + vals
            x[0, :, j] = torch.tensor(vals, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            y = self._model(x, self._a_norm)[0]  # (t_out, n)

        # Return only requested edges; for others, fall back to persistence
        model_idx = {eid: j for j, eid in enumerate(self.edge_ids)}
        out: dict[int, dict[str, float]] = {}
        for h, k in zip(horizons_s, step_idx):
            pred_h: dict[str, float] = {}
            for eid in edge_ids:
                j = model_idx.get(eid)
                if j is None:
                    pred_h[eid] = float(self.last_congestion.get(eid, 0.0))
                else:
                    pred_h[eid] = float(y[k, j].clamp(0.0, 1.0).item())
            out[h] = pred_h
        return out
