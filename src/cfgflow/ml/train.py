from __future__ import annotations

import logging
import math
import pathlib
import sqlite3
from dataclasses import dataclass

from cfgflow.ml.graph import build_edge_adjacency
from cfgflow.ml.model import ModelMeta, build_model, build_row_normalized_adjacency, require_torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    max_edges: int
    t_in: int
    t_out: int
    epochs: int
    batch_size: int
    lr: float
    device: str


def _top_edges_by_count(con: sqlite3.Connection, max_edges: int) -> list[str]:
    rows = con.execute(
        "SELECT edge_id, COUNT(*) AS c FROM edge_state GROUP BY edge_id ORDER BY c DESC LIMIT ?",
        (int(max_edges),),
    ).fetchall()
    return [str(r[0]) for r in rows]


def _load_series_matrix(con: sqlite3.Connection, edge_ids: list[str]):
    require_torch()
    import torch

    t_rows = con.execute(
        "SELECT DISTINCT t FROM edge_state ORDER BY t ASC",
    ).fetchall()
    times = [float(r[0]) for r in t_rows]
    if len(times) < 10:
        raise RuntimeError("Not enough recorded timesteps. Run with --sqlite and let it record longer.")

    # Estimate dt as median diff
    diffs = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    diffs_sorted = sorted(diffs)
    dt_s = float(diffs_sorted[len(diffs_sorted) // 2]) if diffs_sorted else 1.0
    dt_s = dt_s if dt_s > 0 else 1.0

    t_index = {t: i for i, t in enumerate(times)}
    n_t = len(times)
    n = len(edge_ids)
    idx = {eid: j for j, eid in enumerate(edge_ids)}

    x = torch.zeros((n_t, n), dtype=torch.float32)
    q = "SELECT edge_id,t,congestion FROM edge_state WHERE edge_id IN (%s)" % ",".join(["?"] * n)
    for eid, t, c in con.execute(q, edge_ids):
        j = idx.get(str(eid))
        i = t_index.get(float(t))
        if j is None or i is None:
            continue
        x[i, j] = float(c)

    return x, dt_s


class _WindowDataset:
    def __init__(self, series, t_in: int, t_out: int, indices: list[int]):
        self.series = series
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        i = self.indices[k]
        x = self.series[i : i + self.t_in, :]
        y = self.series[i + self.t_in : i + self.t_in + self.t_out, :]
        return x, y


def train_model(
    *,
    net_path: str,
    sqlite_path: str,
    out_path: str,
    max_edges: int,
    t_in: int,
    t_out: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> None:
    require_torch()
    import torch
    from torch.utils.data import DataLoader

    net_p = pathlib.Path(net_path)
    sqlite_p = pathlib.Path(sqlite_path)
    out_p = pathlib.Path(out_path)
    if not net_p.exists():
        raise FileNotFoundError(str(net_p))
    if not sqlite_p.exists():
        raise FileNotFoundError(str(sqlite_p))

    con = sqlite3.connect(str(sqlite_p))
    try:
        edge_ids = _top_edges_by_count(con, int(max_edges))
        if len(edge_ids) < 20:
            raise RuntimeError("Not enough recorded edges in sqlite (edge_state table is too small).")

        # Build global adjacency then filter to selected edges
        all_edge_ids, arcs = build_edge_adjacency(net_p)
        all_idx = {eid: i for i, eid in enumerate(all_edge_ids)}
        selected = [eid for eid in edge_ids if eid in all_idx]
        edge_ids = selected
        sel_idx = {eid: i for i, eid in enumerate(edge_ids)}
        sel_arcs: list[tuple[int, int]] = []
        for i, j in arcs:
            ei = all_edge_ids[i]
            ej = all_edge_ids[j]
            ii = sel_idx.get(ei)
            jj = sel_idx.get(ej)
            if ii is not None and jj is not None:
                sel_arcs.append((ii, jj))

        series, dt_s = _load_series_matrix(con, edge_ids)
    finally:
        con.close()

    cfg = TrainConfig(
        max_edges=int(max_edges),
        t_in=int(t_in),
        t_out=int(t_out),
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        device=str(device),
    )

    n_t, n = series.shape
    n_samples = n_t - (cfg.t_in + cfg.t_out)
    if n_samples <= 20:
        raise RuntimeError(
            f"Not enough timesteps for windows (T={n_t}, t_in={cfg.t_in}, t_out={cfg.t_out})."
        )

    split = int(0.8 * n_samples)
    train_idx = list(range(0, split))
    val_idx = list(range(split, n_samples))

    train_ds = _WindowDataset(series, cfg.t_in, cfg.t_out, train_idx)
    val_ds = _WindowDataset(series, cfg.t_in, cfg.t_out, val_idx)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    a_norm = build_row_normalized_adjacency(n=n, arcs=sel_arcs).to(cfg.device)
    model = build_model(n=n, t_out=cfg.t_out, hidden=48, kernel=3, alpha=0.7).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.MSELoss()

    def eval_dl(dl) -> tuple[float, float]:
        model.eval()
        mse_sum = 0.0
        mae_sum = 0.0
        n_items = 0
        with torch.no_grad():
            for x, y in dl:
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                pred = model(x, a_norm)
                mse = torch.mean((pred - y) ** 2).item()
                mae = torch.mean(torch.abs(pred - y)).item()
                b = x.shape[0]
                mse_sum += mse * b
                mae_sum += mae * b
                n_items += b
        return mse_sum / max(1, n_items), mae_sum / max(1, n_items)

    for ep in range(cfg.epochs):
        model.train()
        for x, y in train_dl:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            pred = model(x, a_norm)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        val_mse, val_mae = eval_dl(val_dl)
        rmse = math.sqrt(val_mse)
        print(f"epoch {ep+1}/{cfg.epochs}  val_rmse={rmse:.4f}  val_mae={val_mae:.4f}")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    meta = ModelMeta(
        version=1,
        dt_s=float(dt_s),
        t_in=cfg.t_in,
        t_out=cfg.t_out,
        edge_ids=edge_ids,
        arcs=sel_arcs,
        config={"hidden": 48, "kernel": 3, "alpha": 0.7},
    )
    torch.save(
        {
            "meta": meta.__dict__,
            "state_dict": model.state_dict(),
        },
        str(out_p),
    )
    print(f"Saved model: {out_p}")
