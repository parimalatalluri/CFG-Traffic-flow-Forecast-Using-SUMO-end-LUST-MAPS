from __future__ import annotations

import pathlib
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceBatch:
    # shapes are left as "array-like" to avoid forcing numpy/torch for baseline usage
    x: list[list[float]]  # [T_in][N]
    y: list[list[float]]  # [T_out][N]
    edge_ids: list[str]


def load_edge_series(
    *, sqlite_path: pathlib.Path, edge_ids: list[str], t_from: float | None = None, t_to: float | None = None
) -> dict[str, list[tuple[float, float]]]:
    """
    Returns per-edge list of (t, congestion).
    """
    con = sqlite3.connect(str(sqlite_path))
    try:
        q = "SELECT edge_id,t,congestion FROM edge_state WHERE edge_id IN (%s)" % (
            ",".join(["?"] * len(edge_ids))
        )
        params: list[object] = list(edge_ids)
        if t_from is not None:
            q += " AND t >= ?"
            params.append(t_from)
        if t_to is not None:
            q += " AND t <= ?"
            params.append(t_to)
        q += " ORDER BY t ASC"

        rows = con.execute(q, params).fetchall()
        out: dict[str, list[tuple[float, float]]] = {eid: [] for eid in edge_ids}
        for eid, t, c in rows:
            out[str(eid)].append((float(t), float(c)))
        return out
    finally:
        con.close()

