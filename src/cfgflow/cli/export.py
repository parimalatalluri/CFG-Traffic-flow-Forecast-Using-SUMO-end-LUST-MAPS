from __future__ import annotations

import csv
import pathlib
import sqlite3


def export_sqlite_to_csv(*, sqlite_path: str, out_csv: str, limit: int = 0) -> None:
    sqlite_p = pathlib.Path(sqlite_path)
    out_p = pathlib.Path(out_csv)
    if not sqlite_p.exists():
        raise FileNotFoundError(str(sqlite_p))

    con = sqlite3.connect(str(sqlite_p))
    try:
        q = "SELECT edge_id,t,speed_mps,veh_n,occupancy_pct,congestion FROM edge_state ORDER BY t ASC"
        if limit and limit > 0:
            q += f" LIMIT {int(limit)}"
        rows = con.execute(q)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["edge_id", "t", "speed_mps", "veh_n", "occupancy_pct", "congestion"])
            for r in rows:
                w.writerow(r)
        print(f"Wrote {out_p}")
    finally:
        con.close()

