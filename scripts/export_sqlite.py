from __future__ import annotations

import argparse
import csv
import pathlib
import sqlite3


def main() -> None:
    p = argparse.ArgumentParser(description="Export recorded edge_state from sqlite to CSV.")
    p.add_argument("--sqlite", required=True, help="Path to sqlite db")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--limit", type=int, default=0, help="Optional row limit (0 = no limit)")
    args = p.parse_args()

    sqlite_path = pathlib.Path(args.sqlite)
    out_path = pathlib.Path(args.out)
    con = sqlite3.connect(str(sqlite_path))
    try:
        q = "SELECT edge_id,t,speed_mps,veh_n,occupancy_pct,congestion FROM edge_state ORDER BY t ASC"
        if args.limit and args.limit > 0:
            q += f" LIMIT {int(args.limit)}"
        rows = con.execute(q)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["edge_id", "t", "speed_mps", "veh_n", "occupancy_pct", "congestion"])
            for r in rows:
                w.writerow(r)
    finally:
        con.close()


if __name__ == "__main__":
    main()

