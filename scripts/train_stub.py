from __future__ import annotations

import argparse
import pathlib

from cfgflow.ml.graph import build_edge_adjacency


def main() -> None:
    p = argparse.ArgumentParser(description="Training scaffold (plug-in STGCN/GraphWaveNet here).")
    p.add_argument("--net", required=True, help="Path to net.xml")
    p.add_argument("--sqlite", required=True, help="Path to sqlite db recorded by cfgflow")
    args = p.parse_args()

    net_path = pathlib.Path(args.net)
    sqlite_path = pathlib.Path(args.sqlite)
    if not net_path.exists():
        raise SystemExit(f"net not found: {net_path}")
    if not sqlite_path.exists():
        raise SystemExit(f"sqlite not found: {sqlite_path}")

    edge_ids, arcs = build_edge_adjacency(net_path)
    print(f"Edges: {len(edge_ids)}")
    print(f"Arcs:  {len(arcs)}")
    print("Next: load sequences from sqlite and train your deep spatiotemporal model.")


if __name__ == "__main__":
    main()

