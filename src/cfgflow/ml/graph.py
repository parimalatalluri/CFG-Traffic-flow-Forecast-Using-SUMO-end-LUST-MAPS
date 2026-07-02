from __future__ import annotations

import logging
import pathlib

from cfgflow.sim.sumo_env import import_sumo_libs

logger = logging.getLogger(__name__)


def build_edge_adjacency(net_path: pathlib.Path) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Build a simple directed edge-to-edge adjacency (line-graph style):
    (e_i -> e_j) if the 'to' node of e_i is the 'from' node of e_j.

    Returns:
      edge_ids: stable list of edge ids (no internal edges)
      arcs: list of (i, j) indices into edge_ids
    """
    import_sumo_libs()
    import sumolib  # type: ignore

    net = sumolib.net.readNet(str(net_path))
    edges = [e for e in net.getEdges() if not e.getID().startswith(":")]
    edge_ids = [e.getID() for e in edges]
    idx = {eid: i for i, eid in enumerate(edge_ids)}

    arcs: list[tuple[int, int]] = []
    for e in edges:
        to_node = e.getToNode()
        if to_node is None:
            continue
        for out_e in to_node.getOutgoing():
            oid = out_e.getID()
            if oid.startswith(":"):
                continue
            arcs.append((idx[e.getID()], idx[oid]))

    logger.info("Adjacency: %d edges, %d arcs", len(edge_ids), len(arcs))
    return edge_ids, arcs

