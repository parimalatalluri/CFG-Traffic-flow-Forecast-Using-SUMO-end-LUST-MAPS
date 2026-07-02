from __future__ import annotations

import os
import pathlib

import pytest


def _has_sumo() -> bool:
    if os.environ.get("SUMO_HOME"):
        return True
    try:
        import sumolib  # noqa: F401
        import traci  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_sumo(), reason="SUMO libs not available (set SUMO_HOME)")
def test_net_cache_loads() -> None:
    from cfgflow.net.net_cache import NetCache

    net_path = pathlib.Path("scenario") / "lust.net.xml"
    if not net_path.exists():
        pytest.skip("LUST net.xml not present in this workspace")

    cache = NetCache(net_path)
    edges = cache.edges()
    assert len(edges) > 1000
    bbox = cache.bbox()
    assert bbox["maxX"] > bbox["minX"]


@pytest.mark.skipif(not _has_sumo(), reason="SUMO libs not available (set SUMO_HOME)")
def test_nearest_edge_id() -> None:
    from cfgflow.net.net_cache import NetCache

    net_path = pathlib.Path("scenario") / "lust.net.xml"
    if not net_path.exists():
        pytest.skip("LUST net.xml not present in this workspace")

    cache = NetCache(net_path)
    b = cache.bbox()
    x = (b["minX"] + b["maxX"]) / 2
    y = (b["minY"] + b["maxY"]) / 2
    eid = cache.nearest_edge_id(x=x, y=y)
    assert eid is None or isinstance(eid, str)

