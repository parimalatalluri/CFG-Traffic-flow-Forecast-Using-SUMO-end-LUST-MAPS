from __future__ import annotations

import logging
from typing import Any

from fastapi import WebSocket
from pydantic import BaseModel, Field

from cfgflow.state import AppState

logger = logging.getLogger(__name__)


class RouteRequest(BaseModel):
    start_edge: str = Field(..., min_length=1)
    end_edge: str = Field(..., min_length=1)


class PredictRouteRequest(BaseModel):
    edges: list[str]
    horizons_s: list[int] = Field(default_factory=lambda: [300, 600, 900])


def register_api(*, ng: Any, state: AppState) -> None:
    # NiceGUI's `app` is already a FastAPI-like app object in recent versions.
    api = ng

    @api.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "connected": state.sim.is_connected}

    @api.get("/api/net/xy")
    def net_xy() -> dict[str, Any]:
        return state.net.as_xy_json()

    @api.get("/api/net/nearest_edge")
    def nearest_edge(x: float, y: float) -> dict[str, Any]:
        edge_id = state.net.nearest_edge_id(x=x, y=y)
        return {"edge_id": edge_id}

    @api.post("/api/sim/connect")
    def sim_connect() -> dict[str, Any]:
        ok, msg = state.sim.connect_or_start()
        return {"ok": ok, "message": msg}

    @api.post("/api/sim/stop")
    def sim_stop() -> dict[str, Any]:
        state.sim.stop()
        return {"ok": True}

    @api.post("/api/route")
    def route(req: RouteRequest) -> dict[str, Any]:
        r = state.sim.find_route(req.start_edge, req.end_edge)
        return {"edges": r.edges, "travel_time_s": r.travel_time_s}

    @api.post("/api/predict/route")
    def predict_route(req: PredictRouteRequest) -> dict[str, Any]:
        pred = state.forecaster.predict(edge_ids=req.edges, horizons_s=req.horizons_s)
        # hot segments by worst predicted at the furthest horizon
        far_h = max(req.horizons_s) if req.horizons_s else 0
        scores = pred.get(far_h, {})
        hot = [e for e, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)][:10]
        return {"hot_segments": hot, "horizons_s": req.horizons_s, "pred": pred}

    @api.websocket("/ws/live")
    async def ws_live(ws: WebSocket) -> None:
        await ws.accept()
        if not state.hub.is_attached:
            state.hub.attach_loop()
        q = state.hub.register()
        try:
            while True:
                message = await q.get()
                await ws.send_text(message)
        except Exception:
            pass
        finally:
            state.hub.unregister(q)
