from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from cfgflow.data.db import SqliteRecorder
from cfgflow.ml.baseline import BaselineForecaster
from cfgflow.net.net_cache import NetCache
from cfgflow.sim.controller import SumoController


@dataclass
class AppState:
    net: NetCache
    sim: SumoController
    hub: "LiveHub"
    recorder: SqliteRecorder | None
    forecaster: BaselineForecaster


class LiveHub:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[asyncio.Queue[str]] = set()

    @property
    def is_attached(self) -> bool:
        return self._loop is not None

    def attach_loop(self) -> None:
        self._loop = asyncio.get_running_loop()

    def register(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=5)
        self._clients.add(q)
        return q

    def unregister(self, q: asyncio.Queue[str]) -> None:
        self._clients.discard(q)

    def publish_threadsafe(self, payload: dict[str, Any]) -> None:
        if self._loop is None:
            return
        message = json.dumps(payload, separators=(",", ":"))
        self._loop.call_soon_threadsafe(self._publish_now, message)

    def _publish_now(self, message: str) -> None:
        dead: list[asyncio.Queue[str]] = []
        for q in self._clients:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                pass
            except Exception:
                dead.append(q)
        for q in dead:
            self._clients.discard(q)
