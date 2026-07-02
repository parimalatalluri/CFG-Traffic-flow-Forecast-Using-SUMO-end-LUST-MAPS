from __future__ import annotations

import logging
import pathlib
import subprocess
import threading
import time
from dataclasses import dataclass

from cfgflow.data.db import SqliteRecorder
from cfgflow.ml.baseline import BaselineForecaster
from cfgflow.net.net_cache import NetCache
from cfgflow.sim.metrics import compute_congestion
from cfgflow.sim.sumo_env import import_sumo_libs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteResult:
    edges: list[str]
    travel_time_s: float


class SumoController:
    def __init__(
        self,
        *,
        sumocfg_path: pathlib.Path,
        net: NetCache,
        hub,
        recorder: SqliteRecorder | None,
        forecaster: BaselineForecaster,
        sumo_binary: str,
        traci_port: int,
        step_length_s: float,
        publish_every_steps: int,
    ) -> None:
        self.sumocfg_path = sumocfg_path
        self.net = net
        self.hub = hub
        self.recorder = recorder
        self.forecaster = forecaster
        self.sumo_binary = sumo_binary
        self.traci_port = traci_port
        self.step_length_s = step_length_s
        self.publish_every_steps = max(1, int(publish_every_steps))

        self._proc: subprocess.Popen[str] | None = None
        self._connected = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._edge_ids: list[str] = []
        self._tc_speed: int | None = None
        self._tc_veh_n: int | None = None
        self._tc_occ: int | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect_or_start(self) -> tuple[bool, str]:
        try:
            import_sumo_libs()
            import traci  # type: ignore
            from traci import constants as tc  # type: ignore
        except Exception as e:
            return False, str(e)

        if not self.sumocfg_path.exists():
            return False, f"sumocfg not found: {self.sumocfg_path}"

        if self._connected:
            return True, "already connected"

        if self._proc is None or self._proc.poll() is not None:
            cmd = [
                self.sumo_binary,
                "-c",
                str(self.sumocfg_path),
                "--remote-port",
                str(self.traci_port),
                "--step-length",
                str(self.step_length_s),
            ]
            logger.info("Starting SUMO: %s", " ".join(cmd))
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            time.sleep(0.8)

        logger.info("Connecting TraCI on port %d...", self.traci_port)
        try:
            # traci.init sets the default connection used by traci.* module functions
            traci.init(self.traci_port)
        except Exception:
            traci.connect(port=self.traci_port)
        self._connected = True
        self._tc_speed = int(tc.LAST_STEP_MEAN_SPEED)
        self._tc_veh_n = int(tc.LAST_STEP_VEHICLE_NUMBER)
        self._tc_occ = int(tc.LAST_STEP_OCCUPANCY)

        self._edge_ids = [e["id"] for e in self.net.as_xy_json()["edges"]]
        logger.info("Subscribing %d edges for live metrics...", len(self._edge_ids))
        for eid in self._edge_ids:
            try:
                traci.edge.subscribe(
                    eid,
                    (
                        self._tc_speed,
                        self._tc_veh_n,
                        self._tc_occ,
                    ),
                )
            except Exception:
                continue

        if self.recorder:
            self.recorder.ensure_schema()

        # best-effort: prediction step size equals publish cadence
        try:
            self.forecaster.set_dt_s(self.step_length_s * float(self.publish_every_steps))
        except Exception:
            pass

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="sumo-loop", daemon=True)
        self._thread.start()
        return True, "connected"

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._connected:
                import traci  # type: ignore

                traci.close(False)
        except Exception:
            pass
        self._connected = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._proc = None

    def find_route(self, start_edge: str, end_edge: str) -> RouteResult:
        if not self._connected:
            return RouteResult(edges=[], travel_time_s=float("nan"))
        import traci  # type: ignore

        try:
            r = traci.simulation.findRoute(start_edge, end_edge)
            return RouteResult(edges=list(r.edges), travel_time_s=float(r.travelTime))
        except Exception:
            return RouteResult(edges=[], travel_time_s=float("nan"))

    def _loop(self) -> None:
        import traci  # type: ignore

        step = 0
        while not self._stop.is_set():
            try:
                traci.simulationStep()
                step += 1
                if step % self.publish_every_steps != 0:
                    continue

                t = float(traci.simulation.getTime())
                subs = traci.edge.getAllSubscriptionResults()
                edges_out: dict[str, dict] = {}
                db_rows: list[tuple] = []

                tc_speed = self._tc_speed or 64
                tc_veh_n = self._tc_veh_n or 33
                tc_occ = self._tc_occ or 36

                for eid, vals in subs.items():
                    speed = float(vals.get(tc_speed, 0.0))
                    veh_n = int(vals.get(tc_veh_n, 0))
                    occ = float(vals.get(tc_occ, 0.0))
                    ff = self.net.edge_speed_mps(eid)
                    # TraCI uses -1 for "no data" (typically no vehicles); treat as uncongested.
                    if veh_n <= 0 or speed < 0:
                        cong = 0.0
                        speed = float(ff or 0.0)
                        occ = 0.0 if occ < 0 else occ
                    else:
                        cong = compute_congestion(speed_mps=speed, freeflow_speed_mps=ff)

                    # Publish/record only active edges to keep payload and DB size manageable.
                    if veh_n > 0 or cong > 0.01 or occ > 0.01:
                        edges_out[eid] = {"s": speed, "n": veh_n, "o": occ, "c": cong}
                        self.forecaster.update(eid, cong, veh_n=veh_n, occupancy_pct=occ)
                        if self.recorder:
                            db_rows.append((eid, t, speed, veh_n, occ, cong))

                if self.recorder and db_rows:
                    self.recorder.insert_edge_states(db_rows)

                self.hub.publish_threadsafe({"t": t, "n": len(edges_out), "edges": edges_out})
            except Exception as e:
                logger.warning("SUMO loop error: %s", e)
                time.sleep(0.4)
