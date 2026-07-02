from __future__ import annotations

import pathlib
import shutil

from cfgflow.sim.sumo_env import import_sumo_libs


def run_doctor(*, sumocfg: str, net: str, sumo_binary: str) -> None:
    sumocfg_path = pathlib.Path(sumocfg)
    net_path = pathlib.Path(net)

    print("CFGFlow doctor")
    print(f"- sumocfg exists: {sumocfg_path.exists()} ({sumocfg_path})")
    print(f"- net exists:    {net_path.exists()} ({net_path})")
    print(f"- sumo binary:   {sumo_binary} -> {shutil.which(sumo_binary) or 'NOT FOUND IN PATH'}")

    try:
        import_sumo_libs()
        import sumolib  # noqa: F401
        import traci  # noqa: F401

        print("- SUMO python libs: OK (sumolib + traci importable)")
    except Exception as e:
        print(f"- SUMO python libs: ERROR ({e})")

