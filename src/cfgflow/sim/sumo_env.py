from __future__ import annotations

import os
import sys


def import_sumo_libs() -> None:
    # Prefer installed packages, but support SUMO_HOME python tools.
    try:
        import sumolib  # noqa: F401
        import traci  # noqa: F401
        return
    except Exception:
        pass

    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        raise RuntimeError(
            "SUMO python libraries not found. Set SUMO_HOME (recommended) or install 'traci' and 'sumolib'."
        )

    tools = os.path.join(sumo_home, "tools")
    if tools not in sys.path:
        sys.path.append(tools)

