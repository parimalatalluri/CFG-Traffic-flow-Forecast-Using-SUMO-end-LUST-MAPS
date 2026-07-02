from __future__ import annotations

import pathlib
import sqlite3
from typing import Iterable

from cfgflow.data.schema import SCHEMA_SQL


class SqliteRecorder:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        # SQLite can create the file, but not missing parent directories.
        parent = self.path.parent
        if str(parent) and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.path), timeout=30.0)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def ensure_schema(self) -> None:
        con = self._connect()
        try:
            con.executescript(SCHEMA_SQL)
            con.commit()
        finally:
            con.close()

    def insert_edge_states(self, rows: Iterable[tuple]) -> None:
        con = self._connect()
        try:
            con.executemany(
                "INSERT OR REPLACE INTO edge_state(edge_id,t,speed_mps,veh_n,occupancy_pct,congestion) VALUES(?,?,?,?,?,?)",
                rows,
            )
            con.commit()
        finally:
            con.close()
