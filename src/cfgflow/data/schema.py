SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS edge_state (
  edge_id TEXT NOT NULL,
  t REAL NOT NULL,
  speed_mps REAL NOT NULL,
  veh_n INTEGER NOT NULL,
  occupancy_pct REAL NOT NULL,
  congestion REAL NOT NULL,
  PRIMARY KEY (edge_id, t)
);
CREATE INDEX IF NOT EXISTS idx_edge_state_t ON edge_state(t);
"""

