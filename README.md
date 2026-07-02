# CFG-Aware Traffic Flow Forecasting (LUST + SUMO) - Local Windows App

This is a fully local Python application that:

- runs the LUST SUMO scenario via TraCI (live connection)
- shows live congestion on the map
- lets the user choose Start -> Destination on the map and computes a route
- forecasts future congestion along the route (+5 / +10 / +15 minutes)

## Prerequisites

1) SUMO installed (`sumo-gui` or `sumo`)
2) Recommended: set `SUMO_HOME` (so Python can import SUMO tools), for example:
   - `C:\Program Files (x86)\Eclipse\Sumo`
3) Python available in your Anaconda environment

Your scenario files are expected to exist here:

- `scenario\due.actuated.sumocfg`
- `scenario\lust.net.xml`

## One-time install (CMD + Anaconda)

Open CMD:

```cmd
cd /d D:\Sumo_kits\simulation
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Optional check:

```cmd
python -m cfgflow doctor --sumocfg scenario\due.actuated.sumocfg --net scenario\lust.net.xml --sumo-binary sumo-gui
```

## Demo (spoon-fed, step-by-step)

### Part A - Run the app (live)

1) Start the app:

```cmd
cd /d D:\Sumo_kits\simulation
python -m cfgflow run --sumocfg scenario\due.actuated.sumocfg --net scenario\lust.net.xml
```

2) Open the UI:

```cmd
start http://127.0.0.1:8088
```

3) In the UI:

- Click `Connect` (SUMO starts and TraCI connects)
- Wait until `Sim time (s)` increases and roads change color (live congestion)

### Part B - Route + forecast (in the UI)

1) Make sure left panel shows `Mode = START`
2) Click a road on the canvas -> `Start edge` becomes a value
3) Click `Toggle Start/End` (now `Mode = END`)
4) Click another road -> `End edge` becomes a value
5) Click `Route`
6) Hover any part of the highlighted route to see forecast text for:
   - +5 minutes
   - +10 minutes
   - +15 minutes

### Part C - Collect data (SQLite recording)

1) Stop the app (close the CMD window or press `Ctrl+C`)
2) Run with SQLite enabled:

```cmd
cd /d D:\Sumo_kits\simulation
python -m cfgflow run --sumocfg scenario\due.actuated.sumocfg --net scenario\lust.net.xml --sqlite data\lust.sqlite
```

3) Open UI again:

```cmd
start http://127.0.0.1:8088
```

4) In the UI:

- Click `Connect`
- Let it run for 5-20 minutes (more = better training data)
- Click `Stop` (or close the app)

5) Confirm DB file exists:

```cmd
dir data\lust.sqlite
```

### Part D - Export CSV from the DB

```cmd
cd /d D:\Sumo_kits\simulation
python -m cfgflow export --sqlite data\lust.sqlite --out data\edge_state.csv
dir data\edge_state.csv
```

### Part E - Train a model (optional, requires PyTorch)

Install PyTorch (choose ONE):

Option 1 (recommended, conda CPU):

```cmd
conda install pytorch cpuonly -c pytorch
```

Option 2 (pip extras):

```cmd
cd /d D:\Sumo_kits\simulation
python -m pip install -e ".[ml]"
```

Train:

```cmd
cd /d D:\Sumo_kits\simulation
python -m cfgflow train --net scenario\lust.net.xml --sqlite data\lust.sqlite --out models\st_model.pt --max-edges 1200 --epochs 10 --batch 32 --device cpu
dir models\st_model.pt
```

### Part F - Run the app using the trained model

```cmd
cd /d D:\Sumo_kits\simulation
python -m cfgflow run --sumocfg scenario\due.actuated.sumocfg --net scenario\lust.net.xml --model models\st_model.pt
start http://127.0.0.1:8088
```

Then use the UI again:

`Connect` -> pick Start/End -> `Route` -> hover to see forecasts.

## Useful variants

Run in a desktop window (native):

```cmd
python -m cfgflow run --sumocfg scenario\due.actuated.sumocfg --net scenario\lust.net.xml --native
```

Run headless SUMO:

```cmd
python -m cfgflow run --sumocfg scenario\due.actuated.sumocfg --net scenario\lust.net.xml --sumo-binary sumo
```

## Troubleshooting

- If the UI looks old after updates: hard refresh `Ctrl+F5`.
- If `Connect` fails: verify SUMO binary exists (`sumo-gui --version`) and `SUMO_HOME` is set.
- If SQLite recording fails: ensure you can write to `D:\Sumo_kits\simulation\data\` (the app auto-creates it).

