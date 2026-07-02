from __future__ import annotations

import pathlib

from nicegui import ui

from nicegui import app as ng_app

from cfgflow.api.routes import register_api
from cfgflow.data.db import SqliteRecorder
from cfgflow.logging_config import configure_logging
from cfgflow.ml.baseline import BaselineForecaster
from cfgflow.net.net_cache import NetCache
from cfgflow.sim.controller import SumoController
from cfgflow.state import AppState, LiveHub


def _canvas_page() -> None:
    ui.add_head_html(
        """
        <style>
          html, body { height: 100%; margin: 0; }
          #wrap { display:flex; height: calc(100vh - 56px); }
          #left { width: 360px; padding: 10px; overflow:auto; border-right: 1px solid #3a2a1c; background:#16110c; }
          #main { flex:1; position:relative; }
          #netCanvas { width:100%; height:100%; display:block; background:#1a120b; }
          .kv { display:flex; justify-content:space-between; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
          .kv span:first-child { color:#c7b8a4; }
          .kv span:last-child { color:#f2d7b2; }
          .badge { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; padding:2px 6px; border-radius: 6px; background:#2a2017; color:#f2d7b2; border: 1px solid rgba(255,183,77,0.18); }
          button { background:#2a2017; color:#f2d7b2; border:1px solid rgba(255,183,77,0.25); border-radius:8px; padding:6px 10px; cursor:pointer; }
          button:hover { background:#3a2a1c; }
          a { color:#ffb74d; text-decoration:none; }
          a:hover { text-decoration:underline; }
          #tip { position:absolute; pointer-events:none; padding:8px 10px; border-radius:10px; background:rgba(22,17,12,0.96); border:1px solid rgba(255,183,77,0.20); color:#f2d7b2; font-family: ui-monospace, Menlo, Consolas, monospace; font-size:12px; display:none; max-width: 320px; }
        </style>
        """
    )

    ui.add_body_html(
        """
        <div id="wrap">
          <div id="left">
            <div style="display:flex; gap:8px; align-items:center; margin-bottom:10px;">
              <span class="badge">CFGFlow</span>
              <span id="simStatus" class="badge">DISCONNECTED</span>
              <span style="margin-left:auto;"><a href="/help">Help</a></span>
            </div>
            <div style="display:flex; gap:8px; margin-bottom:10px;">
              <button id="btnConnect">Connect</button>
              <button id="btnStop">Stop</button>
            </div>

            <div style="margin: 10px 0; font-weight:600; color:#f2d7b2;">Selection</div>
            <div class="kv"><span>Mode</span><span id="selMode">START</span></div>
            <div class="kv"><span>Start edge</span><span id="startEdge">-</span></div>
            <div class="kv"><span>End edge</span><span id="endEdge">-</span></div>
            <div style="display:flex; gap:8px; margin-top:8px;">
              <button id="btnMode">Toggle Start/End</button>
              <button id="btnRoute">Route</button>
            </div>

            <div style="margin: 16px 0 10px; font-weight:600; color:#f2d7b2;">Live</div>
            <div class="kv"><span>Sim time (s)</span><span id="simTime">-</span></div>
            <div class="kv"><span>Edges updated</span><span id="edgeCount">-</span></div>
            <div class="kv"><span>Last publish</span><span id="lastPub">-</span></div>

            <div style="margin: 16px 0 10px; font-weight:600; color:#f2d7b2;">Route Forecast</div>
            <div class="kv"><span>Hot segments</span><span id="hotSeg">-</span></div>
            <div style="color:#c7b8a4; font-size: 12px; margin-top:6px;">
              Click on the network canvas to select edges. Route forecast is a baseline.
            </div>
          </div>
          <div id="main">
            <canvas id="netCanvas"></canvas>
            <div id="tip"></div>
          </div>
        </div>
        <script>
        (function() {
          const canvas = document.getElementById('netCanvas');
          const ctx = canvas.getContext('2d');
          const tip = document.getElementById('tip');

          let net = null;
          let live = new Map();
          let routeEdges = new Set();
          let routePred = {}; // {horizonSec: {edgeId: congestion}}
          let routePredH = 300; // default for route coloring (5 min)

          let mode = 'START';
          let startEdge = null;
          let endEdge = null;

          const view = { scale: 1, tx: 0, ty: 0, dragging:false, lastX:0, lastY:0 };

          function resize() {
            const rect = canvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            canvas.width = Math.max(1, Math.floor(rect.width * dpr));
            canvas.height = Math.max(1, Math.floor(rect.height * dpr));
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            draw();
          }
          window.addEventListener('resize', resize);

          function worldToScreen(x, y) { return { x: x * view.scale + view.tx, y: -y * view.scale + view.ty }; }
          function screenToWorld(sx, sy) { return { x: (sx - view.tx) / view.scale, y: -(sy - view.ty) / view.scale }; }

          function fit() {
            if (!net) return;
            const w = canvas.getBoundingClientRect().width;
            const h = canvas.getBoundingClientRect().height;
            const bw = net.bbox.maxX - net.bbox.minX;
            const bh = net.bbox.maxY - net.bbox.minY;
            const s = 0.92 * Math.min(w / bw, h / bh);
            view.scale = s;
            view.tx = (w - bw * s) / 2 - net.bbox.minX * s;
            view.ty = (h - bh * s) / 2 + net.bbox.maxY * s;
          }

          function colorForCongestion(c) {
            // orange -> red palette (warm colors only)
            const x = Math.max(0, Math.min(1, c));
            const lo = { r: 255, g: 193, b: 120 }; // light orange
            const hi = { r: 180, g: 30, b: 30 };   // deep red
            const r = Math.round(lo.r + (hi.r - lo.r) * x);
            const g = Math.round(lo.g + (hi.g - lo.g) * x);
            const b = Math.round(lo.b + (hi.b - lo.b) * x);
            return `rgb(${r},${g},${b})`;
          }

          function draw() {
            const w = canvas.getBoundingClientRect().width;
            const h = canvas.getBoundingClientRect().height;
            ctx.clearRect(0, 0, w, h);
            if (!net) {
              ctx.fillStyle = '#c7b8a4';
              ctx.font = '14px ui-monospace, Menlo, Consolas, monospace';
              ctx.fillText('Loading network...', 16, 24);
              return;
            }
            ctx.lineCap = 'round';

            function strokeEdge(e, strokeStyle, lineWidth) {
              ctx.strokeStyle = strokeStyle;
              ctx.lineWidth = lineWidth;
              ctx.beginPath();
              for (let i = 0; i < e.shape.length; i++) {
                const p = e.shape[i];
                const s = worldToScreen(p[0], p[1]);
                if (i === 0) ctx.moveTo(s.x, s.y); else ctx.lineTo(s.x, s.y);
              }
              ctx.stroke();
            }

            for (const e of net.edges) {
              const m = live.get(e.id);
              const c = m ? m.c : 0;
              const isRoute = routeEdges.has(e.id);
              const rp = (isRoute && routePred[routePredH] && routePred[routePredH][e.id] != null) ? routePred[routePredH][e.id] : null;
              if (isRoute) {
                // route highlighter: strong red outline, then forecast color on top
                strokeEdge(e, 'rgb(255,60,60)', 4.2);
                strokeEdge(e, colorForCongestion(rp != null ? rp : c), 2.6);
              } else {
                strokeEdge(e, colorForCongestion(c), 1.0);
              }
            }
          }

          async function loadNet() {
            const res = await fetch('/api/net/xy');
            net = await res.json();
            fit();
            draw();
          }

          function uiSet(id, text) { document.getElementById(id).textContent = text; }

          async function connectSim() {
            const res = await fetch('/api/sim/connect', { method: 'POST' });
            const j = await res.json();
            uiSet('simStatus', j.ok ? 'CONNECTED' : 'ERROR');
          }
          async function stopSim() {
            const res = await fetch('/api/sim/stop', { method: 'POST' });
            const j = await res.json();
            uiSet('simStatus', j.ok ? 'STOPPED' : 'ERROR');
          }

          async function computeRoute() {
            if (!startEdge || !endEdge) return;
            const res = await fetch('/api/route', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ start_edge: startEdge, end_edge: endEdge })
            });
            const j = await res.json();
            routeEdges = new Set(j.edges || []);
            routePred = {};
            const pred = await fetch('/api/predict/route', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ edges: Array.from(routeEdges), horizons_s: [300, 600, 900] })
            }).then(r => r.json());
            uiSet('hotSeg', (pred.hot_segments || []).slice(0, 3).join(', ') || '-');
            routePred = pred.pred || {};
            draw();
          }

          document.getElementById('btnConnect').addEventListener('click', connectSim);
          document.getElementById('btnStop').addEventListener('click', stopSim);
          document.getElementById('btnMode').addEventListener('click', () => {
            mode = (mode === 'START') ? 'END' : 'START';
            uiSet('selMode', mode);
          });
          document.getElementById('btnRoute').addEventListener('click', computeRoute);

          canvas.addEventListener('wheel', (ev) => {
            ev.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const sx = ev.clientX - rect.left;
            const sy = ev.clientY - rect.top;
            const before = screenToWorld(sx, sy);
            const k = ev.deltaY < 0 ? 1.12 : 0.89;
            view.scale = Math.max(0.02, Math.min(40, view.scale * k));
            const after = screenToWorld(sx, sy);
            view.tx += (after.x - before.x) * view.scale;
            view.ty += -(after.y - before.y) * view.scale;
            draw();
          }, { passive:false });

          canvas.addEventListener('mousedown', (ev) => { view.dragging = true; view.lastX = ev.clientX; view.lastY = ev.clientY; });
          window.addEventListener('mouseup', () => { view.dragging = false; });
          window.addEventListener('mousemove', (ev) => {
            if (!view.dragging) return;
            const dx = ev.clientX - view.lastX;
            const dy = ev.clientY - view.lastY;
            view.lastX = ev.clientX;
            view.lastY = ev.clientY;
            view.tx += dx;
            view.ty += dy;
            draw();
          });

          canvas.addEventListener('click', async (ev) => {
            const rect = canvas.getBoundingClientRect();
            const sx = ev.clientX - rect.left;
            const sy = ev.clientY - rect.top;
            const w = screenToWorld(sx, sy);
            const res = await fetch(`/api/net/nearest_edge?x=${encodeURIComponent(w.x)}&y=${encodeURIComponent(w.y)}`);
            const j = await res.json();
            if (!j.edge_id) return;
            if (mode === 'START') { startEdge = j.edge_id; uiSet('startEdge', startEdge); }
            else { endEdge = j.edge_id; uiSet('endEdge', endEdge); }
          });

          function startWS() {
            const scheme = (location.protocol === 'https:') ? 'wss://' : 'ws://';
            const ws = new WebSocket(scheme + location.host + '/ws/live');
            ws.onmessage = (ev) => {
              const msg = JSON.parse(ev.data);
              uiSet('simTime', String(msg.t ?? '-'));
              uiSet('edgeCount', String(msg.n ?? '-'));
              uiSet('lastPub', new Date().toLocaleTimeString());
              if (msg.edges) {
                live.clear();
                for (const [id, m] of Object.entries(msg.edges)) live.set(id, m);
                draw();
              }
            };
            ws.onclose = () => setTimeout(startWS, 1500);
          }

          function levelText(c) {
            if (c < 0.20) return 'LOW';
            if (c < 0.40) return 'MILD';
            if (c < 0.70) return 'MODERATE';
            return 'HEAVY';
          }

          function segDist2(px, py, ax, ay, bx, by) {
            const abx = bx - ax, aby = by - ay;
            const apx = px - ax, apy = py - ay;
            const denom = abx*abx + aby*aby;
            let t = 0;
            if (denom > 1e-12) t = (apx*abx + apy*aby) / denom;
            if (t < 0) t = 0; else if (t > 1) t = 1;
            const cx = ax + t*abx, cy = ay + t*aby;
            const dx = px - cx, dy = py - cy;
            return dx*dx + dy*dy;
          }

          function polyMinDist2Screen(sx, sy, shape) {
            let best = 1e30;
            for (let i = 0; i < shape.length - 1; i++) {
              const a = worldToScreen(shape[i][0], shape[i][1]);
              const b = worldToScreen(shape[i+1][0], shape[i+1][1]);
              best = Math.min(best, segDist2(sx, sy, a.x, a.y, b.x, b.y));
            }
            return best;
          }

          canvas.addEventListener('mousemove', (ev) => {
            if (!net || routeEdges.size === 0) { tip.style.display = 'none'; return; }
            const rect = canvas.getBoundingClientRect();
            const sx = ev.clientX - rect.left;
            const sy = ev.clientY - rect.top;

            // only scan route edges (fast)
            let bestId = null;
            let bestD2 = 14*14;
            for (const e of net.edges) {
              if (!routeEdges.has(e.id)) continue;
              const d2 = polyMinDist2Screen(sx, sy, e.shape);
              if (d2 < bestD2) { bestD2 = d2; bestId = e.id; }
            }

            if (!bestId) { tip.style.display = 'none'; return; }
            const p5 = (routePred[300] && routePred[300][bestId] != null) ? routePred[300][bestId] : null;
            const p10 = (routePred[600] && routePred[600][bestId] != null) ? routePred[600][bestId] : null;
            const p15 = (routePred[900] && routePred[900][bestId] != null) ? routePred[900][bestId] : null;

            const lines = [];
            lines.push(`<div><b>${bestId}</b></div>`);
            if (p5 != null) lines.push(`<div>+5 min: ${levelText(p5)} traffic may occur (${p5.toFixed(2)})</div>`);
            if (p10 != null) lines.push(`<div>+10 min: ${levelText(p10)} (${p10.toFixed(2)})</div>`);
            if (p15 != null) lines.push(`<div>+15 min: ${levelText(p15)} (${p15.toFixed(2)})</div>`);
            if (p5 == null && p10 == null && p15 == null) lines.push(`<div>No forecast (route not predicted yet)</div>`);

            tip.innerHTML = lines.join('');
            tip.style.left = Math.min(rect.width - 10, sx + 14) + 'px';
            tip.style.top = Math.min(rect.height - 10, sy + 14) + 'px';
            tip.style.display = 'block';
          });

          loadNet().then(() => { resize(); startWS(); });
        })();
        </script>
        """
    )


def run_app(
    *,
    sumocfg: str,
    net_path: str,
    sumo_binary: str,
    traci_port: int,
    step_length_s: float,
    publish_every_steps: int,
    sqlite_path: str,
    host: str,
    ui_port: int,
    native: bool,
    model_path: str,
) -> None:
    configure_logging()

    net = NetCache(pathlib.Path(net_path))
    hub = LiveHub()
    recorder = SqliteRecorder(pathlib.Path(sqlite_path)) if sqlite_path else None

    forecaster: BaselineForecaster
    if model_path:
        try:
            from cfgflow.ml.torch_forecaster import TorchForecaster

            tf = TorchForecaster()
            tf.load(pathlib.Path(model_path))
            forecaster = tf
        except Exception:
            forecaster = BaselineForecaster()
    else:
        forecaster = BaselineForecaster()

    sim = SumoController(
        sumocfg_path=pathlib.Path(sumocfg),
        net=net,
        hub=hub,
        recorder=recorder,
        forecaster=forecaster,
        sumo_binary=sumo_binary,
        traci_port=traci_port,
        step_length_s=step_length_s,
        publish_every_steps=publish_every_steps,
    )

    state = AppState(net=net, sim=sim, hub=hub, recorder=recorder, forecaster=forecaster)
    register_api(ng=ng_app, state=state)

    def _header() -> None:
        with ui.header().classes("items-center justify-between").style("background:#1a120b;"):
            with ui.row().classes("items-center"):
                ui.label("CFGFlow (LUST + SUMO)").style("color:#f2d7b2; font-weight:600;")
                ui.link("Live", "/").style("color:#ffb74d;").classes("q-ml-md")
                ui.link("Help", "/help").style("color:#ffb74d;").classes("q-ml-md")
                ui.link("Training", "/training").style("color:#ffb74d;").classes("q-ml-md")
            ui.label("Local").style("color:#f2d7b2;")

    @ui.page("/")
    def _index() -> None:
        _header()
        _canvas_page()

    @ui.page("/help")
    def _help() -> None:
        _header()
        ui.markdown(
            """
## What to do

1. Open **Live** page.
2. Click **Connect** (starts SUMO + attaches via TraCI).
3. Wait until **Sim time (s)** increases.
4. Click the network canvas to pick **START** and **END** edges (use **Toggle Start/End**).
5. Click **Route** to overlay the path and see a baseline forecast.

## Record data (for ML training)

Run with SQLite enabled:

```bash
python -m cfgflow run --sumocfg scenario\\due.actuated.sumocfg --net scenario\\lust.net.xml --sqlite data\\lust.sqlite
```

Export to CSV:

```bash
python -m cfgflow export --sqlite data\\lust.sqlite --out data\\edge_state.csv
```

## Common notes

- If you only see the top buttons, **scroll** inside the left panel (it has its own scrollbar).
- The map is drawn from `scenario\\lust.net.xml` and updated via `/ws/live`.
"""
        ).classes("max-w-3xl q-pa-md")

    @ui.page("/training")
    def _training() -> None:
        _header()
        ui.markdown(
            """
## Training (local, offline)

1) Record a dataset first (SQLite):

```bash
python -m cfgflow run --sumocfg scenario\\due.actuated.sumocfg --net scenario\\lust.net.xml --sqlite data\\lust.sqlite
```

2) Install ML extras (PyTorch + NumPy). If `pip install -e .[ml]` is slow on Windows, prefer conda PyTorch.

3) Train:

```bash
python -m cfgflow train --net scenario\\lust.net.xml --sqlite data\\lust.sqlite --out models\\st_model.pt --max-edges 1200 --epochs 10
```

4) Run the app using the trained model:

```bash
python -m cfgflow run --sumocfg scenario\\due.actuated.sumocfg --net scenario\\lust.net.xml --model models\\st_model.pt
```

This is a small **graph + temporal CNN** baseline you can extend into your thesis model (CFG-aware conditioning).
"""
        ).classes("max-w-3xl q-pa-md")

    ui.run(
        host=host,
        port=ui_port,
        native=native,
        title="CFGFlow (LUST + SUMO)",
        reload=False,
    )
