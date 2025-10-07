# Super-prompt for Amazon Q — **Autonomous Convoy Protection System (Hackathon MVP)**

Use this prompt to instruct Amazon Q to design and *generate code + assets* for a hackathon-ready Autonomous Convoy Protection System (ACPS) prototype.  Requirements: **do not integrate AWS now** (design for adding AWS later), deliver a working local/cloud-simulated stack, and produce a black-and-yellow military-style dashboard. Produce a full repo scaffold, runnable simulation, and a clear demo script.

**Primary goals for the agent**

1. Build a local end-to-end prototype: convoy simulator + tethered drone network + edge perception (YOLO) + path planner + cloud fusion service + React dashboard.
2. Dashboard: map-based UI where the operator can draw/select Areas of Interest (AOIs) for drones to inspect, view live detections, threat heatmap, convoy and drone telemetry, and accept/reject suggested reroutes. Minimal color palette: black background, yellow accent, white text.
3. Simulation: runnable locally (Linux/macOS/Windows) using Microsoft AirSim (preferred) *or* a lightweight simulator fallback (Python-based simulation + simple physics) if AirSim cannot be installed. Must include scripts to spawn a convoy and tethered drones that follow convoy vehicles.
4. Edge/cloud workload separation: clearly mark which components run on edge (per-drone inference, local control), and which run on cloud/central (global fusion, mission planning, dashboard). Provide interfaces for later AWS integration.

---

## Deliverables (explicit)

Amazon Q must output a single repo (zip or Git) with these items:

1. `README.md` — setup, run, and demo script (one-liners to run full demo).
2. `backend/` — Python FastAPI service that:

   * Simulates convoy movement (or connects to AirSim)
   * Hosts a mission planner / fusion service
   * Exposes WebSocket + REST APIs for the dashboard
   * Offers a message broker abstraction (MQTT or local Redis/pubsub) for edge-cloud comms
3. `edge/` — per-drone edge agent (Python) that:

   * Connects to AirSim or simulator camera feed
   * Runs YOLO inference on frames (use Ultralytics YOLOv8/YOLOv5 PyTorch)
   * Publishes compact detection messages (JSON) to the broker
   * Accepts local commands (hold, inspect AOI, adjust position)
   * Contains a fallback dummy agent for machines without GPU
4. `planner/` — mission planner module that:

   * Ingests detections, builds threat heatmap and per-cell cost
   * Runs a path optimization (A*/Dijkstra with dynamic cost map or simple multi-agent RL stub)
   * Emits recommended re-route/formation actions
5. `frontend/` — React + Vite dashboard:

   * Black background, yellow accents (CSS variables)
   * Map (Leaflet or Mapbox) showing convoy path, drones, AOIs, threat heatmap overlay
   * AOI drawing tool (polygon/rect/circle)
   * Live feed thumbnails for each drone (compressed frames)
   * Controls for sim start/stop, manual override, and replay
   * Minimal, accessible UI — no flashy gradients, two-tone look
6. `sim_configs/` — sample scenario configs: convoy size, drone tethers, threat spawn points, weather parameters (wind, rain), latency/packet loss emulation.
7. `models/` — instructions and small example model or a script that downloads a small pretrained YOLO model (COCO) and a sample finetune script (no huge weights checked into repo).
8. `docker-compose.yml` — spins backend, simple broker (Mosquitto or Redis), dashboard (dev), and a simulated edge agent (or multiple).
9. `tests/` — unit test stubs and an integration test that runs a 3-vehicle convoy + 3 drones scenario and asserts detections are delivered and planner issues a reroute.
10. `slides/one_pager.pdf` — one-page architecture diagram + dataflow (edge↔cloud) and a short bullet list of what to present in hackathon.

---

## Implementation constraints & tech stack (explicit)

* **Frontend**: React (Vite), TypeScript optional, Tailwind CSS with custom theme (black `#0b0b0b`, yellow accent `#FFD400` or `#FFDD00`), Leaflet (open) for map or Mapbox (optional, require token).
* **Backend**: Python 3.10+, FastAPI, Uvicorn, asyncio WebSockets.
* **Edge agent**: Python, OpenCV, PyTorch, Ultralytics YOLO (v8/v5). Provide CPU fallback (smaller model).
* **Messaging**: MQTT (Eclipse Mosquitto) for realistic edge-cloud pub/sub; fallback Redis pubsub.
* **Simulator**: Prefer **AirSim** (include AirSim API usage + instructions). If AirSim not available, include a `simulator/simple_sim.py` (Python) that simulates vehicles and tethered drones on a 2D plane and returns synthetic camera frames (or static images with synthetic objects) for YOLO testing.
* **Path planner**: Grid-based costmap with A*; threat cells increase cost. Implement multi-drone assignment heuristic (greedy or Hungarian for AOI allocation).
* **Security**: Local dev only — use simple token auth for WebSocket. Leave TODOs for real device identity and TLS (to add when integrating AWS).
* **No external/proprietary APIs** required; use public datasets and pretrained COCO for YOLO.

---

## Exact APIs & message schemas (copy-paste ready)

### WebSocket channel: `/ws/telemetry`

Edge → Backend (detection)

```json
{
  "type": "detection",
  "drone_id": "drone-01",
  "vehicle_id": "veh-02",
  "timestamp": "2025-10-07T15:04:05Z",
  "bbox": [x1, y1, x2, y2],        // image coords
  "class": "person",
  "confidence": 0.87,
  "geo": {"lat": 42.123, "lon": -71.234}, // coarse
  "thumbnail_s3": null,             // optional pointer; local dev: base64 optional
  "priority": 7
}
```

### Planner → Edge (command)

```json
{
  "type": "command",
  "drone_id": "drone-01",
  "command": "inspect_aoi",
  "aoi_id": "aoi-14",
  "params": {"altitude_m": 30, "loiter_time_s": 12}
}
```

### Dashboard REST endpoints

* `GET /api/convoy` — convoy status
* `POST /api/aoi` — create AOI `{type, geometry, priority}`
* `POST /api/mission/start` — start scenario
* `GET /api/heatmap` — threat heatmap tiles (geojson or raster)

---

## Edge / Cloud workload split (must be included verbatim)

**Edge (per drone / vehicle):**

* Capture camera frames, preprocess (resize, compress)
* Run YOLO inference, produce detection messages (bbox, class, confidence)
* Run local safety rules (immediate evasive action, hold position)
* Buffer evidence and low-priority telemetry when offline
* Accept and apply planner commands (inspect AOI, loiter, return-to-tether)

**Cloud (backend/planner/dashboard):**

* Ingest prioritized telemetry from all drones
* Fuse multi-drone detections into a global threat map
* Compute path/formation recommendations (global optimization)
* Host operator dashboard and allow manual overrides
* Store telemetry and lightweight video evidence for replay/training

(Include TODO notes for later AWS service points: IoT Core for device identity, Greengrass for edge orchestration, SageMaker for model retraining)

---

## YOLO specifics (exact)

* Use Ultralytics YOLOv8 (or YOLOv5 if preferred). Provide code to:

  * Load a pretrained model (`yolov8n.pt`) for CPU dev
  * Run inference on a frame and output JSON per detection
  * Postprocess to map pixel bbox → geo-coordinates using simple pinhole camera model and known drone altitude & orientation (approximation acceptable for MVP)
* Provide tiny script `edge/benchmark_inference.py` to measure per-frame latency and memory.

---

## Path-planning & threat scoring (explicit algorithm)

1. Represent the terrain as a 2D grid centered on convoy; each cell has base cost (distance, terrain) and dynamic cost (threat score).
2. Threat fusion: each detection projects to grid cell(s); add weighted score by confidence/class.
3. Planner runs A* for candidate routes; cost = distance + α * threat_score + β * weather_penalty.
4. Drone assignment: for each AOI, compute nearest available drone considering tether length. Prefer tethered assignment: drones have max tether radius; cannot be assigned beyond that without vehicle repositioning.
5. Output recommended convoy maneuver: change lane, slow down, detour, or flag human review.

---

## Simulation scenarios to include (explicit)

* **Scenario A — Ambush test**: convoy of 3 vehicles, 3 tethered drones. Randomized “person-with-gun” spawn near road shoulder. Weather: clear. Expectation: drone detects and planner issues formation change to staggered formation.
* **Scenario B — Weather + Visibility**: heavy rain reduces camera detection probability; show how planner increases drone density and recommends slower speed.
* **Scenario C — Intermittent connectivity**: simulate packet loss 40% between drone and cloud; show local decisions still executed and cloud fusion reconciles later.

Each scenario should have a JSON config and script: `scripts/run_scenario.py --scenario ambush`.

---

## Dashboard style & UX (exact)

* Base CSS variables:

  ```css
  --bg: #0b0b0b; /* black */
  --accent: #FFD400; /* yellow */
  --muted: #bfbfbf; /* greys */
  ```
* Layout: left column (controls + AOI list + scenario controls), center map (majority), right column (drone cards + live thumbnails + timeline). Top bar with status (convoy health, connectivity, current scenario).
* AOI drawing: one-click draw mode, supports polygon/rect/circle, AOI appears on map with dashed yellow outline.
* Minimal color usage: use shades of grey for noncritical UI; yellow only for highlights, active borders, and threat markers. Threat heatmap uses semi-transparent yellow→dark red but keep saturation low.
* Provide accessible keyboard shortcuts: `D` = toggle draw AOI, `S` = start/pause sim, `L` = toggle log.

---

## Testing & validation plan (explicit)

* Unit tests for message serialization/deserialization
* Integration test: `tests/integration/test_end_to_end.py` runs docker-compose, boots backend + 3 edge agents (simulator) and asserts:

  * At least one detection message reaches planner within 5s
  * Planner emits a command within 10s of first confirmed detection
* Provide simple metrics logging (detections/sec, planner latency, percent messages dropped) exposed via `/metrics` (Prometheus simple text output).

---

## Demo script (must be included in README)

1. `docker-compose up --build` (starts backend + broker + one simulated edge)
2. In another terminal: `python scripts/spawn_convoy.py --vehicles 3 --drones 3`
3. Open `http://localhost:3000` — draw an AOI on map (left panel) and press `Start Scenario`.
4. Observe drone thumbnails, detection popups, and planner suggested reroute. Click `Accept` to apply.
5. Toggle `Network Conditions` to simulate packet loss and show local-only behavior.

---

## Scaffolding & project structure (exact)

```
acps-superprompt/
├─ README.md
├─ docker-compose.yml
├─ backend/
│  ├─ app.py (FastAPI)
│  ├─ planner/
│  └─ sim/
├─ edge/
│  ├─ agent.py
│  ├─ camera_adapter.py
│  └─ yolo_infer.py
├─ frontend/
│  ├─ src/
│  └─ tailwind.config.js
├─ planner/
├─ sim_configs/
├─ scripts/
├─ tests/
└─ slides/
```

---

## Extra requirements for Amazon Q (meta)

* Produce complete, runnable code (no placeholders) for the simple simulation and dashboard with mocked camera frames so reviewers can run the demo without GPU or AirSim.
* Where heavy dependencies are required (AirSim, pretrained large models), provide commented alternatives and a `USE_MINIMAL=true` dev mode that uses CPU-friendly tiny models and synthetic frames. Provide clear `pip`/`conda` env file or `requirements.txt`.
* Add TODO markers where AWS integration should be added later, and list exactly which AWS services would hook into each component (IoT Core for device identity, Greengrass for edge orchestration, S3 for evidence, SageMaker for model retraining, Kinesis for telemetry).
* Generate `one_pager.pdf` architecture diagram and include sequence diagram images (PNG or SVG).

---

## Tone & documentation style

* Keep code comments concise and practical.
* README should include a short “pitch” paragraph for judges and a bullet list of what to demo (3 minutes max).
* Keep UI copy terse and militaristic: “CONVOY STATUS”, “THREAT MAP”, “AOI: Inspect”, “ENGAGE (HUMAN)”.

---

## Final output format (how Amazon Q should return results)

1. Provide a link to a downloadable zip of the full repo (or the repo file contents).
2. Show terminal commands to bootstrap and run the full demo.
3. Attach `README.md` content in the chat, plus the `one_pager.pdf` and at least 3 screenshots of the dashboard UI mockup (desktop viewport) — if image generation not possible, produce the UI mockup as HTML/CSS preview files in `frontend/mockup/`.
4. List exact next steps to integrate AWS (3 bullet plan with which services to add first).

---

If anything about hardware assumptions is required, assume the hackathon machines are standard laptops (no GPU). Build a `dev` mode that runs entirely on CPU with tiny detection models and synthetic frames — but still architect everything so swapping to real drones or AirSim is straightforward.

---

Okay Amazon Q: deliver the repo described above with runnable demo, documentation, and assets. Keep the design modular, well-tested, and ready for a subsequent AWS integration phase.
