# ShadowGrid — Implementation Audit Report

> **Audited**: 2026-04-28 | **Scope**: Full codebase vs. [Implementation Plan](file:///d:/Projects/ShadowGrid/Implementation_plan.md)
> **Auditor**: Automated source-level audit
> **Project root**: `d:\Projects\ShadowGrid`

---

## Executive Summary

The ShadowGrid implementation plan describes a **person-tracking** surveillance system using YOLOv8 + DeepSORT + CLIP ReID, with a Socket.io WebSocket backend, an indoor floor-plan canvas dashboard, and features like uncertainty radius, camera-graph prediction, drone dispatch, and text search.

The actual codebase is a **vehicle-tracking** crime detection system inherited from the "ZeroSpot" project. It uses YOLOv8 for crime detection, ByteTrack for vehicle tracking, OSNet for Re-ID (not CLIP), a REST-only FastAPI backend (no WebSocket/Socket.io), and a polished React + Leaflet map dashboard. It includes additional subsystems not in the plan — a MIL anomaly detector, a GeoVision C++ engine integration, and geo-prioritised camera scheduling.

**The plan and the implementation have significant architectural divergences.** Many plan features are absent, while the implementation has several production-quality features not described in the plan at all.

---

## Structural Comparison

### Planned File Structure vs. Actual

| Planned Directory | Actual Equivalent | Status |
|---|---|---|
| `detection/detector.py` | [crime_detector.py](file:///d:/Projects/ShadowGrid/CrimeDetector/models/crime_detector.py) | ⚠️ Modified — detects crime, not just persons |
| `tracking/camera_tracker.py` | [car_tracker.py](file:///d:/Projects/ShadowGrid/CrimeDetector/models/car_tracker.py) | ⚠️ Modified — uses ByteTrack, not DeepSORT |
| `tracking/global_registry.py` | [state.py](file:///d:/Projects/ShadowGrid/CrimeDetector/core/state.py) + [suspect_matcher.py](file:///d:/Projects/ShadowGrid/CrimeDetector/models/suspect_matcher.py) | ⚠️ Partial — no GlobalTrack dataclass with trajectory/velocity |
| `reid/embedder.py` (CLIP) | [model_manager.py](file:///d:/Projects/ShadowGrid/CrimeDetector/models/model_manager.py) (OSNet) | ❌ Different model — no CLIP, no text embedding |
| `ingestion/stream_manager.py` | [frame_ingestor.py](file:///d:/Projects/ShadowGrid/CrimeDetector/core/frame_ingestor.py) | ✅ Equivalent — per-camera threading, auto-reconnect |
| `prediction/uncertainty.py` | ❌ Not implemented | ❌ Missing |
| `prediction/graph_predictor.py` | ❌ Not implemented | ❌ Missing |
| `backend/main.py` (FastAPI+Socket.io) | [main.py](file:///d:/Projects/ShadowGrid/CrimeDetector/main.py) (FastAPI only) | ⚠️ Partial — no Socket.io/WebSocket |
| `backend/orchestrator.py` | [consumer.py](file:///d:/Projects/ShadowGrid/CrimeDetector/core/consumer.py) | ⚠️ Modified — priority-based round-robin, not async gather |
| `frontend/src/useSocket.js` | [AppContext.jsx](file:///d:/Projects/ShadowGrid/UI/src/store/AppContext.jsx) | ⚠️ HTTP polling instead of WebSocket |
| `frontend/src/components/RoomMap.jsx` | [MapView.jsx](file:///d:/Projects/ShadowGrid/UI/src/components/map/MapView.jsx) | ⚠️ Different — Leaflet geo-map, not canvas floor plan |
| `frontend/src/components/PredictionPanel.jsx` | ❌ Not implemented | ❌ Missing |
| `frontend/src/components/UncertaintyPanel.jsx` | ❌ Not implemented | ❌ Missing |
| `frontend/src/components/TextSearch.jsx` | ❌ Not implemented | ❌ Missing |
| `frontend/src/components/ReacquiredBanner.jsx` | ❌ Not implemented | ❌ Missing |
| `.env` | ❌ Does not exist | ❌ Missing |
| `start.sh` | ❌ Does not exist | ❌ Missing |
| `videos/` (test footage) | ❌ Does not exist | ❌ Missing |

### Files Present but NOT in Plan

| File | Description |
|---|---|
| [core/priority_manager.py](file:///d:/Projects/ShadowGrid/CrimeDetector/core/priority_manager.py) | Geo-aware camera priority scoring with decay |
| [core/camera_loader.py](file:///d:/Projects/ShadowGrid/CrimeDetector/core/camera_loader.py) | JSON-based camera registry |
| [core/tracking_module.py](file:///d:/Projects/ShadowGrid/CrimeDetector/core/tracking_module.py) | Cross-camera suspect tracking threads |
| [models/suspect_matcher.py](file:///d:/Projects/ShadowGrid/CrimeDetector/models/suspect_matcher.py) | Cosine similarity matching engine |
| [training/](file:///d:/Projects/ShadowGrid/CrimeDetector/training) (6 files) | MIL anomaly detection pipeline (train, eval, inference) |
| [routers/cameras.py](file:///d:/Projects/ShadowGrid/CrimeDetector/routers/cameras.py) | Camera CRUD + MJPEG streaming endpoints |
| [routers/suspects.py](file:///d:/Projects/ShadowGrid/CrimeDetector/routers/suspects.py) | Suspect management REST endpoints |
| [routers/occurrences.py](file:///d:/Projects/ShadowGrid/CrimeDetector/routers/occurrences.py) | Occurrence query endpoints |
| [UI/src/api/geovision.js](file:///d:/Projects/ShadowGrid/UI/src/api/geovision.js) | GeoVision C++ engine API client |
| [UI/src/components/panels/IntelligencePanel.jsx](file:///d:/Projects/ShadowGrid/UI/src/components/panels/IntelligencePanel.jsx) | Intelligence analysis dashboard |
| [UI/src/components/panels/EmbeddingChart.jsx](file:///d:/Projects/ShadowGrid/UI/src/components/panels/EmbeddingChart.jsx) | Embedding similarity visualization |
| [CrimeDetector/ui/](file:///d:/Projects/ShadowGrid/CrimeDetector/ui) | Legacy vanilla JS dashboard |

---

## Sprint-by-Sprint Audit

### SPRINT 1 — Detection & Tracking (Days 1–3)

#### Day 1 — YOLOv8 Detection

| Exit Condition | Status | Notes |
|---|---|---|
| Detects persons on all 3 test video files | ⚠️ Modified | Detects **crimes** (not persons); also detects **vehicles** via separate model |
| Bounding boxes drawn correctly | ✅ Functional | Crime bboxes returned as `(x1,y1,x2,y2,conf,cls)` |
| Runs at min 15 FPS on GPU | ⚠️ Unknown | `TARGET_FPS=15` configured; actual FPS not benchmarked |
| No crashes on empty frames | ✅ Handled | `None` checks in detector + consumer |

**Key Deviation**: The plan calls for `ShadowDetector` wrapping YOLOv8 with `classes=[0]` (person only). The implementation has two separate YOLOv8 models:
1. **Crime detector** — fine-tuned model for violence/crime ([crime_detector.py](file:///d:/Projects/ShadowGrid/CrimeDetector/models/crime_detector.py))
2. **Vehicle detector** — YOLOv8n with `CAR_CLASSES=[2,3,5,7]` for vehicles ([car_tracker.py](file:///d:/Projects/ShadowGrid/CrimeDetector/models/car_tracker.py))

Neither detects persons as the plan requires.

---

#### Day 2 — DeepSORT Per-Camera Tracking

| Exit Condition | Status | Notes |
|---|---|---|
| Stable track IDs within one camera | ✅ Functional | ByteTrack provides stable IDs via `persist=True` |
| Track survives partial occlusion for 30 frames | ⚠️ Partial | ByteTrack handles occlusion but `max_age` not configurable via config |
| Multiple objects get different IDs | ✅ Functional | Track IDs are unique per `box.id` |
| ID does NOT persist when object leaves and re-enters | ✅ Correct | ByteTrack resets IDs. Re-ID handled separately |

**Key Deviation**: Plan uses `deep_sort_realtime.DeepSort`. Implementation uses **ByteTrack** built into ultralytics (`TRACKER_TYPE = "bytetrack"`). ByteTrack is simpler and faster but does not use appearance features for association (relies purely on IoU).

---

#### Day 3 — Multi-Stream Ingestion

| Exit Condition | Status | Notes |
|---|---|---|
| All streams reading simultaneously | ✅ Implemented | One daemon thread per camera in [frame_ingestor.py](file:///d:/Projects/ShadowGrid/CrimeDetector/core/frame_ingestor.py) |
| No frame lag — queue always has latest frame | ✅ Implemented | `deque(maxlen=30)` drops oldest frames automatically |
| Graceful handling when stream drops | ✅ Improved | Handles both hard failures (10 consecutive) AND frozen streams (perceptual hash detection) — **exceeds plan** |
| Swap between file and RTSP with one line change | ✅ Implemented | Camera URLs are loaded from `cameras.json` |

**Assessment**: Stream ingestion is **better than planned** — includes frozen-frame detection via MD5 hashing, auto-reconnect with configurable delays, and proper frame queue clearing on disconnect.

---

### SPRINT 2 — ReID & Cross-Camera Matching (Days 4–7)

#### Day 4 — CLIP Embedding Engine

| Exit Condition | Status | Notes |
|---|---|---|
| Embedding generates without error on any person crop | ⚠️ Partial | Works on **vehicle** crops. Uses OSNet, not CLIP |
| Same person across 2 frames → sim > 0.80 | ❌ Not applicable | System tracks **vehicles**, not people |
| Different people → sim < 0.70 | ❌ Not applicable | `SUSPECT_MATCH_THRESHOLD=0.70` set for vehicles |
| Text query returns meaningful similarity scores | ❌ Missing | No text embedding capability at all |
| Inference time under 150ms on CPU | ⚠️ Unknown | Not benchmarked |

**Key Deviation**: This is the **most significant divergence** from the plan:

| Planned | Implemented |
|---|---|
| CLIP ViT-B/32 | OSNet x0_25 (torchreid) |
| Person Re-ID | Vehicle Re-ID |
| 512-dim embeddings | 512-dim embeddings ✅ |
| `embed_text()` for text search | ❌ Not implemented |
| `embed_image()` on BGR crops | ✅ `extract_embedding()` on BGR crops |
| `cosine_sim()` static method | ✅ `cosine_similarity()` utility |

The `embed_text()` capability (CLIP's killer feature for "find person wearing red jacket") is **entirely absent**. The suspects endpoint has a `POST /suspects/description` that accepts text but stores a **zero embedding** — no actual text-to-visual matching occurs.

---

#### Day 5 — Global Track Registry

| Exit Condition | Status | Notes |
|---|---|---|
| GlobalTrack dataclass with trajectory/velocity | ❌ Missing | No `GlobalTrack` class. Suspects/occurrences are separate models |
| Cross-camera matching fires correctly | ✅ Functional | [tracking_module.py](file:///d:/Projects/ShadowGrid/CrimeDetector/core/tracking_module.py) matches across cameras |
| `match_or_create()` with time/space feasibility | ⚠️ Simplified | Only cosine similarity matching, no time-gap or speed-cap checks |
| `mark_lost_if_absent()` | ❌ Missing | No concept of "lost" status or automatic lost marking |
| Trajectory recording | ❌ Missing | No trajectory arrays tracked per identity |

The planned `GlobalRegistry` with spatiotemporal feasibility checks (travel time, max speed, room position) does not exist. The implementation uses a **flat suspects list** with embedding-based matching — simpler but less sophisticated.

---

#### Days 6–7 — Uncertainty Radius + Prediction Engine

| Exit Condition | Status | Notes |
|---|---|---|
| Cross-camera match fires correctly on test footage | ✅ Functional | Via tracking module + suspect matcher |
| False positive rate under 10% | ⚠️ Unknown | Not measured |
| Uncertainty radius grows correctly over time | ❌ Missing | No `UncertaintyEngine` implemented |
| Drone trigger fires at 150m threshold | ❌ Missing | No drone dispatch system |
| Probability bars show correct ranked predictions | ❌ Missing | No `CameraGraphPredictor` or NetworkX graph |

**Entire prediction subsystem is absent** — no uncertainty radius, no camera graph, no NetworkX dependency.

However, the GeoVision C++ engine (referenced in the UI but not in the plan) appears to provide some **equivalent** functionality:
- Suspect position estimation with radius
- Escape probability computation
- Police deployment orders
- This is consumed via [geovision.js](file:///d:/Projects/ShadowGrid/UI/src/api/geovision.js) API

> [!IMPORTANT]
> The GeoVision engine is **not part of this repository**. It's a separate C++ project that the UI polls on `localhost:9000`. Its source code is not present in `ShadowGrid/`.

---

### SPRINT 3 — Backend & API (Days 8–9)

#### Day 8 — FastAPI Server + WebSocket

| Exit Condition | Status | Notes |
|---|---|---|
| FastAPI server starts without errors | ✅ Functional | [main.py](file:///d:/Projects/ShadowGrid/CrimeDetector/main.py) with lifespan handler |
| WebSocket connection opens from browser | ❌ Missing | **No Socket.io or WebSocket** — REST polling only |
| All 5 event types emit correctly | ❌ Missing | No event emission system |
| `/api/text_search` returns results | ❌ Missing | Only `POST /suspects/description` (zero embedding) |
| `/api/dispatch_drone` triggers drone event | ❌ Missing | No drone endpoints |

**Key Deviation**: Plan uses `socketio.AsyncServer` for real-time push events (`track_new`, `track_updated`, `track_lost`, `track_reacquired`, `uncertainty`, `drone_dispatched`, `text_search_result`). Implementation uses **HTTP REST polling** from the frontend at 3–15 second intervals.

The actual API is **richer than planned** in REST endpoints:

| Endpoint | Planned | Implemented |
|---|---|---|
| `GET /` | ❌ | ✅ Health check |
| `GET /health` | ❌ | ✅ Detailed health with queue sizes |
| `GET /cameras` | ❌ | ✅ Camera list |
| `GET /cameras/{id}/frame` | ❌ | ✅ Single frame JPEG |
| `GET /cameras/{id}/stream` | ❌ | ✅ MJPEG live stream |
| `DELETE /cameras/{id}/suspect` | ❌ | ✅ Bulk suspect removal by camera |
| `GET /suspects` | ❌ | ✅ Paginated, delta-aware query |
| `DELETE /suspects/{id}` | ❌ | ✅ Individual suspect removal |
| `POST /suspects/upload` | ❌ | ✅ Image-based suspect creation |
| `POST /suspects/description` | ❌ | ✅ Text description suspect |
| `GET /occurrences` | ❌ | ✅ Full occurrence query with filters |
| `POST /api/text_search` | ✅ Planned | ❌ Not implemented |
| `POST /api/dispatch_drone` | ✅ Planned | ❌ Not implemented |

---

#### Day 9 — Pipeline Orchestrator

| Exit Condition | Status | Notes |
|---|---|---|
| Async orchestrator with `asyncio.gather` | ⚠️ Different | Synchronous `_consumer_loop` with `threading` |
| All cameras processed each cycle | ✅ Functional | Round-robin with priority ordering |
| Frame + detect + embed + match pipeline | ✅ Functional | Full pipeline in consumer.py + tracking_module.py |

The implementation uses a **single consumer thread** with priority-based round-robin scheduling instead of the planned async+executor model. Additionally, it includes a **MIL anomaly scoring layer** not in the plan — the consumer accumulates frames into `MILFrameBuffer` and triggers anomaly events when the MIL score exceeds the threshold.

---

### SPRINT 4 — Frontend Dashboard (Days 10–13)

#### Day 10 — Layout + Socket Connection

| Exit Condition | Status | Notes |
|---|---|---|
| Socket.io connection from React | ❌ Missing | HTTP REST polling via `AppContext.jsx` |
| `useSocket()` hook manages all state | ⚠️ Equivalent | `useApp()` hook via React Context + `useReducer` |
| Real-time track/uncertainty/alert state | ⚠️ Polling-based | 3–15 second polling intervals, not real-time push |

---

#### Days 11–12 — Room Map + All Animations

| Exit Condition | Status | Notes |
|---|---|---|
| Canvas-based floor plan with grid | ❌ Different approach | **Leaflet.js geo-map** with real lat/lon coordinates |
| Camera markers with active/inactive colors | ✅ Implemented | Camera markers with live/offline status |
| Trajectory lines per track | ⚠️ Different | Escape probability paths, not per-track trajectories |
| Pulsing uncertainty circle | ⚠️ Via GeoVision | Suspect radius circles rendered from GeoVision data |
| Suspect dots with glow effect | ✅ Implemented | Custom DivIcon markers with CSS glow + pulse animation |
| `requestAnimationFrame` render loop | ❌ Different | Leaflet handles map rendering; overlays rebuilt on data change |

**Key Deviation**: The plan describes a **custom canvas renderer** for an indoor floor plan. The implementation uses **Leaflet.js** for a real-world geographic map with OpenStreetMap tiles. This is actually more sophisticated — it supports:
- Real-world lat/lon positioning
- Police deployment routes
- Airport/railway/border escape targets
- Road-following path rendering
- Interactive map controls (zoom, pan)
- Map legend

---

#### Day 13 — Supporting Panels

| Panel | Planned | Implemented |
|---|---|---|
| PredictionPanel (probability bars) | ✅ Planned | ❌ Missing |
| ReacquiredBanner (green notification) | ✅ Planned | ❌ Missing |
| UncertaintyPanel (search zone stats) | ✅ Planned | ❌ Missing |
| Dispatch Drone button | ✅ Planned | ❌ Missing |
| TrajectoryLog | ✅ Planned | ❌ Missing |
| TextSearch | ✅ Planned | ❌ Missing |
| **SuspectsPanel** | ❌ Not planned | ✅ Implemented — full CRUD |
| **OccurrencesPanel** | ❌ Not planned | ✅ Implemented — timeline view |
| **CamerasPanel** | ❌ Not planned | ✅ Implemented — live feed grid |
| **IntelligencePanel** | ❌ Not planned | ✅ Implemented — GeoVision integration |
| **StatusBar** | ❌ Not planned | ✅ Implemented — dual backend connectivity |
| **NavRail** | ❌ Not planned | ✅ Implemented — tab navigation |
| **MapSidebar** | ❌ Not planned | ✅ Implemented — suspect/escape details |
| **EmbeddingChart** | ❌ Not planned | ✅ Implemented — similarity visualization |

---

### SPRINT 5 & 6 — Integration & Demo Hardening (Days 14–18)

| Checklist Item | Status | Notes |
|---|---|---|
| All WebSocket events rendering on dashboard | ❌ N/A | No WebSocket system |
| REACQUIRED fires at correct moment | ❌ Missing | No reacquisition event system |
| Text search returns correct crops | ❌ Missing | No CLIP text search |
| Drone dispatch button triggers and animates | ❌ Missing | No drone system |
| Switch to live ESP32-CAM streams | ⚠️ Equivalent | Using IP Webcam MJPEG (same concept) |
| 30 minute continuous run, no memory leak | ⚠️ Known issue | `purge_stale_tracks()` exists but is never called (BUG-05 from prior audit) |
| No debug print statements | ✅ Clean | Uses proper `logging` module throughout |
| Error handling on every API endpoint | ✅ Good | HTTPExceptions, try/except in consumer |
| Frontend shows graceful message if backend offline | ✅ Implemented | StatusBar shows `cdOnline`/`gvOnline` connectivity |
| All secrets/IPs in `.env` | ❌ Not done | IPs hardcoded in `cameras.json`, no `.env` file |
| Backup system (prerecorded fallback) | ❌ Missing | No fallback mechanism |
| Single command startup script | ❌ Missing | No `start.sh` |

---

## Feature Completeness Matrix

| Feature | Plan | Implemented | Gap |
|---|---|---|---|
| **Person Detection** (YOLOv8) | ✅ | ❌ — Crime + Vehicle detection | 🔴 Different target |
| **DeepSORT Tracking** | ✅ | ❌ — ByteTrack | ⚠️ Simpler alternative |
| **CLIP ReID** (image + text) | ✅ | ❌ — OSNet (image only) | 🔴 No text search |
| **Global Track Registry** | ✅ | ⚠️ — Flat suspect list | 🟠 No trajectories |
| **Uncertainty Radius** | ✅ | ❌ — (GeoVision external) | 🔴 Missing from codebase |
| **Camera Graph Prediction** | ✅ | ❌ | 🔴 Missing |
| **WebSocket Push Events** | ✅ | ❌ — REST polling | 🟠 Functional alternative |
| **Socket.io Real-time** | ✅ | ❌ | 🔴 Missing |
| **Canvas Floor Plan Map** | ✅ | ❌ — Leaflet Geo Map | ⚠️ Better alternative |
| **Drone Dispatch** | ✅ | ❌ | 🔴 Missing |
| **Text Search** (CLIP) | ✅ | ❌ | 🔴 Missing |
| **Reacquired Banner** | ✅ | ❌ | 🔴 Missing |
| **Prediction Bars** | ✅ | ❌ | 🔴 Missing |
| **Uncertainty Stats Panel** | ✅ | ❌ | 🔴 Missing |
| **Multi-camera Ingestion** | ✅ | ✅ | ✅ Complete |
| **Cross-camera Re-ID matching** | ✅ | ✅ | ✅ Implemented |
| **MIL Anomaly Detection** | ❌ | ✅ | ➕ Extra feature |
| **Priority-based Scheduling** | ❌ | ✅ | ➕ Extra feature |
| **GeoVision Map Intelligence** | ❌ | ✅ | ➕ Extra feature |
| **MJPEG Live Streaming** | ❌ | ✅ | ➕ Extra feature |
| **Suspects CRUD API** | ❌ | ✅ | ➕ Extra feature |
| **Occurrences Tracking** | ❌ | ✅ | ➕ Extra feature |
| **Police Deployment Orders** | ❌ | ✅ (via GeoVision) | ➕ Extra feature |
| **Image-based Suspect Upload** | ❌ | ✅ | ➕ Extra feature |
| **Crime Cooldown Dedup** | ❌ | ✅ | ➕ Extra feature |
| **Frozen Stream Detection** | ❌ | ✅ | ➕ Extra feature |

---

## Known Issues (Carried from Prior Audit + New Findings)

### 🔴 Critical / High

| ID | Issue | File |
|---|---|---|
| SEC-01 | Wide-open CORS `allow_origins=["*"]` with `allow_credentials=True` | [main.py#L90-96](file:///d:/Projects/ShadowGrid/CrimeDetector/main.py#L90-L96) |
| SEC-02 | Hardcoded internal IPs & personal names in `cameras.json` | [cameras.json](file:///d:/Projects/ShadowGrid/CrimeDetector/cameras.json) |
| SEC-04 | No authentication on any API endpoint | [main.py](file:///d:/Projects/ShadowGrid/CrimeDetector/main.py) |
| BUG-02 | Unbounded thread spawning per crime event — no `ThreadPoolExecutor` | [tracking_module.py#L159-L171](file:///d:/Projects/ShadowGrid/CrimeDetector/core/tracking_module.py#L159-L171) |
| BUG-03 | Race condition on `frame_queues` / `last_push_times` — no lock | [state.py#L178-L183](file:///d:/Projects/ShadowGrid/CrimeDetector/core/state.py#L178-L183) |
| BUG-04 | `torch.load(weights_only=False)` in MIL inference — arbitrary code exec | [inference_mil.py#L57](file:///d:/Projects/ShadowGrid/CrimeDetector/training/inference_mil.py#L57) |

### 🟠 Medium

| ID | Issue | File |
|---|---|---|
| BUG-05 | `purge_stale_tracks()` defined but never called — memory leak | [car_tracker.py#L195-L201](file:///d:/Projects/ShadowGrid/CrimeDetector/models/car_tracker.py#L195-L201) |
| BUG-06 | Re-embedding ALL buffered crops every frame — wasteful | [car_tracker.py#L167-178](file:///d:/Projects/ShadowGrid/CrimeDetector/models/car_tracker.py#L167-L178) |
| BUG-07 | `_was_recently_seen()` scans entire deque under lock | [tracking_module.py#L73-L84](file:///d:/Projects/ShadowGrid/CrimeDetector/core/tracking_module.py#L73-L84) |
| BUG-08 | MIL buffer stores 512 full-res frames (~460MB per camera) | [inference_mil.py#L147-L159](file:///d:/Projects/ShadowGrid/CrimeDetector/training/inference_mil.py#L147-L159) |
| BUG-11 | Unused `trigger_frame` parameter in `start_tracking_thread()` | [tracking_module.py#L159](file:///d:/Projects/ShadowGrid/CrimeDetector/core/tracking_module.py#L159) |
| NEW-01 | `package.json` name is `"reprosense"`, not `"shadowgrid"` | [package.json](file:///d:/Projects/ShadowGrid/UI/package.json#L2) |
| NEW-02 | README references ZeroSpot, not ShadowGrid; project structure doesn't match | [README.md](file:///d:/Projects/ShadowGrid/README.md) |

### 🟡 Low

| ID | Issue | File |
|---|---|---|
| BUG-16 | `suspects_page` off-by-one edge case (`<` vs `<=`) | [state.py#L318](file:///d:/Projects/ShadowGrid/CrimeDetector/core/state.py#L318) |
| BUG-17 | Server binds to `0.0.0.0` by default | [config.py#L91](file:///d:/Projects/ShadowGrid/CrimeDetector/config.py#L91) |
| BUG-18 | No rate limiting on API endpoints | [main.py](file:///d:/Projects/ShadowGrid/CrimeDetector/main.py) |
| BUG-19 | Consumer poll interval 10ms may saturate CPU | [config.py#L87](file:///d:/Projects/ShadowGrid/CrimeDetector/config.py#L87) |
| NEW-03 | No `.env` file or `.env.example` despite config using `os.getenv` | [config.py](file:///d:/Projects/ShadowGrid/CrimeDetector/config.py) |
| NEW-04 | No test footage in `videos/` directory | Root directory |
| NEW-05 | No `start.sh` startup script | Root directory |

---

## Implementation Quality Assessment

### Strengths (Things Done Well)

1. **Clean Architecture** — Clear separation of concerns across `core/`, `models/`, `routers/`, `training/`
2. **Comprehensive Logging** — Every module uses Python's `logging` module with proper levels
3. **Robust Stream Handling** — Frozen-frame detection, auto-reconnect, configurable staleness window
4. **MIL Anomaly Pipeline** — Full train/evaluate/inference pipeline with real-time frame buffering
5. **Priority Scheduling** — Geo-aware priority with exponential decay is novel and well-implemented
6. **React Dashboard** — Well-structured with Context/Reducer pattern, delta polling, proper error handling
7. **Leaflet Map** — Rich visualization with escape routes, police deployments, suspect radii, legends
8. **Thread Safety** — Suspect/occurrence deques use proper locks; per-camera locks prevent concurrent tracking
9. **Configurable** — All thresholds centralized in `config.py` with clear documentation
10. **Dual-model Crime Detection** — YOLO + MIL provides both per-frame and temporal anomaly detection

### Weaknesses (Critical Gaps vs. Plan)

1. ❌ **No WebSocket/real-time push** — Frontend polls every 3–15s; latency-sensitive for live tracking
2. ❌ **No CLIP text search** — The plan's headline feature ("find person wearing red jacket") is absent
3. ❌ **No person tracking** — Tracks vehicles, not people, despite the plan focusing on person Re-ID
4. ❌ **No uncertainty radius engine** — No expanding search zone visualization
5. ❌ **No camera graph prediction** — No NetworkX-based next-camera probability
6. ❌ **No drone dispatch** — Feature removed entirely
7. ❌ **No trajectory logging** — No per-identity movement history
8. ❌ **No reacquisition events** — No "SUSPECT REACQUIRED" alerts

---

## Recommendations (Priority Order)

### To Align with Plan

1. **Add Socket.io** — Replace REST polling with `python-socketio` on the backend and `socket.io-client` on the frontend for real-time push
2. **Implement CLIP embedding** — Add `clip` dependency, create `reid/clip_embedder.py` with `embed_text()` for text search
3. **Add UncertaintyEngine** — Port the planned `prediction/uncertainty.py` to compute expanding radius
4. **Add CameraGraphPredictor** — Port the planned `prediction/graph_predictor.py` with NetworkX
5. **Add person detection** — Enable YOLOv8 `classes=[0]` alongside vehicle detection
6. **Create missing frontend panels** — PredictionPanel, UncertaintyPanel, TextSearch, ReacquiredBanner, TrajectoryLog

### To Fix Existing Issues

7. **Fix CORS** — Replace `allow_origins=["*"]` with explicit frontend origins
8. **Bound thread spawning** — Use `ThreadPoolExecutor` in tracking module
9. **Add frame queue locks** — Per-camera lock for `push_frame` / `get_latest_frame` / `clear_frame_queue`
10. **Call `purge_stale_tracks()`** — Add to `CarTracker.update()` to prevent memory leaks
11. **Downsample MIL buffer frames** — Reduce from ~460MB/camera to ~256KB/camera
12. **Update README** — Rename from ZeroSpot to ShadowGrid; update project structure to match reality

### Operational

13. **Create `.env.example`** — Document all environment variables
14. **Create `start.sh`** — Single-command startup script
15. **Add test footage** — Place demo videos in `videos/` directory
16. **Rename UI package** — Change `"reprosense"` to `"shadowgrid"` in `package.json`

---

## Summary Statistics

| Metric | Count |
|---|---|
| Planned features fully implemented | **3** / 14 (21%) |
| Planned features partially implemented | **5** / 14 (36%) |
| Planned features completely missing | **6** / 14 (43%) |
| Extra features not in plan | **12** |
| Open bugs (carried + new) | **18** |
| Critical/High severity issues | **6** |
| Source files in project | **~50** |
| Lines of Python (CrimeDetector) | **~2,200** |
| Lines of JSX/JS (UI) | **~1,200** |

---

> **Bottom Line**: The project is a **functional crime-detection and vehicle-tracking system** with a polished React dashboard and rich GeoVision intelligence integration. However, it has **significant divergence** from the ShadowGrid implementation plan — particularly the absence of CLIP text search, WebSocket real-time updates, person tracking, uncertainty radius, and drone dispatch. The codebase is production-quality in its existing features but needs substantial work to align with the plan's vision.
