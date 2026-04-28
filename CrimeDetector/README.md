# AI Surveillance Server

A real-time, multi-camera crime-detection and cross-camera suspect tracking server built with **FastAPI**, **YOLOv8 + ByteTrack**, and **OSNet Re-ID**.  
Streams live MJPEG from Android IP Webcam phones, detects crime events, embeds suspect vehicles, tracks them across all cameras, and exposes a full REST API plus a dark-theme browser dashboard.

---

## Architecture

```
cameras.json
     │
     ▼
CameraLoader ──► CameraRegistry (state.py)
                      │
          ┌───────────┴────────────────────┐
          │  FrameIngestor threads          │  one daemon thread per camera
          │  OpenCV MJPEG pull              │  auto-reconnects on failure
          └───────────┬────────────────────┘
                      │  push_frame()
                      ▼
         frame_queues[cam_id]               deque, drop-oldest, maxlen=30
                      │
                      ▼
     ┌────────────────────────────────┐
     │  Consumer  (single daemon)     │  ◄── priority_manager
     │  1. detect_crime() on frame    │       (geo boost + exponential decay)
     │  2. per-camera cooldown check  │
     │  3. crop from crime bboxes     │
     │  4. batch Re-ID embed          │
     │  5. intra-event dedup          │
     └──────┬─────────────────────────┘
            │  suspects created
            ├──► add SuspectEntry ──► suspects  deque, maxlen=500
            ├──► boost_on_crime()
            └──► start_tracking_thread()
                          │
                          ▼
          ┌───────────────────────────────┐
          │  TrackingModule               │  daemon thread, 30 passes × 0.1 s
          │  CarTracker (YOLOv8+ByteTrack)│  per-camera stateful tracker
          │  Re-ID embed (always returned)│  single-frame → averaged
          │  SuspectMatcher (cosine sim)  │
          └───────────┬───────────────────┘
                      │  match ≥ 0.70
                      ├──► add OccurrenceEntry ──► occurrences  deque, maxlen=2000
                      └──► boost_on_occurrence()

REST API  (FastAPI, auto-docs at /docs)
  GET  /cameras                                  list all cameras
  GET  /cameras/{id}/frame                       latest JPEG snapshot
  GET  /cameras/{id}/stream                      live MJPEG stream
  DELETE /cameras/{id}/suspect                   remove camera's suspects

  GET  /suspects                                 recent suspects (newest-first)
  GET  /suspects?since_timestamp=<f>&limit=<n>   suspects after a timestamp
  POST /suspects/upload                          add suspect from image file
  POST /suspects/description                     add suspect from text
  DELETE /suspects/{suspect_id}                  remove one suspect

  GET  /occurrences                              recent occurrences (newest-first)
  GET  /occurrences?since_timestamp=<f>&limit=<n> occurrences after a timestamp
  GET  /occurrences/suspect/{suspect_id}         filter by suspect
  GET  /occurrences/camera/{camera_id}           filter by camera

  GET  /health                                   server status + queue depths
```

---

## Project Structure

```
surveillance_server/
├── main.py                  # FastAPI app + startup lifespan
├── config.py                # Every tuneable constant (edit here, not in code)
├── cameras.json             # Camera definitions
│
├── core/
│   ├── state.py             # All shared state: deques, locks, data models,
│   │                        # getters (snapshot, since, page)
│   ├── camera_loader.py     # Parses cameras.json → registers cameras
│   ├── frame_ingestor.py    # Per-camera MJPEG pull daemon threads
│   ├── consumer.py          # Round-robin consumer, crime handling,
│   │                        # per-camera cooldown, intra-event dedup
│   ├── tracking_module.py   # Cross-camera suspect tracking daemon threads
│   └── priority_manager.py  # Gaussian geo boost + exponential decay
│
├── models/
│   ├── model_manager.py     # Loads all .pt models; extract_embedding(),
│   │                        # extract_embeddings_batch()
│   ├── crime_detector.py    # YOLOv8 crime inference wrapper
│   ├── car_tracker.py       # YOLOv8 + ByteTrack + Re-ID per-camera tracker;
│   │                        # always returns embedding (single-frame or averaged)
│   └── suspect_matcher.py   # Cosine similarity search over suspects deque
│
├── routers/
│   ├── cameras.py           # /cameras endpoints
│   ├── suspects.py          # /suspects endpoints
│   └── occurrences.py       # /occurrences endpoints
│
├── utils/
│   ├── geo.py               # Haversine distance
│   └── embedding.py         # Embedding serialise / deserialise helpers
│
├── model_weights/
│   ├── README.md            # How to obtain each .pt file
│   ├── crime_detect.pt      # ← your YOLOv8 crime weights
│   ├── yolov8n.pt           # ← auto-downloaded on first run if absent
│   └── osnet_x0_25.pt       # ← OSNet Re-ID weights
│
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Recommended — full OSNet Re-ID support (falls back to ResNet-18 without it):
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

### 2. Add model weights

See `model_weights/README.md` for download links.  
`yolov8n.pt` is fetched automatically from the Ultralytics hub on first run.

### 3. Configure cameras

Edit `cameras.json`. Every phone running **IP Webcam** (Android) exposes an MJPEG stream at `http://<phone-ip>:8080/video` by default.

```json
[
  {
    "id": "cam_01",
    "url": "http://192.168.1.101:8080/video",
    "metadata": {
      "location_name": "Front Gate",
      "latitude": 22.5726,
      "longitude": 88.3639
    }
  },
  {
    "id": "cam_02",
    "url": "http://192.168.1.102:8080/video",
    "metadata": {
      "location_name": "Parking Lot A",
      "latitude": 22.5731,
      "longitude": 88.3645
    }
  }
]
```

**IP Webcam app settings:** Resolution → 640×480, FPS → 15, disable authentication for LAN use.

### 4. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Interactive API docs: **http://localhost:8000/docs**  
Browser dashboard (open `index.html` from the UI package): enter `http://<server-ip>:8000` and click **CONNECT**.

---

## Complete config.py Reference

All constants are in `config.py`. Restart the server after any change.

### Paths

| Constant | Default | Description |
|---|---|---|
| `CAMERAS_FILE` | `cameras.json` | Camera definitions file |
| `CRIME_MODEL_PATH` | `model_weights/crime_detect.pt` | YOLOv8 crime detector weights |
| `CAR_DETECT_MODEL` | `model_weights/yolov8n.pt` | YOLOv8 vehicle detector weights |
| `REID_MODEL_PATH` | `model_weights/osnet_x0_25.pt` | OSNet Re-ID weights |

### Camera & stream

| Constant | Default | Description |
|---|---|---|
| `FRAME_WIDTH` | `640` | Expected frame width in pixels |
| `FRAME_HEIGHT` | `480` | Expected frame height in pixels |
| `TARGET_FPS` | `15` | Target FPS for the `/stream` MJPEG endpoint |
| `STREAM_JPEG_QUALITY` | `80` | JPEG compression quality for stream output |
| `DEQUE_MAX_LEN` | `30` | Frame buffer depth per camera (~2 s at 15 fps); oldest dropped when full |

### Detection thresholds

| Constant | Default | Description |
|---|---|---|
| `CRIME_CONFIDENCE` | `0.55` | Minimum YOLOv8 confidence to trigger a crime event. Lower = more sensitive, more false positives. |
| `CAR_CONFIDENCE` | `0.40` | Minimum YOLOv8 confidence to accept a vehicle detection in the tracker. |
| `CAR_CLASSES` | `[2,3,5,7]` | COCO class IDs tracked as vehicles: car, motorcycle, bus, truck. |

### Re-ID & suspect matching

| Constant | Default | Description |
|---|---|---|
| `REID_EMBED_DIM` | `512` | Output dimension of the Re-ID model. |
| `SUSPECT_MATCH_THRESHOLD` | `0.70` | Cosine similarity ≥ this → occurrence confirmed. Lower = looser matching. |
| `REID_BATCH_SIZE` | `8` | Max crops per Re-ID forward pass. |

### Tracking

| Constant | Default | Description |
|---|---|---|
| `TRACKER_TYPE` | `bytetrack` | Ultralytics built-in tracker name. |
| `MIN_TRACK_FRAMES` | `4` | Frames a track must be seen before its embedding is averaged. Single-frame embedding is used before this threshold. |

### Suspect flood control

| Constant | Default | Description |
|---|---|---|
| `CRIME_COOLDOWN_SECONDS` | `30` | After suspects are created from a camera, further crime events on that camera are suppressed for this long. Prevents the same scene from flooding the suspects list. Cooldown is only stamped when at least one suspect is actually created. |
| `SUSPECT_DEDUP_SIMILARITY` | `0.92` | Within a single crime event, crops whose embeddings are ≥ this similar to each other are treated as the same object (e.g. overlapping bboxes on one knife). Does **not** compare against historical suspects. |

### Occurrence dedup

| Constant | Default | Description |
|---|---|---|
| `OCCURRENCE_DEDUP_SECONDS` | `60` | A car already logged as an occurrence on the same camera within this window is not logged again. |

### Bounded in-memory lists

| Constant | Default | Description |
|---|---|---|
| `MAX_SUSPECTS` | `500` | Hard cap on suspects deque. Oldest entry evicted automatically on overflow. |
| `MAX_OCCURRENCES` | `2000` | Hard cap on occurrences deque. Oldest entry evicted automatically on overflow. |

### Query pagination

| Constant | Default | Description |
|---|---|---|
| `QUERY_DEFAULT_LIMIT` | `100` | Items returned when `?limit` is not supplied. |
| `QUERY_MAX_LIMIT` | `1000` | Absolute ceiling a caller may request via `?limit`. |

### Camera priority

| Constant | Default | Description |
|---|---|---|
| `PRIORITY_CRIME_BOOST` | `100.0` | Points added to nearby cameras when a crime fires. |
| `PRIORITY_OCCURRENCE_BOOST` | `50.0` | Points added to a camera when it confirms a suspect occurrence. |
| `PRIORITY_DECAY_RATE` | `0.05` | Exponential decay per second: `priority *= e^(-rate * elapsed)`. |
| `PRIORITY_GEO_MAX_KM` | `10.0` | Radius within which neighbouring cameras receive a crime boost. |
| `PRIORITY_GEO_FALLOFF` | `2.0` | Gaussian σ in km for the geo boost weight. Cameras at 2 km get ~60% of the boost; at 4 km ~14%. |

### Consumer scheduler

| Constant | Default | Description |
|---|---|---|
| `CONSUMER_POLL_INTERVAL` | `0.01` | Sleep between processing each camera in the round-robin loop (seconds). |
| `CONSUMER_SKIP_EMPTY_MS` | `5` | Sleep when all camera queues are empty (milliseconds). |

---

## API Reference

### Cameras

```bash
# List all registered cameras (from cameras.json)
GET /cameras

# Latest JPEG snapshot — returns 204 if no frame yet
GET /cameras/{camera_id}/frame

# Live MJPEG stream — paste into browser <img> or VLC
GET /cameras/{camera_id}/stream

# Remove all SuspectEntry records that originated from this camera
DELETE /cameras/{camera_id}/suspect
```

### Suspects

```bash
# 100 most recent suspects (newest-first)
GET /suspects

# Up to 50 suspects added after Unix timestamp 1748000000.0
GET /suspects?since_timestamp=1748000000.0&limit=50

# Add a suspect from a car image (computes Re-ID embedding server-side)
POST /suspects/upload
  -F file=@car.jpg
  -F camera_id=cam_01          # optional
  -F description="Red sedan"   # optional

# Add a suspect from text only (zero embedding stored — no Re-ID matching)
POST /suspects/description
  -F description="White SUV, tinted windows"
  -F camera_id=cam_02          # optional

# Remove one suspect by ID
DELETE /suspects/{suspect_id}
```

### Occurrences

```bash
# 100 most recent occurrences (newest-first)
GET /occurrences

# Up to 200 occurrences after a timestamp
GET /occurrences?since_timestamp=1748000000.0&limit=200

# All occurrences for one suspect
GET /occurrences/suspect/{suspect_id}

# All occurrences recorded by one camera
GET /occurrences/camera/{camera_id}
```

### Health

```bash
GET /health
# Returns: status, cameras_registered, suspects_count, occurrences_count,
#          queues: { cam_id: current_queue_depth, ... }
#          "cameras_live": {cam_id: is_camera_live(cam_id) }
```

---

## Response Schemas

### SuspectEntry
```json
{
  "suspect_id":  "uuid-string",
  "camera_id":   "cam_01",
  "car_id":      0,
  "timestamp":   1748001234.56,
  "embedding":   [0.012, -0.034, ...],   // 512 floats, L2-normalised
  "source":      "crime_detection",      // | "manual_upload" | "manual_description"
  "description": null                    // or string if provided
}
```

### OccurrenceEntry
```json
{
  "occurrence_id": "uuid-string",
  "suspect_id":    "uuid-string",
  "camera_id":     "cam_02",
  "car_id":        7,
  "embedding":     [0.021, -0.011, ...], // 512 floats, L2-normalised
  "similarity":    0.8412,               // cosine similarity to matched suspect
  "timestamp":     1748001300.12
}
```

---

## Model Choices & Rationale

| Task | Model | Reason |
|---|---|---|
| Crime detection | YOLOv8 (custom `.pt`) | Real-time at 640×480, fine-tuneable on violence/crime datasets (RWF-2000, UCF-Crime) |
| Vehicle detection | YOLOv8n | Fast and lightweight; COCO vehicle classes (car, bus, truck, motorcycle) built-in |
| Multi-object tracking | ByteTrack (built into Ultralytics) | Robust to occlusions; `persist=True` maintains IDs across frames within a session |
| Re-ID / embedding | OSNet x0_25 | 512-d cosine embedding space; lightweight; pre-trained on VeRi-776 vehicle Re-ID. Falls back to ResNet-18 if `torchreid` is not installed |

---

## Known Behaviours & Design Decisions

**Suspect creation uses crime bboxes directly, not ByteTrack.**  
ByteTrack requires multiple consecutive frames to assign stable IDs. On a single crime frame it always returns zero results. The consumer therefore crops directly from the crime detector's bounding boxes and embeds immediately. The tracking module later assigns real ByteTrack IDs for cross-camera matching.

**CarTracker always returns an embedding.**  
Before `MIN_TRACK_FRAMES` the tracker returns a single-frame embedding. After `MIN_TRACK_FRAMES` it returns an averaged, re-normalised multi-frame embedding. `is_averaged_embedding` on `TrackResult` indicates which. The tracking module does not skip early-track results.

**`_crop_buffer` stores raw BGR crops, not embeddings.**  
The buffer accumulates pixel arrays so that averaging is done by re-embedding all crops in one batch — not by averaging pre-computed embedding vectors, which would cause a double-embedding error.

**Intra-event dedup only.**  
`SUSPECT_DEDUP_SIMILARITY` compares crops within the same crime event's batch to remove overlapping bboxes on one object. It does **not** compare against historical suspects — doing so causes permanent lockout because the same scene viewed repeatedly produces near-identical embeddings.

**Cooldown stamps only on actual creation.**  
`CRIME_COOLDOWN_SECONDS` is stamped only when `entries_created > 0`. If all crops are degenerate or the embedding forward pass fails, the cooldown is not activated and the next crime frame gets a fresh attempt.

**Suspects and occurrences are in-memory only.**  
State is lost on server restart. For persistence, replace the deque-backed stores in `state.py` with a SQLite or PostgreSQL backend.

---

## GPU Acceleration

By default all inference runs on CPU. To use a CUDA GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Set `CUDA_VISIBLE_DEVICES=0` before starting the server. The model manager auto-detects `torch.cuda.is_available()` and moves models to GPU at load time. No code changes needed.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Cannot open stream` in logs | Phone IP changed or stream not started | Check phone IP, open IP Webcam app, tap "Start server" |
| Crime detected but 0 suspects | All bboxes smaller than 20 px | Lower `CRIME_CONFIDENCE` or move camera closer |
| Only 1 suspect ever created | Old bug (fixed) — ensure you have the latest `consumer.py` | Pull latest files |
| Suspects flooding (40+ per minute) | Old bug (fixed) — ensure `CRIME_COOLDOWN_SECONDS` is set | Check `config.py` |
| No occurrences ever | `SUSPECT_MATCH_THRESHOLD` too high, or tracking thread not seeing frames | Lower threshold to 0.55, check `/health` queue depths |
| `torchreid not installed` warning | torchreid optional dep missing — ResNet-18 used as fallback | `pip install git+https://github.com/KaiyangZhou/deep-person-reid.git` |
| Server slow on large suspect list | `snapshot_suspects()` copies entire deque | Lower `MAX_SUSPECTS` or use `?limit=` on API calls |

