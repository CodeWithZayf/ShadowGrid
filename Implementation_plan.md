# ShadowGrid — Software Development Plan

---

## Team Role Assignment

```
Zafir (You)      →  ML Pipeline Lead
                    Detection, ReID, Cross-cam matching,
                    Kalman prediction

Teammate 2       →  Backend Lead
                    FastAPI, WebSocket, Camera graph,
                    Uncertainty radius engine

Teammate 3       →  Frontend Lead
                    React dashboard, Leaflet map,
                    Animations, UI polish
```

---

## Master Timeline

```
Week 1   │  Days 1–3    │  Detection + Tracking
Week 1   │  Days 4–5    │  CLIP ReID + Embeddings
Week 2   │  Days 6–7    │  Cross-camera matching
Week 2   │  Days 8–9    │  Backend + WebSocket
Week 3   │  Days 10–11  │  Frontend map dashboard
Week 3   │  Days 12–13  │  Integration end-to-end
Week 4   │  Days 14–15  │  Text search + Drone trigger
Week 4   │  Days 16–17  │  Polish + stress test
Week 4   │  Day  18     │  Demo rehearsal. Code freeze.
```

---

## SPRINT 1 — Detection & Tracking
### Days 1–3 | Owner: Zafir

---

### Day 1 — YOLOv8 Detection

```python
# detection/detector.py

from ultralytics import YOLO
import cv2, numpy as np

class ShadowDetector:
    def __init__(self, model="yolov8n.pt", conf=0.5):
        self.model = YOLO(model)
        self.conf  = conf

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model(
            frame, classes=[0],   # person only
            conf=self.conf, verbose=False
        )[0]

        out = []
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            out.append({
                "bbox": [x1,y1,x2-x1,y2-y1],
                "conf": conf,
                "crop": crop,
                "xyxy": [x1,y1,x2,y2]
            })
        return out
```

**Day 1 exit condition:**
```
✓ Detects persons on all 3 test video files
✓ Bounding boxes drawn correctly on frame
✓ Runs at minimum 15 FPS on your GPU
✓ No crashes on empty frames
```

---

### Day 2 — DeepSORT Per-Camera Tracking

```python
# tracking/camera_tracker.py

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

class CameraTracker:
    def __init__(self, cam_id: str, max_age=30):
        self.cam_id  = cam_id
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=3,
            nms_max_overlap=0.7,
            max_cosine_distance=0.3
        )

    def update(self, detections: list,
               frame: np.ndarray) -> list[dict]:
        ds_input = [
            (d["bbox"], d["conf"], "person")
            for d in detections
        ]
        tracks = self.tracker.update_tracks(
            ds_input, frame=frame
        )
        confirmed = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            l,to,w,h = t.to_ltwh()
            confirmed.append({
                "local_id":  t.track_id,
                "cam_id":    self.cam_id,
                "bbox_ltwh": [l,to,w,h],
                "bbox_center": [l + w/2, to + h/2]
            })
        return confirmed
```

**Day 2 exit condition:**
```
✓ Stable track IDs within one camera — no flickering
✓ Track survives partial occlusion for 30 frames
✓ Multiple people get different IDs
✓ ID does NOT persist when person leaves and re-enters
   (that's the ReID problem — handled next)
```

---

### Day 3 — Multi-Stream Ingestion

```python
# ingestion/stream_manager.py

import cv2, threading
from collections import deque

class CameraStream:
    def __init__(self, cam_id: str, source):
        """
        source: int (webcam) | str (file path or RTSP URL)
        For ESP32-CAM: "http://192.168.1.101/stream"
        """
        self.cam_id = cam_id
        self.source = source
        self.queue  = deque(maxlen=2)  # always latest frame
        self.running = False
        self._lock   = threading.Lock()

    def start(self):
        self.running = True
        t = threading.Thread(
            target=self._capture, daemon=True
        )
        t.start()
        return self

    def _capture(self):
        cap = cv2.VideoCapture(self.source)
        while self.running:
            ret, frame = cap.read()
            if ret:
                with self._lock:
                    self.queue.append(frame)
        cap.release()

    def get_frame(self):
        with self._lock:
            return self.queue[-1] if self.queue else None

    def stop(self):
        self.running = False


# Usage — swap source for ESP32 URL on demo day
streams = {
    "CAM_01": CameraStream("CAM_01", "videos/cam1.mp4").start(),
    "CAM_02": CameraStream("CAM_02", "videos/cam2.mp4").start(),
    "CAM_03": CameraStream("CAM_03", "videos/cam3.mp4").start(),
}
```

**Day 3 exit condition:**
```
✓ All 3 streams reading simultaneously
✓ No frame lag — queue always has latest frame
✓ Graceful handling when one stream drops
✓ Swap between file and RTSP with one line change
```

---

## SPRINT 2 — ReID & Cross-Camera Matching
### Days 4–7 | Owner: Zafir

---

### Day 4 — CLIP Embedding Engine

```python
# reid/embedder.py

import clip, torch
import numpy as np
from PIL import Image
import cv2

class CLIPEmbedder:
    def __init__(self, device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load(
            "ViT-B/32", device=device
        )
        self.model.eval()

    def embed_image(self,
                    crop_bgr: np.ndarray) -> torch.Tensor:
        """
        Takes BGR numpy crop, returns normalized 512-dim tensor
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        img = Image.fromarray(
            cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        )
        tensor = self.preprocess(img).unsqueeze(0).to(
            self.device
        )
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
        return (emb / emb.norm(
            dim=-1, keepdim=True
        )).squeeze(0)

    def embed_text(self, query: str) -> torch.Tensor:
        """
        Encodes text description for suspect search
        """
        tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
        return (emb / emb.norm(
            dim=-1, keepdim=True
        )).squeeze(0)

    @staticmethod
    def cosine_sim(e1: torch.Tensor,
                   e2: torch.Tensor) -> float:
        return torch.cosine_similarity(
            e1.unsqueeze(0), e2.unsqueeze(0)
        ).item()
```

**Day 4 exit condition:**
```
✓ Embedding generates without error on any person crop
✓ Same person across 2 different frames → sim > 0.80
✓ Different people → sim < 0.70
✓ Text query returns meaningful similarity scores
✓ Inference time under 150ms on CPU
```

---

### Day 5 — Global Track Registry

```python
# tracking/global_registry.py

import time
from dataclasses import dataclass, field
from typing import Optional
import torch

@dataclass
class GlobalTrack:
    global_id:    int
    cam_id:       str
    local_id:     int
    timestamp:    float
    embedding:    torch.Tensor
    room_pos:     tuple           # (x_pct, y_pct) on floor plan
    velocity:     tuple = (0, 0)  # estimated m/s
    status:       str  = "active" # active|lost|reacquired
    trajectory:   list = field(default_factory=list)
    lost_since:   Optional[float] = None

    def mark_lost(self):
        self.status     = "lost"
        self.lost_since = time.time()

    def reacquire(self, cam_id, embedding,
                  room_pos, timestamp):
        self.cam_id    = cam_id
        self.embedding = embedding
        self.room_pos  = room_pos
        self.timestamp = timestamp
        self.status    = "reacquired"
        self.lost_since = None
        self.trajectory.append((room_pos, timestamp))


class GlobalRegistry:
    def __init__(self):
        self.tracks:  dict[int, GlobalTrack] = {}
        self._next_id = 1
        self.THRESHOLD     = 0.80
        self.MAX_TIME_GAP  = 120   # seconds
        self.MAX_SPEED_MPS = 8.0   # sprint cap

    def match_or_create(self,
                        embedding:  torch.Tensor,
                        cam_id:     str,
                        local_id:   int,
                        room_pos:   tuple,
                        timestamp:  float) -> tuple[int, str]:
        """
        Returns (global_id, event_type)
        event_type: 'new' | 'reacquired' | 'updated'
        """
        best_id, best_score = None, 0.0

        for gid, track in self.tracks.items():

            # Same camera — DeepSORT handles continuity
            if track.cam_id == cam_id:
                continue

            # Time feasibility check
            gap = timestamp - track.timestamp
            if gap > self.MAX_TIME_GAP or gap < 0:
                continue

            # Space feasibility check
            dist = self._room_dist(
                track.room_pos, room_pos
            )
            max_dist = gap * self.MAX_SPEED_MPS
            if dist > max_dist:
                continue

            sim = torch.cosine_similarity(
                embedding.unsqueeze(0),
                track.embedding.unsqueeze(0)
            ).item()

            if sim > best_score:
                best_score = sim
                best_id    = gid

        if best_score >= self.THRESHOLD and best_id:
            event = "reacquired" if \
                self.tracks[best_id].status == "lost" \
                else "updated"
            self.tracks[best_id].reacquire(
                cam_id, embedding, room_pos, timestamp
            )
            return best_id, event

        # New identity
        gid = self._next_id
        self._next_id += 1
        self.tracks[gid] = GlobalTrack(
            global_id  = gid,
            cam_id     = cam_id,
            local_id   = local_id,
            timestamp  = timestamp,
            embedding  = embedding,
            room_pos   = room_pos,
            trajectory = [(room_pos, timestamp)]
        )
        return gid, "new"

    def mark_lost_if_absent(self, active_cam_ids: list):
        """Call every 2 seconds to update lost status"""
        now = time.time()
        for track in self.tracks.values():
            if (track.status == "active" and
                    track.cam_id not in active_cam_ids):
                if now - track.timestamp > 3.0:
                    track.mark_lost()

    def _room_dist(self, p1: tuple, p2: tuple) -> float:
        """Euclidean distance in room percentage units"""
        return ((p1[0]-p2[0])**2 +
                (p1[1]-p2[1])**2) ** 0.5
```

---

### Days 6–7 — Uncertainty Radius + Prediction Engine

```python
# prediction/uncertainty.py

import time, math

class UncertaintyEngine:
    def __init__(self):
        self.DEFAULT_SPEED = 1.4   # m/s walking
        self.MAX_RADIUS    = 500   # metres
        self.DRONE_TRIGGER = 150   # metres

    def compute(self, lost_since: float,
                speed_mps: float = None) -> dict:
        if lost_since is None:
            return None
        speed   = speed_mps or self.DEFAULT_SPEED
        elapsed = time.time() - lost_since
        radius  = min(elapsed * speed, self.MAX_RADIUS)
        conf    = max(0.0, 1.0 - radius / self.MAX_RADIUS)

        return {
            "radius_m":       round(radius, 1),
            "elapsed_sec":    round(elapsed, 1),
            "confidence_pct": round(conf * 100, 1),
            "drone_trigger":  radius >= self.DRONE_TRIGGER,
            "status": "ZONE_EXCEEDED"
                      if radius >= self.MAX_RADIUS
                      else "ACTIVE_SEARCH"
        }


# prediction/graph_predictor.py

import networkx as nx

class CameraGraphPredictor:
    def __init__(self):
        self.G = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """
        Nodes = camera IDs
        Edge weight = avg observed travel time (seconds)
        Customize for your actual room layout
        """
        cameras = ["CAM_01", "CAM_02", "CAM_03"]
        self.G.add_nodes_from(cameras)
        self.G.add_weighted_edges_from([
            ("CAM_01", "CAM_02", 18),
            ("CAM_01", "CAM_03", 35),
            ("CAM_02", "CAM_01", 18),
            ("CAM_02", "CAM_03", 20),
            ("CAM_03", "CAM_02", 20),
            ("CAM_03", "CAM_01", 35),
        ])

    def predict_next(self,
                     current_cam: str) -> list[dict]:
        if current_cam not in self.G:
            return []

        neighbors = list(self.G.successors(current_cam))
        if not neighbors:
            return []

        weights = {
            n: self.G[current_cam][n]["weight"]
            for n in neighbors
        }
        total = sum(1/w for w in weights.values())
        predictions = [
            {
                "camera_id":   n,
                "probability": round((1/w) / total, 2),
                "eta_seconds": w
            }
            for n, w in weights.items()
        ]
        return sorted(
            predictions, key=lambda x: -x["probability"]
        )
```

**Days 6–7 exit condition:**
```
✓ Cross-camera match fires correctly on test footage
✓ False positive rate under 10% on 20 test sequences
✓ Uncertainty radius grows correctly over time
✓ Drone trigger fires at 150m threshold
✓ Probability bars show correct ranked predictions
```

---

## SPRINT 3 — Backend & API
### Days 8–9 | Owner: Teammate 2

---

### Day 8 — FastAPI Server + WebSocket

```python
# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio, asyncio, time

app = FastAPI()
sio = socketio.AsyncServer(
    async_mode="asgi", cors_allowed_origins="*"
)
socket_app = socketio.ASGIApp(sio, app)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"]
)

# ── Event types emitted to frontend ──────────────────
# "track_new"        → new identity detected
# "track_updated"    → same person, same camera
# "track_lost"       → left all camera zones
# "track_reacquired" → matched across cameras
# "uncertainty"      → radius update (every 500ms)
# "drone_dispatched" → drone trigger fired
# "text_search_result" → CLIP text search response

@app.post("/api/dispatch_drone")
async def dispatch_drone(payload: dict):
    global_id = payload.get("global_id")
    await sio.emit("drone_dispatched", {
        "global_id": global_id,
        "timestamp": time.time()
    })
    return {"status": "dispatched"}

@app.post("/api/text_search")
async def text_search(payload: dict):
    query = payload.get("query", "")
    # Calls CLIPEmbedder.embed_text + searches registry
    results = search_registry_by_text(query)
    await sio.emit("text_search_result", {
        "query":   query,
        "matches": results
    })
    return {"matches": results}

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
```

---

### Day 9 — Pipeline Orchestrator

```python
# backend/orchestrator.py
# Ties all components together into one running loop

import asyncio, time, cv2
from ingestion.stream_manager import CameraStream
from detection.detector       import ShadowDetector
from tracking.camera_tracker  import CameraTracker
from reid.embedder             import CLIPEmbedder
from tracking.global_registry import GlobalRegistry
from prediction.uncertainty   import UncertaintyEngine
from prediction.graph_predictor import CameraGraphPredictor

# Camera position map — (x%, y%) on floor plan SVG
CAM_POSITIONS = {
    "CAM_01": (0.12, 0.50),
    "CAM_02": (0.50, 0.50),
    "CAM_03": (0.88, 0.50),
}

class ShadowGridOrchestrator:
    def __init__(self, sio):
        self.sio        = sio
        self.detector   = ShadowDetector()
        self.embedder   = CLIPEmbedder()
        self.registry   = GlobalRegistry()
        self.uncertainty= UncertaintyEngine()
        self.predictor  = CameraGraphPredictor()

        self.streams = {
            "CAM_01": CameraStream(
                "CAM_01", "videos/cam1.mp4"
            ).start(),
            "CAM_02": CameraStream(
                "CAM_02", "videos/cam2.mp4"
            ).start(),
            "CAM_03": CameraStream(
                "CAM_03", "videos/cam3.mp4"
            ).start(),
        }
        self.trackers = {
            cam_id: CameraTracker(cam_id)
            for cam_id in self.streams
        }

    async def run(self):
        """Main loop — runs forever"""
        loop = asyncio.get_event_loop()
        while True:
            tasks = [
                loop.run_in_executor(
                    None, self._process_camera, cam_id
                )
                for cam_id in self.streams
            ]
            results = await asyncio.gather(*tasks)

            # Emit all events from this frame cycle
            for events in results:
                for event in events:
                    await self.sio.emit(
                        event["type"], event["data"]
                    )

            # Update lost tracks
            active = [
                r["cam_id"]
                for events in results
                for r in events
                if r.get("type") == "track_updated"
            ]
            self.registry.mark_lost_if_absent(active)

            # Emit uncertainty updates
            await self._emit_uncertainty()

            await asyncio.sleep(0.033)  # ~30 FPS target

    def _process_camera(self, cam_id: str) -> list:
        frame = self.streams[cam_id].get_frame()
        if frame is None:
            return []

        detections = self.detector.detect(frame)
        tracks     = self.trackers[cam_id].update(
            detections, frame
        )
        events = []

        for track in tracks:
            # Match crop to track bounding box
            lx,ly,lw,lh = [
                int(v) for v in track["bbox_ltwh"]
            ]
            crop = frame[ly:ly+lh, lx:lx+lw]
            emb  = self.embedder.embed_image(crop)
            if emb is None:
                continue

            gid, event_type = self.registry.match_or_create(
                embedding  = emb,
                cam_id     = cam_id,
                local_id   = track["local_id"],
                room_pos   = CAM_POSITIONS[cam_id],
                timestamp  = time.time()
            )

            events.append({
                "type": f"track_{event_type}",
                "data": {
                    "global_id":  gid,
                    "cam_id":     cam_id,
                    "room_pos":   CAM_POSITIONS[cam_id],
                    "bbox":       track["bbox_ltwh"],
                    "timestamp":  time.time(),
                    "predictions": self.predictor.predict_next(
                        cam_id
                    )
                }
            })
        return events

    async def _emit_uncertainty(self):
        for gid, track in self.registry.tracks.items():
            if track.status != "lost":
                continue
            data = self.uncertainty.compute(
                track.lost_since
            )
            if data:
                data["global_id"] = gid
                data["room_pos"]  = track.room_pos
                await self.sio.emit("uncertainty", data)
```

**Days 8–9 exit condition:**
```
✓ FastAPI server starts without errors
✓ WebSocket connection opens from browser
✓ All 5 event types emit correctly
✓ /api/text_search returns results
✓ /api/dispatch_drone triggers drone event
```

---

## SPRINT 4 — Frontend Dashboard
### Days 10–13 | Owner: Teammate 3

---

### Day 10 — Layout + Socket Connection

```jsx
// src/useSocket.js
import { useEffect, useState } from "react";
import { io } from "socket.io-client";

const socket = io("http://localhost:8000");

export function useSocket() {
  const [tracks,      setTracks]      = useState({});
  const [uncertainty, setUncertainty] = useState(null);
  const [alerts,      setAlerts]      = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [drone,       setDrone]       = useState(false);

  useEffect(() => {
    socket.on("track_new",        handleTrack);
    socket.on("track_updated",    handleTrack);
    socket.on("track_reacquired", handleReacquired);
    socket.on("track_lost",       handleLost);
    socket.on("uncertainty",      setUncertainty);
    socket.on("drone_dispatched", () => setDrone(true));
    socket.on("text_search_result", handleSearch);
    return () => socket.removeAllListeners();
  }, []);

  const handleTrack = (data) => {
    setTracks(prev => ({
      ...prev,
      [data.global_id]: {
        ...prev[data.global_id],
        ...data,
        trajectory: [
          ...(prev[data.global_id]?.trajectory || []),
          data.room_pos
        ]
      }
    }));
    if (data.predictions?.length) {
      setPredictions(data.predictions);
    }
  };

  const handleReacquired = (data) => {
    handleTrack(data);
    setAlerts(prev => [{
      type: "REACQUIRED",
      cam_id: data.cam_id,
      global_id: data.global_id,
      time: new Date().toLocaleTimeString()
    }, ...prev.slice(0,9)]);
  };

  const handleLost = (data) => {
    setTracks(prev => ({
      ...prev,
      [data.global_id]: {
        ...prev[data.global_id],
        status: "lost"
      }
    }));
  };

  const handleSearch = (data) => {
    setAlerts(prev => [{
      type: "SEARCH",
      query: data.query,
      matches: data.matches
    }, ...prev]);
  };

  const dispatchDrone = (global_id) => {
    fetch("/api/dispatch_drone", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ global_id })
    });
  };

  const textSearch = (query) => {
    fetch("/api/text_search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });
  };

  return {
    tracks, uncertainty, alerts,
    predictions, drone,
    dispatchDrone, textSearch
  };
}
```

---

### Days 11–12 — Room Map + All Animations

```jsx
// src/components/RoomMap.jsx
import { useEffect, useRef } from "react";

export default function RoomMap({
  tracks, uncertainty, cameras
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let animId;

    const W = canvas.width;
    const H = canvas.height;

    const animate = () => {
      ctx.clearRect(0, 0, W, H);

      // Floor plan grid lines
      ctx.strokeStyle = "rgba(255,255,255,0.04)";
      for (let x=0; x<W; x+=40) {
        ctx.beginPath();
        ctx.moveTo(x,0); ctx.lineTo(x,H);
        ctx.stroke();
      }
      for (let y=0; y<H; y+=40) {
        ctx.beginPath();
        ctx.moveTo(0,y); ctx.lineTo(W,y);
        ctx.stroke();
      }

      // Camera markers
      cameras.forEach(cam => {
        const cx = cam.x * W;
        const cy = cam.y * H;
        ctx.beginPath();
        ctx.arc(cx, cy, 10, 0, Math.PI*2);
        ctx.fillStyle = cam.active
          ? "rgba(34,197,94,0.3)" : "rgba(85,85,85,0.3)";
        ctx.fill();
        ctx.strokeStyle = cam.active ? "#22C55E" : "#555";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = "#AAAAAA";
        ctx.font = "10px monospace";
        ctx.fillText(cam.id, cx-20, cy+24);
      });

      // Trajectory lines per track
      Object.values(tracks).forEach(track => {
        if (!track.trajectory?.length) return;
        ctx.beginPath();
        ctx.strokeStyle = track.status === "lost"
          ? "rgba(204,0,0,0.4)" : "rgba(59,130,246,0.7)";
        ctx.lineWidth   = 2;
        ctx.setLineDash(track.status === "lost"
          ? [5,5] : []
        );
        track.trajectory.forEach(([px,py], i) => {
          const x = px * W, y = py * H;
          i === 0
            ? ctx.moveTo(x, y)
            : ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
      });

      // Uncertainty circle — pulsing
      if (uncertainty) {
        const cx = uncertainty.room_pos[0] * W;
        const cy = uncertainty.room_pos[1] * H;
        const r  = (uncertainty.radius_m / 500) *
                   Math.min(W,H) * 0.45;
        const pulse = r + Math.sin(Date.now()/300) * 6;

        // Outer glow
        const grad = ctx.createRadialGradient(
          cx,cy,r*0.5, cx,cy,pulse
        );
        grad.addColorStop(0, "rgba(204,0,0,0.15)");
        grad.addColorStop(1, "rgba(204,0,0,0)");
        ctx.beginPath();
        ctx.arc(cx, cy, pulse, 0, Math.PI*2);
        ctx.fillStyle = grad;
        ctx.fill();

        // Circle border
        ctx.beginPath();
        ctx.arc(cx, cy, pulse, 0, Math.PI*2);
        ctx.strokeStyle = "#CC0000";
        ctx.lineWidth   = 1.5;
        ctx.setLineDash([6,4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Radius label
        ctx.fillStyle = "#FF2222";
        ctx.font      = "11px monospace";
        ctx.fillText(
          `${uncertainty.radius_m}m`,
          cx+pulse+6, cy
        );
      }

      // Suspect dots
      Object.values(tracks).forEach(track => {
        if (!track.room_pos) return;
        const [px,py] = track.room_pos;
        const sx = px*W, sy = py*H;
        const isLost = track.status === "lost";

        // Glow
        const glow = ctx.createRadialGradient(
          sx,sy,2, sx,sy,18
        );
        glow.addColorStop(0,
          isLost ? "rgba(204,0,0,0.6)"
                 : "rgba(255,34,34,0.4)"
        );
        glow.addColorStop(1, "transparent");
        ctx.beginPath();
        ctx.arc(sx, sy, 18, 0, Math.PI*2);
        ctx.fillStyle = glow;
        ctx.fill();

        // Dot
        ctx.beginPath();
        ctx.arc(sx, sy, 7, 0, Math.PI*2);
        ctx.fillStyle = isLost ? "#555" : "#FF2222";
        ctx.fill();
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth   = 1.5;
        ctx.stroke();

        // Label
        ctx.fillStyle = "#FFFFFF";
        ctx.font      = "10px monospace";
        ctx.fillText(
          `#${track.global_id}`, sx+10, sy-10
        );
      });

      animId = requestAnimationFrame(animate);
    };

    animate();
    return () => cancelAnimationFrame(animId);
  }, [tracks, uncertainty, cameras]);

  return (
    <canvas
      ref={canvasRef}
      width={700} height={420}
      className="w-full h-full rounded-lg"
      style={{ background: "#0A0A0A" }}
    />
  );
}
```

---

### Day 13 — All Supporting Panels

```jsx
// Probability bars
function PredictionPanel({ predictions }) {
  return (
    <div className="space-y-2">
      <p className="text-xs text-gray-500 tracking-widest
                    font-bold uppercase mb-3">
        Next Predicted Camera
      </p>
      {predictions.map(p => (
        <div key={p.camera_id}>
          <div className="flex justify-between
                          text-xs mb-1">
            <span className="text-white font-mono">
              {p.camera_id}
            </span>
            <span className="text-red-400 font-mono">
              {Math.round(p.probability * 100)}%
            </span>
          </div>
          <div className="h-1.5 bg-gray-800 rounded">
            <div
              className="h-full bg-red-600 rounded
                         transition-all duration-700"
              style={{
                width: `${p.probability * 100}%`
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

// REACQUIRED banner
function ReacquiredBanner({ alerts }) {
  const latest = alerts.find(a => a.type==="REACQUIRED");
  if (!latest) return null;
  return (
    <div className="fixed inset-x-0 top-0 z-50
                    bg-green-500 text-black font-black
                    text-xl py-3 text-center
                    animate-in slide-in-from-top
                    duration-300">
      ✓ SUSPECT REACQUIRED — {latest.cam_id}
      — {latest.time}
    </div>
  );
}

// Uncertainty stats card
function UncertaintyPanel({ data, onDispatch }) {
  if (!data) return null;
  return (
    <div className="border border-red-900 rounded p-3
                    bg-black/40 space-y-2">
      <p className="text-red-500 text-xs font-bold
                    tracking-widest uppercase">
        Search Zone
      </p>
      <div className="grid grid-cols-2 gap-2">
        {[
          ["Radius",     `${data.radius_m}m`],
          ["Elapsed",    `${data.elapsed_sec}s`],
          ["Confidence", `${data.confidence_pct}%`],
          ["Status",     data.status],
        ].map(([k,v]) => (
          <div key={k} className="bg-gray-900 rounded p-2">
            <p className="text-gray-500 text-xs">{k}</p>
            <p className="text-white font-mono
                          text-sm font-bold">{v}</p>
          </div>
        ))}
      </div>
      {data.drone_trigger && (
        <button
          onClick={onDispatch}
          className="w-full bg-red-700 hover:bg-red-500
                     text-white font-bold py-2 text-sm
                     tracking-widest uppercase
                     transition-colors duration-200">
          DISPATCH DRONE
        </button>
      )}
    </div>
  );
}
```

---

## SPRINT 5 — Integration & Testing
### Days 14–16 | All Team Members

---

### Integration Checklist

```
Day 14 — Connect frontend to backend
  ✓ All WebSocket events rendering on dashboard
  ✓ Map animating correctly from real pipeline data
  ✓ Probability bars updating on each track event
  ✓ Uncertainty radius growing in sync with backend

Day 15 — End-to-end test on recorded video
  ✓ Run full demo sequence on cam1/cam2/cam3.mp4
  ✓ REACQUIRED fires at correct moment
  ✓ Text search returns correct crops
  ✓ Drone dispatch button triggers and animates
  ✓ Trajectory log updating correctly

Day 16 — Switch to live ESP32-CAM streams
  ✓ Replace video file paths with RTSP URLs
  ✓ All 3 streams stable for 30 minutes continuous
  ✓ No memory leak — RAM stable over time
  ✓ Cross-camera match fires on real hardware footage
  ✓ Tune THRESHOLD for real lighting conditions
```

---

### Known Issues & Fixes

```
Issue                         Fix
──────────────────────────────────────────────────────
CLIP slow on first inference  Pre-warm: run dummy embed
                              at startup before demo

Stream drops mid-demo         maxlen=2 queue + auto
                              reconnect in CameraStream

Memory grows over time        Clear embeddings older
                              than 5 min from registry

ReID false positives          Raise threshold to 0.83
                              if same-gender crowd

Frontend map lag              Throttle uncertainty
                              emit to every 500ms
                              not every frame
```

---

## SPRINT 6 — Demo Hardening
### Days 17–18 | All Team Members

---

### Day 17 — Backup System

```python
# One-line swap for demo fallback
# In orchestrator.py, change source:

# LIVE (demo day)
"CAM_01": CameraStream("CAM_01",
    "http://192.168.1.101/stream").start()

# FALLBACK (if hardware fails)
"CAM_01": CameraStream("CAM_01",
    "videos/cam1_prerecorded.mp4").start()
```

**Pre-record a perfect run of the full demo sequence the day before. Store as `demo_backup.mp4`. Keep it on your phone.**

---

### Day 18 — Final Freeze Checklist

```
CODE
  ✓ No debug print statements in production
  ✓ Error handling on every API endpoint
  ✓ Frontend shows graceful message if backend offline
  ✓ All secrets / IPs in .env file not hardcoded

PERFORMANCE
  ✓ End-to-end latency under 300ms measured
  ✓ 30 minute continuous run with no crashes
  ✓ Memory usage stable (check with htop)
  ✓ CPU under 80% with 3 streams running

DEMO SEQUENCE
  ✓ Full 6-minute run rehearsed 3 times
  ✓ Volunteer knows exact path
  ✓ Text search query pre-typed
  ✓ Drone positioned and tested
  ✓ Backup recording ready

DEPLOYMENT
  ✓ Backend starts with: uvicorn backend.main:socket_app
  ✓ Frontend built: npm run build, served via FastAPI
  ✓ Single command startup script written
  ✓ Laptop plugged in — not on battery
```

---

## File Structure — Complete Project

```
shadowgrid/
│
├── backend/
│   ├── main.py              # FastAPI + Socket.io
│   ├── orchestrator.py      # Main pipeline loop
│   └── routes/
│       ├── text_search.py
│       └── drone.py
│
├── detection/
│   └── detector.py          # YOLOv8 wrapper
│
├── tracking/
│   ├── camera_tracker.py    # DeepSORT per camera
│   └── global_registry.py  # Cross-camera identity
│
├── reid/
│   └── embedder.py          # CLIP embedding + search
│
├── prediction/
│   ├── uncertainty.py       # Radius engine
│   └── graph_predictor.py   # NetworkX camera graph
│
├── ingestion/
│   └── stream_manager.py    # Multi-stream reader
│
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── useSocket.js
│       └── components/
│           ├── RoomMap.jsx
│           ├── PredictionPanel.jsx
│           ├── UncertaintyPanel.jsx
│           ├── TrajectoryLog.jsx
│           ├── TextSearch.jsx
│           └── ReacquiredBanner.jsx
│
├── videos/                  # Test footage
│   ├── cam1.mp4
│   ├── cam2.mp4
│   └── cam3.mp4
│
├── .env                     # CAM IPs, thresholds
├── requirements.txt
└── start.sh                 # Single command launch
```

---

## Single Command Startup Script

```bash
#!/bin/bash
# start.sh — run this on demo day

echo "Starting ShadowGrid..."

# Start backend
uvicorn backend.main:socket_app \
  --host 0.0.0.0 --port 8000 &

# Serve frontend build
cd frontend && npx serve -s build -p 3000 &

echo "Dashboard: http://localhost:3000"
echo "API:       http://localhost:8000"
echo "Ready."
```