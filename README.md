# ShadowGrid

A Real-Time Multi-Camera Suspect Tracking and Crime Detection System Using Computer Vision

## Project Overview

ShadowGrid is an AI-powered surveillance platform that performs real-time crime detection, person tracking, and cross-camera re-identification across multiple IP camera feeds. It combines YOLOv8-based detection, anomaly scoring (MIL), and an interactive React dashboard with live map visualization.

## Features

- **Real-Time Crime Detection** — YOLOv8 fine-tuned for violence/crime events
- **MIL Anomaly Scoring** — Temporal anomaly detection using Multiple Instance Learning on frame sequences
- **Multi-Camera Ingestion** — Concurrent MJPEG stream readers with auto-reconnect and frozen-frame detection
- **Cross-Camera Re-ID** — Suspect matching via cosine-similarity on Re-ID embeddings
- **Priority Scheduling** — Geo-aware camera priority with exponential decay
- **Live Dashboard** — React + Leaflet map with camera markers, suspect overlays, and alert toasts
- **REST API** — Full CRUD for cameras, suspects, and occurrences with delta polling support
- **MJPEG Streaming** — Browser-compatible live video feeds per camera

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **Detection** | YOLOv8 (Ultralytics), MIL Anomaly Model |
| **Re-ID** | OSNet x0_25 (torchreid) |
| **Tracking** | DeepSORT (person tracking) |
| **Frontend** | React 18, Vite 5, Leaflet.js |
| **ML Framework** | PyTorch, torchvision |
| **CV** | OpenCV (headless) |

## Project Structure

```
ShadowGrid/
├── CrimeDetector/                # Python backend
│   ├── main.py                   # FastAPI entry point
│   ├── config.py                 # Centralized configuration
│   ├── cameras.json              # Camera IP/location config
│   ├── requirements.txt          # Python dependencies
│   ├── core/                     # Pipeline core
│   │   ├── state.py              # Shared state (thread-safe registries)
│   │   ├── consumer.py           # Frame consumer + crime detection loop
│   │   ├── frame_ingestor.py     # Per-camera MJPEG reader threads
│   │   ├── tracking_module.py    # Cross-camera tracking coordinator
│   │   └── priority_manager.py   # Camera priority scoring
│   ├── models/                   # AI models
│   │   ├── model_manager.py      # Singleton model loader
│   │   ├── crime_detector.py     # YOLOv8 crime detection
│   │   └── suspect_matcher.py    # Cosine similarity matching
│   ├── detection/                # Person detection (Phase 1)
│   ├── reid/                     # CLIP ReID engine (Phase 2)
│   ├── prediction/               # Uncertainty + graph prediction (Phase 3)
│   ├── routers/                  # FastAPI route handlers
│   │   ├── cameras.py            # Camera endpoints
│   │   ├── suspects.py           # Suspect CRUD + image upload
│   │   └── occurrences.py        # Occurrence query endpoints
│   ├── training/                 # MIL anomaly model training
│   │   ├── model.py              # AnomalyMILModel architecture
│   │   ├── train.py              # Training script
│   │   ├── evaluate.py           # Evaluation + metrics
│   │   └── inference_mil.py      # Runtime inference wrapper
│   └── model_weights/            # Trained model files (gitignored)
├── UI/                           # React frontend
│   ├── src/
│   │   ├── App.jsx               # Root layout
│   │   ├── store/AppContext.jsx   # Central state + polling
│   │   ├── api/crimedetector.js  # Backend API client
│   │   ├── components/           # UI components
│   │   └── utils/                # Formatting helpers
│   └── package.json
├── Implementation_plan.md        # Target architecture spec
└── report.md                     # Implementation audit report
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- NVIDIA GPU (recommended for real-time inference)

### Backend
```bash
cd CrimeDetector
pip install -r requirements.txt
cp cameras.json.example cameras.json
# Edit cameras.json with your camera IPs

python main.py
# Server starts at http://localhost:8000
```

### Frontend
```bash
cd UI
npm install
npm run dev
# Dashboard at http://localhost:5173
```

### Camera Setup
Cameras are configured in `CrimeDetector/cameras.json`:
```json
[
  {
    "id": "cam_01",
    "url": "http://<CAMERA_IP>:8080/video",
    "metadata": {
      "location_name": "Front Gate",
      "latitude": 22.5726,
      "longitude": 88.3639
    }
  }
]
```
Use any MJPEG-compatible source (IP Webcam app, RTSP cameras, etc).

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server health + camera status |
| `GET` | `/cameras` | List all cameras |
| `GET` | `/cameras/{id}/frame` | Latest JPEG frame |
| `GET` | `/cameras/{id}/stream` | MJPEG live stream |
| `GET` | `/suspects` | List suspects (supports delta polling) |
| `POST` | `/suspects/upload` | Add suspect from image |
| `POST` | `/suspects/description` | Add suspect from text |
| `DELETE` | `/suspects/{id}` | Remove suspect |
| `GET` | `/occurrences` | List occurrences |
