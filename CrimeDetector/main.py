"""
main.py — FastAPI application entry point.

Startup sequence
----------------
1. Load cameras from cameras.json → register them in state.
2. Load all AI models into memory (YOLO + Re-ID).
3. Start one frame-ingestor daemon thread per camera.
4. Start the frame-consumer daemon thread.
5. Mount all API routers.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import HOST, LOG_LEVEL, PORT
from core.camera_loader import load_cameras
from core.consumer import start_consumer
from core.frame_ingestor import start_ingestors
from models.model_manager import load_all_models
from routers import cameras, occurrences, suspects

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks; yield; run shutdown tasks (none needed here)."""
    logger.info("=== Surveillance Server starting up ===")

    # 1. Camera registry
    logger.info("Loading cameras...")
    registered_cameras = load_cameras()
    logger.info("%d camera(s) registered.", len(registered_cameras))

    # 2. AI models
    logger.info("Loading AI models (this may take a moment)...")
    load_all_models()
    logger.info("All models loaded.")

    # 3. Frame ingestors (one thread per camera)
    logger.info("Starting frame ingestors...")
    start_ingestors(registered_cameras)

    # 4. Consumer
    logger.info("Starting frame consumer...")
    start_consumer()

    logger.info("=== Server ready. Listening on %s:%d ===", HOST, PORT)
    yield
    logger.info("=== Server shutting down. ===")


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Surveillance Server",
    description=(
        "Receives IP Webcam MJPEG streams, runs person detection and "
        "re-identification, tracks suspects across multiple cameras, and "
        "exposes REST APIs for suspects, occurrences, and live feeds."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allowed frontend origins — restrict in production
_ALLOWED_ORIGINS = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:3000",   # Production build
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────────────────────
app.include_router(cameras.router)
app.include_router(suspects.router)
app.include_router(occurrences.router)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "AI Surveillance Server"}


@app.get("/health", tags=["Health"])
def health():
    from core.state import camera_registry, frame_queues, occurrences, suspects, is_camera_live
    return {
        "status": "ok",
        "cameras_registered": len(camera_registry),
        "suspects_count": len(suspects),
        "occurrences_count": len(occurrences),
        "queues": {
            cam_id: len(q) for cam_id, q in frame_queues.items()
        },
        "cameras_live": {
            cam_id: is_camera_live(cam_id) for cam_id in camera_registry
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
        reload=False,
    )
