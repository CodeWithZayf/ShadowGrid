"""
config.py — Global configuration constants for the surveillance server.
All tuneable parameters live here; no magic numbers elsewhere.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CAMERAS_FILE    = BASE_DIR / "cameras.json"
MODELS_DIR      = BASE_DIR / "model_weights"

# .pt model weight files (place your trained weights here)
CRIME_MODEL_PATH   = MODELS_DIR / "crime_detect.pt"      # YOLOv8 fine-tuned for violence/crime
PERSON_DETECT_MODEL = MODELS_DIR / "yolov8n.pt"          # YOLOv8n for person detection
REID_MODEL_PATH    = MODELS_DIR / "osnet_x0_25.pt"       # OSNet Re-ID weights (placeholder until CLIP)
MIL_MODEL_PATH     = MODELS_DIR / "best_model.pt"        # MIL anomaly detection (trained on UCF-Crime)

# ── Camera / Stream settings ──────────────────────────────────────────────────
FRAME_WIDTH        = 640
FRAME_HEIGHT       = 480
TARGET_FPS         = 15
STREAM_JPEG_QUALITY = 80                                  # JPEG quality for /stream endpoint

# ── Per-camera frame deque ────────────────────────────────────────────────────
DEQUE_MAX_LEN      = 30                                   # ~2 s buffer @ 15 fps; oldest dropped

# ── Stream liveness ───────────────────────────────────────────────────────────
# A frame is considered stale if no new frame has been pushed within this
# many seconds. The consumer will not process stale frames.
# Set to (DEQUE_MAX_LEN / TARGET_FPS) * 2 as a safe upper bound — at 15fps
# the deque covers 2 s, so 4 s means the camera has been silent for 2x that.
FRAME_STALE_SECONDS = 4.0

# Frozen-stream detection: if this many consecutive frames are pixel-identical
# (same hash), the ingestor declares the stream frozen and reconnects.
# At 15fps, 15 identical frames = 1 second of a frozen feed.
FROZEN_FRAME_COUNT  = 15

# ── Crime / detection thresholds ─────────────────────────────────────────────
CRIME_CONFIDENCE   = 0.70                                 # Min confidence to flag as crime
MIL_ANOMALY_THRESHOLD = 0.0123                            # MIL anomaly score threshold (from eval F1=0.83)
PERSON_CONFIDENCE  = 0.50                                 # Min confidence for person detection

# ── Re-ID / suspect matching ──────────────────────────────────────────────────
REID_EMBED_DIM     = 512                                  # Embedding dimension (OSNet / CLIP)
SUSPECT_MATCH_THRESHOLD = 0.80                            # Cosine similarity ≥ this → match
REID_BATCH_SIZE    = 8                                    # Crops sent to Re-ID model at once

# ── Tracking ─────────────────────────────────────────────────────────────────
MIN_TRACK_FRAMES   = 3                                    # DeepSORT n_init — confirm after N frames

# ── Occurrence dedup window ───────────────────────────────────────────────────
OCCURRENCE_DEDUP_SECONDS = 60     # Ignore same car on same cam within this window

# ── Suspect dedup + crime cooldown ────────────────────────────────────────────
# After a camera fires a crime and adds suspects, suppress further suspect
# creation from that same camera for this many seconds.  This prevents the
# same ongoing scene from flooding the suspects list with duplicate entries.
CRIME_COOLDOWN_SECONDS = 30       # Minimum gap between suspect-creation events per camera

# Two detections from the same camera are considered the "same" suspect if
# their Re-ID embeddings are closer than this cosine similarity threshold.
# Prevents near-duplicate crops (e.g. slightly shifted bbox on next frame)
# from creating redundant suspects even within the cooldown window.
SUSPECT_DEDUP_SIMILARITY = 0.92   # Higher than SUSPECT_MATCH_THRESHOLD (0.70) — very strict

# ── Bounded in-memory lists (drop-oldest policy) ──────────────────────────────
MAX_SUSPECTS     = 500    # Hard cap on suspects list length; oldest entry evicted on overflow
MAX_OCCURRENCES  = 2000   # Hard cap on occurrences list length; oldest entry evicted on overflow

# ── Query pagination defaults ─────────────────────────────────────────────────
QUERY_DEFAULT_LIMIT = 100   # Default max items returned when ?limit is not specified
QUERY_MAX_LIMIT     = 1000  # Absolute ceiling a caller may request via ?limit

# ── Priority decay ────────────────────────────────────────────────────────────
PRIORITY_CRIME_BOOST      = 100.0                        # Boost when crime detected nearby
PRIORITY_OCCURRENCE_BOOST = 50.0                         # Boost when occurrence confirmed
PRIORITY_DECAY_RATE       = 0.05                         # Fraction lost per second (exponential)
PRIORITY_GEO_MAX_KM       = 10.0                         # Radius within which neighbours boosted
PRIORITY_GEO_FALLOFF      = 2.0                          # km — gaussian std for geo boost

# ── Consumer / scheduler ─────────────────────────────────────────────────────
CONSUMER_POLL_INTERVAL  = 0.01                           # seconds between round-robin polls
CONSUMER_SKIP_EMPTY_MS  = 5                              # ms to sleep when all queues empty

# ── Server ────────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
