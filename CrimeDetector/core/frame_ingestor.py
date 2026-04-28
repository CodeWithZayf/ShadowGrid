"""
core/frame_ingestor.py — One non-blocking daemon thread per camera.

Each thread opens the IP Webcam MJPEG stream, decodes incoming JPEG frames,
and pushes them into the camera's deque via state.push_frame().

Disconnection handling
----------------------
Two failure modes are defended against:

1. Hard failure — cap.read() returns ret=False.
   OpenCV's FFmpeg backend may return a few False reads before giving up fully,
   so we wait for FAIL_THRESHOLD consecutive failures before reconnecting.

2. Frozen stream — cap.read() keeps returning ret=True but the pixel content
   is identical frame after frame. This happens when the IP Webcam app is
   still TCP-connected but the Android camera has paused or the phone screen
   is off. We detect this by comparing a cheap 8-byte perceptual hash of
   successive frames. After FROZEN_FRAME_COUNT identical hashes we declare
   the stream frozen, clear the deque (so get_latest_frame() returns None
   immediately), and force a reconnect.

In both cases, clear_frame_queue() is called before reconnecting so that the
consumer never processes a stale or frozen frame.

Public interface
----------------
start_ingestors(cameras: List[CameraInfo]) -> None
    Launch one daemon thread per registered camera.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import List, Optional

import cv2
import numpy as np

from config import FRAME_HEIGHT, FRAME_WIDTH, FROZEN_FRAME_COUNT
from core.state import CameraInfo, clear_frame_queue, push_frame

logger = logging.getLogger(__name__)

_RECONNECT_DELAY    = 3.0   # seconds to wait before reconnecting after any failure
_FAIL_THRESHOLD     = 10    # consecutive cap.read() failures before declaring dead


def _frame_hash(frame: np.ndarray) -> bytes:
    """
    Compute a fast 8-byte hash of a frame for frozen-stream detection.
    We downsample to 16×16 greyscale before hashing so the comparison is
    cheap (~1 µs) and robust to minor JPEG compression variation.
    """
    small = cv2.resize(frame, (16, 16), interpolation=cv2.INTER_AREA)
    grey  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # MD5 is fast; we only need 8 bytes for collision-resistance at this scale
    return hashlib.md5(grey.tobytes(), usedforsecurity=False).digest()[:8]


def _ingest_loop(cam: CameraInfo) -> None:
    """
    Continuously read frames from an MJPEG stream.
    On hard failure or frozen stream: clear the deque, wait, then reconnect.
    Runs forever — designed to be a daemon thread.
    """
    url    = cam.url
    cam_id = cam.id

    while True:
        logger.info("[%s] Connecting to stream: %s", cam_id, url)
        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            logger.warning("[%s] Cannot open stream; retrying in %.1fs",
                           cam_id, _RECONNECT_DELAY)
            clear_frame_queue(cam_id)
            time.sleep(_RECONNECT_DELAY)
            continue

        logger.info("[%s] Stream opened.", cam_id)

        consecutive_failures = 0
        frozen_run           = 0
        last_hash: Optional[bytes] = None
        stream_alive         = True

        while stream_alive:
            ret, frame = cap.read()

            # ── Hard failure path ─────────────────────────────────────────────
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= _FAIL_THRESHOLD:
                    logger.warning(
                        "[%s] Stream lost after %d consecutive failures — reconnecting.",
                        cam_id, consecutive_failures,
                    )
                    stream_alive = False
                time.sleep(0.05)
                continue

            consecutive_failures = 0

            # ── Frozen-stream detection ───────────────────────────────────────
            fh = _frame_hash(frame)
            if fh == last_hash:
                frozen_run += 1
                if frozen_run >= FROZEN_FRAME_COUNT:
                    logger.warning(
                        "[%s] Frozen stream detected (%d identical frames) — reconnecting.",
                        cam_id, frozen_run,
                    )
                    stream_alive = False
                    continue
            else:
                frozen_run = 0
                last_hash  = fh

            # ── Good live frame ───────────────────────────────────────────────
            if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)

            push_frame(cam_id, frame)

        # ── Stream is dead — clear deque immediately so consumer gets None ────
        cap.release()
        clear_frame_queue(cam_id)
        logger.info("[%s] Deque cleared. Reconnecting in %.1fs.",
                    cam_id, _RECONNECT_DELAY)
        time.sleep(_RECONNECT_DELAY)


def start_ingestors(cameras: List[CameraInfo]) -> None:
    """Launch one daemon thread per camera to continuously pull frames."""
    for cam in cameras:
        t = threading.Thread(
            target=_ingest_loop,
            args=(cam,),
            name=f"ingestor-{cam.id}",
            daemon=True,
        )
        t.start()
        logger.info("Started ingestor thread for camera '%s'.", cam.id)
