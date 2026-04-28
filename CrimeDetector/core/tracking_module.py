"""
core/tracking_module.py — Cross-camera suspect tracking using person detection.

When a crime is detected on any camera, start_tracking_thread() is called.
It submits a task to a bounded ThreadPoolExecutor that:
  1. Scans all cameras (priority-ordered) for person detections.
  2. Uses ShadowDetector (YOLOv8 person) + CameraTracker (DeepSORT) to get
     stable per-camera person tracks.
  3. Embeds each confirmed track's crop with the Re-ID model.
  4. Matches the embedding against all suspects via cosine similarity.
  5. On a match, records an OccurrenceEntry and boosts camera priority.

Public interface
----------------
start_tracking_thread(trigger_camera_id: str) -> None
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np

from config import (
    OCCURRENCE_DEDUP_SECONDS,
    SUSPECT_MATCH_THRESHOLD,
)
from core import priority_manager
from core.state import (
    OccurrenceEntry,
    add_occurrence,
    camera_registry,
    get_latest_frame,
    occurrences,
    occurrences_lock,
)
from models.suspect_matcher import find_best_match
from models.model_manager import extract_embeddings_batch

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded singletons for detector + per-camera trackers
# ─────────────────────────────────────────────────────────────────────────────

_detector = None          # ShadowDetector (loaded once)
_trackers: Dict[str, object] = {}   # cam_id → CameraTracker
_init_lock = threading.Lock()


def _ensure_detector():
    """Lazy-load the ShadowDetector on first use."""
    global _detector
    if _detector is not None:
        return _detector

    with _init_lock:
        if _detector is not None:
            return _detector
        try:
            from detection.detector import ShadowDetector
            _detector = ShadowDetector()
            logger.info("ShadowDetector loaded for tracking module.")
        except Exception as exc:
            logger.error("Failed to load ShadowDetector: %s", exc)
            _detector = None
    return _detector


def _get_tracker(cam_id: str):
    """Get or create a CameraTracker for the given camera."""
    if cam_id in _trackers:
        return _trackers[cam_id]

    with _init_lock:
        if cam_id in _trackers:
            return _trackers[cam_id]
        try:
            from tracking.camera_tracker import CameraTracker
            tracker = CameraTracker(cam_id=cam_id, max_age=30)
            _trackers[cam_id] = tracker
            logger.info("CameraTracker created for cam=%s", cam_id)
            return tracker
        except Exception as exc:
            logger.error("Failed to create CameraTracker for cam=%s: %s", cam_id, exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Thread pool & per-camera locks
# ─────────────────────────────────────────────────────────────────────────────

# Bounded thread pool — prevents unbounded thread spawning (BUG-02 fix)
_tracking_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tracker")

# Per-camera processing lock — only one tracking thread runs per camera at once
_camera_locks: dict[str, threading.Lock] = {}
_camera_locks_meta = threading.Lock()


def _get_camera_lock(camera_id: str) -> threading.Lock:
    with _camera_locks_meta:
        if camera_id not in _camera_locks:
            _camera_locks[camera_id] = threading.Lock()
        return _camera_locks[camera_id]


# ─────────────────────────────────────────────────────────────────────────────
# Dedup helper
# ─────────────────────────────────────────────────────────────────────────────

def _was_recently_seen(camera_id: str, track_id: int, now: float) -> bool:
    """
    Return True if track_id on camera_id already has an occurrence entry within
    the dedup window, to avoid flooding the occurrences list.
    """
    with occurrences_lock:
        for occ in occurrences:
            if (occ.camera_id == camera_id
                    and occ.car_id == track_id
                    and (now - occ.timestamp) < OCCURRENCE_DEDUP_SECONDS):
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main tracking logic
# ─────────────────────────────────────────────────────────────────────────────

def _process_camera(camera_id: str) -> None:
    """Run one cycle of detect → track → embed → match on a single camera.

    Pipeline:
    1. Get latest frame from the camera's frame queue.
    2. Run ShadowDetector.detect() to find person bounding boxes.
    3. Feed detections to CameraTracker.update() for DeepSORT tracking.
    4. For each confirmed track, extract Re-ID embedding from the crop.
    5. Match embedding against all suspects; create OccurrenceEntry on match.
    """
    cam_lock = _get_camera_lock(camera_id)
    if not cam_lock.acquire(blocking=False):
        return  # Another tracking thread is already on this camera

    try:
        frame = get_latest_frame(camera_id)
        if frame is None:
            return

        # 1. Detect persons
        detector = _ensure_detector()
        if detector is None:
            return

        detections = detector.detect(frame)
        if not detections:
            # Still update tracker with empty detections (ages out old tracks)
            tracker = _get_tracker(camera_id)
            if tracker:
                tracker.update([], frame)
            return

        # 2. Track with DeepSORT
        tracker = _get_tracker(camera_id)
        if tracker is None:
            return

        confirmed_tracks = tracker.update(detections, frame)
        if not confirmed_tracks:
            return

        # 3. Embed confirmed track crops
        crops = [t["crop"] for t in confirmed_tracks]
        embeddings = extract_embeddings_batch(crops)  # shape (N, 512)

        now = time.time()

        # 4. Match each track's embedding against suspects
        for track_info, embedding in zip(confirmed_tracks, embeddings):
            local_id = track_info["local_id"]

            # Dedup check — don't create duplicate occurrences
            if _was_recently_seen(camera_id, local_id, now):
                continue

            # Find best matching suspect
            match_result = find_best_match(embedding)
            if match_result is None:
                continue

            matched_suspect, similarity = match_result
            if similarity < SUSPECT_MATCH_THRESHOLD:
                continue

            # 5. Record occurrence
            occ = OccurrenceEntry(
                occurrence_id=str(uuid.uuid4()),
                suspect_id=matched_suspect.suspect_id,
                camera_id=camera_id,
                car_id=local_id,  # reusing car_id field for track ID
                embedding=embedding.copy(),
                similarity=similarity,
                timestamp=now,
            )
            add_occurrence(occ)
            priority_manager.boost_on_occurrence(camera_id)

            logger.info(
                "Track match: cam=%s track=%d → suspect=%s (sim=%.3f)",
                camera_id, local_id, matched_suspect.suspect_id[:8], similarity,
            )

    except Exception as exc:
        logger.exception("Error processing cam=%s: %s", camera_id, exc)
    finally:
        cam_lock.release()


def _tracking_loop(trigger_camera_id: str) -> None:
    """
    Process all cameras (priority-ordered) looking for suspect matches.
    Runs for MAX_PASSES cycles then exits to keep thread count bounded.
    """
    MAX_PASSES = 30
    PASS_SLEEP = 0.1   # seconds between passes

    for _ in range(MAX_PASSES):
        ordered = priority_manager.get_priority_order()
        for cam_id in ordered:
            try:
                _process_camera(cam_id)
            except Exception as exc:
                logger.exception("Tracking error on cam=%s: %s", cam_id, exc)
        time.sleep(PASS_SLEEP)


def start_tracking_thread(trigger_camera_id: str) -> None:
    """
    Submit a tracking task to the bounded thread pool, triggered by a
    crime event on trigger_camera_id.

    Uses ThreadPoolExecutor to prevent unbounded thread spawning.
    """
    _tracking_pool.submit(_tracking_loop, trigger_camera_id)
    logger.debug("Tracking task submitted (trigger_cam=%s).", trigger_camera_id)
