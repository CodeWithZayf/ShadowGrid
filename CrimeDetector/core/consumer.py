"""
core/consumer.py — Frame consumer with round-robin scheduling.

One daemon thread continuously cycles through all camera queues in priority
order (highest → lowest), pulls the latest frame, and runs crime detection.

On crime detection:
  1. Checks a per-camera cooldown (CRIME_COOLDOWN_SECONDS). If the camera
     handled a crime event recently and created suspects, the frame is skipped.
  2. Crops the frame using the crime detector's own bounding boxes.
  3. Batch-embeds all crops. Within the SAME crime event, deduplicates crops
     against each other (not against historical suspects) — so if the detector
     fires two overlapping boxes on the same object, only one suspect is made.
  4. Creates one SuspectEntry per unique crop. Stamps the cooldown only if at
     least one suspect was actually created.
  5. Boosts camera priorities and spawns one tracking thread.

Public interface
----------------
start_consumer() -> None
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    CONSUMER_POLL_INTERVAL,
    CONSUMER_SKIP_EMPTY_MS,
    CRIME_COOLDOWN_SECONDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MIL_ANOMALY_THRESHOLD,
    SUSPECT_DEDUP_SIMILARITY,
)
from core import priority_manager
from core.state import (
    SuspectEntry,
    add_suspect,
    get_latest_frame,
)
from models.crime_detector import CrimeDetectionResult, detect_crime
from models.model_manager import extract_embeddings_batch

logger = logging.getLogger(__name__)

# (x1, y1, x2, y2, conf, class_id)
BBox = Tuple[int, int, int, int, float, int]

# Minimum crop side length in pixels — smaller boxes are noise
_MIN_CROP_PX = 20

# Per-camera cooldown: camera_id → Unix timestamp of last successful suspect creation.
# Only written when suspects are actually created (entries_created > 0).
# Accessed only from the single consumer thread — no lock needed.
_last_crime_ts: Dict[str, float] = {}

# Per-camera MIL frame buffers for temporal anomaly scoring
_mil_buffers: Dict[str, 'MILFrameBuffer'] = {}


def _on_cooldown(camera_id: str, now: float) -> bool:
    """Return True if this camera created suspects recently and is still cooling down."""
    last = _last_crime_ts.get(camera_id, 0.0)
    return (now - last) < CRIME_COOLDOWN_SECONDS


def _safe_crop(
    frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> Optional[np.ndarray]:
    """Return a clamped crop, or None if the region is too small."""
    h, w = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if (x2 - x1) < _MIN_CROP_PX or (y2 - y1) < _MIN_CROP_PX:
        return None
    return frame[y1:y2, x1:x2]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    return float(np.dot(a, b))


def _deduplicate_within_event(
    embeddings: np.ndarray,
) -> List[int]:
    """
    Given N embeddings from a single crime event, return the indices of
    embeddings that are NOT near-duplicates of any earlier embedding in the
    same batch.

    This handles the case where the detector fires multiple overlapping bboxes
    on the same physical object in a single frame (e.g. a knife detected twice
    at slightly different coordinates). We only keep the first occurrence of
    each unique crop within this event.

    We do NOT compare against historical suspects — that comparison caused the
    permanent lock-out bug (same scene always > 0.92 similarity to prior event).
    """
    kept: List[int] = []
    kept_embeddings: List[np.ndarray] = []

    for i, emb in enumerate(embeddings):
        is_dup = False
        for prev_emb in kept_embeddings:
            if _cosine_sim(emb, prev_emb) >= SUSPECT_DEDUP_SIMILARITY:
                is_dup = True
                break
        if not is_dup:
            kept.append(i)
            kept_embeddings.append(emb)

    return kept


def _handle_crime(
    camera_id: str,
    frame: np.ndarray,
    crime_result: CrimeDetectionResult,
    now: float,
) -> int:
    """
    Process a crime event: crop → embed → dedup within event → add suspects.
    Returns the number of suspects actually created.
    """
    from core.tracking_module import start_tracking_thread  # deferred — avoids circular import

    detections: List[BBox] = crime_result.detections

    # ── Build valid crops from crime bboxes ──────────────────────────────────
    valid_crops: List[np.ndarray] = []
    valid_boxes: List[BBox] = []

    for bbox in detections:
        x1, y1, x2, y2, conf, cls = bbox
        crop = _safe_crop(frame, x1, y1, x2, y2)
        if crop is not None:
            valid_crops.append(crop)
            valid_boxes.append(bbox)

    if not valid_crops:
        # All bboxes were degenerate — embed the full frame as fallback
        logger.warning(
            "Crime on cam=%s: all %d bbox(es) degenerate, using full-frame fallback.",
            camera_id, len(detections),
        )
        valid_crops = [frame]
        valid_boxes = [(0, 0, FRAME_WIDTH, FRAME_HEIGHT, 1.0, -1)]

    # ── Batch embed all crops in one forward pass ─────────────────────────────
    embeddings = extract_embeddings_batch(valid_crops)   # shape (N, D)

    # ── Dedup within this event only ─────────────────────────────────────────
    # Removes overlapping bboxes on the same object. Does NOT touch historical
    # suspects — comparing against historical suspects caused permanent lock-out.
    unique_indices = _deduplicate_within_event(embeddings)

    logger.debug(
        "Crime on cam=%s: %d detections → %d crops → %d unique after intra-event dedup.",
        camera_id, len(detections), len(valid_crops), len(unique_indices),
    )

    # ── Create one SuspectEntry per unique crop ───────────────────────────────
    entries_created = 0

    for rank, idx in enumerate(unique_indices):
        bbox = valid_boxes[idx]
        emb  = embeddings[idx]
        x1, y1, x2, y2, conf, cls = bbox

        suspect_id = str(uuid.uuid4())
        entry = SuspectEntry(
            suspect_id=suspect_id,
            camera_id=camera_id,
            car_id=rank,        # rank within this event; tracker assigns real IDs later
            timestamp=now,
            embedding=emb.copy(),
            source="crime_detection",
        )
        add_suspect(entry)
        entries_created += 1

        logger.info(
            "Suspect added: id=%s cam=%s bbox=(%d,%d,%d,%d) conf=%.2f",
            suspect_id, camera_id, x1, y1, x2, y2, conf,
        )

    # ── Side effects ──────────────────────────────────────────────────────────
    priority_manager.boost_on_crime(camera_id)
    start_tracking_thread(camera_id)

    logger.info(
        "Crime event on cam=%s: %d suspect(s) created from %d detection(s).",
        camera_id, entries_created, len(detections),
    )
    return entries_created


def _consumer_loop() -> None:
    """Main consumer loop — runs forever as a daemon thread."""
    logger.info("Consumer loop started.")
    while True:
        ordered_cams = priority_manager.get_priority_order()
        had_frame = False

        for camera_id in ordered_cams:
            frame = get_latest_frame(camera_id)
            if frame is None:
                continue

            had_frame = True

            try:
                now = time.time()
                result = detect_crime(frame)

                # ── MIL Anomaly Scoring (temporal) ──
                mil_triggered = False
                mil_score = 0.0
                try:
                    from training.inference_mil import is_mil_available, MILFrameBuffer
                    if is_mil_available():
                        if camera_id not in _mil_buffers:
                            _mil_buffers[camera_id] = MILFrameBuffer()
                        buf = _mil_buffers[camera_id]
                        buf.add_frame(frame)
                        if buf.is_ready():
                            mil_score = buf.get_score()
                            if mil_score >= MIL_ANOMALY_THRESHOLD:
                                mil_triggered = True
                                logger.info(
                                    "MIL anomaly on cam=%s: score=%.4f (threshold=%.4f)",
                                    camera_id, mil_score, MIL_ANOMALY_THRESHOLD,
                                )
                except Exception as mil_exc:
                    logger.debug("MIL scoring error on cam=%s: %s", camera_id, mil_exc)

                # ── Crime response: YOLO detection OR MIL anomaly ──
                crime_signal = result.crime_detected or mil_triggered

                if crime_signal:
                    if _on_cooldown(camera_id, now):
                        remaining = CRIME_COOLDOWN_SECONDS - (now - _last_crime_ts.get(camera_id, 0))
                        logger.debug(
                            "Crime on cam=%s suppressed — cooldown %.1fs remaining.",
                            camera_id, remaining,
                        )
                    else:
                        # Not on cooldown — handle the event.
                        source = []
                        if result.crime_detected:
                            source.append("YOLO")
                        if mil_triggered:
                            source.append(f"MIL({mil_score:.4f})")
                        logger.info(
                            "Crime triggered on cam=%s by: %s",
                            camera_id, " + ".join(source),
                        )

                        created = _handle_crime(camera_id, frame, result, now)
                        if created > 0:
                            _last_crime_ts[camera_id] = now
                            # Reset MIL buffer after confirmed event
                            if camera_id in _mil_buffers:
                                _mil_buffers[camera_id].clear()

            except Exception as exc:
                logger.exception("Error in consumer for cam=%s: %s", camera_id, exc)

            time.sleep(CONSUMER_POLL_INTERVAL)

        if not had_frame:
            time.sleep(CONSUMER_SKIP_EMPTY_MS / 1000.0)


def start_consumer() -> None:
    """Launch the consumer loop as a single background daemon thread."""
    t = threading.Thread(
        target=_consumer_loop,
        name="frame-consumer",
        daemon=True,
    )
    t.start()
    logger.info("Frame consumer daemon started.")
