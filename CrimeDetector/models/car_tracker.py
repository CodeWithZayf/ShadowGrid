"""
models/car_tracker.py — Per-camera car detection, tracking, and embedding.

Uses YOLOv8 with the ByteTrack tracker to:
  1. Detect all vehicles in a frame.
  2. Assign stable track IDs across frames.
  3. Accumulate raw BGR crops per track and average their Re-ID embeddings
     once MIN_TRACK_FRAMES have been seen (averaged embedding is more robust).

IMPORTANT — embedding is returned on EVERY frame, not only after MIN_TRACK_FRAMES:
  • Before MIN_TRACK_FRAMES: a single-frame embedding from the current crop.
  • After  MIN_TRACK_FRAMES: the averaged, L2-normalised multi-frame embedding.
  • `is_new_embedding` flips to True on the frame the averaged embedding first
    becomes ready (useful for triggering downstream actions only once).

This ensures the tracking module can do suspect matching immediately without
waiting N frames for a track to mature.

Each CarTracker instance is owned by one camera (consumer or tracking module).

Public interface (per instance)
--------------------------------
tracker = CarTracker(camera_id)
results  = tracker.update(frame)   # List[TrackResult]
tracker.reset()                    # clear all accumulated state
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    CAR_CLASSES,
    CAR_CONFIDENCE,
    MIN_TRACK_FRAMES,
    REID_EMBED_DIM,
    TRACKER_TYPE,
)
from models.model_manager import extract_embedding, extract_embeddings_batch

logger = logging.getLogger(__name__)

# (x1, y1, x2, y2)
BBox = Tuple[int, int, int, int]

# Minimum crop dimension to bother embedding
_MIN_CROP_PX = 20


@dataclass
class TrackResult:
    """Output of the tracker for one tracked vehicle in one frame."""
    track_id: int
    bbox: BBox                              # (x1, y1, x2, y2) in pixel coords
    confidence: float
    class_id: int
    embedding: np.ndarray                   # Always set — single-frame until averaged
    is_averaged_embedding: bool = False     # True once MIN_TRACK_FRAMES reached


class CarTracker:
    """
    Stateful per-camera vehicle tracker.

    Internal state per track_id
    ----------------------------
    _crop_buffer  : BGR crop images (NOT embeddings) — raw pixels for batch Re-ID
    _embeddings   : averaged L2-normalised embedding once MIN_TRACK_FRAMES reached
    _frame_count  : total frames seen for this track
    """

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        # Store raw BGR crops so we can batch-embed them correctly
        self._crop_buffer: Dict[int, List[np.ndarray]] = defaultdict(list)
        self._embeddings: Dict[int, np.ndarray] = {}
        self._frame_count: Dict[int, int] = defaultdict(int)
        self._max_buffer = MIN_TRACK_FRAMES * 2   # rolling window cap

    def reset(self) -> None:
        self._crop_buffer.clear()
        self._embeddings.clear()
        self._frame_count.clear()

    def update(self, frame: np.ndarray) -> List[TrackResult]:
        """
        Run detection + tracking on one BGR frame.
        Returns a list of TrackResult, one per currently active track.
        Embedding is ALWAYS populated — single-frame until MIN_TRACK_FRAMES,
        then the averaged multi-frame embedding.
        """
        from models.model_manager import get_car_detector
        model = get_car_detector()

        # Run YOLOv8 with ByteTrack
        track_results = model.track(
            source=frame,
            conf=CAR_CONFIDENCE,
            classes=CAR_CLASSES,
            tracker=f"{TRACKER_TYPE}.yaml",
            persist=True,
            verbose=False,
        )

        output: List[TrackResult] = []

        if not track_results or track_results[0].boxes is None:
            return output

        boxes = track_results[0].boxes

        # ── Collect raw detections ────────────────────────────────────────────
        raw: List[Tuple[int, BBox, float, int]] = []
        for box in boxes:
            if box.id is None:
                continue
            track_id = int(box.id[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            raw.append((track_id, (x1, y1, x2, y2), conf, cls))

        if not raw:
            return output

        # ── Extract BGR crops (clamped, non-degenerate) ───────────────────────
        h, w = frame.shape[:2]
        crops: List[Optional[np.ndarray]] = []
        for _, (x1, y1, x2, y2), _, _ in raw:
            x1c = max(0, x1); y1c = max(0, y1)
            x2c = min(w, x2); y2c = min(h, y2)
            if (x2c - x1c) >= _MIN_CROP_PX and (y2c - y1c) >= _MIN_CROP_PX:
                crops.append(frame[y1c:y2c, x1c:x2c])
            else:
                crops.append(None)   # degenerate — placeholder

        # ── Batch single-frame embeddings for ALL current crops ───────────────
        # extract_embeddings_batch handles None entries (zero vector)
        valid_crops = [c if c is not None else np.zeros((32, 32, 3), dtype=np.uint8)
                       for c in crops]
        single_frame_embeds = extract_embeddings_batch(valid_crops)  # (N, D)

        # ── Update per-track state and build output ───────────────────────────
        for i, (track_id, bbox, conf, cls) in enumerate(raw):
            crop     = crops[i]
            sf_embed = single_frame_embeds[i]   # single-frame embedding

            self._frame_count[track_id] += 1

            # Store the raw BGR crop (not the embedding) so later averaging is
            # done on fresh per-frame Re-ID outputs, not on pre-embedded vectors
            if crop is not None:
                buf = self._crop_buffer[track_id]
                buf.append(crop)
                if len(buf) > self._max_buffer:
                    buf.pop(0)

            is_averaged = False
            final_embed = sf_embed   # default: use single-frame embedding

            if self._frame_count[track_id] >= MIN_TRACK_FRAMES and self._crop_buffer[track_id]:
                # Re-embed all buffered crops and average
                buffered_embeds = extract_embeddings_batch(
                    self._crop_buffer[track_id]
                )                                        # (K, D)
                avg = buffered_embeds.mean(axis=0)
                norm = np.linalg.norm(avg)
                if norm > 1e-6:
                    avg = avg / norm
                is_averaged = True
                final_embed = avg
                self._embeddings[track_id] = avg

            output.append(TrackResult(
                track_id=track_id,
                bbox=bbox,
                confidence=conf,
                class_id=cls,
                embedding=final_embed,
                is_averaged_embedding=is_averaged,
            ))

        return output

    def get_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Return the averaged embedding for a track, or None if not yet ready."""
        return self._embeddings.get(track_id)

    def purge_stale_tracks(self, active_ids: set) -> None:
        """Remove state for tracks no longer seen by the detector."""
        stale = set(self._frame_count.keys()) - active_ids
        for tid in stale:
            self._crop_buffer.pop(tid, None)
            self._embeddings.pop(tid, None)
            self._frame_count.pop(tid, None)
