"""
tracking/camera_tracker.py — Per-camera DeepSORT person tracker.

Each CameraTracker wraps a deep_sort_realtime.DeepSort instance bound to
one camera feed.  The update() method accepts raw ShadowDetector output
and returns only confirmed tracks (those that survived n_init frames).

Confirmed tracks include a person crop for downstream Re-ID embedding.

Public interface
----------------
CameraTracker(cam_id, max_age=30, n_init=3)
    .update(detections, frame) -> list[dict]
    .reset()

Each returned dict:
    local_id:    int             (DeepSORT track ID — stable within one camera)
    cam_id:      str             (camera that owns this tracker)
    bbox_ltwh:   [l, t, w, h]   (bounding box in pixels)
    bbox_center: [cx, cy]       (centre point)
    crop:        np.ndarray      (BGR crop of tracked person)
    confirmed:   bool            (always True — unconfirmed tracks are filtered)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from config import MIN_TRACK_FRAMES

logger = logging.getLogger(__name__)

# Try to import deep_sort_realtime; provide helpful error if missing
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    _DEEPSORT_AVAILABLE = True
except ImportError:
    _DEEPSORT_AVAILABLE = False
    logger.warning(
        "deep_sort_realtime not installed — CameraTracker will not function. "
        "Install it with: pip install deep-sort-realtime"
    )


class CameraTracker:
    """Per-camera DeepSORT tracker for person tracking.

    Parameters
    ----------
    cam_id : str
        Camera identifier this tracker is bound to.
    max_age : int
        Maximum number of frames a track can be lost before deletion.
    n_init : int
        Number of consecutive detections required to confirm a track.
    """

    def __init__(
        self,
        cam_id: str,
        max_age: int = 30,
        n_init: int = MIN_TRACK_FRAMES,
    ):
        if not _DEEPSORT_AVAILABLE:
            raise RuntimeError(
                "deep_sort_realtime is not installed. "
                "Run: pip install deep-sort-realtime"
            )

        self.cam_id = cam_id
        self.max_age = max_age
        self.n_init = n_init
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=0.7,
            max_cosine_distance=0.3,
        )
        logger.debug(
            "CameraTracker created: cam=%s max_age=%d n_init=%d",
            cam_id, max_age, n_init,
        )

    def update(
        self,
        detections: List[dict],
        frame: np.ndarray,
    ) -> List[dict]:
        """Run one tracking cycle.

        Parameters
        ----------
        detections : list[dict]
            Output from ShadowDetector.detect() — each dict must have
            "bbox" (LTWH), "conf", and optionally "crop".
        frame : np.ndarray
            The current BGR frame (used by DeepSORT for appearance features).

        Returns
        -------
        list[dict]
            Confirmed tracks only, each with local_id, cam_id, bbox_ltwh,
            bbox_center, crop, and confirmed=True.
        """
        if not detections:
            # Still call update with empty input to age existing tracks
            self._tracker.update_tracks([], frame=frame)
            return []

        # Build DeepSORT input: list of (bbox_ltwh, confidence, class_name)
        ds_input = []
        for d in detections:
            bbox = d["bbox"]  # [x, y, w, h]
            conf = d["conf"]
            ds_input.append((bbox, conf, "person"))

        # Run DeepSORT
        tracks = self._tracker.update_tracks(ds_input, frame=frame)

        # Extract confirmed tracks
        confirmed: List[dict] = []
        h_frame, w_frame = frame.shape[:2]

        for t in tracks:
            if not t.is_confirmed():
                continue

            # Get bounding box in LTWH format
            l, top, w, h = t.to_ltwh()
            l = max(0, int(l))
            top = max(0, int(top))
            w = int(w)
            h = int(h)

            # Clamp to frame bounds
            x2 = min(w_frame, l + w)
            y2 = min(h_frame, top + h)
            w = x2 - l
            h = y2 - top

            if w < 5 or h < 5:
                continue

            # Extract crop for Re-ID
            crop = frame[top:top + h, l:l + w]
            if crop.size == 0:
                continue

            confirmed.append({
                "local_id":    t.track_id,
                "cam_id":      self.cam_id,
                "bbox_ltwh":   [l, top, w, h],
                "bbox_center": [l + w / 2, top + h / 2],
                "crop":        crop,
                "confirmed":   True,
            })

        return confirmed

    def reset(self) -> None:
        """Reset the tracker state (e.g. on stream reconnect)."""
        self._tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            nms_max_overlap=0.7,
            max_cosine_distance=0.3,
        )
        logger.debug("CameraTracker reset: cam=%s", self.cam_id)
