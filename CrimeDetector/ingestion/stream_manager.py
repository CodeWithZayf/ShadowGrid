"""
ingestion/stream_manager.py — Adapter around the existing frame ingestion.

The existing core/frame_ingestor.py is already superior to the plan's
CameraStream (it handles frozen-frame detection, auto-reconnect, and
push-based threading). This module provides a thin adapter that presents
the plan's interface on top of the existing state.get_latest_frame().

Public interface
----------------
CameraStreamAdapter(cam_id)
    .get_frame()   -> np.ndarray | None
    .is_live()     -> bool

get_all_frames() -> dict[str, np.ndarray]
    Returns {cam_id: frame} for all cameras that have a live frame.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from core.state import camera_registry, get_latest_frame, is_camera_live

logger = logging.getLogger(__name__)


class CameraStreamAdapter:
    """Adapts the existing frame queue to the plan's CameraStream interface.

    The existing frame_ingestor.py pushes frames into state.frame_queues.
    This adapter simply reads from that queue via get_latest_frame().
    """

    def __init__(self, cam_id: str):
        self.cam_id = cam_id

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame for this camera, or None if unavailable."""
        return get_latest_frame(self.cam_id)

    def is_live(self) -> bool:
        """Return True if the camera stream is currently receiving frames."""
        return is_camera_live(self.cam_id)

    def __repr__(self) -> str:
        live = self.is_live()
        return f"CameraStreamAdapter(cam_id={self.cam_id!r}, live={live})"


def get_all_frames() -> Dict[str, np.ndarray]:
    """Return latest frames from all registered cameras.

    Only cameras with a non-None frame are included.
    """
    frames: Dict[str, np.ndarray] = {}
    for cam_id in camera_registry:
        frame = get_latest_frame(cam_id)
        if frame is not None:
            frames[cam_id] = frame
    return frames
