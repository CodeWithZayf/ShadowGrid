"""
routers/cameras.py — Camera management endpoints.

GET  /cameras              → list all cameras loaded from cameras.json
GET  /cameras/{id}/frame   → latest JPEG frame for a camera
GET  /cameras/{id}/stream  → MJPEG live-stream for a camera
DELETE /cameras/{id}/suspect → remove camera from suspects list (remove all
                               suspect entries that originated from this camera)
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import JSONResponse, Response, StreamingResponse

from config import STREAM_JPEG_QUALITY, TARGET_FPS
from core.state import camera_registry, get_latest_frame, remove_suspect, snapshot_suspects

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cameras", tags=["Cameras"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _frame_to_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    return buf.tobytes()


def _cam_or_404(camera_id: str):
    cam = camera_registry.get(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return cam


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", summary="List all registered cameras")
def list_cameras() -> List[Dict[str, Any]]:
    """Return the full list of cameras as loaded from cameras.json."""
    return [
        {
            "id":            cam.id,
            "url":           cam.url,
            "metadata": {
                "location_name": cam.location_name,
                "latitude":      cam.latitude,
                "longitude":     cam.longitude,
            },
        }
        for cam in camera_registry.values()
    ]


@router.get("/{camera_id}/frame", summary="Get the latest frame from a camera")
def get_frame(camera_id: str = Path(..., description="Camera ID")) -> Response:
    """
    Returns the most recently received frame as a JPEG image.
    Returns 204 if no frame has been received yet.
    """
    _cam_or_404(camera_id)
    frame = get_latest_frame(camera_id)
    if frame is None:
        return Response(status_code=204, content=b"", media_type="image/jpeg")
    return Response(content=_frame_to_jpeg(frame), media_type="image/jpeg")


@router.get("/{camera_id}/stream", summary="Live MJPEG stream from a camera")
def stream_camera(camera_id: str = Path(..., description="Camera ID")) -> StreamingResponse:
    """
    Streams a live MJPEG feed from the camera's deque.
    Compatible with browser <img> tags and most media players.
    """
    _cam_or_404(camera_id)
    frame_interval = 1.0 / TARGET_FPS

    def generate():
        while True:
            t0 = time.time()
            frame = get_latest_frame(camera_id)
            if frame is not None:
                try:
                    jpeg = _frame_to_jpeg(frame)
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                except Exception as exc:
                    logger.warning("Stream encode error cam=%s: %s", camera_id, exc)
            elapsed = time.time() - t0
            sleep_time = max(0.0, frame_interval - elapsed)
            time.sleep(sleep_time)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.delete("/{camera_id}/suspect", summary="Remove a camera from the suspects list")
def remove_camera_suspects(
    camera_id: str = Path(..., description="Camera ID whose suspects should be removed")
) -> Dict[str, Any]:
    """
    Remove all SuspectEntry records that originated from the given camera_id.
    """
    _cam_or_404(camera_id)

    # Collect matching suspect IDs first
    targets = [s.suspect_id for s in snapshot_suspects() if s.camera_id == camera_id]
    removed = 0
    for sid in targets:
        if remove_suspect(sid):
            removed += 1

    logger.info("Removed %d suspect(s) originating from cam=%s", removed, camera_id)
    return {"camera_id": camera_id, "suspects_removed": removed}
