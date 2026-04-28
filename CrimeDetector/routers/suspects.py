"""
routers/suspects.py — Suspects list endpoints.

GET  /suspects                             → all suspects (newest-first, optional limit)
GET  /suspects?since_timestamp=<f>&limit=<i> → suspects added after timestamp
POST /suspects/upload                      → add a suspect from an uploaded image
POST /suspects/description                 → add a suspect from a text description
DELETE /suspects/{suspect_id}              → remove a specific suspect by ID
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from config import QUERY_DEFAULT_LIMIT, QUERY_MAX_LIMIT, REID_EMBED_DIM
from core.state import (
    SuspectEntry,
    add_suspect,
    remove_suspect,
    snapshot_suspects,
    suspects_page,
    suspects_since,
)
from models.model_manager import extract_embedding

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/suspects", tags=["Suspects"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _image_bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode uploaded image.")
    return frame


def _clamp_limit(limit: Optional[int]) -> int:
    """Clamp caller-supplied limit to [1, QUERY_MAX_LIMIT]."""
    if limit is None:
        return QUERY_DEFAULT_LIMIT
    return max(1, min(limit, QUERY_MAX_LIMIT))


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", summary="List suspects (optionally filtered by timestamp)")
def list_suspects(
    since_timestamp: Optional[float] = Query(
        default=None,
        description=(
            "Unix epoch seconds. When supplied, only suspects with "
            "timestamp > since_timestamp are returned."
        ),
    ),
    limit: Optional[int] = Query(
        default=None,
        ge=1,
        le=QUERY_MAX_LIMIT,
        description=(
            f"Maximum number of entries to return (newest-first). "
            f"Defaults to {QUERY_DEFAULT_LIMIT}, hard cap {QUERY_MAX_LIMIT}."
        ),
    ),
) -> List[Dict[str, Any]]:
    """
    Return suspects ordered newest-first.

    - Without query params: returns the `limit` most recent suspects
      (default {QUERY_DEFAULT_LIMIT}).
    - With `since_timestamp`: returns only suspects added **after** that
      Unix timestamp, up to `limit` results.
    """
    clamped = _clamp_limit(limit)

    if since_timestamp is not None:
        entries = suspects_since(since_timestamp, clamped)
    else:
        entries = suspects_page(clamped)

    return [s.to_dict() for s in entries]


@router.delete("/{suspect_id}", summary="Remove a suspect by ID")
def delete_suspect(suspect_id: str) -> Dict[str, Any]:
    """Remove the suspect with the given suspect_id from the list."""
    removed = remove_suspect(suspect_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Suspect '{suspect_id}' not found.")
    logger.info("Suspect manually removed: id=%s", suspect_id)
    return {"suspect_id": suspect_id, "removed": True}


@router.post("/upload", summary="Add a suspect from an uploaded car image")
async def add_suspect_from_image(
    file: UploadFile = File(..., description="Image of the suspect car (JPEG/PNG)"),
    camera_id: Optional[str] = Form(None, description="Camera ID (optional)"),
    description: Optional[str] = Form(None, description="Free-text description (optional)"),
) -> Dict[str, Any]:
    """
    Accept an image of a suspect car, compute its Re-ID embedding,
    and add it to the suspects list.
    """
    raw = await file.read()
    try:
        frame = _image_bytes_to_bgr(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    emb = extract_embedding(frame)
    suspect_id = str(uuid.uuid4())

    entry = SuspectEntry(
        suspect_id=suspect_id,
        camera_id=camera_id or "manual",
        car_id=-1,
        timestamp=time.time(),
        embedding=emb,
        source="manual_upload",
        description=description,
    )
    add_suspect(entry)
    logger.info("Suspect added via image upload: id=%s", suspect_id)
    return entry.to_dict()


@router.post("/description", summary="Add a suspect from a text description")
async def add_suspect_from_description(
    description: str = Form(..., description="Text description of the suspect car"),
    camera_id: Optional[str] = Form(None, description="Camera ID (optional)"),
) -> Dict[str, Any]:
    """
    Accept a text description of a suspect car.
    A zero embedding is stored as a placeholder since we have no image.
    The description is preserved for operator use.

    Note: Without a visual crop, embedding-based matching will not trigger.
    Provide an image via /suspects/upload for full matching capability.
    """
    emb = np.zeros(REID_EMBED_DIM, dtype=np.float32)
    suspect_id = str(uuid.uuid4())

    entry = SuspectEntry(
        suspect_id=suspect_id,
        camera_id=camera_id or "manual",
        car_id=-1,
        timestamp=time.time(),
        embedding=emb,
        source="manual_description",
        description=description,
    )
    add_suspect(entry)
    logger.info("Suspect added via description: id=%s desc='%s'", suspect_id, description[:60])
    return entry.to_dict()
