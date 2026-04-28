"""
routers/occurrences.py — Occurrences list endpoints.

GET /occurrences                                              → recent occurrences (newest-first)
GET /occurrences?since_timestamp=<f>&limit=<i>               → occurrences added after timestamp
GET /occurrences/suspect/{suspect_id}                        → filter by suspect
GET /occurrences/camera/{camera_id}                          → filter by camera
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Path, Query

from config import QUERY_DEFAULT_LIMIT, QUERY_MAX_LIMIT
from core.state import (
    occurrences_page,
    occurrences_since,
    snapshot_occurrences,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/occurrences", tags=["Occurrences"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp_limit(limit: Optional[int]) -> int:
    if limit is None:
        return QUERY_DEFAULT_LIMIT
    return max(1, min(limit, QUERY_MAX_LIMIT))


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", summary="List occurrences (optionally filtered by timestamp)")
def list_occurrences(
    since_timestamp: Optional[float] = Query(
        default=None,
        description=(
            "Unix epoch seconds. When supplied, only occurrences with "
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
    Return occurrences ordered newest-first.

    - Without query params: returns the `limit` most recent occurrences
      (default {QUERY_DEFAULT_LIMIT}).
    - With `since_timestamp`: returns only occurrences recorded **after**
      that Unix timestamp, up to `limit` results.
    """
    clamped = _clamp_limit(limit)

    if since_timestamp is not None:
        entries = occurrences_since(since_timestamp, clamped)
    else:
        entries = occurrences_page(clamped)

    return [o.to_dict() for o in entries]


@router.get(
    "/suspect/{suspect_id}",
    summary="List occurrences for a specific suspect",
)
def occurrences_by_suspect(
    suspect_id: str = Path(..., description="Suspect ID to filter by"),
) -> List[Dict[str, Any]]:
    """Return all occurrences that matched the given suspect_id (oldest-first)."""
    return [
        o.to_dict()
        for o in snapshot_occurrences()
        if o.suspect_id == suspect_id
    ]


@router.get(
    "/camera/{camera_id}",
    summary="List occurrences recorded by a specific camera",
)
def occurrences_by_camera(
    camera_id: str = Path(..., description="Camera ID to filter by"),
) -> List[Dict[str, Any]]:
    """Return all occurrences spotted on the given camera_id (oldest-first)."""
    return [
        o.to_dict()
        for o in snapshot_occurrences()
        if o.camera_id == camera_id
    ]
