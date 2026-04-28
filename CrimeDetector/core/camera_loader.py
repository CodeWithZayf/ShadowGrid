"""
core/camera_loader.py — Reads cameras.json and populates the camera registry.

Public interface
----------------
load_cameras() -> List[CameraInfo]
    Parse cameras.json, register every camera in state, return the list.
"""

from __future__ import annotations

import json
import logging
from typing import List

from config import CAMERAS_FILE
from core.state import CameraInfo, register_camera

logger = logging.getLogger(__name__)


def load_cameras() -> List[CameraInfo]:
    """
    Parse CAMERAS_FILE, register each camera in the global registry,
    and return the resulting list of CameraInfo objects.

    Raises
    ------
    FileNotFoundError  – if cameras.json is missing.
    ValueError         – if any entry is malformed.
    """
    if not CAMERAS_FILE.exists():
        raise FileNotFoundError(f"cameras.json not found at {CAMERAS_FILE}")

    with open(CAMERAS_FILE, "r", encoding="utf-8") as fh:
        raw: list = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError("cameras.json must be a JSON array.")

    cameras: List[CameraInfo] = []
    for idx, entry in enumerate(raw):
        try:
            meta = entry["metadata"]
            cam = CameraInfo(
                id=str(entry["id"]),
                url=str(entry["url"]),
                location_name=str(meta["location_name"]),
                latitude=float(meta["latitude"]),
                longitude=float(meta["longitude"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"cameras.json entry #{idx} missing field: {exc}"
            ) from exc

        register_camera(cam)
        cameras.append(cam)
        logger.info("Registered camera '%s' at %s", cam.id, cam.url)

    logger.info("Loaded %d camera(s) from %s", len(cameras), CAMERAS_FILE)
    return cameras
