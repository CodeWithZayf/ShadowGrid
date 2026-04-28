"""
core/priority_manager.py — Camera priority scoring.

Priority is boosted when:
  • A crime is detected → all cameras within PRIORITY_GEO_MAX_KM get a geo-weighted boost.
  • An occurrence is confirmed → the camera that spotted the car gets a boost.

Priority decays exponentially over time at PRIORITY_DECAY_RATE per second.

Public interface
----------------
boost_on_crime(source_camera_id: str) -> None
boost_on_occurrence(camera_id: str) -> None
decay_priorities() -> None          # call periodically
get_priority_order() -> List[str]   # camera IDs sorted high → low (decayed first)
"""

from __future__ import annotations

import logging
import math
import time
from typing import List

from config import (
    PRIORITY_CRIME_BOOST,
    PRIORITY_DECAY_RATE,
    PRIORITY_GEO_FALLOFF,
    PRIORITY_GEO_MAX_KM,
    PRIORITY_OCCURRENCE_BOOST,
)
from core.state import camera_priorities, camera_registry, priorities_lock

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Haversine distance
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _gaussian_weight(dist_km: float) -> float:
    """Gaussian falloff — 1.0 at dist=0, near 0 at dist >> PRIORITY_GEO_FALLOFF."""
    return math.exp(-(dist_km ** 2) / (2 * PRIORITY_GEO_FALLOFF ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def boost_on_crime(source_camera_id: str) -> None:
    """
    After a crime is detected on source_camera_id, boost all cameras
    proportionally to their geographic proximity to that camera.
    """
    source = camera_registry.get(source_camera_id)
    if source is None:
        return

    now = time.time()
    with priorities_lock:
        for cam_id, priority in camera_priorities.items():
            target = camera_registry.get(cam_id)
            if target is None:
                continue
            dist = _haversine_km(
                source.latitude, source.longitude,
                target.latitude, target.longitude,
            )
            if dist <= PRIORITY_GEO_MAX_KM:
                weight = _gaussian_weight(dist)
                priority.value += PRIORITY_CRIME_BOOST * weight
                priority.last_updated = now
                logger.debug(
                    "Crime boost: cam=%s, dist=%.2f km, weight=%.3f, new_priority=%.1f",
                    cam_id, dist, weight, priority.value,
                )


def boost_on_occurrence(camera_id: str) -> None:
    """Boost a camera's priority when it confirms a suspect occurrence."""
    now = time.time()
    with priorities_lock:
        p = camera_priorities.get(camera_id)
        if p is not None:
            p.value += PRIORITY_OCCURRENCE_BOOST
            p.last_updated = now
            logger.debug("Occurrence boost: cam=%s, new_priority=%.1f", camera_id, p.value)


def decay_priorities() -> None:
    """
    Apply exponential decay based on elapsed time since last update.
    Call this once per scheduling cycle in the consumer loop.
    """
    now = time.time()
    with priorities_lock:
        for p in camera_priorities.values():
            elapsed = now - p.last_updated
            if elapsed > 0 and p.value > 0:
                p.value *= math.exp(-PRIORITY_DECAY_RATE * elapsed)
                p.last_updated = now
                if p.value < 0.01:
                    p.value = 0.0


def get_priority_order() -> List[str]:
    """
    Return camera IDs sorted from highest to lowest priority (after decay).
    Cameras with equal priority maintain their registration order.
    """
    decay_priorities()
    with priorities_lock:
        ordered = sorted(
            camera_priorities.keys(),
            key=lambda cid: camera_priorities[cid].value,
            reverse=True,
        )
    return ordered
