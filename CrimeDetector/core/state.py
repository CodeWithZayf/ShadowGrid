"""
core/state.py — Single source of truth for all shared mutable state.

Every other module imports from here; nothing stores its own private copy
of suspects, occurrences, camera registry, or frame queues.

Suspects and occurrences are stored in collections.deque instances with
maxlen caps defined in config (MAX_SUSPECTS / MAX_OCCURRENCES).  When either
deque is full the oldest entry is silently evicted — identical to the
frame-queue drop-oldest policy used for camera streams.

New internal getters
--------------------
suspects_since(since_ts, limit)   → List[SuspectEntry]   newest-first
occurrences_since(since_ts, limit) → List[OccurrenceEntry] newest-first
suspects_page(limit)              → List[SuspectEntry]   newest-first, no ts filter
occurrences_page(limit)           → List[OccurrenceEntry] newest-first, no ts filter
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import DEQUE_MAX_LEN, MAX_OCCURRENCES, MAX_SUSPECTS, FRAME_STALE_SECONDS


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraInfo:
    """Loaded from cameras.json; immutable after registration."""
    id: str
    url: str
    location_name: str
    latitude: float
    longitude: float


@dataclass
class SuspectEntry:
    """One entry in the global suspects list."""
    suspect_id: str                       # UUID
    camera_id: str                        # Camera that triggered crime detection
    car_id: int                           # Track ID at detection time (legacy name; now person track)
    timestamp: float                      # Unix epoch seconds
    embedding: np.ndarray                 # Re-ID embedding vector
    source: str = "crime_detection"       # "crime_detection" | "manual_upload"
    description: Optional[str] = None    # Free-text description if manually added

    def to_dict(self) -> dict:
        return {
            "suspect_id":   self.suspect_id,
            "camera_id":    self.camera_id,
            "car_id":       self.car_id,
            "timestamp":    self.timestamp,
            "embedding":    self.embedding.tolist(),
            "source":       self.source,
            "description":  self.description,
        }


@dataclass
class OccurrenceEntry:
    """One confirmed sighting of a suspect person at a camera."""
    occurrence_id: str                    # UUID
    suspect_id: str                       # Matched suspect
    camera_id: str                        # Camera that spotted the person
    car_id: int                           # Current tracker ID in that camera (legacy name; now person track)
    embedding: np.ndarray                 # Embedding of the person in this occurrence
    similarity: float                     # Cosine similarity to matched suspect
    timestamp: float                      # Unix epoch seconds

    def to_dict(self) -> dict:
        return {
            "occurrence_id": self.occurrence_id,
            "suspect_id":    self.suspect_id,
            "camera_id":     self.camera_id,
            "car_id":        self.car_id,
            "embedding":     self.embedding.tolist(),
            "similarity":    round(self.similarity, 4),
            "timestamp":     self.timestamp,
        }


@dataclass
class CameraPriority:
    """Mutable priority record for one camera."""
    camera_id: str
    value: float = 0.0                    # Current priority score
    last_updated: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────────────
# Global registries
# ─────────────────────────────────────────────────────────────────────────────

# camera_id → CameraInfo
camera_registry: Dict[str, CameraInfo] = {}

# camera_id → deque[np.ndarray]  (BGR frames, maxlen=DEQUE_MAX_LEN, drops oldest)
frame_queues: Dict[str, deque] = {}

# camera_id → float (Unix epoch of the last successful push_frame() call)
# Used by get_latest_frame() to reject stale frames after camera disconnection.
last_push_times: Dict[str, float] = {}

# Per-camera lock for frame queue operations (push / get / clear).
# Prevents race conditions between ingestor threads and the consumer thread.
frame_locks: Dict[str, threading.Lock] = {}

# ── Bounded deques for suspects and occurrences ───────────────────────────────
# Both use collections.deque with a hard maxlen cap.  When the cap is hit,
# deque.append() automatically evicts the LEFTMOST (oldest) entry — no extra
# code needed.  All consumers must hold the corresponding lock while iterating.

# Ordered oldest-first (index 0 = oldest, index -1 = newest)
suspects: deque[SuspectEntry] = deque(maxlen=MAX_SUSPECTS)
suspects_lock = threading.Lock()

# Ordered oldest-first
occurrences: deque[OccurrenceEntry] = deque(maxlen=MAX_OCCURRENCES)
occurrences_lock = threading.Lock()

# camera_id → CameraPriority
camera_priorities: Dict[str, CameraPriority] = {}
priorities_lock = threading.Lock()

# camera_id → set of tracker IDs currently being processed by tracking module
# (prevents double-processing the same track on concurrent threads)
active_tracks: Dict[str, set] = {}
active_tracks_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation helpers (called once at startup)
# ─────────────────────────────────────────────────────────────────────────────

def register_camera(cam: CameraInfo) -> None:
    """Register a camera and create its frame deque + priority record."""
    camera_registry[cam.id] = cam
    frame_queues[cam.id] = deque(maxlen=DEQUE_MAX_LEN)
    last_push_times[cam.id] = 0.0          # 0.0 → never pushed; treated as stale
    frame_locks[cam.id] = threading.Lock()
    camera_priorities[cam.id] = CameraPriority(camera_id=cam.id)
    active_tracks[cam.id] = set()


def get_latest_frame(camera_id: str) -> Optional[np.ndarray]:
    """
    Return the most recently pushed frame for a camera, or None.

    Returns None if:
      • The deque is empty (camera never connected).
      • The last frame was pushed more than FRAME_STALE_SECONDS ago
        (camera disconnected — prevents the consumer from running crime
        detection on a frozen stale frame after disconnection).
    """
    lock = frame_locks.get(camera_id)
    if lock is None:
        return None
    with lock:
        last_t = last_push_times.get(camera_id, 0.0)
        if last_t == 0.0:
            return None  # never received a frame

        age = time.time() - last_t
        if age > FRAME_STALE_SECONDS:
            return None  # camera is stale / disconnected

        q = frame_queues.get(camera_id)
        if q:
            try:
                return q[-1]
            except IndexError:
                pass
        return None


def push_frame(camera_id: str, frame: np.ndarray) -> None:
    """Push a new frame and record the push timestamp for liveness tracking."""
    lock = frame_locks.get(camera_id)
    if lock is None:
        return
    with lock:
        q = frame_queues.get(camera_id)
        if q is not None:
            q.append(frame)
            last_push_times[camera_id] = time.time()


def clear_frame_queue(camera_id: str) -> None:
    """
    Clear the frame deque and reset the push timestamp for a camera.
    Called by the ingestor when the stream is declared dead so that
    get_latest_frame() immediately returns None rather than serving
    the last frozen frame until FRAME_STALE_SECONDS expires.
    """
    lock = frame_locks.get(camera_id)
    if lock is None:
        return
    with lock:
        q = frame_queues.get(camera_id)
        if q is not None:
            q.clear()
        last_push_times[camera_id] = 0.0


def is_camera_live(camera_id: str) -> bool:
    """Return True if this camera has pushed a frame within FRAME_STALE_SECONDS."""
    last_t = last_push_times.get(camera_id, 0.0)
    if last_t == 0.0:
        return False
    return (time.time() - last_t) <= FRAME_STALE_SECONDS


# ─────────────────────────────────────────────────────────────────────────────
# Suspects — write / delete
# ─────────────────────────────────────────────────────────────────────────────

def add_suspect(entry: SuspectEntry) -> None:
    """
    Append a new SuspectEntry.  If the deque is already at MAX_SUSPECTS
    capacity, the oldest entry is automatically evicted by the deque.
    """
    with suspects_lock:
        suspects.append(entry)


def remove_suspect(suspect_id: str) -> bool:
    """
    Remove every SuspectEntry whose suspect_id matches.
    Returns True if at least one entry was removed.
    """
    with suspects_lock:
        before = len(suspects)
        # Build a fresh deque excluding the matching entries; preserves maxlen.
        to_keep = deque(
            (s for s in suspects if s.suspect_id != suspect_id),
            maxlen=MAX_SUSPECTS,
        )
        suspects.clear()
        suspects.extend(to_keep)
        return len(suspects) < before


# ─────────────────────────────────────────────────────────────────────────────
# Occurrences — write
# ─────────────────────────────────────────────────────────────────────────────

def add_occurrence(entry: OccurrenceEntry) -> None:
    """
    Append a new OccurrenceEntry.  Oldest entry evicted automatically when
    the deque reaches MAX_OCCURRENCES capacity.
    """
    with occurrences_lock:
        occurrences.append(entry)


# ─────────────────────────────────────────────────────────────────────────────
# Internal getters — full snapshots (used by existing consumers / routers)
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_suspects() -> List[SuspectEntry]:
    """Return a shallow copy of the entire suspects deque as a list (oldest→newest)."""
    with suspects_lock:
        return list(suspects)


def snapshot_occurrences() -> List[OccurrenceEntry]:
    """Return a shallow copy of the entire occurrences deque as a list (oldest→newest)."""
    with occurrences_lock:
        return list(occurrences)


# ─────────────────────────────────────────────────────────────────────────────
# Internal getters — paginated / time-filtered  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def suspects_since(since_ts: float, limit: int) -> List[SuspectEntry]:
    """
    Return up to `limit` suspects whose timestamp is strictly greater than
    `since_ts`, ordered newest-first.

    Complexity: O(n) scan from the right (newest) side of the deque,
    stopping as soon as `limit` results are collected or we reach entries
    older than `since_ts`.  Because entries are inserted in chronological
    order, the moment we hit an entry with timestamp ≤ since_ts every
    remaining entry to the left is also ≤ since_ts, so we can break early.
    """
    results: List[SuspectEntry] = []
    with suspects_lock:
        # Iterate right-to-left (newest first)
        for entry in reversed(suspects):
            if entry.timestamp <= since_ts:
                break          # all older entries are also ≤ since_ts
            results.append(entry)
            if len(results) >= limit:
                break
    return results             # already newest-first


def occurrences_since(since_ts: float, limit: int) -> List[OccurrenceEntry]:
    """
    Return up to `limit` occurrences whose timestamp is strictly greater than
    `since_ts`, ordered newest-first.  Same early-exit logic as suspects_since.
    """
    results: List[OccurrenceEntry] = []
    with occurrences_lock:
        for entry in reversed(occurrences):
            if entry.timestamp <= since_ts:
                break
            results.append(entry)
            if len(results) >= limit:
                break
    return results


def suspects_page(limit: int) -> List[SuspectEntry]:
    """
    Return the `limit` most recent suspects regardless of timestamp,
    ordered newest-first.
    """
    with suspects_lock:
        # deque supports negative indexing but not slicing; materialise tail
        items = list(suspects)
    # items is oldest→newest; reverse and take head
    return items[-limit:][::-1] if limit < len(items) else items[::-1]


def occurrences_page(limit: int) -> List[OccurrenceEntry]:
    """
    Return the `limit` most recent occurrences regardless of timestamp,
    ordered newest-first.
    """
    with occurrences_lock:
        items = list(occurrences)
    return items[-limit:][::-1] if limit < len(items) else items[::-1]
