"""
models/suspect_matcher.py — Cosine-similarity search over the suspects list.

Public interface
----------------
find_best_match(embedding: np.ndarray) -> Optional[Tuple[SuspectEntry, float]]
    Return (best_suspect, similarity) if similarity >= SUSPECT_MATCH_THRESHOLD,
    else None.

cosine_similarity(a, b) -> float
    Utility used by other modules.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from config import SUSPECT_MATCH_THRESHOLD
from core.state import SuspectEntry, snapshot_suspects

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity in [-1, 1]; both vectors assumed L2-normalised."""
    return float(np.dot(a, b))


def find_best_match(
    embedding: np.ndarray,
) -> Optional[Tuple[SuspectEntry, float]]:
    """
    Search the suspects list for the highest cosine similarity to `embedding`.
    Returns (suspect, similarity) if the best match >= SUSPECT_MATCH_THRESHOLD,
    otherwise None.
    """
    suspects = snapshot_suspects()
    if not suspects:
        return None

    best_suspect: Optional[SuspectEntry] = None
    best_sim = -1.0

    for s in suspects:
        sim = cosine_similarity(embedding, s.embedding)
        if sim > best_sim:
            best_sim = sim
            best_suspect = s

    if best_sim >= SUSPECT_MATCH_THRESHOLD:
        logger.debug(
            "Suspect match found: suspect_id=%s, similarity=%.4f",
            best_suspect.suspect_id, best_sim,
        )
        return best_suspect, best_sim

    return None
