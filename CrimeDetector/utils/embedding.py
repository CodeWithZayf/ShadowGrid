"""
utils/embedding.py — Helpers for serialising / deserialising Re-ID embeddings.
"""

from __future__ import annotations
import numpy as np
from typing import List


def to_list(embedding: np.ndarray) -> List[float]:
    """Convert a float32 numpy embedding to a plain Python list (for JSON)."""
    return embedding.tolist()


def from_list(values: List[float]) -> np.ndarray:
    """Reconstruct a normalised float32 embedding from a JSON list."""
    arr = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 1e-6:
        arr = arr / norm
    return arr
