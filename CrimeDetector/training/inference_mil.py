"""
training/inference_mil.py — MIL Anomaly Score Inference Wrapper.

Provides a singleton model accessor and a scoring function that the
surveillance server can call on accumulated frame features.

The MIL model operates on pre-extracted C3D features (32 segments × 4096 dims).
For real-time inference, frames must be accumulated into temporal segments
and have their features extracted before scoring.

Since the server currently processes individual frames (not pre-extracted C3D
features), this module also provides a lightweight frame-buffer that
accumulates frames, extracts simple per-segment features, and scores them.

Public interface
----------------
load_mil_model(checkpoint_path) -> None
get_anomaly_score(features: np.ndarray) -> float
get_segment_scores(features: np.ndarray) -> np.ndarray
MILFrameBuffer — accumulate frames and get rolling anomaly scores
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from training.model import AnomalyMILModel

logger = logging.getLogger(__name__)

# ── Singleton ─────────────────────────────────────────────────────────
_mil_model: Optional[AnomalyMILModel] = None
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimal threshold from evaluation (F1=0.8251)
ANOMALY_THRESHOLD = 0.0123


def load_mil_model(checkpoint_path: str = "model_weights/best_model.pt") -> bool:
    """Load the trained MIL model. Returns True if successful."""
    global _mil_model

    cp = Path(checkpoint_path)
    if not cp.exists():
        logger.warning(
            "MIL model checkpoint not found at %s — anomaly scoring disabled.",
            checkpoint_path,
        )
        return False

    try:
        checkpoint = torch.load(str(cp), map_location=_device, weights_only=False)
        # NOTE: weights_only=False is required here because the checkpoint
        # contains non-tensor metadata (args dict, epoch, auc). Consider
        # migrating to torch.save with only state_dict to enable weights_only=True.
        model_args = checkpoint.get("args", {})

        _mil_model = AnomalyMILModel(
            feature_dim=model_args.get("feature_dim", 4096),
            hidden_dim=model_args.get("hidden_dim", 512),
            bottleneck_dim=model_args.get("bottleneck_dim", 32),
            dropout=model_args.get("dropout", 0.6),
        ).to(_device)

        _mil_model.load_state_dict(checkpoint["model_state_dict"])
        _mil_model.eval()

        auc = checkpoint.get("auc", "N/A")
        epoch = checkpoint.get("epoch", "N/A")
        logger.info(
            "MIL anomaly model loaded: epoch=%s, AUC=%s, device=%s",
            epoch, auc, _device,
        )
        return True

    except Exception as e:
        logger.error("Failed to load MIL model: %s", e)
        _mil_model = None
        return False


def is_mil_available() -> bool:
    """Check if the MIL model is loaded and ready."""
    return _mil_model is not None


@torch.no_grad()
def get_anomaly_score(features: np.ndarray) -> float:
    """Score a video's C3D features.

    Args:
        features: (32, 4096) array of segment features

    Returns:
        Float anomaly score (0-1). Higher = more anomalous.
        The max segment score is returned (MIL prediction).
    """
    if _mil_model is None:
        return 0.0

    features_t = torch.from_numpy(features).unsqueeze(0).to(_device)  # (1, 32, 4096)
    scores = _mil_model(features_t)  # (1, 32, 1)
    return float(scores.squeeze().max())


@torch.no_grad()
def get_segment_scores(features: np.ndarray) -> np.ndarray:
    """Get per-segment anomaly scores.

    Args:
        features: (32, 4096) array of segment features

    Returns:
        (32,) array of anomaly scores per segment
    """
    if _mil_model is None:
        return np.zeros(32, dtype=np.float32)

    features_t = torch.from_numpy(features).unsqueeze(0).to(_device)
    scores = _mil_model(features_t)  # (1, 32, 1)
    return scores.squeeze().cpu().numpy()


def is_anomalous(score: float, threshold: float = ANOMALY_THRESHOLD) -> bool:
    """Check if an anomaly score exceeds the detection threshold."""
    return score >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# Frame Buffer for real-time scoring
# ─────────────────────────────────────────────────────────────────────────────

class MILFrameBuffer:
    """Accumulates frames and provides rolling anomaly scores.

    The MIL model expects 32 temporal segments of 4096-dim C3D features.
    Since C3D feature extraction requires a full video clip, this buffer
    uses a lightweight proxy: it accumulates frames, divides them into
    32 temporal bins, and uses simple frame-level statistics as features.

    For production use with full C3D features, replace the _extract_features
    method with an actual C3D forward pass.
    """

    def __init__(self, segment_count: int = 32, frames_per_segment: int = 16):
        self.segment_count = segment_count
        self.frames_per_segment = frames_per_segment
        self.total_capacity = segment_count * frames_per_segment  # 512 frames
        self.frames: list[np.ndarray] = []
        self._last_score: float = 0.0

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the buffer."""
        self.frames.append(frame)
        # Evict oldest if over capacity
        if len(self.frames) > self.total_capacity:
            self.frames.pop(0)

    def is_ready(self) -> bool:
        """Check if we have enough frames for a full scoring pass."""
        return len(self.frames) >= self.total_capacity

    @property
    def fill_ratio(self) -> float:
        """How full the buffer is (0.0 to 1.0)."""
        return len(self.frames) / self.total_capacity

    def get_score(self) -> float:
        """Get current anomaly score. Returns the last score if buffer not ready."""
        if not self.is_ready():
            return self._last_score

        if _mil_model is None:
            return 0.0

        features = self._extract_features()
        self._last_score = get_anomaly_score(features)
        return self._last_score

    def _extract_features(self) -> np.ndarray:
        """Extract 32×4096 features from buffered frames.

        This is a placeholder that uses frame statistics. For full accuracy,
        replace with actual C3D feature extraction.
        """
        import cv2

        features = np.zeros((self.segment_count, 4096), dtype=np.float32)

        for seg_idx in range(self.segment_count):
            start = seg_idx * self.frames_per_segment
            end = start + self.frames_per_segment
            segment_frames = self.frames[start:end]

            # Compute simple statistical features per segment
            seg_features = []
            for frame in segment_frames:
                # Resize to a small fixed size and flatten
                small = cv2.resize(frame, (16, 16))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
                seg_features.append(gray.flatten().astype(np.float32) / 255.0)

            # Stack and compute statistics: mean, std, min, max per pixel position
            seg_array = np.array(seg_features)  # (frames_per_segment, 256)
            mean_feat = seg_array.mean(axis=0)       # (256,)
            std_feat = seg_array.std(axis=0)          # (256,)

            # Compute motion features (frame differences)
            if len(seg_features) > 1:
                diffs = np.diff(seg_array, axis=0)
                motion_mean = np.abs(diffs).mean(axis=0)  # (256,)
                motion_max = np.abs(diffs).max(axis=0)     # (256,)
            else:
                motion_mean = np.zeros(256, dtype=np.float32)
                motion_max = np.zeros(256, dtype=np.float32)

            # Concatenate: 256*4 = 1024 features; tile to 4096
            combined = np.concatenate([mean_feat, std_feat, motion_mean, motion_max])
            # Repeat to fill 4096 dims
            features[seg_idx] = np.tile(combined, 4096 // len(combined) + 1)[:4096]

        return features

    def clear(self) -> None:
        """Clear the buffer."""
        self.frames.clear()
        self._last_score = 0.0
