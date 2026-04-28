"""
models/model_manager.py — Singleton loader for all AI models.

Models loaded
-------------
1. crime_detector  : YOLOv8 fine-tuned for violence / crime detection
2. reid_model      : OSNet x0_25 for person Re-ID embeddings (placeholder until CLIP)

All models are loaded once at startup via load_all_models().
Individual model accessors are module-level functions used by other modules.

Public interface
----------------
load_all_models() -> None
get_crime_detector() -> YOLO
get_reid_model()    -> torch.nn.Module
get_mil_model()     -> AnomalyMILModel | None
extract_embedding(crop: np.ndarray) -> np.ndarray   # shape (REID_EMBED_DIM,)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO

from config import (
    CRIME_MODEL_PATH,
    MIL_MODEL_PATH,
    REID_EMBED_DIM,
    REID_MODEL_PATH,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons (populated by load_all_models)
# ─────────────────────────────────────────────────────────────────────────────
_crime_detector: Optional[YOLO] = None
_reid_model: Optional[torch.nn.Module] = None
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pre-processing pipeline for Re-ID crops (standard ImageNet normalisation)
_reid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),   # standard Re-ID input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# OSNet lightweight wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _OSNetWrapper(torch.nn.Module):
    """
    Thin wrapper around torchreid's OSNet that exposes a single
    forward(x) → L2-normalised feature vector.

    Falls back to a lightweight ResNet-18 backbone if torchreid is not
    installed, so the project still runs for development without Re-ID weights.
    """

    def __init__(self, weights_path: Path) -> None:
        super().__init__()
        try:
            import torchreid  # type: ignore
            self.backbone = torchreid.models.build_model(
                name="osnet_x0_25",
                num_classes=1000,
                pretrained=False,
            )
            if weights_path.exists():
                state = torch.load(weights_path, map_location="cpu")
                # torchreid checkpoints sometimes wrap state under 'state_dict'
                sd = state.get("state_dict", state)
                self.backbone.load_state_dict(sd, strict=False)
                logger.info("OSNet weights loaded from %s", weights_path)
            else:
                logger.warning(
                    "Re-ID weights not found at %s — using random initialisation.",
                    weights_path,
                )
            # Remove classifier head; keep feature extractor only
            self.backbone.classifier = torch.nn.Identity()
            self._embed_dim = 512
        except ImportError:
            logger.warning(
                "torchreid not installed — falling back to ResNet-18 for Re-ID."
            )
            import torchvision.models as tvm
            base = tvm.resnet18(weights=None)
            self.backbone = torch.nn.Sequential(*list(base.children())[:-1])
            self._embed_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        return F.normalize(feat, p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_all_models() -> None:
    """Load all models into memory. Called once at application startup."""
    global _crime_detector, _reid_model

    logger.info("Loading AI models on device: %s", _device)

    # 1. Crime detector
    if CRIME_MODEL_PATH.exists():
        _crime_detector = YOLO(str(CRIME_MODEL_PATH))
        logger.info("Crime detector loaded: %s", CRIME_MODEL_PATH.name)
    else:
        # Fallback to COCO-pretrained for testing (labels won't be crime-specific)
        logger.warning(
            "Crime model not found at %s — loading yolov8n.pt as placeholder.",
            CRIME_MODEL_PATH,
        )
        _crime_detector = YOLO("yolov8n.pt")

    # Move YOLO model to target device
    _crime_detector.to(_device)

    # 2. Re-ID model
    _reid_model = _OSNetWrapper(REID_MODEL_PATH)
    _reid_model.to(_device)
    _reid_model.eval()
    logger.info("Re-ID model ready (embed_dim=%d).", REID_EMBED_DIM)

    # 3. MIL Anomaly model (optional — graceful fallback if not trained yet)
    from training.inference_mil import load_mil_model, is_mil_available
    loaded = load_mil_model(str(MIL_MODEL_PATH))
    if loaded:
        logger.info("MIL anomaly model loaded: %s", MIL_MODEL_PATH.name)
    else:
        logger.warning(
            "MIL anomaly model not available — crime detection will use YOLO only."
        )


def get_crime_detector() -> YOLO:
    if _crime_detector is None:
        raise RuntimeError("Models not loaded. Call load_all_models() first.")
    return _crime_detector



def get_reid_model() -> torch.nn.Module:
    if _reid_model is None:
        raise RuntimeError("Models not loaded. Call load_all_models() first.")
    return _reid_model


def get_mil_model():
    """Return the MIL anomaly model, or None if not loaded."""
    from training.inference_mil import _mil_model
    return _mil_model


@torch.no_grad()
def extract_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Given a BGR crop of a car, return its L2-normalised Re-ID embedding
    as a float32 numpy array of shape (REID_EMBED_DIM,).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros(REID_EMBED_DIM, dtype=np.float32)

    # BGR → RGB for torchvision
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = _reid_transform(rgb).unsqueeze(0).to(_device)   # (1, 3, 256, 128)
    feat: torch.Tensor = _reid_model(tensor)                  # (1, D)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)


@torch.no_grad()
def extract_embeddings_batch(crops_bgr: list) -> np.ndarray:
    """
    Batch version of extract_embedding.

    Parameters
    ----------
    crops_bgr : list of np.ndarray (BGR)

    Returns
    -------
    np.ndarray  shape (N, REID_EMBED_DIM)
    """
    if not crops_bgr:
        return np.empty((0, REID_EMBED_DIM), dtype=np.float32)

    tensors = []
    for crop in crops_bgr:
        if crop is None or crop.size == 0:
            tensors.append(torch.zeros(3, 256, 128))
        else:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(_reid_transform(rgb))

    batch = torch.stack(tensors).to(_device)        # (N, 3, 256, 128)
    feats: torch.Tensor = _reid_model(batch)        # (N, D)
    return feats.cpu().numpy().astype(np.float32)
