"""
detection/detector.py — YOLOv8 person detector.

Wraps ultralytics YOLO with classes=[0] (person only).
Returns a list of detection dicts with bbox, confidence, and crop.

This module is independent of the crime detector — it runs in the tracking
pipeline to find persons, while the crime detector runs in the consumer
loop to trigger crime events.

Public interface
----------------
ShadowDetector(model, conf)
    .detect(frame) -> list[dict]

Each dict:
    bbox:        [x, y, w, h]       (LTWH format for DeepSORT)
    conf:        float
    crop:        np.ndarray          (BGR)
    xyxy:        [x1, y1, x2, y2]   (absolute pixel coords)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from config import PERSON_CONFIDENCE, PERSON_DETECT_MODEL

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once, shared across threads
_detector: Optional[YOLO] = None


def _get_model() -> YOLO:
    """Lazy-load the YOLO model on first use."""
    global _detector
    if _detector is not None:
        return _detector

    model_path = PERSON_DETECT_MODEL
    if model_path.exists():
        _detector = YOLO(str(model_path))
        logger.info("ShadowDetector loaded: %s", model_path.name)
    else:
        logger.info("Downloading YOLOv8n weights for person detection.")
        _detector = YOLO("yolov8n.pt")

    return _detector


class ShadowDetector:
    """YOLOv8 person-only detector.

    Parameters
    ----------
    model : str or Path
        Path to YOLO weights file, or a model name like "yolov8n.pt".
        Defaults to the path configured in config.PERSON_DETECT_MODEL.
    conf : float
        Minimum confidence threshold for detections.
        Defaults to config.PERSON_CONFIDENCE.
    """

    def __init__(
        self,
        model: str | Path | None = None,
        conf: float = PERSON_CONFIDENCE,
    ):
        if model is not None:
            self._model = YOLO(str(model))
        else:
            self._model = _get_model()
        self.conf = conf

    def detect(self, frame: np.ndarray) -> List[dict]:
        """Detect persons in a single BGR frame.

        Returns
        -------
        list[dict]
            Each dict contains:
            - bbox:  [x, y, w, h]     (left, top, width, height — for DeepSORT)
            - conf:  float            (detection confidence)
            - crop:  np.ndarray       (BGR crop of the person)
            - xyxy:  [x1, y1, x2, y2] (absolute pixel coords)
        """
        if frame is None or frame.size == 0:
            return []

        results = self._model(
            frame,
            classes=[0],           # COCO class 0 = person
            conf=self.conf,
            verbose=False,
        )[0]

        out: List[dict] = []
        h_frame, w_frame = frame.shape[:2]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_frame, x2)
            y2 = min(h_frame, y2)

            w = x2 - x1
            h = y2 - y1

            # Skip degenerate boxes
            if w < 10 or h < 10:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out.append({
                "bbox": [x1, y1, w, h],       # LTWH for DeepSORT
                "conf": conf,
                "crop": crop,
                "xyxy": [x1, y1, x2, y2],
            })

        return out
