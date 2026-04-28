"""
models/crime_detector.py — Crime detection on individual frames.

Runs the crime-detection YOLOv8 model on a single frame and returns whether
a crime was detected along with the detection bounding boxes.

Public interface
----------------
detect_crime(frame: np.ndarray) -> CrimeDetectionResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from config import CRIME_CONFIDENCE
from models.model_manager import get_crime_detector

logger = logging.getLogger(__name__)

# (x1, y1, x2, y2, confidence, class_id)
BBox = Tuple[int, int, int, int, float, int]


@dataclass
class CrimeDetectionResult:
    crime_detected: bool
    detections: List[BBox] = field(default_factory=list)
    frame: np.ndarray = field(default=None, repr=False)


def detect_crime(frame: np.ndarray) -> CrimeDetectionResult:
    """
    Run the crime-detection model on a BGR frame.

    Returns a CrimeDetectionResult. crime_detected is True when at least one
    detection exceeds CRIME_CONFIDENCE.
    """
    model = get_crime_detector()

    results = model.predict(
        source=frame,
        conf=CRIME_CONFIDENCE,
        verbose=False,
        device=None,     # model already pinned to its device
    )

    detections: List[BBox] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CRIME_CONFIDENCE:
                continue
            cls  = int(box.cls[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            detections.append((x1, y1, x2, y2, conf, cls))

    crime_detected = len(detections) > 0
    if crime_detected:
        logger.info("Crime detected: %d box(es), top_conf=%.2f",
                    len(detections), max(d[4] for d in detections))

    return CrimeDetectionResult(
        crime_detected=crime_detected,
        detections=detections,
        frame=frame,
    )
