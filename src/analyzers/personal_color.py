"""Personal color inference pipeline built on MediaPipe landmarks."""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Mapping, Optional, Tuple

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import cv2
import numpy as np
from skimage import color as skcolor

from .face_detection import denormalise_landmarks, detect_landmarks
from .skin_roi import DEFAULT_PATCH_SIZE, extract_skin_rois

logger = logging.getLogger(__name__)


DEFAULT_PALETTES: Dict[str, list[str]] = {
    "Spring": ["#F8C2A3", "#F5E6A8", "#F7DAD9", "#FFB7B2"],
    "Summer": ["#A8C8E8", "#E9BFD1", "#9EC3B1", "#F5E6A8"],
    "Autumn": ["#B46A55", "#D39F6B", "#F2C572", "#865640"],
    "Winter": ["#5B6DCE", "#B6CAE3", "#6ED0D4", "#F5F5F5"],
}


THRESHOLDS: Dict[str, Dict[str, float]] = {
    "ita": {"warm": 28.0, "cool": 10.0},
    "L": {"too_dark": 25.0, "too_bright": 90.0},
    "roi": {"min_pixels": 900.0, "min_brightness": 10.0, "min_variance": 5.0},
}


def classify_from_metrics(metrics: Mapping[str, float]) -> Tuple[str, str]:
    """Return season and tone labels derived from LAB metrics."""

    ita = metrics["ITA"]
    l_star = metrics["L"]
    b_star = metrics["b"]

    warm_threshold = THRESHOLDS["ita"]["warm"]
    cool_threshold = THRESHOLDS["ita"]["cool"]

    if ita >= warm_threshold:
        tone = "Warm"
    elif ita <= cool_threshold:
        tone = "Cool"
    else:
        tone = "Warm" if b_star >= 0 else "Cool"

    if tone == "Warm":
        season = "Spring" if l_star >= 55.0 else "Autumn"
    else:
        season = "Summer" if l_star >= 55.0 else "Winter"

    return season, tone


class PersonalColorAnalyzer:
    """Run personal color analysis over a BGR image."""

    def __init__(self, min_valid_rois: int = 1) -> None:
        self.min_valid_rois = min_valid_rois

    def analyze(
        self, image_bgr: np.ndarray, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        opts: Dict[str, Any] = dict(options or {})
        trace_id = opts.get("trace_id")

        landmarks_norm = detect_landmarks(image_bgr)
        if landmarks_norm is None:
            return self._build_response(
                status="guardrail", code="NO_FACE", trace_id=trace_id
            )

        height, width = image_bgr.shape[:2]
        landmarks_px = denormalise_landmarks(landmarks_norm, width, height)

        rois, roi_metrics = extract_skin_rois(
            image_bgr,
            landmarks_px,
            patch_size=DEFAULT_PATCH_SIZE,
            thresholds=THRESHOLDS["roi"],
        )

        if len(rois) < self.min_valid_rois:
            logger.info("ROI guardrail triggered", extra={"roi_metrics": roi_metrics})
            return self._build_response(
                status="guardrail",
                code="LOW_QUALITY",
                trace_id=trace_id,
                landmarks=landmarks_norm,
            )

        metrics = self._compute_color_metrics(rois)
        season, tone = classify_from_metrics(metrics)
        palette = DEFAULT_PALETTES.get(season, DEFAULT_PALETTES["Summer"])

        return self._build_response(
            status="ok",
            season=season,
            tone=tone,
            metrics=metrics,
            palette=palette,
            landmarks=landmarks_norm,
            trace_id=trace_id,
        )

    def _compute_color_metrics(self, rois: Dict[str, np.ndarray]) -> Dict[str, float]:
        lab_values = []
        for name, patch_bgr in rois.items():
            rgb_patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            lab_patch = skcolor.rgb2lab(rgb_patch.astype(np.float32) / 255.0)
            mean_lab = lab_patch.reshape(-1, 3).mean(axis=0)
            lab_values.append(mean_lab)
            logger.debug("ROI %s mean LAB: %s", name, mean_lab)

        lab_array = np.vstack(lab_values)
        avg_L, avg_a, avg_b = lab_array.mean(axis=0)
        ita = math.degrees(math.atan((avg_L - 50.0) / (avg_b + 1e-6)))

        metrics = {
            "L": round(float(avg_L), 2),
            "a": round(float(avg_a), 2),
            "b": round(float(avg_b), 2),
            "ITA": round(float(ita), 2),
        }
        return metrics

    def _build_response(
        self,
        *,
        status: str,
        code: Optional[str] = None,
        season: Optional[str] = None,
        tone: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        palette: Optional[list[str]] = None,
        landmarks: Optional[np.ndarray] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"status": status}
        if code:
            payload["code"] = code
        if season:
            payload["season"] = season
        if tone:
            payload["tone"] = tone
        if metrics:
            payload["metrics"] = metrics
        if palette:
            payload["palette"] = palette
        if landmarks is not None:
            payload["landmarks"] = landmarks.tolist()
        if trace_id:
            payload["traceId"] = trace_id
        return payload


__all__ = ["PersonalColorAnalyzer", "THRESHOLDS", "classify_from_metrics"]
