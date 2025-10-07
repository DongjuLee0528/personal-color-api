"""Skin region extraction from face landmarks."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, MutableMapping, Tuple

import cv2
import numpy as np

# Landmark clusters roughly corresponding to cheeks and forehead in Face Mesh.
ROI_LANDMARKS: Mapping[str, Iterable[int]] = {
    "left_cheek": (234, 93, 137, 132, 58),
    "right_cheek": (454, 323, 366, 361, 288),
    "forehead": (10, 338, 297, 332, 9),
}

DEFAULT_PATCH_SIZE = 32
MIN_PATCH_RATIO = 0.6

ROI_GUARDRAIL_CODES = {
    "min_pixels": "ROI_TOO_SMALL",
    "min_brightness": "ROI_TOO_DARK",
    "min_variance": "ROI_LOW_VARIANCE",
}


def _compute_center(landmarks: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    points = landmarks[list(indices)]
    return np.mean(points, axis=0)


def _crop_patch(
    image_bgr: np.ndarray, center: np.ndarray, size: int
) -> np.ndarray | None:
    half = size // 2
    x, y = center
    top = max(int(round(y)) - half, 0)
    bottom = min(int(round(y)) + half, image_bgr.shape[0])
    left = max(int(round(x)) - half, 0)
    right = min(int(round(x)) + half, image_bgr.shape[1])

    patch = image_bgr[top:bottom, left:right]
    if patch.size == 0:
        return None

    if patch.shape[0] < size * MIN_PATCH_RATIO or patch.shape[1] < size * MIN_PATCH_RATIO:
        return None

    return patch


def _compute_patch_stats(patch_bgr: np.ndarray) -> Tuple[float, float, int]:
    lab_patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab_patch[:, :, 0] * (100.0 / 255.0)
    pixels = int(patch_bgr.shape[0] * patch_bgr.shape[1])
    mean_l = float(np.mean(l_channel))
    variance = float(np.var(l_channel))
    return mean_l, variance, pixels


def extract_skin_rois(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    patch_size: int,
    thresholds: Mapping[str, float],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """Extract cheek and forehead patches guided by facial landmarks.

    Returns both accepted ROI patches and per-ROI metrics (mean L, variance, pixels).
    """

    rois: MutableMapping[str, np.ndarray] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for name, indices in ROI_LANDMARKS.items():
        center = _compute_center(landmarks, indices)
        patch = _crop_patch(image_bgr, center, patch_size)
        if patch is None:
            continue

        mean_l, variance, pixels = _compute_patch_stats(patch)
        stats = {
            "mean_l": mean_l,
            "variance": variance,
            "pixels": float(pixels),
        }

        if pixels < thresholds["min_pixels"]:
            stats["reason"] = ROI_GUARDRAIL_CODES["min_pixels"]
            metrics[name] = stats
            continue
        if mean_l < thresholds["min_brightness"]:
            stats["reason"] = ROI_GUARDRAIL_CODES["min_brightness"]
            metrics[name] = stats
            continue
        if variance < thresholds["min_variance"]:
            stats["reason"] = ROI_GUARDRAIL_CODES["min_variance"]
            metrics[name] = stats
            continue

        rois[name] = patch
        metrics[name] = stats

    return dict(rois), metrics


__all__ = ["extract_skin_rois", "ROI_LANDMARKS", "ROI_GUARDRAIL_CODES"]
