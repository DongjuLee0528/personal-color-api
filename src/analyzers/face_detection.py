"""Face detection helpers built on top of MediaPipe Face Mesh."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from io import BytesIO
from typing import Optional, Tuple

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)


def load_image_to_bgr(
    image_bytes: bytes, exif_correction: bool = True
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Decode raw image bytes into a BGR ndarray and return its size.

    Parameters
    ----------
    image_bytes:
        Raw encoded image payload (e.g. JPEG/PNG).
    exif_correction:
        Whether to apply EXIF orientation transpose before conversion.

    Returns
    -------
    Tuple[np.ndarray, Tuple[int, int]]
        OpenCV-style BGR image and ``(width, height)`` tuple.

    Raises
    ------
    UnidentifiedImageError
        If the payload cannot be parsed as an image.
    """

    with Image.open(BytesIO(image_bytes)) as img:
        if exif_correction:
            img = ImageOps.exif_transpose(img)
        rgb_image = img.convert("RGB")
        np_rgb = np.asarray(rgb_image)

    bgr_image = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr_image.shape[:2]
    return bgr_image, (width, height)


@lru_cache(maxsize=1)
def _get_face_mesh() -> mp.solutions.face_mesh.FaceMesh:
    """Create (and memoize) the MediaPipe FaceMesh graph."""

    logger.info("Initialising MediaPipe FaceMesh")
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


def detect_landmarks(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Detect 468 landmarks and return normalised (x, y) coordinates."""

    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    face_mesh = _get_face_mesh()
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        logger.debug("No face landmarks detected by MediaPipe")
        return None

    face_landmarks = results.multi_face_landmarks[0].landmark
    coords = np.zeros((len(face_landmarks), 2), dtype=np.float32)

    for idx, landmark in enumerate(face_landmarks):
        coords[idx, 0] = landmark.x
        coords[idx, 1] = landmark.y

    logger.debug("Detected %d landmarks", len(face_landmarks))
    return coords


def denormalise_landmarks(
    landmarks: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Convert ``[0, 1]`` landmark coordinates into pixel positions."""

    scale = np.array([width, height], dtype=np.float32)
    return landmarks * scale


__all__ = ["load_image_to_bgr", "detect_landmarks", "denormalise_landmarks"]
