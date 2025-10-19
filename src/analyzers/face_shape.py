"""Face-shape classification utilities (MediaPipe landmarks 기반)."""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import math
import numpy as np

from .face_detection import detect_landmarks  # returns [(x,y) in [0,1]]

# 수평 띠(band) 정의: (세로 위치 비율, 띠의 반높이 비율)
Band = Tuple[float, float]
DEFAULT_BANDS: Dict[str, Band] = {
    "forehead": (0.20, 0.06),
    "cheekbone": (0.50, 0.06),
    "jaw": (0.82, 0.06),
}

# ------------------------- 내부 유틸 ------------------------- #
def _as_ndarray(landmarks: List[Tuple[float, float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("landmarks must have shape (N, 2)")
    if arr.shape[0] < 100:  # MediaPipe 468이 일반적이지만 최소 100개로 방어
        raise ValueError("Expected >=100 landmarks (MediaPipe Face Mesh recommended: 468)")
    return arr

def _band_width(pts: np.ndarray, band: Band) -> float:
    """특정 수평 띠에서의 얼굴 폭(정규화 좌표 기준, 0~1 범위)을 측정."""
    ys = pts[:, 1]
    miny, maxy = ys.min(), ys.max()
    height = maxy - miny
    cy, hh = band
    y0 = miny + cy * height  # 띠의 중심 y
    mask = np.abs(ys - y0) <= (hh * height)
    band_pts = pts[mask]
    # 띠에 점이 부족하면 근접 점으로 보강
    if band_pts.shape[0] < 8:
        k = min(24, pts.shape[0])
        idx = np.argsort(np.abs(ys - y0))[:k]
        band_pts = pts[idx]
    return float(band_pts[:, 0].max() - band_pts[:, 0].min())

def _metrics(landmarks: List[Tuple[float, float]] | np.ndarray) -> Dict[str, float]:
    """주요 지표 계산: 길이/폭, 이마/턱/광대 폭 등."""
    pts = _as_ndarray(landmarks)
    ys = pts[:, 1]
    miny, maxy = ys.min(), ys.max()
    length = float(maxy - miny)

    w_forehead = _band_width(pts, DEFAULT_BANDS["forehead"])
    w_cheek = _band_width(pts, DEFAULT_BANDS["cheekbone"])
    w_jaw = _band_width(pts, DEFAULT_BANDS["jaw"])

    LWR = length / (w_cheek + 1e-6)       # Length-to-Width (cheek)
    FJ  = w_forehead / (w_jaw + 1e-6)     # Forehead/Jaw
    CWF = w_cheek / (w_forehead + 1e-6)   # Cheek/Forehead
    CWJ = w_cheek / (w_jaw + 1e-6)        # Cheek/Jaw

    return {
        "face_length": length,
        "forehead_width": w_forehead,
        "cheekbone_width": w_cheek,
        "jaw_width": w_jaw,
        "LWR": LWR, "FJ": FJ, "CWF": CWF, "CWJ": CWJ,
    }

def _dist_to_interval(x: float, lo: float, hi: float) -> float:
    if lo <= x <= hi:
        return 0.0
    return min(abs(x - lo), abs(x - hi))

def _rule_penalties(m: Dict[str, float]) -> Dict[str, float]:
    """
    규칙에서 벗어난 정도(패널티). 값이 작을수록 해당 클래스일 가능성이 높음.
    임계값은 경험적으로 설정했으며 향후 데이터로 튜닝 가능.
    """
    LWR, FJ, CWF, CWJ = m["LWR"], m["FJ"], m["CWF"], m["CWJ"]
    p: Dict[str, float] = {}
    p["Oblong"]  = max(0.0, 1.55 - LWR) + _dist_to_interval(FJ, 0.88, 1.12)
    p["Round"]   = max(0.0, LWR - 1.05) + max(0.0, CWJ - 1.10)
    p["Square"]  = _dist_to_interval(FJ, 0.93, 1.07) + _dist_to_interval(CWF, 0.93, 1.07) + max(0.0, LWR - 1.20)
    p["Heart"]   = max(0.0, 1.18 - FJ) + max(0.0, 1.05 - CWF) + max(0.0, 1.05 - CWJ)
    p["Diamond"] = max(0.0, 1.12 - CWF) + max(0.0, 1.12 - CWJ) + _dist_to_interval(FJ, 0.92, 1.10)
    p["Oval"]    = _dist_to_interval(LWR, 1.25, 1.50) + _dist_to_interval(FJ, 0.95, 1.10) + _dist_to_interval(CWF, 0.95, 1.10)
    return p

def _softmax_from_penalties(p: Dict[str, float], alpha: float = 4.0) -> Dict[str, float]:
    """작은 패널티 → 큰 확률이 되도록 변환."""
    keys = list(p.keys())
    scores = np.array([math.exp(-alpha * p[k]) for k in keys], dtype=np.float64)
    probs = (scores / scores.sum()).tolist()
    return {k: float(v) for k, v in zip(keys, probs)}

def classify_with_confidence(m: Dict[str, float]) -> Dict[str, object]:
    penalties = _rule_penalties(m)
    probs = _softmax_from_penalties(penalties)
    best = max(probs.items(), key=lambda kv: kv[1])[0]
    top2 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:2]
    return {
        "shape": best,                    # 'Oval' | 'Oblong' | ...
        "confidence": probs[best],        # 0~1
        "top2": [{"label": k, "prob": v} for k, v in top2],
        "penalties": penalties,
        "probs": probs,
    }

# ------------------------- 외부 API ------------------------- #
def analyze_face_shape(
    image_bgr,
    landmarks: Optional[List[Tuple[float, float]]] = None,
    debug: bool = False,
) -> Dict:
    """
    이미지(BGR) 또는 landmarks로 얼굴형을 분석.
    반환: {"status","shape","confidence","top2","metrics", ("debug")}
    """
    if landmarks is None:
        lm = detect_landmarks(image_bgr)
        if lm is None or lm.size == 0:
            return {"status": "guardrail", "code": "NO_FACE"}
        landmarks = lm

    m = _metrics(landmarks)
    res = classify_with_confidence(m)
    out = {
        "status": "ok",
        "shape": res["shape"],
        "confidence": res["confidence"],
        "top2": res["top2"],
        "metrics": m,
    }
    if debug:
        out["debug"] = {"bands": DEFAULT_BANDS, "probs": res["probs"], "penalties": res["penalties"]}
    return out

def classify_face_shape(landmarks: np.ndarray) -> Tuple[str, float]:
    """
    (이전 placeholder와 호환) MediaPipe 468x2 landmarks → (label, confidence)
    label은 영문 enum: 'Oval' | 'Oblong' | 'Round' | 'Square' | 'Heart' | 'Diamond'
    """
    m = _metrics(landmarks)
    res = classify_with_confidence(m)
    return res["shape"], float(res["confidence"])

__all__ = ["analyze_face_shape", "classify_face_shape", "classify_with_confidence"]
