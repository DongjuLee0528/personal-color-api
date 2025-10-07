"""FastAPI application exposing personal color inference APIs."""
from __future__ import annotations

import base64
import binascii
import logging
import re
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

import cv2
from fastapi import Body, FastAPI, Header, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette import status
from starlette.responses import Response

from src.analyzers import PersonalColorAnalyzer, analyze_face_shape
from src.analyzers.face_detection import load_image_to_bgr

logger = logging.getLogger(__name__)
app = FastAPI(title="Personal Color Service", version="1.0.0")

DATA_URL_PATTERN = re.compile(r"^data:image/[^;]+;base64,")

analyzer = PersonalColorAnalyzer()


class AnalyzeOptions(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    trace_id: Optional[str] = Field(default=None, alias="traceId")
    exif_correction: bool = Field(default=True, alias="exif_correction")
    debug: Optional[bool] = Field(default=None, alias="debug")

    @model_validator(mode="before")
    @classmethod
    def _compat(cls, values: Any):  # type: ignore[override]
        if not isinstance(values, dict):
            return values
        # 역호환: camelCase → snake_case
        if "trace_id" not in values and "traceId" in values:
            values["trace_id"] = values.pop("traceId")
        if "exif_correction" not in values and "exifCorrection" in values:
            values["exif_correction"] = values.pop("exifCorrection")
        return values


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    image_base64: str = Field(alias="image_base64")
    options: Optional[AnalyzeOptions] = None

    @model_validator(mode="before")
    @classmethod
    def _compat(cls, values: Any):  # type: ignore[override]
        if not isinstance(values, dict):
            return values
        # 역호환: camelCase → snake_case
        if "image_base64" not in values and "imageBase64" in values:
            values["image_base64"] = values.pop("imageBase64")
        return values


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    status: Literal["ok", "guardrail", "error"]
    code: Optional[str] = None
    season: Optional[str] = None
    tone: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    palette: Optional[list[str]] = None
    traceId: Optional[str] = None
    debug_image_base64: Optional[str] = None


def _decode_base64_image(data: str) -> Optional[bytes]:
    if DATA_URL_PATTERN.match(data):
        _, encoded = data.split(",", 1)
    else:
        encoded = data
    try:
        return base64.b64decode(encoded)
    except (ValueError, binascii.Error):  # pragma: no cover - defensive
        return None


def _draw_landmarks(image_bgr, landmarks):
    overlay = image_bgr.copy()
    height, width = overlay.shape[:2]
    for x_norm, y_norm in landmarks:
        x = int(round(x_norm * width))
        y = int(round(y_norm * height))
        cv2.circle(overlay, (x, y), 1, (0, 255, 0), thickness=-1)
    success, buffer = cv2.imencode(".png", overlay)
    if not success:  # pragma: no cover - OpenCV failure is rare but handled
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _run_analysis(image_bgr, trace_id: str, debug: bool) -> AnalyzeResponse:
    analysis = analyzer.analyze(image_bgr, options={"trace_id": trace_id})
    landmarks = analysis.pop("landmarks", None)

    debug_image = None
    if debug and landmarks is not None:
        debug_image = _draw_landmarks(image_bgr, landmarks)

    analysis.setdefault("traceId", trace_id)
    if debug_image:
        analysis["debug_image_base64"] = debug_image

    return AnalyzeResponse(**analysis)


def _response_with_trace(payload: AnalyzeResponse, trace_id: str) -> JSONResponse:
    response = JSONResponse(
        content=payload.model_dump(exclude_none=True),  # pydantic v2 직렬화
        status_code=status.HTTP_200_OK,
    )
    response.headers["X-Trace-Id"] = trace_id
    return response


def _response_with_trace_dict(payload: Dict[str, Any], trace_id: str) -> JSONResponse:
    """AnalyzeResponse 외 임의 dict 응답에 X-Trace-Id 헤더를 부여."""
    resp = JSONResponse(content=payload, status_code=status.HTTP_200_OK)
    resp.headers["X-Trace-Id"] = trace_id
    return resp


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(
    request: AnalyzeRequest = Body(...),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-Id"),
    debug: bool = Query(default=False),
) -> Response:
    options = request.options
    trace_id = x_trace_id or (options.trace_id if options else None) or str(uuid4())

    effective_debug = debug or (options.debug if options and options.debug is not None else False)
    exif_correction = options.exif_correction if options else True

    image_bytes = _decode_base64_image(request.image_base64)
    if image_bytes is None:
        error_payload = AnalyzeResponse(status="error", code="INVALID_IMAGE", traceId=trace_id)
        return _response_with_trace(error_payload, trace_id)

    try:
        image_bgr, _ = load_image_to_bgr(image_bytes, exif_correction=exif_correction)
    except (UnidentifiedImageError, OSError):
        logger.debug("Failed to decode base64 image", exc_info=True)
        error_payload = AnalyzeResponse(status="error", code="INVALID_IMAGE", traceId=trace_id)
        return _response_with_trace(error_payload, trace_id)

    payload = _run_analysis(image_bgr, trace_id, effective_debug)
    return _response_with_trace(payload, trace_id)


@app.post("/analyze/file")
def analyze_file(
    file: UploadFile,
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-Id"),
    debug: bool = Query(default=False),
    exif_correction: bool = Query(default=True),
) -> Response:
    trace_id = x_trace_id or str(uuid4())

    image_bytes = file.file.read()
    if not image_bytes:
        error_payload = AnalyzeResponse(status="error", code="INVALID_IMAGE", traceId=trace_id)
        return _response_with_trace(error_payload, trace_id)

    try:
        image_bgr, _ = load_image_to_bgr(image_bytes, exif_correction=exif_correction)
    except (UnidentifiedImageError, OSError):  # pragma: no cover
        logger.debug("Failed to decode uploaded image", exc_info=True)
        error_payload = AnalyzeResponse(status="error", code="INVALID_IMAGE", traceId=trace_id)
        return _response_with_trace(error_payload, trace_id)

    payload = _run_analysis(image_bgr, trace_id, debug)
    return _response_with_trace(payload, trace_id)


# ------------------ Face shape endpoints ------------------ #
@app.post("/face-shape")
def face_shape_json(
    request: AnalyzeRequest = Body(...),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-Id"),
    debug: bool = Query(default=False),
) -> Response:
    options = request.options
    trace_id = x_trace_id or (options.trace_id if options else None) or str(uuid4())
    exif_correction = options.exif_correction if options else True

    image_bytes = _decode_base64_image(request.image_base64)
    if image_bytes is None:
        return _response_with_trace_dict(
            {"status": "error", "code": "INVALID_IMAGE", "traceId": trace_id},
            trace_id,
        )

    try:
        image_bgr, _ = load_image_to_bgr(image_bytes, exif_correction=exif_correction)
    except (UnidentifiedImageError, OSError):
        logger.debug("Failed to decode base64 image (face-shape)", exc_info=True)
        return _response_with_trace_dict(
            {"status": "error", "code": "INVALID_IMAGE", "traceId": trace_id},
            trace_id,
        )

    result = analyze_face_shape(image_bgr, debug=debug)
    result.setdefault("traceId", trace_id)
    return _response_with_trace_dict(result, trace_id)


@app.post("/face-shape/file")
def face_shape_file(
    file: UploadFile,
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-Id"),
    debug: bool = Query(default=False),
    exif_correction: bool = Query(default=True),
) -> Response:
    trace_id = x_trace_id or str(uuid4())

    image_bytes = file.file.read()
    if not image_bytes:
        return _response_with_trace_dict(
            {"status": "error", "code": "INVALID_IMAGE", "traceId": trace_id},
            trace_id,
        )

    try:
        image_bgr, _ = load_image_to_bgr(image_bytes, exif_correction=exif_correction)
    except (UnidentifiedImageError, OSError):  # pragma: no cover
        logger.debug("Failed to decode uploaded image (face-shape)", exc_info=True)
        return _response_with_trace_dict(
            {"status": "error", "code": "INVALID_IMAGE", "traceId": trace_id},
            trace_id,
        )

    result = analyze_face_shape(image_bgr, debug=debug)
    result.setdefault("traceId", trace_id)
    return _response_with_trace_dict(result, trace_id)


__all__ = ("app",)
