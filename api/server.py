"""
FastAPI Backend — Steel Defect Detection API
---------------------------------------------
Endpoints:
  POST /predict        — upload image, get prediction JSON + annotated image
  GET  /health         — liveness check
  GET  /classes        — list of defect class names
  GET  /docs           — Swagger UI (auto-generated)

Run
---
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models.classifier import DEFECT_CLASSES
from utils.inference_engine import InferenceEngine


# ── App & CORS ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Steel Surface Defect Detection API",
    description=(
        "Real-time defect detection on steel mill surface images.\n\n"
        "Pipeline: OpenCV edge/blob detection → EfficientNet-B0 CNN → Grad-CAM"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: load model once ────────────────────────────────────────────────

CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "checkpoints/best_model.pt")
DEVICE     = os.environ.get("DEVICE", "cpu")

engine: InferenceEngine = None


@app.on_event("startup")
async def load_engine():
    global engine
    engine = InferenceEngine(
        checkpoint_path=CHECKPOINT,
        device=DEVICE,
        use_gradcam=True,
    )


# ── Response schemas ────────────────────────────────────────────────────────

class RegionSchema(BaseModel):
    x: int
    y: int
    w: int
    h: int
    area: int
    aspect_ratio: float
    mean_intensity: float
    edge_density: float


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    all_probabilities: dict[str, float]
    defect_regions: list[RegionSchema]
    annotated_image_b64: str    # base64-encoded JPEG of the annotated frame
    gradcam_available: bool
    processing_ms: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=engine is not None,
        device=DEVICE,
    )


@app.get("/classes", tags=["System"])
async def list_classes():
    return {"classes": DEFECT_CLASSES}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Steel surface image (JPEG/PNG)"),
    edge_threshold: int = Query(40, ge=5, le=200, description="Sobel edge threshold"),
    return_gradcam: bool = Query(True, description="Include Grad-CAM overlay"),
):
    """
    Analyse a steel surface image and return:
    - CNN defect classification with per-class probabilities
    - OpenCV-detected anomaly regions (bounding boxes)
    - Base64-encoded annotated image (Grad-CAM + bounding boxes)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    raw = await file.read()
    nparr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=422, detail="Cannot decode image.")

    # Update edge threshold per-request (thread-safe enough for demo)
    engine.prep.edge_threshold = edge_threshold
    engine.gradcam = engine.gradcam if return_gradcam else None

    t0 = time.perf_counter()
    result = engine.predict(bgr)
    ms = int((time.perf_counter() - t0) * 1000)

    # Encode annotated image as base64 JPEG
    _, buf = cv2.imencode(".jpg", result.annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    return PredictionResponse(
        label=result.label,
        confidence=round(result.confidence, 4),
        all_probabilities=result.all_probs,
        defect_regions=[
            RegionSchema(
                x=r.x, y=r.y, w=r.w, h=r.h,
                area=r.area,
                aspect_ratio=r.aspect_ratio,
                mean_intensity=r.mean_intensity,
                edge_density=r.edge_density,
            )
            for r in result.regions
        ],
        annotated_image_b64=b64,
        gradcam_available=result.gradcam_heatmap is not None,
        processing_ms=ms,
    )


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(
    files: list[UploadFile] = File(..., description="Up to 16 images"),
):
    """Run prediction on multiple images. Returns list of results."""
    if len(files) > 16:
        raise HTTPException(status_code=400, detail="Max 16 images per batch.")
    results = []
    for f in files:
        raw = await f.read()
        nparr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            results.append({"filename": f.filename, "error": "Cannot decode image"})
            continue
        res = engine.predict(bgr)
        results.append({
            "filename": f.filename,
            "label": res.label,
            "confidence": round(res.confidence, 4),
            "all_probabilities": res.all_probs,
            "defect_regions_count": len(res.regions),
        })
    return {"predictions": results}
