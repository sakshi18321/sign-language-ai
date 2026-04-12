"""
Sign Language Detection API - Fixed for Python 3.10 + Windows
Supports both landmark-based and image-based prediction (server-side MediaPipe).
"""

from __future__ import annotations

import os, json, pickle, base64
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any
from io import BytesIO

import tensorflow as tf
from PIL import Image
import mediapipe as mp

# ── Load model ────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "asl_model.h5"))

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
    meta = json.load(f)

LABELS      = meta["labels"]
FEATURE_DIM = meta["feature_dim"]
print(f"Model loaded | accuracy={meta['accuracy']:.4f} | classes={len(LABELS)}")

# ── MediaPipe (server-side) ───────────────────────────────────────────────────
mp_hands_module = mp.solutions.hands
mp_hands = mp_hands_module.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sign Language Detection API", version="2.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class LandmarkPayload(BaseModel):
    landmarks: List[float]

class ImagePayload(BaseModel):
    image: str  # base64-encoded JPEG/PNG (with or without data-URI prefix)

class PredictionResponse(BaseModel):
    letter: str
    confidence: float
    top5: List[Any]

class TTSPayload(BaseModel):
    text: str

class TTSResponse(BaseModel):
    letters: List[str]
    message: str

# ── Helper ────────────────────────────────────────────────────────────────────
def run_model(flat: list[float]) -> PredictionResponse:
    print("Received landmarks:", len(flat))   # DEBUG
    print("Expected:", FEATURE_DIM)

    arr = np.array(flat, dtype=np.float32).reshape(1, -1)
    print("Input shape:", arr.shape)          # DEBUG

    probs = model.predict(arr, verbose=0)[0]
    print("Raw output:", probs)               # DEBUG

    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [
        {"letter": LABELS[i], "confidence": round(float(probs[i]), 4)}
        for i in top5_idx
    ]

    best = int(np.argmax(probs))

    return PredictionResponse(
        letter=LABELS[best],
        confidence=round(float(probs[best]), 4),
        top5=top5,
    )
# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "labels": LABELS,
        "feature_dim": FEATURE_DIM,
        "model_accuracy": meta["accuracy"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: LandmarkPayload):
    """Accept 63 pre-extracted landmark floats (from browser MediaPipe)."""
    return run_model(payload.landmarks)


@app.post("/predict-image", response_model=PredictionResponse)
def predict_image(payload: ImagePayload):
    """
    Accept a base64 image, run server-side MediaPipe to extract landmarks,
    then classify. This is the reliable fallback when browser MediaPipe fails.
    """
    # Strip data-URI prefix if present  (e.g. "data:image/jpeg;base64,...")
    raw = payload.image
    if "," in raw:
        raw = raw.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    results = mp_hands.process(img_np)

    if not results.multi_hand_landmarks:
        raise HTTPException(status_code=422, detail="No hand detected in frame")

    lm   = results.multi_hand_landmarks[0]
    flat = [v for p in lm.landmark for v in (p.x, p.y, p.z)]
    return run_model(flat)


@app.post("/tts", response_model=TTSResponse)
def text_to_sign(payload: TTSPayload):
    text    = payload.text.upper().strip()
    letters = [
        ch if ch in LABELS else "SPACE"
        for ch in text
        if ch in LABELS or ch == " "
    ]
    return TTSResponse(
        letters=letters,
        message=f"Showing {len(letters)} signs for: '{payload.text}'",
    )


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}