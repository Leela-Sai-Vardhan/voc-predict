"""FastAPI inference hub for VOC-Predict.

POST /predict  ->  pipeline.extract_features  ->  model.joblib predict_proba
GET  /health   ->  status + model meta
GET  /features ->  list of 5 model features
"""

import json
import joblib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Patch Starlette Router to accept on_startup (FastAPI 0.110.0 passes it,
# Starlette 1.10.0 Router doesn't accept it — no-op since startup handlers
# are not used by this service)
import starlette.routing as _sr
_orig_router_init = _sr.Router.__init__
def _router_init(self, *args, on_startup=None, on_shutdown=None, **kwargs):
    return _orig_router_init(self, *args, **kwargs)
_sr.Router.__init__ = _router_init

import pipeline

MODEL_PATH = Path("model.joblib")
META_PATH = Path("model_meta.json")

# Load once at startup
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None

if META_PATH.exists():
    with open(META_PATH) as f:
        meta = json.load(f)
    FEATURES = meta["features"]
    GASES = meta["gases"]
else:
    FEATURES = ["peak_response", "response_time", "rise_rate", "tau_ad_1", "tau_ad_2"]
    GASES = ["acetone", "ethanol", "isopropanol", "methanol"]

app = FastAPI(title="VOC-Predict Inference API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "no_model",
        "model_type": type(model).__name__ if model else None,
        "gases": GASES,
    }


@app.get("/features")
def features():
    return {"features": FEATURES, "gases": GASES}


def _predict_payload(time, resistance):
    """Core inference: extract features -> predict -> return dict."""
    df = pd.DataFrame({"time": np.array(time, dtype=float), "resistance": np.array(resistance, dtype=float)})
    f = pipeline.extract_features(df)
    row = pd.DataFrame([{k: f[k] for k in FEATURES}])
    proba = model.predict_proba(row)[0]
    idx = int(np.argmax(proba))
    return {
        "gas": GASES[idx],
        "probabilities": {GASES[i]: float(round(p, 3)) for i, p in enumerate(proba)},
        "peak_response": float(f["peak_response"]),
        "response_time": float(f["response_time"]) if not np.isnan(f["response_time"]) else None,
    }


@app.post("/predict")
def predict(payload: dict):
    """Predict gas from a `time`/`resistance` payload."""
    time = payload.get("time")
    resistance = payload.get("resistance")

    if time is None or resistance is None:
        raise HTTPException(status_code=400, detail="missing time or resistance")

    if not isinstance(time, list) or not isinstance(resistance, list):
        raise HTTPException(status_code=400, detail="time and resistance must be arrays")

    if len(time) != len(resistance):
        raise HTTPException(status_code=400, detail="time and resistance must have the same length")

    if len(time) < 2:
        raise HTTPException(status_code=400, detail="need at least 2 data points")

    try:
        result = _predict_payload(time, resistance)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {e}")

    return JSONResponse(content=result)