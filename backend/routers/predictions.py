"""
backend/routers/predictions.py  (v3 — LSTM + Tree Ensemble)
============================================================
/api/predict         POST  single (zone, date) prediction  — 60/40 LSTM blend
/api/predict/all     GET   all 21 LAPD divisions for a date
/api/zones           GET   list of available zones
/api/lstm/metrics    GET   LSTM test-set performance metrics
"""
import json
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.schemas import LSTMMetrics, PredictionRequest, PredictionResponse
from src.inference import ALL_ZONES, predict_all_zones, predict_zone_date

router = APIRouter(tags=["Predictions"])


# ─────────────────────────────────────────────────────────────────────────────
@router.post("/predict", response_model=PredictionResponse)
def predict_single(req: PredictionRequest, request: Request):
    """
    Predict crime risk for a single (zone, date).

    The response includes:
    - **risk_score**         — blended probability (tree × 0.6 + LSTM × 0.4)
    - **tree_risk_score**    — XGBoost / LightGBM probability
    - **lstm_high_risk_prob**— LSTM P(High-risk) for the zone
    - **risk_level**         — Low / Medium / High (from blended score)
    """
    app   = request.app
    model = getattr(app.state, "model",        None)
    fcols = getattr(app.state, "feature_cols", None)
    fdf   = getattr(app.state, "features_df",  None)
    thr   = getattr(app.state, "opt_threshold", 0.50)
    lstm  = getattr(app.state, "lstm_bundle",  None)

    result = predict_zone_date(
        req.zone, req.date,
        model, fcols, fdf, thr,
        lstm_bundle=lstm,
        lstm_weight=req.lstm_weight,
        hour=req.hour,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ─────────────────────────────────────────────────────────────────────────────
@router.get("/predict/all", response_model=List[PredictionResponse])
def predict_all(date: str, hour: Optional[int] = None, lstm_weight: float = 0.40, request: Request = None):
    """
    Predict risk for all 21 LAPD divisions on the given date (and optional hour).

    **Query params**
    - `date`         ISO date string e.g. `2023-10-01`
    - `hour`         Optional hour (0-23)
    - `lstm_weight`  blend weight for LSTM (0–1, default 0.40)
    """
    app   = request.app
    model = getattr(app.state, "model",        None)
    fcols = getattr(app.state, "feature_cols", None)
    fdf   = getattr(app.state, "features_df",  None)
    thr   = getattr(app.state, "opt_threshold", 0.50)
    lstm  = getattr(app.state, "lstm_bundle",  None)

    results = predict_all_zones(
        date, model, fcols, fdf, thr,
        lstm_bundle=lstm, lstm_weight=lstm_weight,
        hour=hour
    )
    return [r for r in results if "error" not in r]


# ─────────────────────────────────────────────────────────────────────────────
@router.get("/zones")
def list_zones():
    """Return list of all available LAPD division names."""
    return {"zones": ALL_ZONES, "count": len(ALL_ZONES)}


# ─────────────────────────────────────────────────────────────────────────────
@router.get("/lstm/metrics", response_model=Optional[LSTMMetrics])
def lstm_metrics():
    """
    Return LSTM test-set evaluation metrics.
    Returns 404 if the LSTM has not been trained yet.
    """
    from config import MODELS_DIR
    meta_path = MODELS_DIR / "lstm_meta.json"
    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail="LSTM not trained yet. Run: python src/lstm_training.py",
        )
    with open(meta_path) as f:
        meta = json.load(f)
    return LSTMMetrics(**{k: meta[k] for k in LSTMMetrics.model_fields if k in meta})
