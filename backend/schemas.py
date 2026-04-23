"""
backend/schemas.py - Pydantic request/response models (v3 - LSTM ensemble)
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    zone: str = Field(..., description="Name of the division/area (e.g., '77th Street')")
    date: str = Field(..., description="Target date in YYYY-MM-DD format")
    hour: Optional[int] = Field(None, description="Target hour for prediction (0-23)", ge=0, le=23)
    lstm_weight: Optional[float] = Field(
        0.4, 
        description="Weight of LSTM in the ensemble (0.0 to 1.0)",
        ge=0.0, le=1.0
    )


class PredictionResponse(BaseModel):
    zone:                 str
    date:                 str
    hour:                 Optional[int] = None
    risk_score:           float          # blended probability (after scaling)
    base_risk_score:      Optional[float] = None # original blended prob before scaling
    tod_multiplier:       Optional[float] = None # Time of Day scaling multiplier
    risk_level:           str            # Low | Medium | High
    predicted_high_risk:  int            # 0 or 1
    tree_risk_score:      Optional[float] = None   # XGBoost / LightGBM probability
    lstm_high_risk_prob:  Optional[float] = None   # LSTM P(High risk)
    lstm_risk_label:      Optional[str]  = None   # LSTM risk class
    avg_lat:              Optional[float] = None
    avg_lon:              Optional[float] = None
    zone_id:              Optional[int]  = None


class HotspotCluster(BaseModel):
    cluster: int
    lat: float
    lon: float
    size: int
    method: str


class ZoneStats(BaseModel):
    zone: str
    total_crimes: int
    avg_daily: float
    max_daily: int
    high_risk_days_pct: float


class ModelMetrics(BaseModel):
    model:     str
    accuracy:  float
    precision: float
    recall:    float
    f1:        float
    roc_auc:   float


class LSTMMetrics(BaseModel):
    """Summary metrics from the trained BiLSTM model."""
    test_mae:        float
    test_rmse:       float
    test_risk_acc:   float
    test_risk_f1:    float
    baseline_mae:    float
    improvement:     float
    n_train:         int
    n_val:           int
    n_test:          int
    best_val_loss:   float
