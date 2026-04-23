"""
backend/routers/stats.py
/api/stats - model metrics, crime trends, zone summaries
"""
import sys
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.schemas import ModelMetrics, ZoneStats
from config import REPORTS_DIR, DATA_DIR

router = APIRouter(tags=["Statistics"])


def _load_metrics() -> pd.DataFrame:
    p = REPORTS_DIR / "all_metrics.csv"
    if not p.exists():
        raise HTTPException(503, "Metrics not found. Run the pipeline first.")
    return pd.read_csv(p)


def _load_features() -> pd.DataFrame:
    p = DATA_DIR / "processed" / "features_v2.csv"
    if not p.exists():
        raise HTTPException(503, "Feature table not found. Run the pipeline first.")
    return pd.read_csv(p, parse_dates=["date"])


@router.get("/stats/metrics", response_model=List[ModelMetrics])
def get_model_metrics():
    """Return evaluation metrics for all trained models."""
    df = _load_metrics()
    return df.to_dict(orient="records")


@router.get("/stats/zones", response_model=List[ZoneStats])
def get_zone_stats(request: Request):
    """Return crime summary statistics per LAPD division."""
    fdf = getattr(request.app.state, "features_df", None)
    if fdf is None:
        fdf = _load_features()

    import src.inference as inf

    rows = []
    # Drop duplicates so we only count area_level daily crime once per day
    area_daily = fdf.drop_duplicates(subset=["area_id", "date"]).copy()
    
    for area_id, grp in area_daily.groupby("area_id"):
        if area_id not in inf.AREA_ID_TO_LOC:
            continue
        zone = inf.AREA_ID_TO_LOC[area_id]['zone']
        
        rows.append({
            "zone":             zone,
            "total_crimes":     int(grp["area_crime_t1"].sum()),
            "avg_daily":        round(float(grp["area_crime_t1"].mean()), 2),
            "max_daily":        int(grp["area_crime_t1"].max()),
            "high_risk_days_pct": round(float(grp["target_risk"].mean()) * 100, 1),
        })
    return sorted(rows, key=lambda x: x["total_crimes"], reverse=True)


@router.get("/stats/trend")
def get_crime_trend(zone: str = None):
    """
    Return daily crime counts over time (optionally filtered to one zone).
    Query param: ?zone=Central
    """
    fdf = _load_features()
    import src.inference as inf
    
    # Use area level counts to avoid overcounting (districts have duplicated area counts)
    area_daily = fdf.drop_duplicates(subset=["area_id", "date"]).copy()

    if zone:
        if zone not in inf.ZONE_TO_AREA_ID:
            raise HTTPException(404, f"Zone '{zone}' not found.")
        area_id = inf.ZONE_TO_AREA_ID[zone]
        area_daily = area_daily[area_daily["area_id"] == area_id]
        if area_daily.empty:
            raise HTTPException(404, f"No data for zone '{zone}'.")

    trend = (
        area_daily.groupby("date")["area_crime_t1"]
        .sum()
        .reset_index()
        .rename(columns={"date": "date", "area_crime_t1": "count"})
    )
    trend["date"] = trend["date"].dt.strftime("%Y-%m-%d")
    return trend.to_dict(orient="records")


@router.get("/stats/feature_importance")
def get_feature_importance():
    """Return the top feature importances from SHAP or model."""
    # Try SHAP first, then fall back to model importance
    for pattern in ["shap_importance_*.csv", "feature_importance_*.csv"]:
        files = sorted(REPORTS_DIR.glob(pattern))
        if files:
            df = pd.read_csv(files[-1])
            col = "mean_abs_shap" if "mean_abs_shap" in df.columns else "importance"
            return df.sort_values(col, ascending=False).to_dict(orient="records")
    raise HTTPException(503, "Feature importance not found. Run the pipeline first.")
