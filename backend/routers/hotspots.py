"""
backend/routers/hotspots.py
/api/hotspots - returns cluster centroids as GeoJSON-like list
"""
import sys
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.schemas import HotspotCluster
from config import REPORTS_DIR

router = APIRouter(tags=["Hotspots"])


def _load_centroids() -> pd.DataFrame:
    p = REPORTS_DIR / "hotspot_centroids.csv"
    if not p.exists():
        raise HTTPException(
            status_code=503,
            detail="Hotspot centroids not found. Run the pipeline first."
        )
    return pd.read_csv(p)


@router.get("/hotspots", response_model=List[HotspotCluster])
def get_hotspots(top: int = 50):
    """
    Return the top N crime hotspot cluster centroids.
    Query param: ?top=50 (default)
    """
    df = _load_centroids().head(top)
    return df.to_dict(orient="records")


@router.get("/hotspots/geojson")
def get_hotspots_geojson(top: int = 50):
    """Return hotspots as GeoJSON FeatureCollection for Leaflet."""
    df = _load_centroids().head(top)
    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["lon"], row["lat"]]
            },
            "properties": {
                "cluster": int(row["cluster"]),
                "size":    int(row["size"]),
                "method":  str(row["method"]),
            }
        })
    return {"type": "FeatureCollection", "features": features}
