"""
backend/main.py  —  FastAPI application entry point  (v3 — LSTM ensemble)
Run with: uvicorn backend.main:app --reload --port 8000
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.routers import hotspots, predictions, stats

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crime Hotspot Prediction API",
    description=(
        "Spatio-temporal crime risk forecasting for LAPD divisions. "
        "Ensemble: XGBoost + LightGBM (SHAP explanations) + "
        "BiLSTM with Self-Attention (temporal sequence modelling). "
        "Dataset: LA Crime Data 2020-2023."
    ),
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["Health"])
def health():
    from config import LSTM_WEIGHTS_PATH
    return {
        "status":         "ok",
        "tree_model":     getattr(app.state, "model", None) is not None,
        "lstm_model":     getattr(app.state, "lstm_bundle", None) is not None,
        "lstm_available": LSTM_WEIGHTS_PATH.exists(),
        "service":        "Crime Hotspot Prediction API v3.0",
    }

# ── API Routers ───────────────────────────────────────────────────────────────
app.include_router(predictions.router, prefix="/api")
app.include_router(hotspots.router,    prefix="/api")
app.include_router(stats.router,       prefix="/api")

# ── Dashboard ─────────────────────────────────────────────────────────────────
# Serve React+Vite build from frontend/dist (fall back to old dashboard/ if build missing)
_root = Path(__file__).resolve().parent.parent
dist_dir      = _root / "frontend" / "dist"
dashboard_dir = _root / "dashboard"

@app.get("/", include_in_schema=False)
def serve_dashboard():
    # Prefer the React build
    for candidate in [dist_dir / "index.html", dashboard_dir / "index.html"]:
        if candidate.exists():
            return FileResponse(str(candidate))
    return {"detail": "Dashboard not found. Run: cd frontend && npm run build"}

# Mount static assets — React build takes priority
if (dist_dir / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(dist_dir / "assets")), name="static")
elif dashboard_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(dashboard_dir)), name="static")

# ── Startup: load tree model + LSTM + feature table ──────────────────────────
@app.on_event("startup")
async def startup_event():
    # ── 1. Tree model (XGBoost / LightGBM) ───────────────────────────────────
    try:
        from src.inference import load_model, load_features_df
        app.state.model, app.state.feature_cols, app.state.opt_threshold = load_model()
        app.state.features_df = load_features_df()
        print(f"[Backend] Tree model loaded | threshold={app.state.opt_threshold:.4f}")
    except Exception as e:
        print(f"[Backend] WARNING: Tree model not loaded: {e}")
        print("[Backend]   Run `python final_train.py` first.")
        app.state.model        = None
        app.state.feature_cols = None
        app.state.features_df  = None
        app.state.opt_threshold = 0.50

    # ── 2. LSTM model (optional — graceful fallback) ───────────────────────────
    try:
        from src.inference import load_lstm
        lstm_bundle = load_lstm()
        app.state.lstm_bundle = lstm_bundle
        if lstm_bundle is not None:
            print("[Backend] LSTM model loaded (BiLSTM + Attention)")
        else:
            print("[Backend] LSTM weights not found — tree-only mode "
                  "(run `python src/lstm_training.py` to train)")
    except Exception as e:
        print(f"[Backend] LSTM load failed: {e}")
        app.state.lstm_bundle = None

