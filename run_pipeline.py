"""
run_pipeline.py  –  Predictive Policing System v3
===================================================
Full end-to-end pipeline:
  Step 1 : Data loading & cleaning          (src/data_loader.py)
  Step 2 : Feature engineering v2           (src/feature_engineering.py)
  Step 3 : Spatial hotspot clustering       (src/hotspot_clustering.py)
  Step 4 : XGBoost + LightGBM training      (final_train.py)
  Step 5 : BiLSTM + Attention training      (src/lstm_training.py)
  Step 6 : SHAP + LSTM Attention XAI        (src/interpretability.py)
  Step 7 : Summary & startup instructions

Usage
-----
    python run_pipeline.py
"""
import io
import json
import os
import shutil
import subprocess
import sys
import time

# ── Force UTF-8 stdout on Windows ────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# ── Ensure raw data is in the expected location ───────────────────────────────
raw_dest = Path("data/raw/data2.csv")
raw_dest.parent.mkdir(parents=True, exist_ok=True)
if not raw_dest.exists() and Path("data2.csv").exists():
    shutil.copy("data2.csv", raw_dest)
    print(f"[Setup] Copied data2.csv -> {raw_dest}")

t0 = time.time()
print("=" * 70)
print("  Predictive Policing System v3 — LSTM + Tree Ensemble")
print("  Dataset : Los Angeles Crime Data (2020-2023)")
print("=" * 70)

# =============================================================================
# STEP 1 — Data Loading & Cleaning
# =============================================================================
print("\n[STEP 1/7] Data Loading & Cleaning ...")
from src.data_loader import load_and_clean
df_clean = load_and_clean()

# =============================================================================
# STEP 2 — Feature Engineering v2  (district × day panel)
# =============================================================================
print("\n[STEP 2/7] Feature Engineering v2 (Reporting District Level) ...")
from src.feature_engineering import build_district_panel
feat_df, dist_codes, area_codes = build_district_panel(df_clean)

# =============================================================================
# STEP 3 — Spatial Hotspot Clustering  (DBSCAN / HDBSCAN / KMeans)
# =============================================================================
print("\n[STEP 3/7] Spatial Hotspot Clustering ...")
from src.hotspot_clustering import run_clustering
cluster_results, best_cluster_method, centroids_df = run_clustering(df_clean)

# =============================================================================
# STEP 4 — XGBoost + LightGBM Training  (via final_train.py)
# =============================================================================
print("\n[STEP 4/7] XGBoost + LightGBM Training (anti-overfit, Youden threshold) ...")
try:
    subprocess.run([sys.executable, "final_train.py"], check=True)
    print("  -> Tree model training complete. Artifacts saved -> artifacts/models/")
except subprocess.CalledProcessError as e:
    print(f"  [ERROR] Tree model training failed: {e}")
    sys.exit(1)

# =============================================================================
# STEP 5 — BiLSTM + Attention Training
# =============================================================================
print("\n[STEP 5/7] BiLSTM + Attention Training ...")
lstm_model = None
try:
    import torch   # noqa: F401 — check install before importing heavy stack
    from src.lstm_training import train_lstm
    lstm_model = train_lstm(feat_df)   # pass already-loaded panel to avoid re-read
    print("  -> LSTM training complete. Weights saved -> artifacts/models/lstm_weights.pt")
except ImportError:
    print("  [SKIP] PyTorch not installed. Install with:")
    print("         pip install torch")
    print("         Then re-run this step: python -c \"from src.lstm_training import train_lstm; train_lstm()\"")
except Exception as e:
    print(f"  [WARN] LSTM training encountered an error: {e}")
    import traceback; traceback.print_exc()

# =============================================================================
# STEP 6 — Explainability  (SHAP for trees  +  Attention for LSTM)
# =============================================================================
print("\n[STEP 6/7] Explainability Analysis ...")

# ── 6a. SHAP (tree models) ────────────────────────────────────────────────────
try:
    import joblib, pandas as pd
    from config import MODELS_DIR, REPORTS_DIR
    from src.inference import load_features_df
    from src.interpretability import run_shap

    best_model  = joblib.load(MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols_tree = json.load(f)

    feat_enriched = load_features_df()
    X_shap = feat_enriched[[c for c in feature_cols_tree
                             if c in feat_enriched.columns]].tail(3000)
    run_shap(best_model, X_shap, "best_model")
    print("  -> SHAP analysis complete.")
except Exception as e:
    print(f"  [WARN] SHAP analysis failed: {e}")

# ── 6b. LSTM Attention heatmaps ───────────────────────────────────────────────
if lstm_model is not None:
    try:
        import pickle
        import numpy as np
        import torch
        from config import (LSTM_DISTRICT_META, LSTM_LOOKBACK, LSTM_SCALER_PATH)
        from src.lstm_training import apply_scaler, build_sequences
        from src.interpretability import explain_lstm_attention

        with open(LSTM_DISTRICT_META, "rb") as f:
            meta_bundle = pickle.load(f)
        lstm_fc = meta_bundle["feature_cols"]

        with open(LSTM_SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        device = torch.device("cpu")
        X_all, _, _, dates_all, _, _ = build_sequences(feat_df, lstm_fc,
                                                        lookback=LSTM_LOOKBACK)
        # Use the last 500 test-period samples for XAI
        demo_size = min(500, len(X_all))
        X_demo    = X_all[-demo_size:]
        explain_lstm_attention(lstm_model, X_demo, scaler, lstm_fc, device)
        print("  -> LSTM attention heatmaps saved -> artifacts/figures/")
    except Exception as e:
        print(f"  [WARN] LSTM XAI failed: {e}")
        import traceback; traceback.print_exc()

# =============================================================================
# STEP 7 — Final Summary
# =============================================================================
elapsed = time.time() - t0

print(f"\n{'='*70}")
print(f"  [DONE]  Full pipeline complete in {elapsed:.1f}s")
print(f"  Hotspot Engine  : {best_cluster_method.upper()} ({len(centroids_df)} clusters)")
print(f"  LSTM Model      : {'Trained ✓' if lstm_model is not None else 'Skipped (pip install torch)'}")
print()
print(f"  Key artifacts:")
print(f"    artifacts/models/best_model.joblib       <- XGBoost / LightGBM")
print(f"    artifacts/models/lstm_weights.pt         <- BiLSTM weights")
print(f"    artifacts/models/model_meta.json         <- threshold + AUC")
print(f"    artifacts/figures/lstm_training_history.png")
print(f"    artifacts/figures/lstm_global_attention.png")
print(f"    artifacts/figures/lstm_local_saliency.png")
print(f"    artifacts/figures/shap_importance_best_model.png")
print(f"    artifacts/reports/lstm_test_predictions.csv")
print()
print(f"  Launch dashboard:")
print(f"    python -m uvicorn backend.main:app --reload --port 8000")
print(f"{'='*70}")
