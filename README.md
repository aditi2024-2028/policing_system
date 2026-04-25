 PoliceSight:

Core task: Given all crime activity in an LAPD reporting district up to day t, predict whether any crime will occur there on day t + 1.

This binary risk forecast is aggregated across 590 active reporting districts, assigned a Low / Medium / High label, and surfaced through a REST API and live dashboard so patrol commanders can make data-driven deployment decisions.
Features

53-feature engineering pipeline — temporal lags, rolling statistics, EWMA, crime streaks, time-of-day buckets, spatial spillover, cyclical calendar encodings, and per-district historical baselines
ML ensemble — XGBoost, LightGBM, Random Forest, and MLP with soft-voting; optimal decision threshold via Youden's J statistic
BiLSTM + Additive Attention — 2-layer bidirectional LSTM with Bahdanau-style self-attention over 30-day sequences; dual output heads for count regression and 3-class risk classification
Hotspot clustering — DBSCAN, HDBSCAN, and KMeans comparison with silhouette-score selection
SHAP interpretability — per-prediction feature importance explanations served through the API
Fairness diagnostics — per-zone FPR / FNR analysis to audit geographic bias
Per-district adaptive thresholds — risk calibration using each district's own crime-rate percentiles
Production-ready API — FastAPI v3.0 with OpenAPI docs, health checks, and graceful LSTM fallback
Interactive dashboard — React + Leaflet risk maps, 30-day trend charts, attention heatmaps, and patrol recommendations

Quick Start:
Prerequisites
Requirement: VersionPython3.10 + Node.js18 + RAM4 GB minimum (8 GB for LSTM training)

1 — Clone
bash: git clone https://github.com/aditi2024-2028/policing_system.git
cd policing_system
2 — Install Python dependencies
bash: pip install -r requirements.txt
3 — Add the dataset
Download LAPD Crime Data 2020–Present from the LA Open Data Portal and place it at:
data/raw/data2.csv
4 — Run the full pipeline
bash :python run_pipeline.py
5 — Start the API
bash :uvicorn backend.main:app --reload --port 8000
Interactive API docs: http://localhost:8000/api/docs
6 — Start the dashboard
bash: cd frontend
npm install
npm run dev
Dashboard: http://localhost:5173
