PoliceSight
### Spatio-Temporal Crime Hotspot Prediction System

PoliceSight ingests 800,000+ LAPD incident records and delivers per-district binary risk forecasts (Low / Medium / High) 24 hours in advance — served through a production FastAPI backend and a live React + Leaflet dashboard.

## Overview

PoliceSight is an end-to-end **spatio-temporal crime hotspot prediction system** built on the Los Angeles Police Department (LAPD) crime dataset (2020–2023).

**Core Task:** Given all crime activity in an LAPD reporting district up to day *t*, predict whether any crime will occur there on day *t + 1*.

This binary risk forecast is aggregated across **590 active reporting districts**, assigned a **Low / Medium / High** label, and surfaced through a REST API and live dashboard so patrol commanders can make data-driven deployment decisions.

---

## Features

| Feature | Description |
|---|---|
| 🧠 **53-feature engineering pipeline** | Temporal lags, rolling statistics, EWMA, crime streaks, time-of-day buckets, spatial spillover, cyclical calendar encodings, and per-district historical baselines |
| 🌲 **ML Ensemble** | XGBoost, LightGBM, Random Forest, and MLP with soft-voting; optimal decision threshold via Youden's J statistic |
| 🔁 **BiLSTM + Additive Attention** | 2-layer bidirectional LSTM with Bahdanau-style self-attention over 30-day sequences; dual output heads for count regression and 3-class risk classification |
| 📍 **Hotspot Clustering** | DBSCAN, HDBSCAN, and KMeans comparison with silhouette-score selection |
| 🔍 **SHAP Interpretability** | Per-prediction feature importance explanations served through the API |
| ⚖️ **Fairness Diagnostics** | Per-zone FPR / FNR analysis to audit geographic bias |
| 🎯 **Per-district Adaptive Thresholds** | Risk calibration using each district's own crime-rate percentiles |
| ⚡ **Production-ready API** | FastAPI v3.0 with OpenAPI docs, health checks, and graceful LSTM fallback |
| 🗺️ **Interactive Dashboard** | React + Leaflet risk maps, 30-day trend charts, attention heatmaps, and patrol recommendations |

---

## System Architecture

```
┌──────────────────────────────────────────────────┐
│           Data Layer                             │
│     Raw CSV Ingestion & Cleaning                 │
│     (800K+ LAPD records, 28 columns)             │
└─────────────────────┬────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────┐
│           Feature Layer                          │
│     53-dimensional District-Day Panel            │
│     (~860,000 rows)                              │
└─────────────────────┬────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────┐
│           Model Layer                            │
│   ML Ensemble (XGB + LGBM + RF + MLP)           │
│   + BiLSTM w/ Attention + DBSCAN/HDBSCAN        │
└─────────────────────┬────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────┐
│           Application Layer                      │
│     FastAPI Backend + React Dashboard            │
│     (Risk Maps, SHAP, Patrol Recommendations)   │
└──────────────────────────────────────────────────┘
```

---

## Tech Stack

**Backend & ML**
- Python 3.10+
- XGBoost, LightGBM, scikit-learn
- PyTorch (BiLSTM)
- SHAP
- FastAPI v3.0
- Pandas, NumPy

**Frontend**
- React + Vite
- Leaflet.js (interactive maps)
- Recharts (trend visualisations)

**Clustering**
- DBSCAN, HDBSCAN, KMeans (scikit-learn)

---

## Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| RAM | 4 GB minimum (8 GB for LSTM training) |

### 1 — Clone

```bash
git clone https://github.com/aditi2024-2028/policing_system.git
cd policing_system
```

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — Add the dataset

Download **LAPD Crime Data 2020–Present** from the [LA Open Data Portal](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8) and place it at:

```
data/raw/data2.csv
```

### 4 — Run the full pipeline

```bash
python run_pipeline.py
```

This will clean the data, engineer features, train all models, run clustering, and export artifacts.

### 5 — Start the API

```bash
uvicorn backend.main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/api/docs

### 6 — Start the dashboard

```bash
cd frontend
npm install
npm run dev
```

Dashboard: http://localhost:5173

---

## Model Details

### ML Ensemble

Four base classifiers combined via **soft voting** (averaged predicted probabilities):

| Model | Key Config |
|---|---|
| XGBoost | 500 trees, depth=7, `scale_pos_weight≈1.54` |
| LightGBM | 500 trees, leaves=127, `is_unbalance=True` |
| Random Forest | 300 trees, depth=12, `class_weight='balanced'` |
| MLP | 256→128→64, Adam optimizer, early stopping |

Temporal split: **train < June 1, 2023 / test ≥ June 1, 2023**

### BiLSTM with Additive Attention (CrimeLSTM)

```
Input: 30-day sequence × 53 features
  ↓
Feature Projection: Linear → LayerNorm → GELU
  ↓
BiLSTM (2 layers, 128 hidden units, dropout 0.30)
  ↓
Additive Self-Attention (Bahdanau): c = Σ αt·ht
  ↙              ↘
Count Regression    Risk Classification
  (MSE + MAE)       (Low / Med / High)
```

Training: Adam (lr=3×10⁻⁴), batch size 512, up to 60 epochs, early stopping (patience 10). Temporal split: 70/10/20 train/val/test.

### Hotspot Clustering

| Algorithm | Clusters | Silhouette | Noise % |
|---|---|---|---|
| DBSCAN | ~180 | ~0.45 | ~8% |
| **HDBSCAN** ✅ | ~120 | **~0.52** | ~12% |
| KMeans (k=21) | 21 | ~0.38 | 0% |

**HDBSCAN** achieves the highest silhouette score and identifies geographically coherent hotspots in Downtown LA, Hollywood, South LA, and the San Fernando Valley.

---

## Results

| Metric | Ensemble |
|---|---|
| Accuracy | 65.2% |
| AUC-ROC | 0.6925 |
| F1 Score | 0.5084 |
| Recall | 0.5565 |
| Overfitting Gap | −0.0082 |

> The AUC of ~0.69 reflects the inherent difficulty of predicting binary crime occurrence at individual-district granularity 24 hours in advance. Results are competitive with published fine-grained crime prediction benchmarks.

**Top predictors (SHAP + gain-based):**
1. `crime_t1` — previous day's crime count (strongest predictor)
2. `roll7_mean`, `ewma_7d` — short-term rolling trend
3. `hist_crime_rate` — per-district historical baseline
4. `crime_days_last7`, `days_since_crime` — recency/streak signals
5. `area_crime_t1` — division-level spatial spillover

---

## Dataset

**Source:** [LAPD Crime Data 2020–Present](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8) — Los Angeles Open Data Portal

| Stat | Value |
|---|---|
| Raw records | ~800,000 |
| Date range | January 2020 – December 2023 |
| Active reporting districts | 590 |
| Feature panel rows | ~860,000 |
| Positive class rate (train) | 39.3% |
| Positive class rate (test) | 38.7% |

---

## Fairness & Ethics

PoliceSight explicitly addresses algorithmic bias (cf. Lum & Isaac, 2016):

- **Per-zone FPR/FNR analysis** — false positive and false negative rates are compared across all geographic zones to detect disparate impact.
- **Per-district adaptive thresholds** — each district's Low/Medium/High boundaries are calibrated to its own historical crime-rate percentiles (33rd and 66th), reducing systematic bias toward high-crime districts.

---

## Future Work

- [ ] **Graph Neural Networks** (STGCN / DCRNN) for spatial adjacency modelling
- [ ] **External covariates** — weather, public events, and socioeconomic data
- [ ] **Real-time ingestion** from the LAPD Open Data API
- [ ] **Fairness-constrained training** — equalized odds objectives

---

## Team

| Name | Roll No. |
|---|---|
| Aditi Raj | 2024UGCS016 |
| Vivek Chaudhry | 2024UGCS021 |
| Diya Sahu | 2024UGCS013 |
| Kriti | 2024UGCS103 |

**Supervisor:** Prof. K.K. Singh
**Institution:** Department of Computer Science & Engineering, NIT Jamshedpur
**Year:** 2026

---
Interactive API docs: http://localhost:8000/api/docs
6 — Start the dashboard
bash: cd frontend
npm install
npm run dev
Dashboard: http://localhost:5173
