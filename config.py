"""
config.py – Predictive Policing Pipeline Configuration (v3 – LSTM)
Dataset: Los Angeles Crime Data (2020-2023)
"""
from pathlib import Path

# ── Directory Layout ─────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "data2.csv"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR    = ARTIFACTS_DIR / "models"
FIGURES_DIR   = ARTIFACTS_DIR / "figures"
REPORTS_DIR   = ARTIFACTS_DIR / "reports"

for _d in [DATA_DIR / "raw", PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── LA Crime Dataset Column Mapping ──────────────────────────────────────────
COL_DATE       = "DATE OCC"
COL_TIME       = "TIME OCC"
COL_LAT        = "LAT"
COL_LON        = "LON"
COL_AREA       = "AREA NAME"
COL_AREA_ID    = "AREA"
COL_DIST       = "Rpt Dist No"        # Reporting District (finer spatial unit)
COL_CRIME_DESC = "Crm Cd Desc"
COL_PART       = "Part 1-2"
COL_WEAPON     = "Weapon Used Cd"
COL_PREMIS     = "Premis Desc"
COL_STATUS     = "Status"
COL_VICT_AGE   = "Vict Age"
COL_VICT_SEX   = "Vict Sex"

# ── Los Angeles Bounding Box ─────────────────────────────────────────────────
LA_LAT_MIN, LA_LAT_MAX = 33.70, 34.35
LA_LON_MIN, LA_LON_MAX = -118.70, -118.05

# ── v2 Spatial Configuration ─────────────────────────────────────────────────
# Use reporting districts (1000+ districts) for finer granularity
# Only use districts with enough crime history for reliable patterns
MIN_CRIMES_PER_DIST = 500   # ~590 active districts → 590 × 1461 days = 860k rows
GRID_DECIMALS = 2

# ── Temporal Configuration ───────────────────────────────────────────────────
TRAIN_CUTOFF_DATE = "2023-06-01"
TIME_WINDOW       = "1D"

# ── Target Variable ───────────────────────────────────────────────────────────
TARGET_COL = "target_risk"
# v2 target: will there be ANY crime in this district tomorrow?
# (binary: 0=no crime, 1=crime) – creates real predictive signal

# ── Risk Tier Thresholds ─────────────────────────────────────────────────────
RISK_BINS   = [0.0, 0.30, 0.60, 1.01]
RISK_LABELS = ["Low", "Medium", "High"]

# ── Model Reproducibility ─────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── v2 Feature Columns (rich feature set) ────────────────────────────────────
FEATURE_COLS = [
    # Lag crime counts (most predictive features)
    "crime_t1", "crime_t2", "crime_t3", "crime_t7", "crime_t14",
    # Rolling statistics
    "roll7_mean", "roll7_std", "roll7_max",
    "roll14_mean", "roll14_std",
    "roll30_mean", "roll30_std",
    # Exponential weighted moving average
    "ewma_3d", "ewma_7d", "ewma_14d",
    # Crime streak / recency
    "days_since_crime",   # how many days since last crime in this district
    "crime_days_last7",   # how many of last 7 days had crime
    "crime_days_last14",  # how many of last 14 days had crime
    "crime_days_last30",  # how many of last 30 days had crime
    # Crime severity features
    "part1_rate_7d",      # proportion of serious crimes (Part I) in last 7 days
    "weapon_rate_7d",     # proportion of weapon-involved crimes last 7 days
    # Spatial context: division-level crime (neighbors)
    "area_crime_t1",      # yesterday's total crime in the same LAPD division
    "area_roll7_mean",    # 7-day rolling mean for the whole division
    # Temporal features
    "day_of_week", "is_weekend", "month", "quarter", "season",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "day_of_year_sin", "day_of_year_cos",
    # District identity
    "dist_id",            # district numeric ID (ordinal encoded)
    "area_id",            # LAPD area/division numeric ID
    # Historical baseline for this district
    "hist_mean_crime",    # overall historical average daily crime count for district
    "hist_crime_rate",    # historical fraction of days with any crime
    # Time-of-day features (from actual TIME OCC, no more 12:00 AM dummy)
    "night_crimes_t1",       # last-day crimes between 00:00–05:59
    "morning_crimes_t1",     # last-day crimes between 06:00–11:59
    "afternoon_crimes_t1",   # last-day crimes between 12:00–17:59
    "evening_crimes_t1",     # last-day crimes between 18:00–23:59
    "peak_crimes_t1",        # last-day crimes in peak hours (19–23 & 0)
    "night_crimes_roll7",    # 7-day rolling mean of night crimes
    "evening_crimes_roll7",  # 7-day rolling mean of evening crimes
    "peak_crimes_roll7",     # 7-day rolling mean of peak-hour crimes
    "peak_hour_t1",          # most common crime hour yesterday
]

# ── LSTM Hyperparameters ──────────────────────────────────────────────────────
LSTM_LOOKBACK      = 30        # days of history per sequence
LSTM_HIDDEN_SIZE   = 128       # hidden units per LSTM direction
LSTM_NUM_LAYERS    = 2         # stacked LSTM layers
LSTM_DROPOUT       = 0.30      # dropout between LSTM layers
LSTM_ATTN_HEADS    = 1         # number of attention heads (1 = additive)
LSTM_EPOCHS        = 60        # max training epochs
LSTM_BATCH_SIZE    = 512       # mini-batch size
LSTM_LR            = 3e-4      # initial learning rate (Adam)
LSTM_PATIENCE      = 10        # early-stopping patience (val loss)
LSTM_WEIGHT_DECAY  = 1e-4      # L2 regularisation
LSTM_TRAIN_FRAC    = 0.70      # fraction of timeline for training
LSTM_VAL_FRAC      = 0.10      # fraction for validation (rest = test)

# Risk-level thresholds used by LSTM head (percentile-based per district)
LSTM_RISK_PERCENTILES = [33, 66]   # Low / Medium / High boundaries

# Saved artefact paths
LSTM_WEIGHTS_PATH  = MODELS_DIR / "lstm_weights.pt"
LSTM_SCALER_PATH   = MODELS_DIR / "lstm_scaler.pkl"
LSTM_DISTRICT_META = MODELS_DIR / "lstm_district_meta.pkl"  # per-district stats
