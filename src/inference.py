"""
src/inference.py
Load a trained model and generate risk predictions for any zone-date pair.
Adapted for v2 features (aggregates district-level predictions up to Division).

Changes:
  - Loads optimal decision threshold from model_meta.json (Youden's J)
  - Correctly resolves area_id (raw 1-21) to encoded (0-20) values used in features
  - Falls back to 0.50 threshold if meta JSON is missing (backward compat)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, LSTM_WEIGHTS_PATH, MODELS_DIR, RISK_BINS, RISK_LABELS

_SEASON_MAP = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
               6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}

DIVISION_LOCATIONS = [
    {'area_id': 1.0, 'zone': 'Central', 'avg_lat': 34.04740703526264, 'avg_lon': -118.25048660494626},
    {'area_id': 2.0, 'zone': 'Rampart', 'avg_lat': 34.06329079399474, 'avg_lon': -118.27419413093948},
    {'area_id': 3.0, 'zone': 'Southwest', 'avg_lat': 34.01999696627483, 'avg_lon': -118.31682173656384},
    {'area_id': 4.0, 'zone': 'Hollenbeck', 'avg_lat': 34.05432094828571, 'avg_lon': -118.20321230259903},
    {'area_id': 5.0, 'zone': 'Harbor', 'avg_lat': 33.77202543005181, 'avg_lon': -118.28525731260795},
    {'area_id': 6.0, 'zone': 'Hollywood', 'avg_lat': 34.09941473276697, 'avg_lon': -118.32991064630605},
    {'area_id': 7.0, 'zone': 'Wilshire', 'avg_lat': 34.06169293276544, 'avg_lon': -118.35218568299362},
    {'area_id': 8.0, 'zone': 'West LA', 'avg_lat': 34.051648890620136, 'avg_lon': -118.44085928014958},
    {'area_id': 9.0, 'zone': 'Van Nuys', 'avg_lat': 34.17861939806156, 'avg_lon': -118.44718357082128},
    {'area_id': 10.0, 'zone': 'West Valley', 'avg_lat': 34.18735711293479, 'avg_lon': -118.52096434776637},
    {'area_id': 11.0, 'zone': 'Northeast', 'avg_lat': 34.107267427777224, 'avg_lon': -118.24481379595619},
    {'area_id': 12.0, 'zone': '77th Street', 'avg_lat': 33.97769697986577, 'avg_lon': -118.29715338971522},
    {'area_id': 13.0, 'zone': 'Newton', 'avg_lat': 34.00906500754586, 'avg_lon': -118.26080008416531},
    {'area_id': 14.0, 'zone': 'Pacific', 'avg_lat': 33.98470185189752, 'avg_lon': -118.42527418933248},
    {'area_id': 15.0, 'zone': 'N Hollywood', 'avg_lat': 34.17141403202745, 'avg_lon': -118.3846823563054},
    {'area_id': 16.0, 'zone': 'Foothill', 'avg_lat': 34.2489380778692, 'avg_lon': -118.37804336669778},
    {'area_id': 17.0, 'zone': 'Devonshire', 'avg_lat': 34.250044740483204, 'avg_lon': -118.53923286266718},
    {'area_id': 18.0, 'zone': 'Southeast', 'avg_lat': 33.93881034365911, 'avg_lon': -118.2672202439782},
    {'area_id': 19.0, 'zone': 'Mission', 'avg_lat': 34.256185247546036, 'avg_lon': -118.45066026008455},
    {'area_id': 20.0, 'zone': 'Olympic', 'avg_lat': 34.06009854297098, 'avg_lon': -118.30047461297667},
    {'area_id': 21.0, 'zone': 'Topanga', 'avg_lat': 34.19228045847293, 'avg_lon': -118.60078922588609}
]

ZONE_TO_AREA_ID = {d['zone']: d['area_id'] for d in DIVISION_LOCATIONS}
AREA_ID_TO_LOC = {d['area_id']: d for d in DIVISION_LOCATIONS}
ALL_ZONES = sorted([d['zone'] for d in DIVISION_LOCATIONS])


def load_model():
    """Return (model, feature_cols, opt_threshold) from saved artifacts."""
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    # Load Youden-optimal threshold saved by final_train.py
    meta_path = MODELS_DIR / "model_meta.json"
    opt_threshold = 0.50  # safe default
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        opt_threshold = meta.get("opt_threshold", 0.50)
    return model, feature_cols, opt_threshold


def load_lstm() -> tuple:
    """
    Load the trained CrimeLSTM ensemble component.
    Returns (lstm_model, scaler, dist_meta, lstm_feature_cols, device)
    or None if the LSTM weights haven\'t been trained yet.
    """
    if not LSTM_WEIGHTS_PATH.exists():
        return None
    try:
        from src.lstm_training import load_lstm_model
        return load_lstm_model()
    except Exception as e:
        print(f"[Inference] LSTM load failed ({e}) — using tree model only.")
        return None


def load_features_df() -> pd.DataFrame:
    """Load features_v2.csv and add the interaction features computed at training time."""
    df = pd.read_csv(
        DATA_DIR / "processed" / "features_v2.csv", parse_dates=["date"]
    )
    # ── Replicate exactly the interaction features added in final_train.py ──────
    if "crime_t1" in df.columns and "crime_t7" in df.columns:
        df["crime_momentum"]  = df["crime_t1"] - df["crime_t7"]
        df["crime_accel"]     = df["crime_t1"] - 2*df["crime_t2"] + df["crime_t3"]
        df["vol_ratio"]       = df["roll7_std"] / (df["roll7_mean"].clip(lower=0.01))
        df["streak_x_mean"]  = df["crime_days_last7"] * df["roll7_mean"]
        df["ewma_divergence"] = df["ewma_3d"] - df["ewma_14d"]
        df["area_dist_ratio"] = df["area_crime_t1"] / (df["crime_t1"].clip(lower=0.01))
    return df


def _build_temporal_row(date_str: str) -> dict:
    """Build temporal feature overrides for a given date string."""
    dt = pd.to_datetime(date_str)
    doy = dt.dayofyear
    return {
        "date":        dt,
        "day_of_week": dt.dayofweek,
        "is_weekend":  int(dt.dayofweek >= 5),
        "month":       dt.month,
        "quarter":     dt.quarter,
        "season":      _SEASON_MAP[dt.month],
        "month_sin":   np.sin(2 * np.pi * dt.month / 12),
        "month_cos":   np.cos(2 * np.pi * dt.month / 12),
        "dow_sin":     np.sin(2 * np.pi * dt.dayofweek / 7),
        "dow_cos":     np.cos(2 * np.pi * dt.dayofweek / 7),
        "day_of_year_sin": np.sin(2 * np.pi * doy / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * doy / 365.25),
    }


def _resolve_area_id_in_features(features_df: pd.DataFrame, raw_area_id: float) -> set:
    """
    The features_v2.csv stores ENCODED area_id (0-based ordinal).
    The DIVISION_LOCATIONS table uses raw LAPD area_id (1-21).
    This function finds what encoded value(s) correspond to raw_area_id.
    Strategy: the encoded values are assigned in sorted order of the raw ids,
    so encoded = raw - 1 (assuming areas 1..21 → encoded 0..20).
    We also try a direct match in case the CSV was not re-encoded.
    """
    encoded_guess = raw_area_id - 1.0   # most common mapping
    candidates = {encoded_guess, raw_area_id}
    available  = set(features_df["area_id"].unique())
    matched    = candidates & available
    if not matched:
        # Fallback: pick the closest encoded value
        arr = np.array(sorted(available))
        closest = arr[np.argmin(np.abs(arr - encoded_guess))]
        matched = {closest}
    return matched


def predict_zone_date(
    zone: str,
    date_str: str,
    model=None,
    feature_cols: list = None,
    features_df: pd.DataFrame = None,
    opt_threshold: float = None,
    lstm_bundle: tuple = None,   # (lstm_model, scaler, dist_meta, lstm_fc, device)
    lstm_weight: float = 0.40,   # how much weight to give LSTM vs tree model
    hour: int = None,            # (Optional) requested hour 0-23
) -> dict:
    """
    Predict crime risk for a single (zone, date).
    When an LSTM bundle is provided the output probability is a weighted
    average of the tree model and the LSTM High-risk probability.
    Returns a JSON-serializable dict.
    """
    if model is None or feature_cols is None:
        model, feature_cols, _thr = load_model()
        if opt_threshold is None:
            opt_threshold = _thr
    if opt_threshold is None:
        opt_threshold = 0.50
    if features_df is None:
        features_df = load_features_df()

    if zone not in ZONE_TO_AREA_ID:
        return {"error": f"Zone '{zone}' not found."}

    raw_area_id = ZONE_TO_AREA_ID[zone]

    # Resolve encoded area_id values used in features_v2.csv
    matched_ids = _resolve_area_id_in_features(features_df, raw_area_id)
    area_df = features_df[features_df["area_id"].isin(matched_ids)]

    if area_df.empty:
        return {"error": f"No data for zone '{zone}' (raw_area_id={raw_area_id})."}

    latest_rows = area_df.sort_values("date").groupby("dist", as_index=False).tail(1).copy()

    # Apply temporals for target date
    overrides = _build_temporal_row(date_str)
    for col, val in overrides.items():
        if col in latest_rows.columns:
            latest_rows[col] = val

    X = latest_rows[feature_cols].fillna(0).values
    tree_probs = model.predict_proba(X)[:, 1]
    avg_tree_prob = float(np.mean(tree_probs))

    # ── LSTM ensemble component ───────────────────────────────────────────────
    lstm_prob = None
    lstm_risk_label = None
    if lstm_bundle is not None:
        try:
            lstm_model, scaler, dist_meta, lstm_fc, device = lstm_bundle
            from src.lstm_training import predict_sequences
            # Build 30-day sequences for each district in this zone
            dists_in_zone = area_df["dist"].unique().tolist()
            seq_list = []
            for d in dists_in_zone:
                d_df = area_df[area_df["dist"] == d].sort_values("date")
                if len(d_df) < 30:
                    continue
                cols = [c for c in lstm_fc if c in d_df.columns]
                seq = d_df[cols].fillna(0).values[-30:].astype(np.float32)
                if seq.shape[0] == 30:
                    seq_list.append(seq)
            if seq_list:
                X_seq = np.stack(seq_list, axis=0)   # [n_dists, 30, n_feats]
                # Pad missing features with zero
                n_needed = len(lstm_fc)
                if X_seq.shape[-1] < n_needed:
                    pad = np.zeros(
                        (X_seq.shape[0], 30, n_needed - X_seq.shape[-1]),
                        dtype=np.float32,
                    )
                    X_seq = np.concatenate([X_seq, pad], axis=-1)
                _, risk_proba, _, _ = predict_sequences(
                    lstm_model, X_seq, scaler, device
                )
                lstm_prob = float(risk_proba[:, 2].mean())   # P(High risk)
                lstm_risk_cls = int(risk_proba.mean(axis=0).argmax())
                lstm_risk_label = ["Low", "Medium", "High"][lstm_risk_cls]
        except Exception as e:
            print(f"[Inference] LSTM inference failed ({e})")

    # ── Fuse probabilities ────────────────────────────────────────────────────
    if lstm_prob is not None:
        avg_prob = (1 - lstm_weight) * avg_tree_prob + lstm_weight * lstm_prob
    else:
        avg_prob = avg_tree_prob

    # ── Time-of-Day Scaling ───────────────────────────────────────────────
    multiplier = 1.0
    base_prob = avg_prob
    if hour is not None:
        if 0 <= hour <= 5:
            tod_col = "night_crimes_roll7"
        elif 6 <= hour <= 11:
            tod_col = "morning_crimes_roll7"
        elif 12 <= hour <= 17:
            tod_col = "afternoon_crimes_roll7"
        else:
            tod_col = "evening_crimes_roll7"
            
        if tod_col in latest_rows.columns and "roll7_mean" in latest_rows.columns:
            tod_n = latest_rows[tod_col].sum()
            tot_n = latest_rows["roll7_mean"].sum()
            if tot_n > 0:
                # Uniform baseline for a 6-hour window is 25%
                window_ratio = tod_n / tot_n
                multiplier = float(np.clip(window_ratio / 0.25, 0.4, 2.5))
                avg_prob = np.clip(avg_prob * multiplier, 0.0, 1.0)
                
    risk_level = str(pd.cut([avg_prob], bins=RISK_BINS, labels=RISK_LABELS)[0])
    loc = AREA_ID_TO_LOC[raw_area_id]

    result = {
        "zone":                zone,
        "date":                date_str,
        "hour":                hour,
        "risk_score":          round(avg_prob, 4),
        "base_risk_score":     round(base_prob, 4),
        "tod_multiplier":      round(multiplier, 3),
        "risk_level":          risk_level,
        "predicted_high_risk": int(avg_prob >= opt_threshold),
        "tree_risk_score":     round(avg_tree_prob, 4),
        "avg_lat":             loc["avg_lat"],
        "avg_lon":             loc["avg_lon"],
        "zone_id":             int(raw_area_id),
    }
    if lstm_prob is not None:
        result["lstm_high_risk_prob"] = round(lstm_prob, 4)
        result["lstm_risk_label"]     = lstm_risk_label
    return result


def predict_all_zones(
    date_str: str,
    model=None,
    feature_cols: list = None,
    features_df: pd.DataFrame = None,
    opt_threshold: float = None,
    lstm_bundle: tuple = None,
    lstm_weight: float = 0.40,
    hour: int = None,
) -> list:
    """Predict risk for every zone on a given date (LSTM + tree ensemble)."""
    if model is None or feature_cols is None:
        model, feature_cols, _thr = load_model()
        if opt_threshold is None:
            opt_threshold = _thr
    if opt_threshold is None:
        opt_threshold = 0.50
    if features_df is None:
        features_df = load_features_df()

    return [
        predict_zone_date(z, date_str, model, feature_cols, features_df,
                          opt_threshold, lstm_bundle, lstm_weight, hour)
        for z in ALL_ZONES
    ]


if __name__ == "__main__":
    m, fc, thr = load_model()
    fdf         = load_features_df()
    lstm_b      = load_lstm()      # None if not trained yet
    print(f"Loaded tree model | threshold={thr:.4f}")
    print(f"LSTM available    : {lstm_b is not None}")
    out = predict_all_zones("2023-10-01", m, fc, fdf, thr, lstm_b)
    for o in out[:5]:
        print(o)
