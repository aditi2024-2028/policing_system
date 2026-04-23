# src/__init__.py
# Public API for the predictive policing src package

from src.data_loader   import load_and_clean
from src.feature_engineering import build_district_panel
from src.hotspot_clustering  import run_clustering
from src.model_training      import train_all_models
from src.inference           import predict_zone_date, predict_all_zones, load_model, load_lstm
from src.interpretability    import run_shap, fairness_check, explain_lstm_attention

# LSTM modules (optional – require torch)
try:
    from src.lstm_model    import CrimeLSTM, build_model
    from src.lstm_training import train_lstm, load_lstm_model, predict_sequences
    _LSTM_AVAILABLE = True
except ImportError:
    _LSTM_AVAILABLE = False

__all__ = [
    "load_and_clean",
    "build_district_panel",
    "run_clustering",
    "train_all_models",
    "predict_zone_date",
    "predict_all_zones",
    "load_model",
    "load_lstm",
    "run_shap",
    "fairness_check",
    "explain_lstm_attention",
    # LSTM (torch-dependent)
    "CrimeLSTM",
    "build_model",
    "train_lstm",
    "load_lstm_model",
    "predict_sequences",
    "_LSTM_AVAILABLE",
]
