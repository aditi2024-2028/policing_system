"""Quick check: features_v2.csv status and sequence test."""
import sys
sys.path.insert(0, ".")
from pathlib import Path

fpath = Path("data/processed/features_v2.csv")
print(f"features_v2.csv exists: {fpath.exists()}")

if fpath.exists():
    import pandas as pd
    df = pd.read_csv(fpath, parse_dates=["date"])
    print(f"Rows      : {len(df):,}")
    print(f"Districts : {df['dist'].nunique()}")
    print(f"Columns   : {len(df.columns)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    from config import FEATURE_COLS, LSTM_LOOKBACK
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    print(f"Feature cols available: {len(feat_cols)} / {len(FEATURE_COLS)}")

    from src.lstm_training import build_sequences
    X, y_count, y_risk, dates, districts, dist_meta = build_sequences(
        df, feat_cols, lookback=LSTM_LOOKBACK
    )
    print(f"\nSequence shape : {X.shape}")
    print(f"y_count sample : {y_count[:5]}")
    print(f"y_risk  sample : {y_risk[:5]}   (0=Low, 1=Med, 2=High)")
    print(f"Districts meta : {len(dist_meta)} entries")
    print("\nSEQUENCE BUILD — OK")
else:
    print("features_v2.csv is MISSING.")
    print("Run the pipeline from Step 1:")
    print("  python run_pipeline.py")
    print("Or just steps 1-2:")
    print("  python -c \"from src.data_loader import load_and_clean; "
          "from src.feature_engineering import build_district_panel; "
          "build_district_panel(load_and_clean())\"")
