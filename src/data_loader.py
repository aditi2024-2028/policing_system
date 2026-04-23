"""
src/data_loader.py
Load and clean the LA Crime dataset.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    COL_DATE, COL_TIME, COL_LAT, COL_LON,
    LA_LAT_MIN, LA_LAT_MAX, LA_LON_MIN, LA_LON_MAX,
    PROCESSED_DIR, RAW_DATA_PATH,
)


def load_and_clean(path=None) -> pd.DataFrame:
    """
    Load the LA crime CSV, clean it, and return a tidy DataFrame.

    Cleaning steps:
      1. Drop duplicate report numbers
      2. Build real datetime: date from DATE OCC + actual HH:MM from TIME OCC
         (TIME OCC is a military-time int: 2130 → 21:30, 900 → 09:00, 1 → 00:01)
      3. Drop rows with invalid datetime
      4. Drop rows with missing or zero coordinates
      5. Filter to LA bounding box
      6. Sort chronologically
    """
    path = path or RAW_DATA_PATH
    print(f"[DataLoader] Loading: {path}")

    df = pd.read_csv(path, dtype={"DR_NO": str}, low_memory=False)
    print(f"[DataLoader] Raw rows: {len(df):,}")

    # ── 1. Remove duplicate incident reports ─────────────────────────────────
    df = df.drop_duplicates(subset=["DR_NO"])
    print(f"[DataLoader] After drop_duplicates: {len(df):,}")

    # ── 2. Parse datetime (date from DATE OCC + actual time from TIME OCC) ──────
    # DATE OCC always has a dummy "12:00:00 AM" – extract date portion only.
    date_part = pd.to_datetime(
        df[COL_DATE], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    ).dt.normalize()   # floor to midnight (date only)

    # TIME OCC is a military-time integer: 2130 → 21:30, 900 → 09:00, 1 → 00:01
    time_raw  = pd.to_numeric(df[COL_TIME], errors="coerce").fillna(0).astype(int)
    hours     = (time_raw // 100).clip(0, 23)
    minutes   = (time_raw  % 100).clip(0, 59)

    # Combine: actual datetime = date + hour + minute
    df["datetime"] = (
        date_part
        + pd.to_timedelta(hours,   unit="h")
        + pd.to_timedelta(minutes, unit="m")
    )
    df["hour"] = hours   # keep hour column for downstream use

    df = df.dropna(subset=["datetime"])
    print(f"[DataLoader] After datetime parse: {len(df):,}")

    # ── 3. Filter coordinates ─────────────────────────────────────────────────
    df = df.dropna(subset=[COL_LAT, COL_LON])
    df = df[(df[COL_LAT] != 0) & (df[COL_LON] != 0)]
    df = df[
        df[COL_LAT].between(LA_LAT_MIN, LA_LAT_MAX)
        & df[COL_LON].between(LA_LON_MIN, LA_LON_MAX)
    ]
    print(f"[DataLoader] After coordinate filter: {len(df):,}")

    # ── 4. Rename coords and reorder columns for better visibility ────────────
    df = df.rename(columns={COL_LAT: "latitude", COL_LON: "longitude"})
    
    # Drop only Date Rptd (which is unused and has dummy 12:00:00 AM)
    if "Date Rptd" in df.columns:
        df = df.drop(columns=["Date Rptd"])
        
    # Reorder to put time columns right at the front
    front_cols = ["DR_NO", "datetime", COL_DATE, COL_TIME, "hour"]
    all_cols = df.columns.tolist()
    ordered_cols = front_cols + [c for c in all_cols if c not in front_cols]
    df = df[ordered_cols]

    # ── 5. Sort chronologically ───────────────────────────────────────────────
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"[DataLoader] Clean rows: {len(df):,}")
    print(
        f"[DataLoader] Date range: {df['datetime'].min().date()} "
        f"to {df['datetime'].max().date()}"
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out = PROCESSED_DIR / "crime_clean.csv"
    df.to_csv(out, index=False)
    print(f"[DataLoader] Saved -> {out}")
    return df


if __name__ == "__main__":
    clean = load_and_clean()
    print(clean[["datetime", "latitude", "longitude", "AREA NAME", "Crm Cd Desc"]].head())
