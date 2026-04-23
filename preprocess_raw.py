"""
preprocess_raw.py
=================
Step 1 : Extract data2.csv from data2.csv.zip
Step 2 : Fix DATE OCC -- replace the dummy "12:00:00 AM" with the REAL
          occurrence time from TIME OCC (military format: 2130 -> 21:30:00).
Step 3 : Save the corrected CSV as  data/raw/data2.csv
Step 4 : Delete stale processed files so the pipeline rebuilds cleanly.
Step 5 : Run data_loader  -> crime_clean.csv
Step 6 : Run feature_engineering -> features_v2.csv

Usage
-----
    python preprocess_raw.py
"""
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

ZIP_PATH      = Path("data2.csv.zip")
RAW_OUT       = Path("data/raw/data2.csv")
PROCESSED_DIR = Path("data/processed")

t0 = time.time()
print("=" * 65)
print("  Raw Data Preprocessing: fixing DATE OCC with real TIME OCC")
print("=" * 65)

# ---- Step 1: Extract zip ------------------------------------------------
if not ZIP_PATH.exists():
    sys.exit(f"[ERROR] {ZIP_PATH} not found!")

print(f"\n[Step 1] Extracting {ZIP_PATH} ...")
RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    members = zf.namelist()
    print(f"         Contents: {members}")
    csv_members = [m for m in members if m.lower().endswith(".csv")]
    if not csv_members:
        sys.exit("[ERROR] No CSV found inside zip!")
    extracted_name = csv_members[0]
    zf.extract(extracted_name, path="data/raw/")
    extracted_path = Path("data/raw") / extracted_name
    if extracted_path != RAW_OUT:
        if RAW_OUT.exists():
            RAW_OUT.unlink()
        extracted_path.rename(RAW_OUT)
    print(f"         Extracted -> {RAW_OUT}  ({RAW_OUT.stat().st_size / 1e6:.1f} MB)")

# ---- Step 2: Load and fix DATE OCC ---------------------------------------
print(f"\n[Step 2] Loading raw data and fixing DATE OCC ...")
df = pd.read_csv(RAW_OUT, dtype={"DR_NO": str}, low_memory=False)
print(f"         Rows: {len(df):,}  |  Columns: {len(df.columns)}")

print(f"\n         BEFORE fix (sample DATE OCC): {df['DATE OCC'].iloc[:3].tolist()}")
print(f"         BEFORE fix (sample TIME OCC): {df['TIME OCC'].iloc[:3].tolist()}")

# Parse the date portion only from DATE OCC (ignore its dummy "12:00:00 AM")
date_only = pd.to_datetime(
    df["DATE OCC"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
).dt.normalize()   # floor to midnight -> pure date

# Parse real time from TIME OCC
time_raw = pd.to_numeric(df["TIME OCC"], errors="coerce").fillna(0).astype(int)
hours    = (time_raw // 100).clip(0, 23)
minutes  = (time_raw  % 100).clip(0, 59)

# Build corrected datetime by adding real hours + minutes to the date
corrected_dt = (
    date_only
    + pd.to_timedelta(hours,   unit="h")
    + pd.to_timedelta(minutes, unit="m")
)

# Write back in LAPD string format but with REAL time
df["DATE OCC"] = corrected_dt.dt.strftime("%m/%d/%Y %I:%M:%S %p")

print(f"\n         AFTER fix (sample DATE OCC): {df['DATE OCC'].iloc[:3].tolist()}")
print("\n         Hour distribution in fixed DATE OCC:")
fixed_hours = corrected_dt.dt.hour.value_counts().sort_index()
for h, cnt in fixed_hours.items():
    bar = "#" * (int(cnt) // 2000)
    print(f"           {int(h):02d}:00  {int(cnt):7,}  {bar}")

# ---- Step 3: Save corrected raw file ------------------------------------
print(f"\n[Step 3] Saving corrected data2.csv -> {RAW_OUT} ...")
df.to_csv(RAW_OUT, index=False)
print(f"         Saved: {RAW_OUT.stat().st_size / 1e6:.1f} MB")

# ---- Step 4: Delete stale processed files --------------------------------
print(f"\n[Step 4] Removing stale processed files ...")
for fname in ["crime_clean.csv", "features_v2.csv"]:
    p = PROCESSED_DIR / fname
    if p.exists():
        p.unlink()
        print(f"         Deleted: {p}")
    else:
        print(f"         Not found (already clean): {p}")

# ---- Step 5: Rebuild crime_clean.csv -------------------------------------
print(f"\n[Step 5] Rebuilding crime_clean.csv ...")
from src.data_loader import load_and_clean
df_clean = load_and_clean(RAW_OUT)

hour_check = df_clean["datetime"].dt.hour.value_counts().sort_index()
print("         Hour verification in rebuilt crime_clean.csv:")
for h in [0, 6, 12, 18, 20, 23]:
    print(f"           {h:02d}:xx  {hour_check.get(h, 0):,}")

# ---- Step 6: Rebuild features_v2.csv -------------------------------------
print(f"\n[Step 6] Rebuilding features_v2.csv ...")
from src.feature_engineering import build_district_panel
feat_df, dist_codes, area_codes = build_district_panel(df_clean)

elapsed = time.time() - t0
print(f"\n{'='*65}")
print(f"  [DONE]  Preprocessing complete in {elapsed:.1f}s")
print(f"  data/raw/data2.csv              -> Fixed (real DATE OCC times)")
print(f"  data/processed/crime_clean.csv  -> Rebuilt")
print(f"  data/processed/features_v2.csv  -> Rebuilt")
print(f"\n  Next steps (retrain models on corrected data):")
print(f"    python final_train.py")
print(f"    python src/lstm_training.py")
print(f"{'='*65}")
