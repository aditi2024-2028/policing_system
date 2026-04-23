"""
src/feature_engineering.py  (v2 - High Accuracy)
==================================================
Key changes vs v1:
  - Spatial unit: Reporting District (1000+ districts) instead of 21 divisions
  - Full panel: cross-join all active districts x all dates (fills 0-crime days)
  - Binary target: any crime tomorrow? (0/1) - real predictive signal
  - 35+ rich features: lags, EWMA, crime streaks, days_since_crime,
    spatial context (division-level), historical baseline, cyclical encoding
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PROCESSED_DIR, TARGET_COL,
    COL_AREA_ID, COL_DIST, COL_PART, COL_WEAPON,
    MIN_CRIMES_PER_DIST, FEATURE_COLS, RANDOM_STATE,
)

_SEASON = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
           6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}


# ─────────────────────────────────────────────────────────────────────────────
def build_district_panel(df: pd.DataFrame) -> tuple:
    """
    Build a full (district x date) panel with all features and binary target.
    Returns (features_df, dist_codes, area_codes).
    """
    print("[FeatEng v2] Building full district-day panel ...")

    df = df.copy()
    df["date"]       = df["datetime"].dt.normalize()
    df["hour"]       = df["datetime"].dt.hour          # ← real hour from TIME OCC
    df["dist"]       = df[COL_DIST].astype(float).astype(int)
    df["area_id"]    = df[COL_AREA_ID].astype(int)
    df["is_part1"]   = (df[COL_PART] == 1).astype(int)
    df["has_weapon"] = df[COL_WEAPON].notna().astype(int)

    # Time-of-day buckets (applied per individual crime record)
    df["is_night"]   = df["hour"].between(0,  5).astype(int)   # 00:00–05:59
    df["is_morning"] = df["hour"].between(6, 11).astype(int)   # 06:00–11:59
    df["is_afternoon"]= df["hour"].between(12,17).astype(int)  # 12:00–17:59
    df["is_evening"] = df["hour"].between(18,23).astype(int)   # 18:00–23:59
    df["is_peak"]    = df["hour"].isin([19,20,21,22,23,0]).astype(int)  # peak crime hours

    # ── 1. Select active reporting districts ──────────────────────────────────
    dist_totals  = df.groupby("dist").size()
    active_dists = dist_totals[dist_totals >= MIN_CRIMES_PER_DIST].index
    dist_area_map = df.drop_duplicates("dist").set_index("dist")["area_id"].to_dict()
    df = df[df["dist"].isin(active_dists)]
    print(f"[FeatEng v2] Active districts: {len(active_dists)} "
          f"(min {MIN_CRIMES_PER_DIST} crimes each)")

    # ── 2. Daily aggregates per district (now includes time-of-day counts) ────
    dd = (
        df.groupby(["dist", "date"]).agg(
            crime_count   = ("datetime",    "count"),
            part1_count   = ("is_part1",    "sum"),
            weapon_count  = ("has_weapon",  "sum"),
            night_crimes  = ("is_night",    "sum"),
            morning_crimes= ("is_morning",  "sum"),
            afternoon_crimes=("is_afternoon","sum"),
            evening_crimes= ("is_evening",  "sum"),
            peak_crimes   = ("is_peak",     "sum"),
            peak_hour     = ("hour",         lambda h: h.value_counts().idxmax() if len(h) else 12),
        ).reset_index()
    )

    # ── 3. Build FULL panel (all districts x all dates, fill 0) ───────────────
    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    all_dists = sorted(active_dists.tolist())
    print(f"[FeatEng v2] Panel size: {len(all_dists)} districts x "
          f"{len(all_dates)} dates = {len(all_dists)*len(all_dates):,} rows")

    idx = pd.MultiIndex.from_product([all_dists, all_dates], names=["dist", "date"])
    panel = (
        dd.set_index(["dist", "date"])
          .reindex(idx, fill_value=0)
          .reset_index()
    )
    panel["area_id"] = panel["dist"].map(dist_area_map).fillna(0).astype(int)
    panel = panel.sort_values(["dist", "date"]).reset_index(drop=True)

    # ── 3b. Lag versions of time-of-day features (no leakage) ────────────────
    for tod_col in ["night_crimes", "morning_crimes", "afternoon_crimes",
                    "evening_crimes", "peak_crimes"]:
        panel[f"{tod_col}_t1"] = panel.groupby("dist")[tod_col].shift(1).fillna(0)
        roll7 = (panel.groupby("dist")[tod_col]
                     .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()))
        panel[f"{tod_col}_roll7"] = roll7.fillna(0)
    panel["peak_hour_t1"] = panel.groupby("dist")["peak_hour"].shift(1).fillna(12)

    # ── 4. Division-level daily crime (spatial context) ───────────────────────
    area_daily = (
        panel.groupby(["area_id", "date"])["crime_count"]
             .sum()
             .reset_index()
             .rename(columns={"crime_count": "area_daily"})
    )
    panel = panel.merge(area_daily, on=["area_id", "date"], how="left")

    # ── 5. Historical baseline per district (computed on full history) ─────────
    hist = panel.groupby("dist")["crime_count"].agg(
        hist_mean_crime = "mean",
        hist_crime_rate = lambda s: (s > 0).mean(),
    ).reset_index()
    panel = panel.merge(hist, on="dist", how="left")

    # ── 6. Per-district lag / rolling / EWMA features ─────────────────────────
    panel = _add_lag_features(panel)

    # ── 7. Division-level lag (spatial context) ───────────────────────────────
    area_lag = area_daily.sort_values(["area_id", "date"])
    area_lag["area_crime_t1"]   = area_lag.groupby("area_id")["area_daily"].shift(1).fillna(0)
    area_lag["area_roll7_mean"] = (
        area_lag.groupby("area_id")["area_daily"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        .fillna(0)
    )
    panel = panel.merge(
        area_lag[["area_id", "date", "area_crime_t1", "area_roll7_mean"]],
        on=["area_id", "date"], how="left"
    )

    # ── 8. Temporal features ──────────────────────────────────────────────────
    panel = _add_temporal_features(panel)

    # ── 9. Ordinal encodings ──────────────────────────────────────────────────
    dist_codes = {d: i for i, d in enumerate(sorted(all_dists))}
    area_codes = {a: i for i, a in enumerate(sorted(panel["area_id"].unique()))}
    panel["dist_id"]  = panel["dist"].map(dist_codes)
    panel["area_id"]  = panel["area_id"].map(area_codes)

    # ── 10. Target: any crime TOMORROW? ──────────────────────────────────────
    # Shift crime_count forward by -1 within each district (no data leakage)
    panel[TARGET_COL] = (
        panel.groupby("dist")["crime_count"].shift(-1) > 0
    ).astype("Int64")

    # Drop last day per district (no future label)
    panel = panel.dropna(subset=[TARGET_COL])
    panel[TARGET_COL] = panel[TARGET_COL].astype(int)

    # Keep only feature columns + metadata
    keep = ["dist", "date", TARGET_COL] + [c for c in FEATURE_COLS if c in panel.columns]
    panel = panel[keep].reset_index(drop=True)

    bal = panel[TARGET_COL].value_counts()
    print(f"[FeatEng v2] Final panel: {len(panel):,} rows x {len(panel.columns)} cols")
    print(f"[FeatEng v2] Target balance: 0={bal.get(0,0):,}  1={bal.get(1,0):,}  "
          f"({bal.get(1,0)/len(panel)*100:.1f}% crime days)")

    out = PROCESSED_DIR / "features_v2.csv"
    panel.to_csv(out, index=False)
    print(f"[FeatEng v2] Saved -> {out}")
    return panel, dist_codes, area_codes


# ─────────────────────────────────────────────────────────────────────────────
def _add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add all lag, rolling, EWMA, and streak features per district."""

    g = panel.groupby("dist")["crime_count"]

    # Point lags
    panel["crime_t1"]  = g.shift(1).fillna(0)
    panel["crime_t2"]  = g.shift(2).fillna(0)
    panel["crime_t3"]  = g.shift(3).fillna(0)
    panel["crime_t7"]  = g.shift(7).fillna(0)
    panel["crime_t14"] = g.shift(14).fillna(0)

    # Rolling mean / std / max
    def roll(n, fn):
        return g.shift(1).rolling(n, min_periods=1).agg(fn).reset_index(level=0, drop=True)

    panel["roll7_mean"]  = roll(7,  "mean").fillna(0)
    panel["roll7_std"]   = roll(7,  "std").fillna(0)
    panel["roll7_max"]   = roll(7,  "max").fillna(0)
    panel["roll14_mean"] = roll(14, "mean").fillna(0)
    panel["roll14_std"]  = roll(14, "std").fillna(0)
    panel["roll30_mean"] = roll(30, "mean").fillna(0)
    panel["roll30_std"]  = roll(30, "std").fillna(0)

    # EWMA (exponential weighted moving average)
    def ewma(span):
        return (
            g.shift(1)
             .transform(lambda s: s.ewm(span=span, adjust=False).mean())
             .fillna(0)
        )

    panel["ewma_3d"]  = ewma(3)
    panel["ewma_7d"]  = ewma(7)
    panel["ewma_14d"] = ewma(14)

    # Crime days in last N days (streak features)
    def crime_days(n):
        return (
            (g.shift(1) > 0)
            .rolling(n, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    panel["crime_days_last7"]  = crime_days(7)
    panel["crime_days_last14"] = crime_days(14)
    panel["crime_days_last30"] = crime_days(30)

    # Days since last crime (strong predictor)
    binary = (panel["crime_count"] > 0).astype(int)
    cumsum = binary.groupby(panel["dist"]).cumsum()
    # days_since_crime: count days backward from previous day
    panel["days_since_crime"] = (
        panel.groupby("dist").apply(
            lambda grp: _days_since_last(grp["crime_count"].values)
        ).explode().reset_index(drop=True)
        .astype(float).fillna(30)
    )

    # 7-day part1 and weapon rates
    gp1 = panel.groupby("dist")["part1_count"]
    gw  = panel.groupby("dist")["weapon_count"]

    roll_crimes = g.shift(1).rolling(7, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0)
    roll_p1     = gp1.shift(1).rolling(7, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0)
    roll_wpn    = gw.shift(1).rolling(7, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0)

    panel["part1_rate_7d"]  = (roll_p1  / roll_crimes.clip(lower=1)).fillna(0)
    panel["weapon_rate_7d"] = (roll_wpn / roll_crimes.clip(lower=1)).fillna(0)

    return panel


def _days_since_last(arr: np.ndarray) -> np.ndarray:
    """For each position i, compute how many days since last crime before i."""
    result = np.full(len(arr), 30.0)  # cap at 30
    since = 30
    for i in range(len(arr)):
        result[i] = since
        # After recording, update using the CURRENT day's crime status
        if arr[i] > 0:
            since = 0
        else:
            since = min(since + 1, 30)
    return result


def _add_temporal_features(panel: pd.DataFrame) -> pd.DataFrame:
    dt = panel["date"]
    panel["day_of_week"] = dt.dt.dayofweek
    panel["is_weekend"]  = (panel["day_of_week"] >= 5).astype(int)
    panel["month"]       = dt.dt.month
    panel["quarter"]     = dt.dt.quarter
    panel["season"]      = panel["month"].map(_SEASON)
    doy = dt.dt.day_of_year
    panel["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365)
    panel["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365)
    panel["month_sin"]   = np.sin(2 * np.pi * panel["month"] / 12)
    panel["month_cos"]   = np.cos(2 * np.pi * panel["month"] / 12)
    panel["dow_sin"]     = np.sin(2 * np.pi * panel["day_of_week"] / 7)
    panel["dow_cos"]     = np.cos(2 * np.pi * panel["day_of_week"] / 7)
    return panel


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "crime_clean.csv", parse_dates=["datetime"])
    tbl, dc, ac = build_district_panel(df)
    print(tbl.head(3))
    print("\nSample target distribution by district (first 5):")
    print(tbl.groupby("dist")[TARGET_COL].mean().head())
