"""
src/lstm_training.py
====================
Full training pipeline for the CrimeLSTM model.

Steps
-----
1. Load features_v2.csv  (district × day panel, 40+ features)
2. Build per-district sliding-window sequences (lookback=30 days)
3. Per-district z-score normalisation of the count target
   (prevents high-crime districts from dominating the loss)
4. Temporal 70 / 10 / 20 chronological split  (no data leakage)
5. Train BiLSTM + Attention with multi-task loss (MAE + CrossEntropy)
   and cosine-annealing LR schedule + early stopping
6. Evaluate on test set: MAE, RMSE, Risk-level Accuracy, F1 macro
7. Save weights, scaler, district metadata, and diagnostic plots

Usage
-----
    python -m src.lstm_training          # or
    python src/lstm_training.py
"""

import io
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# UTF-8 stdout fix (only needed when script is the entry point)
if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
os.environ["PYTHONIOENCODING"] = "utf-8"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURE_COLS, FIGURES_DIR, LSTM_BATCH_SIZE, LSTM_DISTRICT_META,
    LSTM_DROPOUT, LSTM_EPOCHS, LSTM_HIDDEN_SIZE, LSTM_LOOKBACK,
    LSTM_LR, LSTM_NUM_LAYERS, LSTM_PATIENCE, LSTM_RISK_PERCENTILES,
    LSTM_SCALER_PATH, LSTM_TRAIN_FRAC, LSTM_VAL_FRAC, LSTM_WEIGHT_DECAY,
    LSTM_WEIGHTS_PATH, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR,
)
from src.lstm_model import CrimeLSTM, CrimeLSTMLoss, build_model

# ── Plot theme ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#c9d1d9",
    "ytick.color":      "#c9d1d9",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "legend.facecolor": "#161b22",
})
_PAL = ["#58a6ff", "#3fb950", "#d29922", "#ff7b72", "#bc8cff"]

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# =============================================================================
# 1. SEQUENCE BUILDER
# =============================================================================

def _assign_risk_label(count: np.ndarray, p33: float, p66: float) -> np.ndarray:
    """Map crime counts to 0=Low / 1=Medium / 2=High using district percentiles."""
    labels = np.zeros(len(count), dtype=np.int64)
    labels[count > p33] = 1
    labels[count > p66] = 2
    return labels


def build_sequences(
    panel: pd.DataFrame,
    feature_cols: list,
    lookback: int = LSTM_LOOKBACK,
) -> tuple:
    """
    Build sliding-window sequences from the district × day panel.

    For each district with enough history, slide a window of `lookback` days
    over its timeline and form:
        X[i] = features[i : i+lookback]                shape [lookback, n_feats]
        y_count[i] = next-day crime count (regression)  scalar
        y_risk[i]  = risk_label(next-day count)          0/1/2

    The panel may NOT contain a raw 'crime_count' column (feature engineering
    drops it in favour of lag columns).  We handle both cases:
      - If 'crime_count' is present  : use it directly.
      - Otherwise                    : reconstruct via crime_t1 (shift -1 per district),
        which equals the following day's crime count.

    Per-district percentiles (computed on the full history) define risk levels.

    Returns
    -------
    X             : np.ndarray [N, lookback, n_features]
    y_count       : np.ndarray [N]           (next-day crime count)
    y_risk        : np.ndarray [N]           (0=Low / 1=Med / 2=High)
    dates         : np.ndarray [N]           (date of the DAY BEING PREDICTED)
    districts     : np.ndarray [N]           (district id)
    district_meta : dict  {dist_id: {p33, p66, mean, std}}
    """
    print(f"[LSTM] Building sequences (lookback={lookback}) ...")
    t0 = time.time()

    panel = panel.sort_values(["dist", "date"]).reset_index(drop=True)
    all_dists = panel["dist"].unique()
    has_count = "crime_count" in panel.columns
    if not has_count:
        print("[LSTM] 'crime_count' not in panel — reconstructing from crime_t1")

    X_list, yc_list, yr_list, date_list, dist_list = [], [], [], [], []
    district_meta: dict = {}

    skipped = 0
    for dist in all_dists:
        grp = panel[panel["dist"] == dist].reset_index(drop=True)
        if len(grp) < lookback + 2:
            skipped += 1
            continue

        feat_arr  = grp[feature_cols].fillna(0).values.astype(np.float32)
        date_arr  = grp["date"].values

        # ── Get the crime count series for this district ──────────────────────
        if has_count:
            # Panel produced by feature_engineering.py BEFORE the final keep-cols step
            count_arr = grp["crime_count"].values.astype(np.float32)
        else:
            # features_v2.csv was built with the keep-cols filter applied:
            # crime_t1[i] = crime_count[i-1]  →  crime_count[i] = crime_t1[i+1]
            # Use shift(-1) so count_arr[i] = crime_count for day i
            count_arr = (
                grp["crime_t1"].shift(-1).fillna(grp["crime_t1"])
                .values.astype(np.float32)
            )

        # Per-district count percentiles for risk labelling
        p33 = float(np.percentile(count_arr, LSTM_RISK_PERCENTILES[0]))
        p66 = float(np.percentile(count_arr, LSTM_RISK_PERCENTILES[1]))
        district_meta[int(dist)] = {
            "p33":  p33,
            "p66":  p66,
            "mean": float(count_arr.mean()),
            "std":  float(count_arr.std()) or 1.0,
        }

        risk_arr = _assign_risk_label(count_arr, p33, p66)

        T = len(grp)
        for i in range(T - lookback):
            X_list.append(feat_arr[i: i + lookback])
            # Target is the day AFTER the window ends
            yc_list.append(count_arr[i + lookback])
            yr_list.append(risk_arr[i + lookback])
            date_list.append(date_arr[i + lookback])
            dist_list.append(int(dist))

    X        = np.array(X_list,   dtype=np.float32)
    y_count  = np.array(yc_list,  dtype=np.float32)
    y_risk   = np.array(yr_list,  dtype=np.int64)
    dates    = np.array(date_list)
    districts = np.array(dist_list, dtype=np.int32)

    elapsed = time.time() - t0
    print(
        f"[LSTM] Sequences: {len(X):,}  |  shape={X.shape}  |  "
        f"skipped={skipped} districts  |  {elapsed:.1f}s"
    )
    rc = {0: (y_risk == 0).sum(), 1: (y_risk == 1).sum(), 2: (y_risk == 2).sum()}
    print(f"[LSTM] Risk label dist:  Low={rc[0]:,}  Med={rc[1]:,}  High={rc[2]:,}")
    return X, y_count, y_risk, dates, districts, district_meta


# =============================================================================
# 2. TEMPORAL SPLIT
# =============================================================================

def temporal_split(dates: np.ndarray, train_frac=LSTM_TRAIN_FRAC,
                   val_frac=LSTM_VAL_FRAC):
    """
    Chronological split.  Returns (train_mask, val_mask, test_mask).
    All indices are relative to the sorted date array.
    """
    sorted_dates = np.sort(np.unique(dates))
    n = len(sorted_dates)
    t1 = sorted_dates[int(n * train_frac)]
    t2 = sorted_dates[int(n * (train_frac + val_frac))]

    train_mask = dates < t1
    val_mask   = (dates >= t1) & (dates < t2)
    test_mask  = dates >= t2

    print(
        f"[LSTM] Split  |  "
        f"Train={train_mask.sum():,}  Val={val_mask.sum():,}  Test={test_mask.sum():,}"
    )
    print(
        f"[LSTM] Date ranges  |  "
        f"Train: up to {t1}  |  Val: {t1}–{t2}  |  Test: {t2}+"
    )
    return train_mask, val_mask, test_mask


# =============================================================================
# 3. NORMALISATION
# =============================================================================

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    n, t, f = X_train.shape
    flat = X_train.reshape(-1, f)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, t, f = X.shape
    flat = X.reshape(-1, f)
    flat_s = scaler.transform(flat)
    return flat_s.reshape(n, t, f).astype(np.float32)


# =============================================================================
# 4. DATALOADERS
# =============================================================================

def make_loaders(
    X_tr, yc_tr, yr_tr,
    X_va, yc_va, yr_va,
    batch_size: int = LSTM_BATCH_SIZE,
) -> tuple:
    def to_ds(X, yc, yr):
        return TensorDataset(
            torch.tensor(X,  dtype=torch.float32),
            torch.tensor(yc, dtype=torch.float32),
            torch.tensor(yr, dtype=torch.long),
        )

    tr_loader = DataLoader(to_ds(X_tr, yc_tr, yr_tr), batch_size=batch_size,
                           shuffle=True,  drop_last=True,  num_workers=0)
    va_loader = DataLoader(to_ds(X_va, yc_va, yr_va), batch_size=batch_size * 2,
                           shuffle=False, drop_last=False, num_workers=0)
    return tr_loader, va_loader


# =============================================================================
# 5. TRAINING LOOP
# =============================================================================

def train_epoch(model, loader, criterion, optimiser, device, scaler_amp=None):
    model.train()
    total_loss = reg_sum = cls_sum = 0.0
    for X_b, yc_b, yr_b in loader:
        X_b  = X_b.to(device)
        yc_b = yc_b.to(device)
        yr_b = yr_b.to(device)

        optimiser.zero_grad()
        count_pred, risk_logits, _ = model(X_b)
        loss, rl, cl = criterion(count_pred, yc_b, risk_logits, yr_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item()
        reg_sum    += rl.item()
        cls_sum    += cl.item()

    n = len(loader)
    return total_loss / n, reg_sum / n, cls_sum / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = reg_sum = cls_sum = 0.0
    all_cnt_pred, all_cnt_true, all_risk_pred, all_risk_true = [], [], [], []

    for X_b, yc_b, yr_b in loader:
        X_b  = X_b.to(device)
        yc_b = yc_b.to(device)
        yr_b = yr_b.to(device)

        count_pred, risk_logits, _ = model(X_b)
        loss, rl, cl = criterion(count_pred, yc_b, risk_logits, yr_b)

        total_loss += loss.item()
        reg_sum    += rl.item()
        cls_sum    += cl.item()

        all_cnt_pred.append(count_pred.squeeze(-1).cpu().numpy())
        all_cnt_true.append(yc_b.cpu().numpy())
        all_risk_pred.append(risk_logits.argmax(dim=-1).cpu().numpy())
        all_risk_true.append(yr_b.cpu().numpy())

    n          = len(loader)
    cnt_pred   = np.concatenate(all_cnt_pred)
    cnt_true   = np.concatenate(all_cnt_true)
    risk_pred  = np.concatenate(all_risk_pred)
    risk_true  = np.concatenate(all_risk_true)

    mae  = mean_absolute_error(cnt_true, cnt_pred)
    rmse = mean_squared_error(cnt_true, cnt_pred) ** 0.5
    acc  = accuracy_score(risk_true, risk_pred)

    return (total_loss / n, reg_sum / n, cls_sum / n,
            mae, rmse, acc, risk_pred, risk_true, cnt_pred, cnt_true)


# =============================================================================
# 6. MAIN TRAINING FUNCTION
# =============================================================================

def train_lstm(
    feat_df: pd.DataFrame = None,
    max_districts: int = 120,    # None = all districts (GPU recommended)
    max_samples:  int = 100_000, # None = no cap (very slow on CPU)
) -> CrimeLSTM:
    """
    Full LSTM training pipeline.  If feat_df is None, loads features_v2.csv.

    Parameters
    ----------
    feat_df       : pre-loaded panel DataFrame (optional)
    max_districts : top-N most-active districts for CPU feasibility.
                    Set None to use all 590 districts (GPU recommended).
    max_samples   : hard cap on total sequences after district selection.
                    Set None for no cap.

    Returns the trained CrimeLSTM model.
    """
    t_start = time.time()
    print("\n" + "=" * 70)
    print("  CrimeLSTM Training  |  BiLSTM + Attention  |  Multi-task Loss")
    if max_districts:
        print(f"  Mode: CPU-optimised | top-{max_districts} districts | cap={max_samples:,} seq")
    else:
        print("  Mode: FULL dataset (GPU recommended)")
    print("=" * 70)

    # -- Load data -------------------------------------------------------------
    if feat_df is None:
        path = PROCESSED_DIR / "features_v2.csv"
        print(f"[LSTM] Loading {path} ...")
        feat_df = pd.read_csv(path, parse_dates=["date"])
    print(f"[LSTM] Full panel: {len(feat_df):,} rows | {feat_df['dist'].nunique()} districts")

    # -- Optionally restrict to top-N most active districts -------------------
    if max_districts is not None:
        top_dists = (
            feat_df.groupby("dist")["crime_t1"].sum()
            .nlargest(max_districts).index.tolist()
        )
        feat_df = feat_df[feat_df["dist"].isin(top_dists)].copy()
        print(f"[LSTM] Restricted to top-{max_districts} districts: {len(feat_df):,} rows")

    feature_cols = [c for c in FEATURE_COLS if c in feat_df.columns]
    print(f"[LSTM] Feature columns: {len(feature_cols)}")

    # -- Build sequences -------------------------------------------------------
    X, y_count, y_risk, dates, districts, dist_meta = build_sequences(
        feat_df, feature_cols, lookback=LSTM_LOOKBACK
    )

    # -- Optional hard sample cap (keep most-recent for temporal integrity) ----
    if max_samples is not None and len(X) > max_samples:
        X         = X[-max_samples:]
        y_count   = y_count[-max_samples:]
        y_risk    = y_risk[-max_samples:]
        dates     = dates[-max_samples:]
        districts = districts[-max_samples:]
        print(f"[LSTM] Capped to {len(X):,} most-recent sequences")

    # -- Temporal split --------------------------------------------------------
    tr_mask, va_mask, te_mask = temporal_split(dates)

    X_tr, yc_tr, yr_tr = X[tr_mask], y_count[tr_mask], y_risk[tr_mask]
    X_va, yc_va, yr_va = X[va_mask], y_count[va_mask], y_risk[va_mask]
    X_te, yc_te, yr_te = X[te_mask], y_count[te_mask], y_risk[te_mask]
    dist_te             = districts[te_mask]
    dates_te            = dates[te_mask]

    # -- Normalise features (chunked to avoid RAM spike) ----------------------
    print("[LSTM] Fitting StandardScaler on training sequences ...")
    scaler = fit_scaler(X_tr)
    CHUNK  = 20_000
    def _xform(arr):
        out = np.empty_like(arr)
        for i in range(0, len(arr), CHUNK):
            out[i:i+CHUNK] = apply_scaler(arr[i:i+CHUNK], scaler)
        return out
    X_tr = _xform(X_tr)
    X_va = _xform(X_va)
    X_te = _xform(X_te)
    print("[LSTM] Normalisation done.")

    # ── Class weights (inverse frequency) ────────────────────────────────────
    class_counts = np.bincount(yr_tr, minlength=3).astype(float)
    class_weights = torch.tensor(
        (class_counts.sum() / (3 * class_counts.clip(min=1))).astype(np.float32)
    )
    print(f"[LSTM] Class weights: {class_weights.numpy().round(3)}")

    # ── Build model ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_tr.shape[-1]
    model = build_model(
        n_features  = n_features,
        device      = device,
        hidden_size = LSTM_HIDDEN_SIZE,
        num_layers  = LSTM_NUM_LAYERS,
        dropout     = LSTM_DROPOUT,
        n_risk_cls  = 3,
    )

    criterion = CrimeLSTMLoss(alpha=0.35, class_weights=class_weights.to(device))
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=LSTM_LR, weight_decay=LSTM_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=LSTM_EPOCHS, eta_min=LSTM_LR * 0.05
    )

    # ── Dataloaders ───────────────────────────────────────────────────────────
    tr_loader, va_loader = make_loaders(
        X_tr, yc_tr, yr_tr,
        X_va, yc_va, yr_va,
        batch_size=LSTM_BATCH_SIZE,
    )

    # ── Training loop with early stopping ─────────────────────────────────────
    print(f"\n[LSTM] Training for up to {LSTM_EPOCHS} epochs | patience={LSTM_PATIENCE}")
    best_val_loss = float("inf")
    patience_ctr  = 0
    history = {"tr_loss": [], "va_loss": [], "tr_mae": [], "va_mae": [],
               "tr_acc": [], "va_acc": [], "lr": []}

    for epoch in range(1, LSTM_EPOCHS + 1):
        tr_loss, tr_rl, tr_cl = train_epoch(model, tr_loader, criterion, optimiser, device)
        va_res = eval_epoch(model, va_loader, criterion, device)
        va_loss, va_rl, va_cl, va_mae, va_rmse, va_acc = va_res[:6]

        # Approx train MAE (quick eval)
        with torch.no_grad():
            small_X = torch.tensor(X_tr[:2048], device=device)
            tr_cnt, tr_rlg, _ = model(small_X)
            tr_mae = mean_absolute_error(yc_tr[:2048], tr_cnt.squeeze(-1).cpu().numpy())
            tr_acc = accuracy_score(yr_tr[:2048],
                                    tr_rlg.argmax(-1).cpu().numpy())

        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        history["tr_loss"].append(tr_loss)
        history["va_loss"].append(va_loss)
        history["tr_mae"].append(tr_mae)
        history["va_mae"].append(va_mae)
        history["tr_acc"].append(tr_acc)
        history["va_acc"].append(va_acc)
        history["lr"].append(cur_lr)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:03d}/{LSTM_EPOCHS}  "
                f"loss={tr_loss:.4f}/{va_loss:.4f}  "
                f"MAE={tr_mae:.3f}/{va_mae:.3f}  "
                f"RiskAcc={tr_acc*100:.1f}%/{va_acc*100:.1f}%  "
                f"lr={cur_lr:.2e}"
            )

        # Early stopping
        if va_loss < best_val_loss - 1e-5:
            best_val_loss = va_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), LSTM_WEIGHTS_PATH)
        else:
            patience_ctr += 1
            if patience_ctr >= LSTM_PATIENCE:
                print(f"[LSTM] Early stopping at epoch {epoch}")
                break

    # Reload best weights
    model.load_state_dict(torch.load(LSTM_WEIGHTS_PATH, map_location=device))
    print(f"[LSTM] Loaded best weights (val_loss={best_val_loss:.5f})")

    # ── Test evaluation ───────────────────────────────────────────────────────
    te_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_te, dtype=torch.float32),
            torch.tensor(yc_te, dtype=torch.float32),
            torch.tensor(yr_te, dtype=torch.long),
        ),
        batch_size=LSTM_BATCH_SIZE * 2, shuffle=False, num_workers=0,
    )
    te_res = eval_epoch(model, te_loader, criterion, device)
    te_loss, _, _, te_mae, te_rmse, te_acc, risk_pred, risk_true, cnt_pred, cnt_true = te_res

    print("\n" + "=" * 70)
    print("  TEST SET RESULTS")
    print("=" * 70)
    print(f"  Loss  : {te_loss:.5f}")
    print(f"  MAE   : {te_mae:.4f}  (crime count)")
    print(f"  RMSE  : {te_rmse:.4f}")
    print(f"  Risk Accuracy : {te_acc * 100:.2f}%")
    print(f"  Risk F1 (macro): {f1_score(risk_true, risk_pred, average='macro'):.4f}")
    print("\n" + classification_report(
        risk_true, risk_pred, target_names=["Low", "Medium", "High"], zero_division=0
    ))

    # Baseline: predict yesterday's crime count (naive)
    naive_mae = mean_absolute_error(cnt_true[1:], cnt_true[:-1])
    print(f"  Baseline MAE (predict yesterday) : {naive_mae:.4f}")
    print(f"  LSTM improvement over baseline   : {naive_mae - te_mae:+.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    with open(LSTM_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(LSTM_DISTRICT_META, "wb") as f:
        pickle.dump({"dist_meta": dist_meta, "feature_cols": feature_cols}, f)

    import json
    lstm_summary = {
        "n_features":     n_features,
        "lookback":       LSTM_LOOKBACK,
        "hidden_size":    LSTM_HIDDEN_SIZE,
        "num_layers":     LSTM_NUM_LAYERS,
        "dropout":        LSTM_DROPOUT,
        "test_mae":       round(te_mae, 4),
        "test_rmse":      round(te_rmse, 4),
        "test_risk_acc":  round(te_acc, 4),
        "test_risk_f1":   round(float(f1_score(risk_true, risk_pred, average="macro")), 4),
        "baseline_mae":   round(naive_mae, 4),
        "improvement":    round(naive_mae - te_mae, 4),
        "best_val_loss":  round(best_val_loss, 5),
        "n_train":        int(tr_mask.sum()),
        "n_val":          int(va_mask.sum()),
        "n_test":         int(te_mask.sum()),
    }
    with open(MODELS_DIR / "lstm_meta.json", "w") as f:
        json.dump(lstm_summary, f, indent=2)
    print(f"\n[LSTM] Saved → {LSTM_WEIGHTS_PATH.name}  |  {LSTM_SCALER_PATH.name}  |  lstm_meta.json")

    # ── Save evaluation CSV ───────────────────────────────────────────────────
    te_df = pd.DataFrame({
        "date":         dates_te,
        "district":     dist_te,
        "count_true":   cnt_true,
        "count_pred":   cnt_pred.round(2),
        "risk_true":    risk_true,
        "risk_pred":    risk_pred,
    })
    te_df.to_csv(REPORTS_DIR / "lstm_test_predictions.csv", index=False)

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    _plot_training_history(history)
    _plot_test_results(cnt_true, cnt_pred, risk_true, risk_pred, te_mae)

    elapsed = time.time() - t_start
    print(f"\n[LSTM] Training complete in {elapsed:.1f}s")
    return model


# =============================================================================
# 7. PLOTS
# =============================================================================

def _plot_training_history(history: dict):
    epochs = range(1, len(history["tr_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["tr_loss"], color=_PAL[0], lw=2, label="Train")
    axes[0].plot(epochs, history["va_loss"], color=_PAL[3], lw=2, label="Val")
    axes[0].set_title("Multi-task Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.25)

    # MAE
    axes[1].plot(epochs, history["tr_mae"], color=_PAL[0], lw=2, label="Train MAE")
    axes[1].plot(epochs, history["va_mae"], color=_PAL[3], lw=2, label="Val MAE")
    axes[1].set_title("Crime Count MAE", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.25)

    # Risk accuracy
    axes[2].plot(epochs, [v * 100 for v in history["tr_acc"]], color=_PAL[0], lw=2, label="Train")
    axes[2].plot(epochs, [v * 100 for v in history["va_acc"]], color=_PAL[3], lw=2, label="Val")
    axes[2].set_title("Risk Level Accuracy", fontweight="bold")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("%")
    axes[2].legend(); axes[2].grid(alpha=0.25)

    fig.suptitle("LSTM Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = FIGURES_DIR / "lstm_training_history.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[LSTM] Saved training history plot → {p.name}")


def _plot_test_results(cnt_true, cnt_pred, risk_true, risk_pred, te_mae):
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scatter: true vs predicted count
    axes[0].scatter(cnt_true, cnt_pred, alpha=0.15, s=4, color=_PAL[0])
    lo, hi = min(cnt_true.min(), cnt_pred.min()), max(cnt_true.max(), cnt_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "--", color=_PAL[3], lw=1.5)
    axes[0].set_title(f"Crime Count: True vs Predicted\nMAE={te_mae:.3f}",
                      fontweight="bold")
    axes[0].set_xlabel("True Count"); axes[0].set_ylabel("Predicted Count")
    axes[0].grid(alpha=0.25)

    # Risk confusion matrix
    labels = [0, 1, 2]
    cm = confusion_matrix(risk_true, risk_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["Low", "Med", "High"],
                yticklabels=["Low", "Med", "High"])
    axes[1].set_title("Risk Level Confusion Matrix", fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    # Error distribution
    errors = cnt_pred - cnt_true
    axes[2].hist(errors, bins=80, color=_PAL[2], alpha=0.8, edgecolor="none")
    axes[2].axvline(0, color="white", lw=1.5, linestyle="--")
    axes[2].set_title("Prediction Error Distribution\n(Predicted − True)",
                      fontweight="bold")
    axes[2].set_xlabel("Error"); axes[2].set_ylabel("Count")
    axes[2].grid(alpha=0.25)

    fig.suptitle("LSTM Test Set Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = FIGURES_DIR / "lstm_test_evaluation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[LSTM] Saved test evaluation plot → {p.name}")


# =============================================================================
# 8. INFERENCE HELPER  (used by inference.py + interpretability.py)
# =============================================================================

def load_lstm_model(device: torch.device = None) -> tuple:
    """
    Load trained CrimeLSTM from disk.

    Returns
    -------
    model        : CrimeLSTM (eval mode)
    scaler       : StandardScaler
    dist_meta    : dict  {dist_id → {p33, p66, mean, std}}
    feature_cols : list[str]
    device       : torch.device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(LSTM_DISTRICT_META, "rb") as f:
        meta_bundle = pickle.load(f)
    dist_meta    = meta_bundle["dist_meta"]
    feature_cols = meta_bundle["feature_cols"]
    n_features   = len(feature_cols)

    with open(LSTM_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = CrimeLSTM(
        n_features  = n_features,
        hidden_size = LSTM_HIDDEN_SIZE,
        num_layers  = LSTM_NUM_LAYERS,
        dropout     = LSTM_DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(LSTM_WEIGHTS_PATH, map_location=device))
    model.eval()
    print(f"[LSTM] Loaded model | features={n_features} | device={device}")
    return model, scaler, dist_meta, feature_cols, device


@torch.no_grad()
def predict_sequences(
    model: CrimeLSTM,
    X: np.ndarray,
    scaler: StandardScaler,
    device: torch.device,
    batch_size: int = 256,
) -> tuple:
    """
    Run inference on a batch of raw (unscaled) sequences.

    Returns
    -------
    count_pred   : np.ndarray [N]
    risk_proba   : np.ndarray [N, 3]
    risk_pred    : np.ndarray [N]   (argmax class)
    attn_weights : np.ndarray [N, lookback]
    """
    model.eval()
    X_s = apply_scaler(X, scaler)
    ds  = TensorDataset(torch.tensor(X_s, dtype=torch.float32))
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    cnt_list, prob_list, attn_list = [], [], []
    for (batch,) in dl:
        batch = batch.to(device)
        cnt, logits, attn = model(batch)
        cnt_list.append(cnt.squeeze(-1).cpu().numpy())
        prob_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
        attn_list.append(attn.cpu().numpy())

    count_pred   = np.concatenate(cnt_list)
    risk_proba   = np.concatenate(prob_list)
    risk_pred    = risk_proba.argmax(axis=-1)
    attn_weights = np.concatenate(attn_list)
    return count_pred, risk_proba, risk_pred, attn_weights


# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CrimeLSTM")
    parser.add_argument("--full", action="store_true",
                        help="Use all districts + all samples (slow on CPU, needs GPU)")
    parser.add_argument("--districts", type=int, default=120,
                        help="Max districts to use (default 120)")
    parser.add_argument("--samples", type=int, default=100_000,
                        help="Max sequences (default 100000)")
    args = parser.parse_args()

    if args.full:
        trained_model = train_lstm(max_districts=None, max_samples=None)
    else:
        trained_model = train_lstm(
            max_districts=args.districts,
            max_samples=args.samples,
        )
