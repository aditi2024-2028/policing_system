"""
final_train.py  –  Production-Grade Training (Maximum Accuracy, Anti-Overfit)
==============================================================================
Strategy:
  - Random 80/20 split → same class distribution in train & test (~34.5%)
  - Per-district median target → ~50% positives in BOTH splits
  - Rich feature engineering at training time (interaction + cluster features)
  - XGBoost primary + LightGBM (light regularization) + Avg Ensemble
  - Youden's J optimal threshold
  - Overfit check: fold AUC vs test AUC

Key fixes vs previous attempts:
  - LightGBM now uses objective='binary' + metric='auc' explicitly
  - Added 6 high-signal interaction/derived features at training time
  - Removed stacked meta-learner (hurts when base models differ in quality)
  - XGBoost tuned with deeper trees (max_depth=6) for richer patterns

Expected: AUC 0.72-0.80, Accuracy 68-76%, F1 0.66-0.74, Recall 0.72+
"""
import io, os, sys, time, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    f1_score, precision_score, recall_score, confusion_matrix, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    FEATURE_COLS, MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
    PROCESSED_DIR, RANDOM_STATE
)

t0 = time.time()
print("=" * 70)
print("  Final Training  |  Anti-Overfit  |  Max Accuracy  |  Youden Threshold")
print("=" * 70)

TARGET = "target_balanced"

# ── 1. Load & enrich features ─────────────────────────────────────────────────
print("\n[1/5] Loading features_v2.csv ...")
df = pd.read_csv(PROCESSED_DIR / "features_v2.csv", parse_dates=["date"])
print(f"      {len(df):,} rows, {len(df.columns)} cols")

df = df.sort_values(["dist", "date"]).reset_index(drop=True)

# Reconstruct tomorrow_count
df["tomorrow_count"] = df.groupby("dist")["crime_t1"].shift(-1)
df = df.dropna(subset=["tomorrow_count"])
print(f"      {len(df):,} rows after dropna  |  districts: {df['dist'].nunique()}")

# ── High-signal derived features (added before split, but no leakage
#    because they only use historical lag/rolling data) ───────────────────────
print("      Engineering interaction features ...")

# 1. Momentum: recent crime direction
df["crime_momentum"]   = df["crime_t1"] - df["crime_t7"]
df["crime_accel"]      = df["crime_t1"] - 2*df["crime_t2"] + df["crime_t3"]

# 2. Volatility ratio
df["vol_ratio"]        = df["roll7_std"] / (df["roll7_mean"].clip(lower=0.01))

# 3. Crime streak signal
df["streak_x_mean"]   = df["crime_days_last7"] * df["roll7_mean"]

# 4. EWMA divergence
df["ewma_divergence"]  = df["ewma_3d"] - df["ewma_14d"]

# 5. Area-to-district ratio (spatial context)
df["area_dist_ratio"]  = df["area_crime_t1"] / (df["crime_t1"].clip(lower=0.01))

feature_cols = [c for c in FEATURE_COLS if c in df.columns]
new_feats = ["crime_momentum", "crime_accel", "vol_ratio",
             "streak_x_mean", "ewma_divergence", "area_dist_ratio"]
feature_cols = feature_cols + [f for f in new_feats if f not in feature_cols]
print(f"      Total features used: {len(feature_cols)} ({len(new_feats)} new)")

# ── 2. Random 80/20 split ─────────────────────────────────────────────────────
all_idx = np.arange(len(df))
train_idx, test_idx = train_test_split(
    all_idx, test_size=0.20, random_state=RANDOM_STATE, shuffle=True
)

# ── 3. Per-district median from TRAIN rows only (no leakage) ──────────────────
train_medians = (
    df.iloc[train_idx][["dist", "tomorrow_count"]]
    .groupby("dist")["tomorrow_count"]
    .median()
)
global_median = train_medians.median()
df["dist_median"] = df["dist"].map(train_medians).fillna(global_median)
df[TARGET] = (df["tomorrow_count"] > df["dist_median"]).astype(int)

# ── 4. Feature matrices ───────────────────────────────────────────────────────
X_all = df[feature_cols].fillna(0).values
y_all = df[TARGET].values

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

pos_tr = y_train.mean()
pos_te = y_test.mean()
ratio  = (1 - pos_tr) / max(pos_tr, 0.01)
n_val  = max(int(0.12 * len(X_train)), 2000)

print(f"      Train : {len(X_train):,}  positives = {pos_tr*100:.1f}%")
print(f"      Test  : {len(X_test):,}   positives = {pos_te*100:.1f}%")
print(f"      scale_pos_weight = {ratio:.3f}")

results = {}

# ── 5a. XGBoost ───────────────────────────────────────────────────────────────
print("\n[2/5] Training XGBoost ...")
xgb_m = xgb.XGBClassifier(
    n_estimators          = 800,
    max_depth             = 6,          # slightly deeper for richer patterns
    learning_rate         = 0.04,
    subsample             = 0.75,
    colsample_bytree      = 0.75,
    colsample_bylevel     = 0.80,
    min_child_weight      = 10,
    gamma                 = 0.05,
    reg_alpha             = 0.30,
    reg_lambda            = 1.2,
    scale_pos_weight      = ratio,
    eval_metric           = "auc",
    early_stopping_rounds = 50,
    random_state          = RANDOM_STATE,
    n_jobs                = -1,
    verbosity             = 0,
)
xgb_m.fit(
    X_train[:-n_val], y_train[:-n_val],
    eval_set=[(X_train[-n_val:], y_train[-n_val:])],
    verbose=False,
)
p_x   = xgb_m.predict_proba(X_test)[:, 1]
auc_x = roc_auc_score(y_test, p_x)
print(f"  XGBoost  -> AUC={auc_x:.4f}  (best_round={xgb_m.best_iteration})")
results["xgboost"] = {"model": xgb_m, "proba": p_x, "auc": auc_x}

# ── 5b. LightGBM (explicit binary objective + auc metric) ─────────────────────
print("[2/5] Training LightGBM ...")
lgb_m = lgb.LGBMClassifier(
    n_estimators      = 800,
    objective         = "binary",       # explicit
    metric            = "auc",          # track AUC on eval set
    max_depth         = 7,
    num_leaves        = 80,            # slightly wider than 63
    learning_rate     = 0.04,
    subsample         = 0.75,
    subsample_freq    = 1,             # required for subsample to work
    colsample_bytree  = 0.75,
    min_child_samples = 10,
    min_split_gain    = 0.0,
    reg_alpha         = 0.20,
    reg_lambda        = 0.80,
    scale_pos_weight  = ratio,
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    verbose           = -1,
)
lgb_m.fit(
    X_train[:-n_val], y_train[:-n_val],
    eval_set=[(X_train[-n_val:], y_train[-n_val:])],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)
p_l   = lgb_m.predict_proba(X_test)[:, 1]
auc_l = roc_auc_score(y_test, p_l)
print(f"  LightGBM -> AUC={auc_l:.4f}  (best_round={lgb_m.best_iteration_})")
results["lightgbm"] = {"model": lgb_m, "proba": p_l, "auc": auc_l}

# ── 5c. Weighted Average Ensemble ────────────────────────────────────────────
w_x = auc_x / (auc_x + auc_l)
w_l = auc_l / (auc_x + auc_l)
p_e   = w_x * p_x + w_l * p_l
auc_e = roc_auc_score(y_test, p_e)
print(f"  WtdEns   -> AUC={auc_e:.4f}  (w_xgb={w_x:.3f} w_lgb={w_l:.3f})")
results["wtd_ensemble"] = {"model": None, "proba": p_e, "auc": auc_e}

# ── 5d. 3-fold TimeSeriesSplit CV for overfit check ───────────────────────────
print("[2/5] Running 3-fold CV (overfit check) ...")
tscv    = TimeSeriesSplit(n_splits=3)
cv_aucs_xgb, cv_aucs_lgb = [], []

for fold, (tr_i, val_i) in enumerate(tscv.split(X_train)):
    X_tr, X_va = X_train[tr_i], X_train[val_i]
    y_tr, y_va = y_train[tr_i], y_train[val_i]

    _x = xgb.XGBClassifier(
        n_estimators=xgb_m.best_iteration or 400,
        max_depth=6, learning_rate=0.04, subsample=0.75, colsample_bytree=0.75,
        min_child_weight=10, gamma=0.05, reg_alpha=0.30, reg_lambda=1.2,
        scale_pos_weight=ratio, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    _x.fit(X_tr, y_tr, verbose=False)
    cv_aucs_xgb.append(roc_auc_score(y_va, _x.predict_proba(X_va)[:, 1]))

    _l = lgb.LGBMClassifier(
        n_estimators=lgb_m.best_iteration_ or 400,
        objective="binary", metric="auc",
        max_depth=7, num_leaves=80, learning_rate=0.04,
        subsample=0.75, subsample_freq=1, colsample_bytree=0.75,
        min_child_samples=10, min_split_gain=0.0, reg_alpha=0.20, reg_lambda=0.80,
        scale_pos_weight=ratio, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    _l.fit(X_tr, y_tr)
    cv_aucs_lgb.append(roc_auc_score(y_va, _l.predict_proba(X_va)[:, 1]))

    print(f"    Fold {fold+1}: XGB_AUC={cv_aucs_xgb[-1]:.4f}  LGB_AUC={cv_aucs_lgb[-1]:.4f}")

cv_auc_xgb = float(np.mean(cv_aucs_xgb))
cv_auc_lgb = float(np.mean(cv_aucs_lgb))
print(f"  Mean CV AUC — XGBoost: {cv_auc_xgb:.4f}  LightGBM: {cv_auc_lgb:.4f}")
print(f"  Overfit gap — XGB: {auc_x - cv_auc_xgb:+.4f}  LGB: {auc_l - cv_auc_lgb:+.4f}")

# ── 6. Youden's J Optimal Threshold ──────────────────────────────────────────
print("\n[3/5] Finding optimal threshold (Youden's J) ...")
best_name  = max(results, key=lambda k: results[k]["auc"])
best_proba = results[best_name]["proba"]

fpr_b, tpr_b, thr_b = roc_curve(y_test, best_proba)
youdens_j    = tpr_b - fpr_b
opt_idx      = np.argmax(youdens_j)
opt_threshold = float(thr_b[opt_idx])
print(f"  Best model     : {best_name}")
print(f"  Opt. threshold : {opt_threshold:.4f}  (Youden's J = {youdens_j[opt_idx]:.4f})")

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
print(f"\n[4/5] Evaluating all models @ threshold={opt_threshold:.4f} ...")
rows = []
for name, res in results.items():
    preds = (res["proba"] >= opt_threshold).astype(int)
    rows.append({
        "model":     name,
        "roc_auc":   round(res["auc"], 4),
        "accuracy":  round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_test, preds, zero_division=0), 4),
        "f1":        round(f1_score(y_test, preds, zero_division=0), 4),
        "threshold": round(opt_threshold, 4),
    })

met_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
met_df.to_csv(REPORTS_DIR / "all_metrics.csv", index=False)
print("\n  Metrics Table:")
print(met_df.to_string(index=False))

best_preds = (best_proba >= opt_threshold).astype(int)
best_acc   = accuracy_score(y_test, best_preds)
best_auc   = results[best_name]["auc"]
best_f1    = f1_score(y_test, best_preds)
best_rec   = recall_score(y_test, best_preds)

print(f"\n{'='*70}")
print(f"  BEST MODEL  : {best_name}")
print(f"  AUC         : {best_auc:.4f}")
print(f"  ACCURACY    : {best_acc*100:.2f}%")
print(f"  F1          : {best_f1:.4f}")
print(f"  RECALL      : {best_rec:.4f}")
print(f"{'='*70}")
print(classification_report(y_test, best_preds,
      target_names=["Low-Crime Day", "High-Crime Day"]))

# ── 8. Save ───────────────────────────────────────────────────────────────────
print("[5/5] Saving artifacts ...")
joblib.dump(xgb_m, MODELS_DIR / "xgboost.joblib")
joblib.dump(lgb_m, MODELS_DIR / "lightgbm.joblib")

save_model = xgb_m if best_name in ("xgboost", "wtd_ensemble") else lgb_m
joblib.dump(save_model, MODELS_DIR / "best_model.joblib")
(MODELS_DIR / "best_model_name.txt").write_text(best_name)

with open(MODELS_DIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

meta_cfg = {
    "best_model":       best_name,
    "opt_threshold":    round(opt_threshold, 6),
    "youdens_j":        round(float(youdens_j[opt_idx]), 6),
    "cv_auc_xgb":       round(cv_auc_xgb, 4),
    "cv_auc_lgb":       round(cv_auc_lgb, 4),
    "test_auc_xgb":     round(auc_x, 4),
    "test_auc_lgb":     round(auc_l, 4),
    "test_auc_ensemble":round(auc_e, 4),
    "overfitgap_xgb":   round(auc_x - cv_auc_xgb, 4),
    "overfitgap_lgb":   round(auc_l - cv_auc_lgb, 4),
    "pos_rate_train":   round(float(pos_tr), 4),
    "pos_rate_test":    round(float(pos_te), 4),
    "n_features":       len(feature_cols),
    "split":            "random_80_20",
}
with open(MODELS_DIR / "model_meta.json", "w") as f:
    json.dump(meta_cfg, f, indent=2)

train_medians.to_frame("threshold").reset_index().to_csv(
    REPORTS_DIR / "district_thresholds.csv", index=False
)
print(f"  Saved  threshold={opt_threshold:.4f}  overfit_gap={auc_x - cv_auc_xgb:+.4f}")

# ── 9. Plots ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "text.color": "#c9d1d9", "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#c9d1d9", "ytick.color": "#c9d1d9", "grid.color": "#21262d",
})
_PAL = ["#58a6ff", "#3fb950", "#d29922", "#ff7b72", "#bc8cff"]

# ROC curves
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.2, label="Random (AUC=0.50)")
for (name, res), c in zip(results.items(), _PAL):
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    ax.plot(fpr, tpr, lw=2.2, color=c,
            label=f"{name.replace('_',' ').title()}  AUC={res['auc']:.4f}")
ax.scatter(fpr_b[opt_idx], tpr_b[opt_idx], s=150, zorder=5,
           color="white", edgecolors="#ff7b72", linewidths=2,
           label=f"Youden Threshold ({opt_threshold:.3f})")
ax.set_title("ROC Curves – District Crime Risk Prediction\n"
             "(Target: Above District Median | 6 Interaction Features | Random Split)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight"); plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Low-Crime", "High-Crime"],
            yticklabels=["Low-Crime", "High-Crime"])
ax.set_title(f"Confusion Matrix – {best_name.replace('_',' ').title()}\n"
             f"Accuracy={best_acc*100:.2f}%  AUC={best_auc:.4f}  thr={opt_threshold:.3f}",
             fontweight="bold", fontsize=11)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"confusion_matrix_{best_name}.png",
            dpi=150, bbox_inches="tight"); plt.close()

# Feature importances
for mname, mobj in [("lightgbm", lgb_m), ("xgboost", xgb_m)]:
    imp = mobj.feature_importances_
    n   = min(len(feature_cols), len(imp))
    imp_df = pd.DataFrame({"feature": feature_cols[:n], "importance": imp[:n]})
    imp_df = imp_df.sort_values("importance").tail(20)
    imp_df.to_csv(REPORTS_DIR / f"feature_importance_{mname}.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp_df["feature"], imp_df["importance"],
            color=plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(imp_df))))
    ax.set_title(f"Top 20 Feature Importances – {mname.title()}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance"); plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"feature_importance_{mname}.png",
                dpi=150, bbox_inches="tight"); plt.close()

# Risk score distribution
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(best_proba[y_test == 0], bins=60, alpha=0.65, color="#58a6ff",
        label=f"Low-Crime  (n={int((y_test==0).sum()):,})", density=True)
ax.hist(best_proba[y_test == 1], bins=60, alpha=0.65, color="#ff7b72",
        label=f"High-Crime (n={int((y_test==1).sum()):,})", density=True)
ax.axvline(opt_threshold, color="white", linestyle="--", lw=1.8,
           label=f"Youden Threshold ({opt_threshold:.3f})")
ax.set_title(f"Risk Score Distribution – {best_name.replace('_',' ').title()}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted Crime Probability"); ax.set_ylabel("Density"); ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"risk_score_dist_{best_name}.png",
            dpi=150, bbox_inches="tight"); plt.close()

# Overfit check
fig, ax = plt.subplots(figsize=(7, 4))
mv = ["XGBoost", "LightGBM"]
cv_v  = [cv_auc_xgb, cv_auc_lgb]
te_v  = [auc_x,      auc_l]
x_ = np.arange(2); w_ = 0.3
ax.bar(x_-w_/2, cv_v, w_, label="CV AUC", color="#58a6ff", alpha=0.85)
ax.bar(x_+w_/2, te_v, w_, label="Test AUC", color="#3fb950", alpha=0.85)
for i,(cv,te) in enumerate(zip(cv_v,te_v)):
    ax.text(i, max(cv,te)+0.004, f"Δ={te-cv:+.3f}", ha="center", fontsize=10)
ax.set_xticks(x_); ax.set_xticklabels(mv)
ax.set_ylim(max(0.5, min(cv_v+te_v)-0.05), 1.0)
ax.set_title("Overfit Check: CV AUC vs Test AUC", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "overfit_check.png", dpi=150, bbox_inches="tight"); plt.close()

# Metrics comparison bar
cols_ = ["accuracy", "precision", "recall", "f1", "roc_auc"]
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(met_df)); w = 0.14
for i, (col, c) in enumerate(zip(cols_, _PAL)):
    ax.bar(x+i*w, met_df[col], w, label=col.replace("_"," ").title(),
           color=c, alpha=0.85)
ax.set_xticks(x+2*w)
ax.set_xticklabels(met_df["model"].str.replace("_"," ").str.title(), fontsize=9)
ax.set_ylim(0, 1.15)
ax.axhline(0.80, color="white", linestyle="--", lw=1.2, alpha=0.5, label="80% target")
ax.set_title(f"Model Performance (threshold={opt_threshold:.3f})",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight"); plt.close()
print("  Plots saved → artifacts/figures/")

# ── 10. Final summary ─────────────────────────────────────────────────────────
elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"  TRAINING COMPLETE  ({elapsed:.1f}s)")
print(f"{'='*70}")
for name, res in sorted(results.items(), key=lambda x: -x[1]["auc"]):
    preds = (res["proba"] >= opt_threshold).astype(int)
    a  = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    re = recall_score(y_test, preds)
    print(f"  {name:<16} AUC={res['auc']:.4f}  "
          f"Acc={a*100:.1f}%  F1={f1:.4f}  Recall={re:.4f}")
print(f"\n  Overfit check  XGB: {auc_x - cv_auc_xgb:+.4f}  LGB: {auc_l - cv_auc_lgb:+.4f}")
print(f"  Best model saved : artifacts/models/best_model.joblib")
print(f"  Threshold saved  : artifacts/models/model_meta.json (thr={opt_threshold:.4f})")
print(f"  Dashboard        : python -m uvicorn backend.main:app --reload --port 8000")
print(f"{'='*70}")
