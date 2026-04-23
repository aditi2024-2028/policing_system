"""
fast_train.py
=============
Skips feature engineering (features_v2.csv already built).
Trains only XGBoost + LightGBM - fastest high-accuracy models.
Completes in ~5-15 minutes on 700k+ rows.
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
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve, f1_score
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    FEATURE_COLS, MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
    TARGET_COL, TRAIN_CUTOFF_DATE, RANDOM_STATE, PROCESSED_DIR
)

t0 = time.time()
print("=" * 65)
print("  Fast Train v2  |  XGBoost + LightGBM on District Panel")
print("=" * 65)

# ── Load pre-built feature table ─────────────────────────────────────────────
feat_path = PROCESSED_DIR / "features_v2.csv"
print(f"\n[1/4] Loading {feat_path.name}  ({feat_path.stat().st_size / 1e6:.0f} MB)...")
df = pd.read_csv(feat_path, parse_dates=["date"])
print(f"      {len(df):,} rows  x  {len(df.columns)} cols")

# ── Feature / Target split ────────────────────────────────────────────────────
feature_cols = [c for c in FEATURE_COLS if c in df.columns]
print(f"      Using {len(feature_cols)} features: {feature_cols[:6]}...")

cutoff   = pd.Timestamp(TRAIN_CUTOFF_DATE)
train_df = df[df["date"] <  cutoff]
test_df  = df[df["date"] >= cutoff]

X_train = train_df[feature_cols].fillna(0).values
y_train = train_df[TARGET_COL].values
X_test  = test_df[feature_cols].fillna(0).values
y_test  = test_df[TARGET_COL]

neg    = int((y_train == 0).sum())
pos    = int((y_train == 1).sum())
ratio  = neg / max(pos, 1)

print(f"\n[2/4] Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
print(f"      Class balance: 0={neg:,}  1={pos:,}  ratio={ratio:.2f}")

results = {}

# ── XGBoost ───────────────────────────────────────────────────────────────────
print("\n[3/4] Training XGBoost (200 trees, GPU-safe CPU mode)...")
n_val = int(0.10 * len(X_train))
xgb_m = xgb.XGBClassifier(
    n_estimators          = 200,
    max_depth             = 6,
    learning_rate         = 0.1,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 10,
    gamma                 = 0.1,
    reg_alpha             = 0.5,
    reg_lambda            = 2.0,
    scale_pos_weight      = ratio,
    eval_metric           = "logloss",
    early_stopping_rounds = 20,
    random_state          = RANDOM_STATE,
    n_jobs                = -1,
    verbosity             = 0,
)
xgb_m.fit(
    X_train[:-n_val], y_train[:-n_val],
    eval_set=[(X_train[-n_val:], y_train[-n_val:])],
    verbose=False,
)
p_xgb = xgb_m.predict_proba(X_test)[:, 1]
auc_x  = roc_auc_score(y_test, p_xgb)
acc_x  = accuracy_score(y_test, (p_xgb >= 0.5))
results["xgboost"] = {"model": xgb_m, "proba": p_xgb, "auc": auc_x}
print(f"  XGBoost  ->  AUC={auc_x:.4f}   Accuracy={acc_x*100:.1f}%")

# ── LightGBM ──────────────────────────────────────────────────────────────────
print("\n[3b/4] Training LightGBM (200 trees)...")
lgb_m = lgb.LGBMClassifier(
    n_estimators      = 200,
    max_depth         = 7,
    num_leaves        = 63,
    learning_rate     = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_samples = 30,
    reg_alpha         = 0.5,
    reg_lambda        = 2.0,
    is_unbalance      = True,
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    verbose           = -1,
)
lgb_m.fit(
    X_train[:-n_val], y_train[:-n_val],
    eval_set=[(X_train[-n_val:], y_train[-n_val:])],
    callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)],
)
p_lgb = lgb_m.predict_proba(X_test)[:, 1]
auc_l  = roc_auc_score(y_test, p_lgb)
acc_l  = accuracy_score(y_test, (p_lgb >= 0.5))
results["lightgbm"] = {"model": lgb_m, "proba": p_lgb, "auc": auc_l}
print(f"  LightGBM ->  AUC={auc_l:.4f}   Accuracy={acc_l*100:.1f}%")

# ── Soft Ensemble (weighted average) ─────────────────────────────────────────
print("\n[3c/4] Computing soft ensemble (XGB + LGB average)...")
p_ens   = 0.5 * p_xgb + 0.5 * p_lgb
auc_e   = roc_auc_score(y_test, p_ens)
acc_e   = accuracy_score(y_test, (p_ens >= 0.5))
results["ensemble_avg"] = {"model": None, "proba": p_ens, "auc": auc_e, "_skip_save": True}
print(f"  Ensemble ->  AUC={auc_e:.4f}   Accuracy={acc_e*100:.1f}%")

# ── Pick best ─────────────────────────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["auc"])
best_proba = results[best_name]["proba"]
best_preds = (best_proba >= 0.5).astype(int)

print("\n" + "=" * 65)
print(f"  BEST MODEL : {best_name}   AUC={results[best_name]['auc']:.4f}")
best_acc = accuracy_score(y_test, best_preds)
print(f"  ACCURACY   : {best_acc*100:.1f}%")
print("=" * 65)
print()
print(classification_report(y_test, best_preds,
      target_names=["No Crime (0)", "Crime (1)"]))

# ── Save models & metadata ────────────────────────────────────────────────────
print("\n[4/4] Saving artifacts...")
joblib.dump(xgb_m, MODELS_DIR / "xgboost.joblib")
joblib.dump(lgb_m, MODELS_DIR / "lightgbm.joblib")

# Best model points to the actual best (not ensemble avg)
save_name  = best_name if best_name != "ensemble_avg" else "lightgbm"
best_model = results[save_name]["model"]
joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
with open(MODELS_DIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f)
(MODELS_DIR / "best_model_name.txt").write_text(save_name)

# Metrics CSV
rows = []
for name, res in results.items():
    acc_  = accuracy_score(y_test, (res["proba"] >= 0.5))
    f1_   = f1_score(y_test, (res["proba"] >= 0.5), zero_division=0)
    rows.append({"model": name, "roc_auc": round(res["auc"], 4),
                 "accuracy": round(acc_, 4), "f1": round(f1_, 4)})
met_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
met_df.to_csv(REPORTS_DIR / "all_metrics.csv", index=False)
print("\nModel Comparison:")
print(met_df.to_string(index=False))

# ── Plots ─────────────────────────────────────────────────────────────────────
_PAL = ["#58a6ff", "#3fb950", "#d29922"]
plt.rcParams.update({"figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
                     "text.color": "#c9d1d9", "axes.labelcolor": "#c9d1d9",
                     "xtick.color": "#c9d1d9", "ytick.color": "#c9d1d9",
                     "grid.color": "#21262d"})

# ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.2, label="Random baseline")
for (name, res), color in zip(results.items(), _PAL):
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    ax.plot(fpr, tpr, lw=2.2, color=color,
            label=f"{name.replace('_',' ').title()}  AUC={res['auc']:.4f}")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves - District-Level Crime Prediction", fontsize=13, fontweight="bold")
ax.legend(loc="lower right"); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Confusion matrix for best model
cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(6, 5))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Crime", "Crime"],
            yticklabels=["No Crime", "Crime"])
ax.set_title(f"Confusion Matrix - {save_name.replace('_',' ').title()}\n"
             f"Accuracy = {best_acc*100:.1f}%", fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"confusion_matrix_{save_name}.png", dpi=150, bbox_inches="tight")
plt.close()

# Feature importance (best tree model)
best_tree = lgb_m if save_name in ("lightgbm", "ensemble_avg") else xgb_m
imp = best_tree.feature_importances_
imp_df = pd.DataFrame({"feature": feature_cols[:len(imp)], "importance": imp})
imp_df = imp_df.sort_values("importance").tail(15)
imp_df.to_csv(REPORTS_DIR / f"feature_importance_{save_name}.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(imp_df)))
ax.barh(imp_df["feature"], imp_df["importance"], color=colors)
ax.set_title(f"Top 15 Feature Importances\n{save_name.replace('_',' ').title()}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"feature_importance_{save_name}.png", dpi=150, bbox_inches="tight")
plt.close()

# Risk score distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(best_proba[y_test == 0], bins=50, alpha=0.65, color="#58a6ff",
        label="No Crime Tomorrow", density=True)
ax.hist(best_proba[y_test == 1], bins=50, alpha=0.65, color="#ff7b72",
        label="Crime Tomorrow", density=True)
ax.axvline(0.5, color="white", linestyle="--", lw=1.5, label="Threshold 0.5")
ax.set_xlabel("Predicted Crime Probability")
ax.set_ylabel("Density")
ax.set_title(f"Risk Score Distribution - {save_name.replace('_',' ').title()}",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"risk_score_dist_{save_name}.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nFigures saved -> artifacts/figures/")

elapsed = time.time() - t0
print(f"\n{'='*65}")
print(f"  DONE in {elapsed:.1f}s")
print(f"  Best Model : {save_name}")
print(f"  ROC-AUC    : {results[best_name]['auc']:.4f}")
print(f"  Accuracy   : {best_acc*100:.2f}%")
print(f"\n  To start backend + dashboard:")
print(f"  python -m uvicorn backend.main:app --reload --port 8000")
print(f"{'='*65}")
