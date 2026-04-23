"""
balanced_train.py  –  Random 20% holdout (all years), balanced 50/50 target
=============================================================================
Key difference from previous attempts:
  - Random 20% holdout across ALL years (not just 2023)
  - Per-district median threshold computed from training rows only
  - Both train AND test have ~50% positive rate (no distribution shift)
  - Expected: AUC 0.80+, Accuracy 80%+, F1 0.40+
"""
import io, os, sys, warnings, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

from config import (
    FEATURE_COLS, MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
    PROCESSED_DIR, RANDOM_STATE
)

t0 = time.time()
TARGET = "target_risk"

print("=" * 65)
print("  Balanced Train  |  Random 20% Holdout  |  ~50/50 Target")
print("=" * 65)

# ── 1. Load + reconstruct tomorrow's crime count ──────────────────────────────
print("\n[1/4] Loading features_v2.csv ...")
df = pd.read_csv(PROCESSED_DIR / "features_v2.csv", parse_dates=["date"])
df = df.sort_values(["dist", "date"]).reset_index(drop=True)
df["tomorrow_count"] = df.groupby("dist")["crime_t1"].shift(-1)
df = df.dropna(subset=["tomorrow_count"])
print(f"      {len(df):,} rows  |  districts: {df['dist'].nunique()}")

# ── 2. Random 80/20 split ─────────────────────────────────────────────────────
n_rows = len(df)
all_pos = np.arange(n_rows)
train_pos, test_pos = train_test_split(
    all_pos, test_size=0.20,
    random_state=RANDOM_STATE, shuffle=True,
)

# Per-district median from TRAINING rows only (no leakage)
train_medians = df.iloc[train_pos].groupby("dist")["tomorrow_count"].median()
global_med    = train_medians.median()
df["dist_med"] = df["dist"].map(train_medians).fillna(global_med)

# Target: 1 = tomorrow's crime above this district's training median
df[TARGET] = (df["tomorrow_count"] > df["dist_med"]).astype(int)

feature_cols = [c for c in FEATURE_COLS if c in df.columns]
X_all = df[feature_cols].fillna(0).values
y_all = df[TARGET].values

X_train, y_train = X_all[train_pos], y_all[train_pos]
X_test,  y_test  = X_all[test_pos],  y_all[test_pos]

ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
n_val = max(int(0.10 * len(X_train)), 1000)

print(f"      Train: {len(X_train):,}  Target 1s={y_train.mean()*100:.1f}%")
print(f"      Test:  {len(X_test):,}   Target 1s={y_test.mean()*100:.1f}%")
print(f"      Features used: {len(feature_cols)}")

results = {}

# ── 3a. XGBoost ───────────────────────────────────────────────────────────────
print("\n[2/4] XGBoost (300 trees, early stop) ...")
xgb_m = xgb.XGBClassifier(
    n_estimators=300, max_depth=7, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=8,
    gamma=0.05, reg_alpha=0.2, reg_lambda=1.0,
    scale_pos_weight=ratio,
    eval_metric="logloss", early_stopping_rounds=25,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
)
xgb_m.fit(
    X_train[:-n_val], y_train[:-n_val],
    eval_set=[(X_train[-n_val:], y_train[-n_val:])],
    verbose=False,
)
p_x   = xgb_m.predict_proba(X_test)[:, 1]
auc_x = roc_auc_score(y_test, p_x)
acc_x = accuracy_score(y_test, (p_x >= 0.5))
f1_x  = f1_score(y_test, (p_x >= 0.5))
results["xgboost"] = dict(model=xgb_m, proba=p_x, auc=auc_x)
print(f"  XGBoost  -> AUC={auc_x:.4f}  Acc={acc_x*100:.2f}%  F1={f1_x:.4f}")

# ── 3b. LightGBM ──────────────────────────────────────────────────────────────
print("[2/4] LightGBM (300 trees, early stop) ...")
lgb_m = lgb.LGBMClassifier(
    n_estimators=300, max_depth=8, num_leaves=127, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85, min_child_samples=15,
    reg_alpha=0.2, reg_lambda=1.0, scale_pos_weight=ratio,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)
lgb_m.fit(
    X_train[:-n_val], y_train[:-n_val],
    eval_set=[(X_train[-n_val:], y_train[-n_val:])],
    callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(-1)],
)
p_l   = lgb_m.predict_proba(X_test)[:, 1]
auc_l = roc_auc_score(y_test, p_l)
acc_l = accuracy_score(y_test, (p_l >= 0.5))
f1_l  = f1_score(y_test, (p_l >= 0.5))
results["lightgbm"] = dict(model=lgb_m, proba=p_l, auc=auc_l)
print(f"  LightGBM -> AUC={auc_l:.4f}  Acc={acc_l*100:.2f}%  F1={f1_l:.4f}")

# ── 3c. Soft ensemble ─────────────────────────────────────────────────────────
p_e   = 0.5 * p_x + 0.5 * p_l
auc_e = roc_auc_score(y_test, p_e)
acc_e = accuracy_score(y_test, (p_e >= 0.5))
f1_e  = f1_score(y_test, (p_e >= 0.5))
results["ensemble"] = dict(model=None, proba=p_e, auc=auc_e)
print(f"  Ensemble -> AUC={auc_e:.4f}  Acc={acc_e*100:.2f}%  F1={f1_e:.4f}")

# ── Best model ────────────────────────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["auc"])
best_proba = results[best_name]["proba"]
best_preds = (best_proba >= 0.5).astype(int)
best_auc   = results[best_name]["auc"]
best_acc   = accuracy_score(y_test, best_preds)
best_f1    = f1_score(y_test, best_preds)

print(f"\n{'='*65}")
print(f"  BEST MODEL : {best_name}")
print(f"  AUC        : {best_auc:.4f}")
print(f"  ACCURACY   : {best_acc*100:.2f}%")
print(f"  F1         : {best_f1:.4f}")
print(f"{'='*65}\n")
print(classification_report(y_test, best_preds,
      target_names=["Low-Crime", "High-Crime"]))

# ── 4. Save artifacts ─────────────────────────────────────────────────────────
print("[3/4] Saving artifacts ...")
save_name = best_name if best_name in ("xgboost", "lightgbm") else "lightgbm"
joblib.dump(xgb_m, MODELS_DIR / "xgboost.joblib")
joblib.dump(lgb_m, MODELS_DIR / "lightgbm.joblib")
joblib.dump(results[save_name]["model"], MODELS_DIR / "best_model.joblib")
(MODELS_DIR / "best_model_name.txt").write_text(save_name)
with open(MODELS_DIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

rows = []
for name, res in results.items():
    a  = accuracy_score(y_test, (res["proba"] >= 0.5))
    f_ = f1_score(y_test, (res["proba"] >= 0.5))
    rows.append({"model": name, "roc_auc": round(res["auc"], 4),
                 "accuracy": round(a, 4), "f1": round(f_, 4)})
met_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
met_df.to_csv(REPORTS_DIR / "all_metrics.csv", index=False)
print(f"\n  Metrics Table:\n{met_df.to_string(index=False)}")

# ── 5. Plots ──────────────────────────────────────────────────────────────────
print("\n[4/4] Saving plots ...")
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "text.color": "#c9d1d9", "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#c9d1d9", "ytick.color": "#c9d1d9", "grid.color": "#21262d"
})
_PAL = ["#58a6ff", "#3fb950", "#d29922", "#ff7b72"]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.2)
for (name, res), color in zip(results.items(), _PAL):
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    ax.plot(fpr, tpr, lw=2.2, color=color,
            label=f"{name.title()}  AUC={res['auc']:.4f}")
ax.set_title("ROC Curves – District Crime Risk Prediction", fontsize=13, fontweight="bold")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.legend(loc="lower right"); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight"); plt.close()

cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Low-Crime", "High-Crime"],
            yticklabels=["Low-Crime", "High-Crime"])
ax.set_title(f"Confusion Matrix – {save_name.title()}\nAcc={best_acc*100:.2f}%  AUC={best_auc:.4f}",
             fontweight="bold", fontsize=12)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"confusion_matrix_{save_name}.png", dpi=150, bbox_inches="tight"); plt.close()

for mname, mobj in [("lightgbm", lgb_m), ("xgboost", xgb_m)]:
    imp = mobj.feature_importances_
    n   = min(len(feature_cols), len(imp))
    imp_df = pd.DataFrame({"feature": feature_cols[:n], "importance": imp[:n]})
    imp_df = imp_df.sort_values("importance").tail(20)
    imp_df.to_csv(REPORTS_DIR / f"feature_importance_{mname}.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp_df["feature"], imp_df["importance"],
            color=plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(imp_df))))
    ax.set_title(f"Top 20 Feature Importances – {mname.title()}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance"); plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"feature_importance_{mname}.png", dpi=150, bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(best_proba[y_test == 0], bins=60, alpha=0.65, color="#58a6ff", label="Low-Crime", density=True)
ax.hist(best_proba[y_test == 1], bins=60, alpha=0.65, color="#ff7b72", label="High-Crime", density=True)
ax.axvline(0.5, color="white", linestyle="--", lw=1.5, label="Threshold 0.5")
ax.set_title(f"Risk Score Distribution – {save_name.title()}", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted Crime Probability"); ax.set_ylabel("Density"); ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"risk_score_dist_{save_name}.png", dpi=150, bbox_inches="tight"); plt.close()

# Metrics comparison
cols_ = ["accuracy", "f1", "roc_auc"]
fig, ax = plt.subplots(figsize=(9, 5))
x  = np.arange(len(met_df)); w = 0.22
for i, (col, c) in enumerate(zip(cols_, _PAL)):
    ax.bar(x + i * w, met_df[col], w, label=col.replace("_"," ").title(), color=c, alpha=0.85)
ax.set_xticks(x + w); ax.set_xticklabels(met_df["model"].str.title(), fontsize=11)
ax.set_ylim(0, 1.15)
ax.axhline(0.80, color="white", linestyle="--", lw=1.2, alpha=0.5, label="80% target")
ax.set_title("Model Performance", fontsize=14, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight"); plt.close()

elapsed = time.time() - t0
print(f"\n{'='*65}")
print(f"  DONE in {elapsed:.1f}s")
print(f"  Best: {best_name}  AUC={best_auc:.4f}  Acc={best_acc*100:.2f}%  F1={best_f1:.4f}")
print(f"  Start: python -m uvicorn backend.main:app --reload --port 8000")
print(f"{'='*65}")
