"""
patch_target_and_train.py
=========================
Reads the already-built features_v2.csv, replaces the binary
"any crime?" target (unstable, 17% in test) with a BALANCED target:
  "will tomorrow's crime be ABOVE this district's 28-day rolling median?"

This is ALWAYS ~50/50 by construction and much more stable over time.
Then retrains XGBoost + LightGBM and reports final results.
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
    confusion_matrix, f1_score, precision_score, recall_score
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    FEATURE_COLS, MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
    PROCESSED_DIR, RANDOM_STATE, TRAIN_CUTOFF_DATE
)

t0 = time.time()
print("=" * 65)
print("  Patch Target + Retrain  |  District-Level Median Split")
print("=" * 65)

TARGET = "target_risk"

# ── Step 1: Load existing features + rebuild target ───────────────────────────
print(f"\n[1/5] Loading features_v2.csv...")
feat_path = PROCESSED_DIR / "features_v2.csv"
df = pd.read_csv(feat_path, parse_dates=["date"])
print(f"      {len(df):,} rows, {len(df.columns)} cols")

# We need the REAL crime_count for tomorrow to rebuild target.
# Strategy: crime_t1[t+1] = crime_count[t], so shift crime_t1 forward per district
# i.e., "tomorrow crime count" for row t = crime_t1 at row t+1 (same district)
print("[1/5] Rebuilding balanced target (above EXPANDING median - always 50/50)...")

df = df.sort_values(["dist", "date"]).reset_index(drop=True)

# tomorrow_count = next row's crime_t1
df["tomorrow_count"] = df.groupby("dist")["crime_t1"].shift(-1)

# Expanding median: for each row, the median of ALL past crime counts per district.
# This adapts to the actual crime level at any time, so the target stays ~50/50
# in BOTH train AND test periods — no distribution shift.
df["expand_med"] = (
    df.groupby("dist")["crime_t1"]
      .transform(lambda s: s.expanding(min_periods=7).median())
)

# Target: will tomorrow's crime be ABOVE the CURRENT running median?
df[TARGET] = (df["tomorrow_count"] > df["expand_med"]).astype("Int64")
df = df.dropna(subset=[TARGET, "tomorrow_count", "expand_med"])
df[TARGET] = df[TARGET].astype(int)

bal = df[TARGET].value_counts()
print(f"      Target balance: 0={bal.get(0,0):,}  1={bal.get(1,0):,}  "
      f"({bal.get(1,0)/len(df)*100:.1f}% above-median days)")

# ── Step 2: Train/Test split ──────────────────────────────────────────────────
feature_cols = [c for c in FEATURE_COLS if c in df.columns]
cutoff   = pd.Timestamp(TRAIN_CUTOFF_DATE)
train_df = df[df["date"] <  cutoff]
test_df  = df[df["date"] >= cutoff]

X_train = train_df[feature_cols].fillna(0).values
y_train = train_df[TARGET].values
X_test  = test_df[feature_cols].fillna(0).values
y_test  = test_df[TARGET]

neg   = int((y_train == 0).sum())
pos   = int((y_train == 1).sum())
ratio = neg / max(pos, 1)
tneg  = int((y_test  == 0).sum())
tpos  = int((y_test  == 1).sum())

print(f"\n[2/5] Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
print(f"      Train: 0={neg:,}  1={pos:,}  ({pos/len(train_df)*100:.1f}% above-median)")
print(f"      Test:  0={tneg:,}  1={tpos:,}  ({tpos/len(test_df)*100:.1f}% above-median)")
print(f"      Using {len(feature_cols)} features")

results = {}
n_val = max(int(0.10 * len(X_train)), 1000)

# ── Step 3a: XGBoost ─────────────────────────────────────────────────────────
print("\n[3/5] Training XGBoost...")
xgb_m = xgb.XGBClassifier(
    n_estimators          = 300,
    max_depth             = 7,
    learning_rate         = 0.08,
    subsample             = 0.85,
    colsample_bytree      = 0.85,
    min_child_weight      = 8,
    gamma                 = 0.05,
    reg_alpha             = 0.3,
    reg_lambda            = 1.5,
    scale_pos_weight      = ratio,
    eval_metric           = "logloss",
    early_stopping_rounds = 25,
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
print(f"  XGBoost  ->  AUC={auc_x:.4f}   Accuracy={acc_x*100:.2f}%")

# ── Step 3b: LightGBM ────────────────────────────────────────────────────────
print("[3/5] Training LightGBM...")
lgb_m = lgb.LGBMClassifier(
    n_estimators      = 300,
    max_depth         = 8,
    num_leaves        = 127,
    learning_rate     = 0.08,
    subsample         = 0.85,
    colsample_bytree  = 0.85,
    min_child_samples = 20,
    reg_alpha         = 0.3,
    reg_lambda        = 1.5,
    class_weight      = "balanced",
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    verbose           = -1,
)
lgb_m.fit(
    X_train[:-n_val], y_train[:-n_val],
    eval_set=[(X_train[-n_val:], y_train[-n_val:])],
    callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(-1)],
)
p_lgb = lgb_m.predict_proba(X_test)[:, 1]
auc_l  = roc_auc_score(y_test, p_lgb)
acc_l  = accuracy_score(y_test, (p_lgb >= 0.5))
results["lightgbm"] = {"model": lgb_m, "proba": p_lgb, "auc": auc_l}
print(f"  LightGBM ->  AUC={auc_l:.4f}   Accuracy={acc_l*100:.2f}%")

# ── Step 3c: Weighted Soft Ensemble ──────────────────────────────────────────
p_ens = 0.5 * p_xgb + 0.5 * p_lgb
auc_e  = roc_auc_score(y_test, p_ens)
acc_e  = accuracy_score(y_test, (p_ens >= 0.5))
results["ensemble"] = {"model": None, "proba": p_ens, "auc": auc_e}
print(f"  Ensemble ->  AUC={auc_e:.4f}   Accuracy={acc_e*100:.2f}%")

# ── Best model ────────────────────────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["auc"])
best_proba = results[best_name]["proba"]
best_preds = (best_proba >= 0.5).astype(int)
best_auc   = results[best_name]["auc"]
best_acc   = accuracy_score(y_test, best_preds)

print(f"\n{'='*65}")
print(f"  BEST MODEL : {best_name}   AUC={best_auc:.4f}   "
      f"Accuracy={best_acc*100:.2f}%")
print(f"{'='*65}")
print()
print(classification_report(y_test, best_preds,
      target_names=["Low-Crime Day (0)", "High-Crime Day (1)"]))

# ── Step 4: Save artifacts ────────────────────────────────────────────────────
print("[4/5] Saving models & reports...")
save_name = "xgboost" if best_name == "xgboost" else "lightgbm"
best_model_obj = results[save_name]["model"]
joblib.dump(best_model_obj, MODELS_DIR / "best_model.joblib")
joblib.dump(xgb_m, MODELS_DIR / "xgboost.joblib")
joblib.dump(lgb_m, MODELS_DIR / "lightgbm.joblib")
(MODELS_DIR / "best_model_name.txt").write_text(save_name)
with open(MODELS_DIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

# Metrics CSV
rows = []
for name, res in results.items():
    if res["model"] is None: continue
    a = accuracy_score(y_test, (res["proba"] >= 0.5))
    rows.append({
        "model":     name,
        "roc_auc":   round(res["auc"], 4),
        "accuracy":  round(a, 4),
        "precision": round(precision_score(y_test, (res["proba"]>=0.5), zero_division=0), 4),
        "recall":    round(recall_score(y_test, (res["proba"]>=0.5), zero_division=0), 4),
        "f1":        round(f1_score(y_test, (res["proba"]>=0.5), zero_division=0), 4),
    })
met_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
met_df.to_csv(REPORTS_DIR / "all_metrics.csv", index=False)
print("\nMetrics:")
print(met_df.to_string(index=False))

# Also save the patched feature table
patch_path = PROCESSED_DIR / "features_v2.csv"
df[[c for c in df.columns if c != "tomorrow_count"]].to_csv(patch_path, index=False)
print(f"\nPatched features saved -> {patch_path}")

# ── Step 5: Plots ─────────────────────────────────────────────────────────────
print("\n[5/5] Generating plots...")
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "text.color": "#c9d1d9", "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#c9d1d9", "ytick.color": "#c9d1d9", "grid.color": "#21262d"
})
_PAL = ["#58a6ff", "#3fb950", "#d29922", "#ff7b72", "#bc8cff"]

# ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.2)
for (name, res), col in zip(results.items(), _PAL):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    ax.plot(fpr, tpr, lw=2.2, color=col,
            label=f"{name.title()}  AUC={res['auc']:.4f}")
ax.set_title("ROC Curves - District Crime Prediction\n(Target: Above Rolling Median)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.legend(loc="lower right"); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Low-Crime", "High-Crime"],
            yticklabels=["Low-Crime", "High-Crime"])
ax.set_title(f"Confusion Matrix - {save_name.title()}\nAccuracy = {best_acc*100:.2f}%",
             fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"confusion_matrix_{save_name}.png", dpi=150, bbox_inches="tight")
plt.close()

# Feature Importance (LightGBM or XGBoost, top 20)
for mname, mobj in [("lightgbm", lgb_m), ("xgboost", xgb_m)]:
    imp = mobj.feature_importances_
    n = min(len(feature_cols), len(imp))
    imp_df = pd.DataFrame({"feature": feature_cols[:n], "importance": imp[:n]})
    imp_df = imp_df.sort_values("importance").tail(20)
    imp_df.to_csv(REPORTS_DIR / f"feature_importance_{mname}.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(imp_df)))
    ax.barh(imp_df["feature"], imp_df["importance"], color=colors)
    ax.set_title(f"Top 20 Feature Importances - {mname.title()}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"feature_importance_{mname}.png", dpi=150, bbox_inches="tight")
    plt.close()

# Risk Score Distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(best_proba[y_test == 0], bins=60, alpha=0.65, color="#58a6ff",
        label="Low-Crime Day (below median)", density=True)
ax.hist(best_proba[y_test == 1], bins=60, alpha=0.65, color="#ff7b72",
        label="High-Crime Day (above median)", density=True)
ax.axvline(0.5, color="white", linestyle="--", lw=1.5, label="Threshold 0.5")
ax.set_xlabel("Predicted Crime Probability"); ax.set_ylabel("Density")
ax.set_title(f"Risk Score Distribution - {save_name.title()}", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"risk_score_dist_{save_name}.png", dpi=150, bbox_inches="tight")
plt.close()

# Metrics comparison bar chart
cols_ = ["accuracy", "precision", "recall", "f1", "roc_auc"]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(met_df)); w = 0.15
for i, (col, c) in enumerate(zip(cols_, _PAL)):
    ax.bar(x + i*w, met_df[col], w, label=col.replace("_"," ").title(), color=c, alpha=0.85)
ax.set_xticks(x + 2*w)
ax.set_xticklabels(met_df["model"].str.title(), fontsize=11)
ax.set_ylim(0, 1.1)
ax.axhline(0.80, color="white", linestyle="--", lw=1.2, alpha=0.6, label="80% Target")
ax.set_title("Model Performance Comparison v2", fontsize=14, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nAll plots saved -> artifacts/figures/")

elapsed = time.time() - t0
print(f"\n{'='*65}")
print(f"  FINAL RESULTS")
print(f"{'='*65}")
for name, res in sorted(results.items(), key=lambda x: -x[1]["auc"]):
    a = accuracy_score(y_test, (res["proba"] >= 0.5))
    print(f"  {name:<20}  AUC={res['auc']:.4f}   Accuracy={a*100:.2f}%")
print(f"\n  Completed in {elapsed:.1f}s")
print(f"  Best: {save_name} | AUC={best_auc:.4f} | Accuracy={best_acc*100:.2f}%")
print(f"  Start server: python -m uvicorn backend.main:app --reload --port 8000")
print(f"{'='*65}")
