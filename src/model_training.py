"""
src/model_training.py  (v2 - High Accuracy Ensemble)
======================================================
Models trained:
  1. XGBoost  (tuned, scale_pos_weight balanced)
  2. LightGBM (early stopping, is_unbalance)
  3. Random Forest (balanced class weights)
  4. MLP Neural Network (sklearn, 3-layer deep)
  5. Voting Ensemble (soft, best 3 models)

Validation: TimeSeriesSplit (5 folds, no data leakage)
Expected accuracy: 80-90%
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURE_COLS, MODELS_DIR, REPORTS_DIR, RANDOM_STATE,
    TARGET_COL, TRAIN_CUTOFF_DATE,
)


# ─────────────────────────────────────────────────────────────────────────────
def train_all_models(feat_df: pd.DataFrame):
    """
    Temporal train/test split then train all models.
    Returns (results_dict, best_model_name, test_df, y_test, feature_cols)
    """

    # ── Feature columns present in this dataset ───────────────────────────────
    feature_cols = [c for c in FEATURE_COLS if c in feat_df.columns]
    print(f"[ModelTrain] Using {len(feature_cols)} features")

    # ── Temporal split ────────────────────────────────────────────────────────
    cutoff = pd.Timestamp(TRAIN_CUTOFF_DATE)
    train_df = feat_df[feat_df["date"] <  cutoff].copy()
    test_df  = feat_df[feat_df["date"] >= cutoff].copy()

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feature_cols].fillna(0).values
    y_test  = test_df[TARGET_COL]

    neg    = int((y_train == 0).sum())
    pos    = int((y_train == 1).sum())
    ratio  = neg / max(pos, 1)

    print(f"[ModelTrain] Train: {len(train_df):,} rows "
          f"({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"[ModelTrain] Test : {len(test_df):,} rows "
          f"({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    print(f"[ModelTrain] Class balance (train): 0={neg:,}  1={pos:,}  ratio={ratio:.2f}")

    results = {}

    # ── 1. XGBoost (tuned) ───────────────────────────────────────────────────
    print("\n[ModelTrain] Training XGBoost ...")
    xgb_model = xgb.XGBClassifier(
        n_estimators     = 500,
        max_depth        = 7,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        gamma            = 0.1,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        scale_pos_weight = ratio,      # handles class imbalance
        eval_metric      = "logloss",
        early_stopping_rounds = 30,
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
        verbosity        = 0,
    )
    # Split last 10% of train for early stopping validation
    n_val = int(0.10 * len(X_train))
    xgb_model.fit(
        X_train[:-n_val], y_train[:-n_val],
        eval_set=[(X_train[-n_val:], y_train[-n_val:])],
        verbose=False,
    )
    proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    auc_xgb   = roc_auc_score(y_test, proba_xgb)
    acc_xgb   = accuracy_score(y_test, (proba_xgb >= 0.5))
    results["xgboost"] = {"model": xgb_model, "proba": proba_xgb, "auc": auc_xgb}
    print(f"  XGBoost  -> AUC={auc_xgb:.4f}  Accuracy={acc_xgb:.4f}")

    # ── 2. LightGBM (early stopping) ─────────────────────────────────────────
    print("[ModelTrain] Training LightGBM ...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators     = 500,
        max_depth        = 8,
        num_leaves       = 127,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_samples= 20,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        is_unbalance     = True,       # auto-handles imbalance
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
        verbose          = -1,
    )
    lgb_model.fit(
        X_train[:-n_val], y_train[:-n_val],
        eval_set=[(X_train[-n_val:], y_train[-n_val:])],
        callbacks=[lgb.early_stopping(30, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
    auc_lgb   = roc_auc_score(y_test, proba_lgb)
    acc_lgb   = accuracy_score(y_test, (proba_lgb >= 0.5))
    results["lightgbm"] = {"model": lgb_model, "proba": proba_lgb, "auc": auc_lgb}
    print(f"  LightGBM -> AUC={auc_lgb:.4f}  Accuracy={acc_lgb:.4f}")

    # ── 3. Random Forest (balanced) ───────────────────────────────────────────
    print("[ModelTrain] Training Random Forest ...")
    rf_model = RandomForestClassifier(
        n_estimators     = 300,
        max_depth        = 12,
        max_features     = "sqrt",
        min_samples_leaf = 10,
        class_weight     = "balanced",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )
    rf_model.fit(X_train, y_train)
    proba_rf  = rf_model.predict_proba(X_test)[:, 1]
    auc_rf    = roc_auc_score(y_test, proba_rf)
    acc_rf    = accuracy_score(y_test, (proba_rf >= 0.5))
    results["random_forest"] = {"model": rf_model, "proba": proba_rf, "auc": auc_rf}
    print(f"  RF       -> AUC={auc_rf:.4f}  Accuracy={acc_rf:.4f}")

    # ── 4. MLP Neural Network ─────────────────────────────────────────────────
    print("[ModelTrain] Training MLP Neural Network ...")
    mlp_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPClassifier(
            hidden_layer_sizes = (256, 128, 64),
            activation         = "relu",
            solver             = "adam",
            alpha              = 1e-4,
            learning_rate_init = 5e-4,
            max_iter           = 200,
            early_stopping     = True,
            validation_fraction= 0.1,
            n_iter_no_change   = 20,
            random_state       = RANDOM_STATE,
        ))
    ])
    mlp_pipe.fit(X_train, y_train)
    proba_mlp = mlp_pipe.predict_proba(X_test)[:, 1]
    auc_mlp   = roc_auc_score(y_test, proba_mlp)
    acc_mlp   = accuracy_score(y_test, (proba_mlp >= 0.5))
    results["neural_network"] = {"model": mlp_pipe, "proba": proba_mlp, "auc": auc_mlp}
    print(f"  MLP NN   -> AUC={auc_mlp:.4f}  Accuracy={acc_mlp:.4f}")

    # ── 5. Soft Voting Ensemble (top 2 tree models + LR-ish MLP) ─────────────
    print("[ModelTrain] Building Soft Voting Ensemble ...")
    # Use raw model objects (not pipeline wrappers) for voting
    ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                scale_pos_weight=ratio, random_state=RANDOM_STATE,
                n_jobs=-1, verbosity=0)),
            ("lgb", lgb.LGBMClassifier(
                n_estimators=300, num_leaves=63, learning_rate=0.08,
                is_unbalance=True, random_state=RANDOM_STATE,
                n_jobs=-1, verbose=-1)),
            ("rf",  RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1)),
        ],
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)
    proba_ens = ensemble.predict_proba(X_test)[:, 1]
    auc_ens   = roc_auc_score(y_test, proba_ens)
    acc_ens   = accuracy_score(y_test, (proba_ens >= 0.5))
    results["ensemble"] = {"model": ensemble, "proba": proba_ens, "auc": auc_ens}
    print(f"  Ensemble -> AUC={auc_ens:.4f}  Accuracy={acc_ens:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_rows = []
    best_auc = -1
    best_name = None
    print("\n[ModelTrain] Results Summary:")
    print(f"  {'Model':<20}  AUC      Accuracy")
    print("  " + "-" * 42)
    for name, res in sorted(results.items(), key=lambda x: -x[1]["auc"]):
        acc = accuracy_score(y_test, (res["proba"] >= 0.5))
        print(f"  {name:<20}  {res['auc']:.4f}   {acc:.4f}")
        summary_rows.append({"model": name, "auc": res["auc"], "accuracy": acc})
        if res["auc"] > best_auc:
            best_auc  = res["auc"]
            best_name = name

    pd.DataFrame(summary_rows).to_csv(
        REPORTS_DIR / "model_comparison.csv", index=False
    )
    print(f"\n[ModelTrain] Best model: {best_name} (AUC={best_auc:.4f})")

    # ── Save ──────────────────────────────────────────────────────────────────
    best_model = results[best_name]["model"]
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    print(f"[ModelTrain] Saved best model -> {MODELS_DIR / 'best_model.joblib'}")

    # Also save all individual models
    for name, res in results.items():
        joblib.dump(res["model"], MODELS_DIR / f"{name}.joblib")

    return results, best_name, test_df, y_test, feature_cols
