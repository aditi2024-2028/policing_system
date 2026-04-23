"""
src/evaluation.py  (v2 - compatible with district-level panel)
Produces: confusion matrix, ROC curve, PR curve, feature importance,
          metrics comparison, risk score distribution.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, REPORTS_DIR

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
    "legend.edgecolor": "#30363d",
})

_PALETTE = ["#58a6ff", "#ff7b72", "#3fb950", "#d29922", "#bc8cff"]


def evaluate_all(results: dict, y_test: pd.Series, feature_cols: list) -> pd.DataFrame:
    """Evaluate all models and produce all figures/reports."""

    all_metrics = []
    for name, res in results.items():
        proba = res["proba"]
        preds = (proba >= 0.5).astype(int)
        try:
            auc = round(roc_auc_score(y_test, proba), 4)
        except Exception:
            auc = float("nan")
        m = {
            "model":     name,
            "accuracy":  round(accuracy_score(y_test, preds),            4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall":    round(recall_score(y_test, preds,    zero_division=0), 4),
            "f1":        round(f1_score(y_test, preds,         zero_division=0), 4),
            "roc_auc":   auc,
        }
        all_metrics.append(m)
        print("\n" + "-"*55)
        print(f"  Model: {name.upper()}")
        print("-"*55)
        print(classification_report(y_test, preds, target_names=["Low-Risk", "High-Risk"]))

    metrics_df = pd.DataFrame(all_metrics).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(REPORTS_DIR / "all_metrics.csv", index=False)
    print("\n[Eval] Metrics summary:")
    print(metrics_df.to_string(index=False))

    best_name = metrics_df.iloc[0]["model"]
    best_res  = results[best_name]
    best_pred = (best_res["proba"] >= 0.5).astype(int)

    _plot_confusion(y_test, best_pred, best_name)
    _plot_roc(results, y_test)
    _plot_precision_recall(results, y_test)
    _plot_feature_importance(best_res["model"], feature_cols, best_name)
    _plot_metrics_comparison(metrics_df)
    _plot_risk_score_dist(best_res["proba"], y_test, best_name)
    return metrics_df


def _plot_confusion(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Low-Risk", "High-Risk"],
        yticklabels=["Low-Risk", "High-Risk"],
        linewidths=0.5, linecolor="#21262d",
    )
    ax.set_title(
        f"Confusion Matrix\n{model_name.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Predicted"), ax.set_ylabel("Actual")
    plt.tight_layout()
    p = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Saved confusion matrix -> {p.name}")


def _plot_roc(results: dict, y_test: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.2, label="Random baseline")
    for (name, res), color in zip(results.items(), _PALETTE):
        try:
            fpr, tpr, _ = roc_curve(y_test, res["proba"])
            ax.plot(fpr, tpr, lw=2.2, color=color,
                    label=f"{name.replace('_',' ').title()}  AUC={res['auc']:.3f}")
        except Exception:
            pass
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Eval] Saved ROC curves")


def _plot_precision_recall(results: dict, y_test: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, res), color in zip(results.items(), _PALETTE):
        try:
            prec, rec, _ = precision_recall_curve(y_test, res["proba"])
            ax.plot(rec, prec, lw=2.2, color=color,
                    label=name.replace("_", " ").title())
        except Exception:
            pass
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves - All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Eval] Saved PR curves")


def _plot_feature_importance(model, feature_cols: list, model_name: str):
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "named_steps"):
        inner = list(model.named_steps.values())[-1]
        if hasattr(inner, "feature_importances_"):
            imp = inner.feature_importances_
        elif hasattr(inner, "coef_"):
            imp = np.abs(inner.coef_[0])
    # VotingClassifier: average sub-estimator importances
    elif hasattr(model, "estimators_"):
        imps = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                imps.append(est.feature_importances_)
        if imps:
            imp = np.mean(imps, axis=0)

    if imp is None:
        return

    n = min(len(feature_cols), len(imp))
    imp_df = (
        pd.DataFrame({"feature": feature_cols[:n], "importance": imp[:n]})
        .sort_values("importance")
        .tail(15)
    )
    imp_df.to_csv(REPORTS_DIR / f"feature_importance_{model_name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(imp_df["feature"], imp_df["importance"],
                   color=_PALETTE[2], alpha=0.85)
    ax.set_title(
        f"Top 15 Feature Importances\n{model_name.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Importance")
    ax.bar_label(bars, fmt="%.3f", padding=3, color="#c9d1d9", fontsize=8)
    plt.tight_layout()
    p = FIGURES_DIR / f"feature_importance_{model_name}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Saved feature importance -> {p.name}")


def _plot_metrics_comparison(metrics_df: pd.DataFrame):
    cols   = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    models = metrics_df["model"].str.replace("_", " ").str.title()
    x      = np.arange(len(metrics_df))
    w      = 0.14

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (col, color) in enumerate(zip(cols, _PALETTE)):
        ax.bar(x + i * w, metrics_df[col], w,
               label=col.replace("_", " ").title(), color=color, alpha=0.85)

    ax.set_xticks(x + 2 * w)
    ax.set_xticklabels(models, fontsize=10, rotation=10)
    ax.set_ylim(0, 1.18)
    ax.axhline(0.80, color="white", linestyle="--", lw=1.2, alpha=0.5, label="80% target")
    ax.set_title("Model Performance Comparison (v2)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Eval] Saved metrics comparison")


def _plot_risk_score_dist(proba: np.ndarray, y_test: pd.Series, model_name: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(proba[y_test == 0], bins=40, alpha=0.65, color=_PALETTE[0],
            label="Actual Low-Risk (no crime)", density=True)
    ax.hist(proba[y_test == 1], bins=40, alpha=0.65, color=_PALETTE[1],
            label="Actual High-Risk (crime)", density=True)
    ax.axvline(0.5, color="white", linestyle="--", lw=1.5, label="Threshold 0.5")
    ax.set_xlabel("Predicted Crime Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Risk Score Distribution\n{model_name.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"risk_score_dist_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Eval] Saved risk score distribution")
