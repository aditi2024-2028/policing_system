"""
src/interpretability.py
SHAP explanations (global + local), LSTM attention heatmaps,
gradient×input saliency, and spatial fairness analysis.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

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
})


def run_shap(model, X_test: pd.DataFrame, model_name: str,
             n_sample: int = 500) -> pd.DataFrame | None:
    """
    Compute SHAP values and save global importance + beeswarm plots.
    Returns a DataFrame of mean |SHAP| per feature, or None if shap unavailable.
    """
    try:
        import shap
    except ImportError:
        print("[SHAP] 'shap' not installed - skipping interpretability.")
        return None

    print(f"\n[SHAP] Computing SHAP values for {model_name} …")

    # Sample for speed
    X_sample = X_test.sample(min(n_sample, len(X_test)), random_state=42)

    # ── Choose explainer based on model type ──────────────────────────────────
    sv = None
    base_val = None
    try:
        explainer = shap.TreeExplainer(model)
        sv_raw = explainer.shap_values(X_sample)
        # Binary classifiers return a list [class0, class1]
        sv = sv_raw[1] if isinstance(sv_raw, list) else sv_raw
        ev = explainer.expected_value
        base_val = ev[1] if isinstance(ev, (list, np.ndarray)) else float(ev)
    except Exception as e:
        print(f"  TreeExplainer failed ({e}), trying KernelExplainer …")
        try:
            bg = shap.sample(X_sample, 50)
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            sv_raw = explainer.shap_values(X_sample, nsamples=100)
            sv = sv_raw[1] if isinstance(sv_raw, list) else sv_raw
            ev = explainer.expected_value
            base_val = ev[1] if isinstance(ev, (list, np.ndarray)) else float(ev)
        except Exception as e2:
            print(f"  KernelExplainer also failed ({e2}) - skipping SHAP.")
            return None

    # ── Global bar chart ──────────────────────────────────────────────────────
    mean_abs = np.abs(sv).mean(axis=0)
    feat_imp = (
        pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap")
        .tail(15)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(feat_imp)))
    ax.barh(feat_imp["feature"], feat_imp["mean_abs_shap"], color=colors)
    ax.set_title(
        f"SHAP Feature Importance\n{model_name.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    p = FIGURES_DIR / f"shap_importance_{model_name}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved importance bar chart -> {p.name}")

    # ── Beeswarm plot ─────────────────────────────────────────────────────────
    try:
        shap_expl = shap.Explanation(
            values=sv,
            base_values=np.full(len(X_sample), base_val),
            data=X_sample.values,
            feature_names=list(X_sample.columns),
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.beeswarm(shap_expl, max_display=15, show=False)
        plt.tight_layout()
        bp = FIGURES_DIR / f"shap_beeswarm_{model_name}.png"
        plt.savefig(bp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Saved beeswarm plot -> {bp.name}")
    except Exception as e:
        print(f"[SHAP] Beeswarm skipped: {e}")

    feat_imp.to_csv(REPORTS_DIR / f"shap_importance_{model_name}.csv", index=False)
    return feat_imp


def fairness_check(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    group_col: str = "zone",
) -> pd.DataFrame:
    """
    Compute False Positive Rate (FPR) and False Negative Rate (FNR)
    per LAPD division to surface spatial / proxy-demographic bias.
    """
    df = test_df[[group_col]].copy().reset_index(drop=True)
    df["y_true"] = y_true.values
    df["y_pred"] = y_pred.values

    rows = []
    for zone, grp in df.groupby(group_col):
        if len(grp) < 5:
            continue
        tn, fp, fn, tp = confusion_matrix(
            grp["y_true"], grp["y_pred"], labels=[0, 1]
        ).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        rows.append({
            "zone": zone, "n": len(grp),
            "fpr": round(fpr, 4), "fnr": round(fnr, 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })

    fair_df = pd.DataFrame(rows).sort_values("fpr", ascending=False)
    fair_df.to_csv(REPORTS_DIR / "fairness_by_zone.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors_fpr = plt.cm.Reds(np.linspace(0.35, 0.9, len(fair_df)))
    colors_fnr = plt.cm.Blues(np.linspace(0.35, 0.9, len(fair_df)))

    axes[0].barh(fair_df["zone"], fair_df["fpr"],
                 color=colors_fpr[::-1], alpha=0.9)
    axes[0].set_title("False Positive Rate by Division\n(False Alarm Rate)",
                      fontsize=12, fontweight="bold")
    axes[0].set_xlabel("FPR")
    axes[0].axvline(fair_df["fpr"].mean(), color="white",
                    linestyle="--", lw=1.2, label=f"Mean={fair_df['fpr'].mean():.3f}")
    axes[0].legend(fontsize=9)

    axes[1].barh(fair_df["zone"], fair_df["fnr"],
                 color=colors_fnr[::-1], alpha=0.9)
    axes[1].set_title("False Negative Rate by Division\n(Missed Crime Rate)",
                      fontsize=12, fontweight="bold")
    axes[1].set_xlabel("FNR")
    axes[1].axvline(fair_df["fnr"].mean(), color="white",
                    linestyle="--", lw=1.2, label=f"Mean={fair_df['fnr'].mean():.3f}")
    axes[1].legend(fontsize=9)

    fig.suptitle("Spatial Fairness Analysis - LAPD Divisions",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fairness_by_zone.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Fairness] Saved fairness analysis plot")
    print(fair_df.to_string(index=False))

    return fair_df


# =============================================================================
# LSTM EXPLANATIONS
# =============================================================================

def explain_lstm_attention(
    model,
    X_raw: np.ndarray,
    scaler,
    feature_cols: list,
    device,
    sample_idx: int = None,
    n_global: int = 500,
    district_id: int = None,
) -> dict:
    """
    Generate attention-weight explanations for the CrimeLSTM.

    Produces two levels of explanation:

    **Global** (mean across ``n_global`` samples)
        - Bar chart: average attention weight per timestep (day t-30 → t-1)
        - Shows which lag days the model systematically attends to most

    **Local** (single prediction at ``sample_idx``)
        - Heatmap: attention weight × feature value for each (timestep, feature)
        - Forces the analyst to see which specific feature on which day drove
          the current prediction

    Parameters
    ----------
    model        : CrimeLSTM (loaded, eval mode)
    X_raw        : np.ndarray  [N, lookback, n_features]  (UN-scaled)
    scaler       : StandardScaler fitted on training data
    feature_cols : list[str]  feature names in order
    device       : torch.device
    sample_idx   : int  index into X_raw for the local explanation
                   (if None, picks the sample with highest predicted risk)
    n_global     : int  number of samples used for the global average
    district_id  : int  optional, used only for plot titles

    Returns
    -------
    dict with keys:
        'global_attn'  : np.ndarray [lookback]  mean attention per day
        'local_attn'   : np.ndarray [lookback]  attention for sample_idx
        'local_saliency': np.ndarray [lookback, n_features]  grad×input
        'count_pred'   : float
        'risk_pred'    : int  (0/1/2)
        'risk_label'   : str  (Low/Medium/High)
    """
    import torch
    import torch.nn.functional as F
    try:
        from src.lstm_training import apply_scaler
    except ImportError:
        # running as __main__
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.lstm_training import apply_scaler

    RISK_NAMES = ["Low", "Medium", "High"]
    model.eval()

    # ── Global attention (mean over n_global samples) ────────────────────────
    rng   = np.random.default_rng(42)
    idx_g = rng.choice(len(X_raw), min(n_global, len(X_raw)), replace=False)
    X_g   = apply_scaler(X_raw[idx_g], scaler)
    with torch.no_grad():
        _, _, attn_g = model(torch.tensor(X_g, dtype=torch.float32).to(device))
    global_attn = attn_g.cpu().numpy().mean(axis=0)   # [lookback]

    # ── Local: pick sample ────────────────────────────────────────────────────
    if sample_idx is None:
        X_all_s = apply_scaler(X_raw, scaler)
        with torch.no_grad():
            _, logits_all, _ = model(
                torch.tensor(X_all_s, dtype=torch.float32).to(device)
            )
        probs_all  = F.softmax(logits_all, dim=-1).cpu().numpy()   # [N,3]
        sample_idx = int(probs_all[:, 2].argmax())   # highest High-risk probability

    X_loc  = apply_scaler(X_raw[[sample_idx]], scaler)   # [1, T, F]
    X_t    = torch.tensor(X_loc, dtype=torch.float32, requires_grad=True).to(device)

    # Forward
    count_pred, risk_logits, local_attn = model(X_t)
    risk_prob   = F.softmax(risk_logits, dim=-1)
    risk_class  = int(risk_prob.argmax(dim=-1).item())
    count_val   = float(count_pred.squeeze().item())

    # ── Gradient × input saliency ─────────────────────────────────────────────
    # Backprop w.r.t. the highest-risk class logit
    model.zero_grad()
    risk_logits[0, risk_class].backward()

    grad = X_t.grad.squeeze(0).detach().cpu().numpy()   # [T, F]
    inp  = X_loc[0]                                       # [T, F]  (scaled)
    saliency = np.abs(grad * inp)                         # [T, F]

    # ── Plots ─────────────────────────────────────────────────────────────────
    dist_tag = f" – District {district_id}" if district_id else ""
    lookback  = global_attn.shape[0]
    lag_labels = [f"t-{lookback - i}" for i in range(lookback)]

    # (a) Global attention bar chart
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = plt.cm.plasma(global_attn / global_attn.max())
    ax.bar(range(lookback), global_attn, color=colors, edgecolor="none", width=0.85)
    ax.set_xticks(range(0, lookback, 5))
    ax.set_xticklabels([lag_labels[i] for i in range(0, lookback, 5)], rotation=45)
    ax.set_title(f"LSTM — Global Attention Weights{dist_tag}\n"
                 f"(mean over {len(idx_g)} predictions)",
                 fontweight="bold")
    ax.set_xlabel("Lag Day"); ax.set_ylabel("Mean Attention Weight")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    p = FIGURES_DIR / "lstm_global_attention.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[LSTM-XAI] Global attention chart → {p.name}")

    # (b) Local prediction saliency heatmap
    top_feats   = min(15, len(feature_cols))
    feat_saliency_mean = saliency.mean(axis=0)            # mean across time  [F]
    top_feat_idx = np.argsort(feat_saliency_mean)[-top_feats:]  # top 15 features
    heat = saliency[:, top_feat_idx].T                     # [top_feats, T]

    fig, ax = plt.subplots(figsize=(14, 6))
    import matplotlib.cm as _cm
    im = ax.imshow(heat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(top_feats))
    ax.set_yticklabels(
        [feature_cols[i] for i in top_feat_idx],
        fontsize=8,
    )
    ax.set_xticks(range(0, lookback, 5))
    ax.set_xticklabels([lag_labels[i] for i in range(0, lookback, 5)], rotation=45)
    ax.set_xlabel("Lag Day (older → more recent →)")
    ax.set_title(
        f"LSTM — Gradient×Input Saliency{dist_tag}\n"
        f"Prediction: {RISK_NAMES[risk_class]} risk  |  count≈{count_val:.2f}",
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="|Gradient × Input|")
    plt.tight_layout()
    p2 = FIGURES_DIR / "lstm_local_saliency.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[LSTM-XAI] Local saliency heatmap → {p2.name}")

    # (c) Local attention bar (single prediction)
    local_attn_np = local_attn.squeeze(0).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    colors2 = plt.cm.plasma(local_attn_np / local_attn_np.max())
    ax.bar(range(lookback), local_attn_np, color=colors2, edgecolor="none")
    ax.set_xticks(range(0, lookback, 5))
    ax.set_xticklabels([lag_labels[i] for i in range(0, lookback, 5)], rotation=45)
    ax.set_title(
        f"LSTM — Local Attention Weights{dist_tag}\n"
        f"Predicted: {RISK_NAMES[risk_class]} (count≈{count_val:.2f})",
        fontweight="bold",
    )
    ax.set_xlabel("Lag Day"); ax.set_ylabel("Attention Weight")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    p3 = FIGURES_DIR / "lstm_local_attention.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[LSTM-XAI] Local attention bar → {p3.name}")

    return {
        "global_attn":    global_attn,
        "local_attn":     local_attn_np,
        "local_saliency": saliency,
        "count_pred":     count_val,
        "risk_pred":      risk_class,
        "risk_label":     RISK_NAMES[risk_class],
    }
