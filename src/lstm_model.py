"""
src/lstm_model.py
=================
Bidirectional LSTM with Additive Self-Attention for crime hotspot prediction.

Architecture
------------
Input  : [batch, seq_len, n_features]
  → FeatureProjection (Linear → LayerNorm → GELU)
  → BiLSTM  ×  num_layers  (+ dropout)
  → Additive Self-Attention  (returns context vector + attention weights)
  → Dropout
  → FC-head 1 : count_reg  → crime count regression  (MSE / MAE)
  → FC-head 2 : risk_cls   → 3-class risk level      (CrossEntropy)

The attention weights (shape [batch, seq_len]) are exposed so the
interpretability module can generate per-prediction heatmaps.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
)


# ─────────────────────────────────────────────────────────────────────────────
class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention over the time axis.

    Given BiLSTM outputs H of shape [batch, T, 2*hidden],
    computes a context vector c = sum_t(alpha_t * h_t) and
    attention weights alpha of shape [batch, T].
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H: torch.Tensor):
        # H : [batch, T, hidden_dim]
        scores = self.v(torch.tanh(self.W(H)))   # [batch, T, 1]
        alpha  = F.softmax(scores, dim=1)         # [batch, T, 1]
        context = (alpha * H).sum(dim=1)           # [batch, hidden_dim]
        return context, alpha.squeeze(-1)          # [batch, hidden_dim], [batch, T]


# ─────────────────────────────────────────────────────────────────────────────
class CrimeLSTM(nn.Module):
    """
    Bidirectional LSTM Crime Forecaster with dual output heads.

    Parameters
    ----------
    n_features   : int   — number of input features per timestep
    hidden_size  : int   — LSTM hidden units per direction
    num_layers   : int   — stacked LSTM layers
    dropout      : float — dropout probability (between layers + after attention)
    n_risk_cls   : int   — number of risk classes (default 3: Low/Med/High)
    """

    def __init__(
        self,
        n_features : int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers : int = LSTM_NUM_LAYERS,
        dropout    : float = LSTM_DROPOUT,
        n_risk_cls : int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.n_risk_cls  = n_risk_cls
        lstm_out_dim = 2 * hidden_size   # BiLSTM doubles the hidden dim

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        # ── Bidirectional LSTM ────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
            bidirectional = True,
        )

        # ── Attention ─────────────────────────────────────────────────────────
        self.attention = AdditiveAttention(lstm_out_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # ── Shared representation block ───────────────────────────────────────
        self.shared_fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # ── Output head 1: Crime count regression ────────────────────────────
        self.count_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # ── Output head 2: Risk-level classification (3 classes) ─────────────
        self.risk_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_risk_cls),
        )

        # Initialise weights
        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Forget-gate bias trick: set forget gate to 1
                if "bias_ih" in name or "bias_hh" in name:
                    n = param.size(0)
                    param.data[n // 4: n // 2].fill_(1.0)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor  shape [batch, seq_len, n_features]

        Returns
        -------
        count_pred   : [batch, 1]          raw crime count (regression)
        risk_logits  : [batch, n_risk_cls] unnormalised class scores
        attn_weights : [batch, seq_len]    attention weights (for XAI)
        """
        # Project input features
        proj = self.input_proj(x)           # [batch, T, hidden_size]

        # BiLSTM
        lstm_out, _ = self.lstm(proj)       # [batch, T, 2*hidden_size]

        # Attention → context vector
        context, attn_weights = self.attention(lstm_out)   # [batch, 2*H], [batch, T]
        context = self.attn_dropout(context)

        # Shared representation
        shared = self.shared_fc(context)    # [batch, 64]

        # Dual output heads
        count_pred  = self.count_head(shared)   # [batch, 1]
        risk_logits = self.risk_head(shared)    # [batch, 3]

        return count_pred, risk_logits, attn_weights

    # ─────────────────────────────────────────────────────────────────────────
    def predict_risk_proba(self, x: torch.Tensor):
        """Return softmax probabilities over risk classes."""
        _, risk_logits, attn = self.forward(x)
        return F.softmax(risk_logits, dim=-1), attn

    # ─────────────────────────────────────────────────────────────────────────
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
class CrimeLSTMLoss(nn.Module):
    """
    Combined multi-task loss for the CrimeLSTM:
        L = alpha * MAE(count) + (1 - alpha) * CrossEntropy(risk)

    Parameters
    ----------
    alpha        : float  weight for the regression loss (0–1)
    class_weights: Tensor optional class weights for imbalanced risk levels
    """

    def __init__(self, alpha: float = 0.35, class_weights: torch.Tensor = None):
        super().__init__()
        self.alpha  = alpha
        self.mae    = nn.L1Loss()
        self.ce     = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        count_pred  : torch.Tensor,   # [batch, 1]
        count_true  : torch.Tensor,   # [batch]
        risk_logits : torch.Tensor,   # [batch, 3]
        risk_true   : torch.Tensor,   # [batch] long
    ):
        reg_loss = self.mae(count_pred.squeeze(-1), count_true.float())
        cls_loss = self.ce(risk_logits, risk_true.long())
        return self.alpha * reg_loss + (1 - self.alpha) * cls_loss, reg_loss, cls_loss


# ─────────────────────────────────────────────────────────────────────────────
def build_model(n_features: int, device: torch.device = None, **kwargs) -> CrimeLSTM:
    """Convenience factory — creates a CrimeLSTM and moves it to device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrimeLSTM(n_features=n_features, **kwargs).to(device)
    print(
        f"[CrimeLSTM] Built model | n_features={n_features} | "
        f"params={model.n_params:,} | device={device}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke-test
    device = torch.device("cpu")
    m = build_model(n_features=44)
    x  = torch.randn(16, 30, 44)   # batch=16, seq=30, feats=44
    cnt, risk, attn = m(x)
    print(f"count shape   : {cnt.shape}")    # [16, 1]
    print(f"risk shape    : {risk.shape}")   # [16, 3]
    print(f"attn shape    : {attn.shape}")   # [16, 30]
    print(f"attn sum ≈ 1  : {attn.sum(dim=1)}")
