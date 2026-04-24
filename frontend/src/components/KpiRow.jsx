/**
 * KpiRow.jsx — PoliceSight
 * Renders the 5 KPI cards: High/Medium/Low risk counts, Avg Risk Score, Model Accuracy.
 */
export default function KpiRow({ predictions, bestModel }) {
  const high   = predictions.filter(p => p.risk_level === "High").length;
  const medium = predictions.filter(p => p.risk_level === "Medium").length;
  const low    = predictions.filter(p => p.risk_level === "Low").length;
  const avg    = predictions.length
    ? (predictions.reduce((s, p) => s + p.risk_score, 0) / predictions.length).toFixed(3)
    : "—";

  const pctFmt = (v) => (v != null ? (v * 100).toFixed(1) + "%" : "—");
  const modelLabel = bestModel
    ? bestModel.model.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())
    : "—";

  return (
    <section className="kpi-row" id="kpi-row">
      {/* High Risk */}
      <div className="kpi-card kpi-card-accent">
        <div className="kpi-icon red">🔴</div>
        <div className="kpi-body">
          <div className="kpi-val" id="kpi-high">{predictions.length ? high : "—"}</div>
          <div className="kpi-label">High-Risk Divisions</div>
        </div>
      </div>

      {/* Medium Risk */}
      <div className="kpi-card kpi-card-accent">
        <div className="kpi-icon amber">🟡</div>
        <div className="kpi-body">
          <div className="kpi-val" id="kpi-medium">{predictions.length ? medium : "—"}</div>
          <div className="kpi-label">Medium-Risk Divisions</div>
        </div>
      </div>

      {/* Low Risk */}
      <div className="kpi-card kpi-card-accent">
        <div className="kpi-icon green">🟢</div>
        <div className="kpi-body">
          <div className="kpi-val" id="kpi-low">{predictions.length ? low : "—"}</div>
          <div className="kpi-label">Low-Risk Divisions</div>
        </div>
      </div>

      {/* Avg Risk Score */}
      <div className="kpi-card kpi-card-accent">
        <div className="kpi-icon blue">📊</div>
        <div className="kpi-body">
          <div className="kpi-val" id="kpi-avg-risk">{avg}</div>
          <div className="kpi-label">Avg Risk Score</div>
        </div>
      </div>

      {/* Model Accuracy wide card */}
      <div className="kpi-card kpi-card-model">
        <div className="kpi-model-header">
          <span className="kpi-model-label">MODEL ACCURACY</span>
          <span className="kpi-model-name" id="kpi-model-name">{modelLabel}</span>
        </div>
        <div className="kpi-model-metrics">
          <div className="kpi-metric-item">
            <span className="kpi-metric-val" id="kpi-m-acc">{pctFmt(bestModel?.accuracy)}</span>
            <span className="kpi-metric-key">Accuracy</span>
          </div>
          <div className="kpi-metric-divider"></div>
          <div className="kpi-metric-item">
            <span className="kpi-metric-val" id="kpi-m-auc">{bestModel?.roc_auc?.toFixed(3) ?? "—"}</span>
            <span className="kpi-metric-key">ROC-AUC</span>
          </div>
          <div className="kpi-metric-divider"></div>
          <div className="kpi-metric-item">
            <span className="kpi-metric-val" id="kpi-m-f1">{pctFmt(bestModel?.f1)}</span>
            <span className="kpi-metric-key">F1 Score</span>
          </div>
          <div className="kpi-metric-divider"></div>
          <div className="kpi-metric-item">
            <span className="kpi-metric-val" id="kpi-m-prec">{pctFmt(bestModel?.precision)}</span>
            <span className="kpi-metric-key">Precision</span>
          </div>
        </div>
      </div>
    </section>
  );
}
