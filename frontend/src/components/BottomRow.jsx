/**
 * BottomRow.jsx — PoliceSight
 * Contains: Model Performance chart, Feature Importance chart, Division Risk Rankings table.
 */
import { useEffect, useRef } from "react";
import { Chart, registerables } from "chart.js";

Chart.register(...registerables);

function riskColor(level) {
  if (level === "High")   return "#ef4444";
  if (level === "Medium") return "#f59e0b";
  return "#22c55e";
}

export default function BottomRow({
  metrics, featureImportance, predictions, zoneStats, tableDateLabel, onDivisionClick,
}) {
  const metricsChartRef    = useRef(null);
  const importanceChartRef = useRef(null);
  const metricsInstanceRef    = useRef(null);
  const importanceInstanceRef = useRef(null);

  // ── Model metrics chart ────────────────────────────────────────────────────
  useEffect(() => {
    if (!metrics.length || !metricsChartRef.current) return;

    const models  = metrics.map(d => d.model.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()));
    const aucs    = metrics.map(d => d.roc_auc);
    const f1s     = metrics.map(d => d.f1);
    const precs   = metrics.map(d => d.precision);
    const recalls = metrics.map(d => d.recall);
    const accs    = metrics.map(d => d.accuracy);

    if (metricsInstanceRef.current) metricsInstanceRef.current.destroy();

    metricsInstanceRef.current = new Chart(metricsChartRef.current, {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          { label: "Accuracy",  data: accs,    backgroundColor: "#06b6d4cc" },
          { label: "ROC-AUC",   data: aucs,    backgroundColor: "#3b82f6cc" },
          { label: "F1",        data: f1s,     backgroundColor: "#22c55ecc" },
          { label: "Precision", data: precs,   backgroundColor: "#f59e0bcc" },
          { label: "Recall",    data: recalls, backgroundColor: "#ef4444cc" },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#8099c4", font: { size: 10 } } },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(1)}%`,
            },
          },
        },
        scales: {
          x: { ticks: { color: "#4a6080", font: { size: 9 } }, grid: { color: "#1e2d47" } },
          y: {
            min: 0, max: 1,
            ticks: {
              color: "#4a6080", font: { size: 10 },
              callback: v => (v * 100).toFixed(0) + "%",
            },
            grid: { color: "#1e2d47" },
          },
        },
      },
    });
  }, [metrics]);

  // ── Feature importance chart ───────────────────────────────────────────────
  useEffect(() => {
    if (!featureImportance.length || !importanceChartRef.current) return;

    const col  = featureImportance[0]?.mean_abs_shap !== undefined ? "mean_abs_shap" : "importance";
    const top  = featureImportance.slice(0, 10).reverse();
    const labels = top.map(d => d.feature.replace(/_/g, " "));
    const vals   = top.map(d => d[col]);

    if (importanceInstanceRef.current) importanceInstanceRef.current.destroy();

    importanceInstanceRef.current = new Chart(importanceChartRef.current, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          data: vals,
          backgroundColor: vals.map((_, i) => `hsl(${140 + i * 18}, 60%, ${45 + i * 2}%)`),
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#4a6080", font: { size: 9 } }, grid: { color: "#1e2d47" } },
          y: { ticks: { color: "#8099c4", font: { size: 9 } }, grid: { display: false } },
        },
      },
    });
  }, [featureImportance]);

  // cleanup
  useEffect(() => {
    return () => {
      if (metricsInstanceRef.current)    metricsInstanceRef.current.destroy();
      if (importanceInstanceRef.current) importanceInstanceRef.current.destroy();
    };
  }, []);

  // ── Sorted predictions for table ──────────────────────────────────────────
  const sorted = [...predictions].sort((a, b) => b.risk_score - a.risk_score);

  return (
    <section className="bottom-row">
      {/* Model Performance */}
      <div className="card">
        <div className="card-header">
          <h3>Model Performance</h3>
          <span className="card-tag">All Models</span>
        </div>
        <div className="card-body">
          <canvas ref={metricsChartRef} id="metrics-chart" height={180}></canvas>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="card">
        <div className="card-header">
          <h3>Feature Importance</h3>
          <span className="card-tag">SHAP / Model</span>
        </div>
        <div className="card-body">
          <canvas ref={importanceChartRef} id="importance-chart" height={180}></canvas>
        </div>
      </div>

      {/* Division Risk Rankings Table */}
      <div className="card card-wide">
        <div className="card-header">
          <h3>Division Risk Rankings</h3>
          <span className="card-tag" id="table-date-label">{tableDateLabel}</span>
        </div>
        <div className="card-body">
          <div className="risk-table-wrap">
            <table className="risk-table" id="risk-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Division</th>
                  <th>Risk Score</th>
                  <th>Risk Level</th>
                  <th>Total Crimes</th>
                  <th>High-Risk %</th>
                </tr>
              </thead>
              <tbody id="risk-table-body">
                {sorted.length === 0 ? (
                  <tr>
                    <td colSpan={6} style={{ textAlign: "center", padding: "2rem", color: "var(--text-muted)" }}>
                      Run a forecast to populate table…
                    </td>
                  </tr>
                ) : (
                  sorted.map((p, i) => {
                    const zs = zoneStats[p.zone] || {};
                    const color = riskColor(p.risk_level);
                    return (
                      <tr
                        key={p.zone}
                        className={`highlight-${p.risk_level.toLowerCase()}`}
                        onClick={() => onDivisionClick(p)}
                      >
                        <td style={{ color: "var(--text-muted)", fontFamily: "monospace" }}>{i + 1}</td>
                        <td style={{ fontWeight: 600, color: "var(--text-primary)" }}>{p.zone}</td>
                        <td style={{ fontFamily: "monospace", color }}>{p.risk_score.toFixed(4)}</td>
                        <td>
                          <span className={`level-pill ${p.risk_level}`}>{p.risk_level}</span>
                        </td>
                        <td style={{ color: "var(--text-sub)" }}>
                          {zs.total_crimes != null ? zs.total_crimes.toLocaleString() : "—"}
                        </td>
                        <td style={{ color: "var(--text-sub)" }}>
                          {zs.high_risk_days_pct != null ? zs.high_risk_days_pct + "%" : "—"}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}
