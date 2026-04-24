/**
 * DetailPanel.jsx — PoliceSight
 * Right-side panel showing division detail: risk score, stats grid,
 * AI explanation breakdown, and crime trend chart.
 */
import { useEffect, useRef } from "react";
import { Chart, registerables } from "chart.js";

Chart.register(...registerables);

function riskColor(level) {
  if (level === "High")   return "#ef4444";
  if (level === "Medium") return "#f59e0b";
  return "#22c55e";
}

export default function DetailPanel({ division, zoneStats, trendData }) {
  const trendChartRef  = useRef(null);
  const trendInstanceRef = useRef(null);

  // ── Trend chart ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!trendData || !trendChartRef.current) return;

    const labels = trendData.map(d => d.date);
    const counts = trendData.map(d => d.count);

    if (trendInstanceRef.current) trendInstanceRef.current.destroy();

    trendInstanceRef.current = new Chart(trendChartRef.current, {
      type: "line",
      data: {
        labels,
        datasets: [{
          data: counts,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59,130,246,0.1)",
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
          tension: 0.4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            display: false,
            ticks: { color: "#4a6080" },
            grid:  { color: "#1e2d47" },
          },
          y: {
            ticks: { color: "#4a6080", font: { size: 10 } },
            grid:  { color: "#1e2d47" },
          },
        },
      },
    });
  }, [trendData]);

  // cleanup on unmount
  useEffect(() => {
    return () => { if (trendInstanceRef.current) trendInstanceRef.current.destroy(); };
  }, []);

  const zs    = division && zoneStats ? zoneStats[division.zone] : null;
  const score = division?.risk_score ?? 0;
  const color = division ? riskColor(division.risk_level) : "#8099c4";

  const bProb  = division ? (division.base_risk_score ?? division.risk_score) : null;
  const tProb  = division?.tree_risk_score ?? null;
  const lProb  = division?.lstm_high_risk_prob ?? null;
  const mult   = division?.tod_multiplier ?? 1.0;
  const multColor = mult > 1.05 ? "#ef4444" : (mult < 0.95 ? "#22c55e" : "#8099c4");

  let dateText = "";
  if (division) {
    dateText = division.date || "";
    if (division.hour !== undefined && division.hour !== null) {
      dateText += ` at ${division.hour}:00`;
    }
  }

  return (
    <div className="detail-panel" id="detail-panel">
      {!division ? (
        /* Placeholder */
        <div className="panel-placeholder" id="panel-placeholder">
          <div className="placeholder-icon">🗺️</div>
          <p>Click a division on the map<br/>or select one from the sidebar</p>
        </div>
      ) : (
        /* Division detail */
        <div className="division-detail" id="division-detail">
          {/* Header */}
          <div className="detail-header">
            <div>
              <h2 className="detail-zone" id="detail-zone-name">{division.zone}</h2>
              <p className="detail-date" id="detail-zone-date">{dateText}</p>
            </div>
            <div className={`risk-badge-large ${division.risk_level}`} id="detail-risk-badge">
              {division.risk_level} Risk
            </div>
          </div>

          {/* Risk meter */}
          <div className="risk-meter-wrap">
            <div className="risk-meter-label">
              <span>Risk Score</span>
              <span className="risk-score-val" id="detail-risk-score">{score.toFixed(4)}</span>
            </div>
            <div className="risk-meter-track">
              <div
                className="risk-meter-fill"
                id="detail-risk-fill"
                style={{ width: (score * 100) + "%" }}
              ></div>
            </div>
          </div>

          {/* Stats grid */}
          <div className="detail-stats-grid">
            <div className="detail-stat">
              <span className="ds-val" id="ds-high-risk-days">
                {zs ? zs.high_risk_days_pct + "%" : "—"}
              </span>
              <span className="ds-key">High-Risk Days %</span>
            </div>
            <div className="detail-stat">
              <span className="ds-val" id="ds-avg-daily">
                {zs ? zs.avg_daily.toFixed(1) : "—"}
              </span>
              <span className="ds-key">Avg Daily Crimes</span>
            </div>
            <div className="detail-stat">
              <span className="ds-val" id="ds-total">
                {zs ? zs.total_crimes.toLocaleString() : "—"}
              </span>
              <span className="ds-key">Total Crimes</span>
            </div>
            <div className="detail-stat">
              <span className="ds-val" id="ds-max">
                {zs ? zs.max_daily : "—"}
              </span>
              <span className="ds-key">Peak Day</span>
            </div>
          </div>

          {/* AI Explanation */}
          <div className="chart-section" style={{ marginTop: "1rem", marginBottom: "1rem" }}>
            <p className="chart-title">AI Explanation (Risk Breakdown)</p>
            <div className="expl-block">
              <div className="expl-row">
                <span style={{ color: "var(--text-sub)", fontSize: "0.85rem" }}>Base Ensemble Risk</span>
                <span id="expl-base-prob" style={{ fontFamily: "monospace", color: "var(--text-primary)" }}>
                  {bProb !== null ? bProb.toFixed(4) : "—"}
                </span>
              </div>
              <div className="expl-row">
                <span style={{ color: "var(--text-sub)", fontSize: "0.85rem" }}>↳ Tree Local Model</span>
                <span id="expl-tree-prob" style={{ fontFamily: "monospace", color: "var(--text-muted)" }}>
                  {typeof tProb === "number" ? tProb.toFixed(4) : "—"}
                </span>
              </div>
              <div className="expl-row">
                <span style={{ color: "var(--text-sub)", fontSize: "0.85rem" }}>↳ LSTM Temporal Risk</span>
                <span id="expl-lstm-prob" style={{ fontFamily: "monospace", color: "var(--text-muted)" }}>
                  {typeof lProb === "number" ? lProb.toFixed(4) : "—"}
                </span>
              </div>
              <div className="expl-divider"></div>
              <div className="expl-row">
                <span style={{ color: "var(--text-primary)", fontSize: "0.85rem", fontWeight: 600 }}>
                  Time-of-Day Multiplier
                </span>
                <span id="expl-multiplier" style={{ fontFamily: "monospace", color: multColor, fontWeight: 600 }}>
                  {mult.toFixed(2)}x
                </span>
              </div>
              <div className="expl-divider"></div>
              <div className="expl-row">
                <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>Scaled Final Risk</span>
                <span id="expl-final-prob" style={{ fontFamily: "monospace", fontWeight: 700, color }}>
                  {score.toFixed(4)}
                </span>
              </div>
            </div>
          </div>

          {/* Trend chart */}
          <div className="chart-section">
            <p className="chart-title">Crime Trend (Division)</p>
            <canvas ref={trendChartRef} id="trend-chart" height={130}></canvas>
          </div>
        </div>
      )}
    </div>
  );
}
