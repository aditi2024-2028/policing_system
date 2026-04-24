/**
 * Sidebar.jsx — PoliceSight
 * Contains: logo, date/time forecast, division drill-down,
 *           risk filter chips, system status, model accuracy panel.
 */
export default function Sidebar({
  date, setDate,
  hour, setHour,
  zones, selectedZone, setSelectedZone,
  riskFilters, setRiskFilters,
  modelStatus, modelStatusText,
  statZones, statAuc,
  bestModel,
  onRunForecast, onForecastZone,
}) {
  const pctFmt = (v) => (v != null ? (v * 100).toFixed(1) + "%" : "—");

  const toggleFilter = (level) => {
    setRiskFilters(prev => ({ ...prev, [level]: !prev[level] }));
  };

  return (
    <aside className="sidebar" id="sidebar">
      {/* Logo */}
      <div className="sidebar-header">
        <div className="logo">
          <div className="logo-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
          </div>
          <div className="logo-text">
            <span className="logo-name">PoliceSight</span>
            <span className="logo-sub">Crime Intelligence</span>
          </div>
        </div>
      </div>

      {/* Date & Time Forecast */}
      <div className="sidebar-section">
        <p className="sidebar-label">DATE &amp; TIME FORECAST</p>
        <div className="input-group">
          <label htmlFor="date-picker">Select Date</label>
          <input
            type="date"
            id="date-picker"
            value={date}
            onChange={e => setDate(e.target.value)}
          />
        </div>
        <div className="input-group" style={{ marginTop: "0.6rem" }}>
          <label htmlFor="hour-picker">Target Hour (0-23)</label>
          <input
            type="number"
            id="hour-picker"
            min="0" max="23"
            placeholder="Time (e.g. 19 for 7 PM)"
            value={hour}
            onChange={e => setHour(e.target.value)}
          />
        </div>
        <button
          className="btn-primary"
          id="btn-forecast"
          onClick={onRunForecast}
          style={{ marginTop: "1rem" }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="5 3 19 12 5 21 5 3"/>
          </svg>
          Run Forecast
        </button>
      </div>

      {/* Division Drill-down */}
      <div className="sidebar-section">
        <p className="sidebar-label">DIVISION DRILL-DOWN</p>
        <div className="input-group">
          <label htmlFor="zone-select">LAPD Division</label>
          <select
            id="zone-select"
            value={selectedZone}
            onChange={e => setSelectedZone(e.target.value)}
          >
            <option value="">— All Divisions —</option>
            {zones.map(z => (
              <option key={z} value={z}>{z}</option>
            ))}
          </select>
        </div>
        <button className="btn-secondary" id="btn-zone" onClick={onForecastZone}>
          Analyse Division
        </button>
      </div>

      {/* Risk Filter */}
      <div className="sidebar-section">
        <p className="sidebar-label">RISK FILTER</p>
        <div className="risk-filter-group">
          {["High", "Medium", "Low"].map(level => (
            <label
              key={level}
              className={`risk-chip ${level.toLowerCase()} ${riskFilters[level] ? "active" : ""}`}
              data-level={level}
              onClick={() => toggleFilter(level)}
            >
              <input type="checkbox" readOnly checked={riskFilters[level]} /> {level}
            </label>
          ))}
        </div>
      </div>

      {/* System */}
      <div className="sidebar-section">
        <p className="sidebar-label">SYSTEM</p>
        <div id="model-status" className={`status-badge ${modelStatus}`}>
          <div className="status-dot"></div>
          <span>{modelStatusText}</span>
        </div>
        <div className="sidebar-stats">
          <div className="stat-item">
            <span className="stat-val" id="stat-zones">{statZones}</span>
            <span className="stat-key">Divisions</span>
          </div>
          <div className="stat-item">
            <span className="stat-val" id="stat-auc">{statAuc}</span>
            <span className="stat-key">Best AUC</span>
          </div>
        </div>
      </div>

      {/* Model Accuracy Panel */}
      <div className="sidebar-section">
        <p className="sidebar-label">BEST MODEL ACCURACY</p>
        <div className="model-name-badge" id="m-name-badge">
          {bestModel
            ? bestModel.model.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())
            : "—"}
        </div>
        <div className="metrics-grid">
          <div className="metric-tile">
            <span className="metric-tile-val" id="m-acc">{pctFmt(bestModel?.accuracy)}</span>
            <span className="metric-tile-key">Accuracy</span>
          </div>
          <div className="metric-tile">
            <span className="metric-tile-val" id="m-auc2">{bestModel?.roc_auc?.toFixed(3) ?? "—"}</span>
            <span className="metric-tile-key">ROC-AUC</span>
          </div>
          <div className="metric-tile">
            <span className="metric-tile-val" id="m-prec">{pctFmt(bestModel?.precision)}</span>
            <span className="metric-tile-key">Precision</span>
          </div>
          <div className="metric-tile">
            <span className="metric-tile-val" id="m-recall">{pctFmt(bestModel?.recall)}</span>
            <span className="metric-tile-key">Recall</span>
          </div>
          <div className="metric-tile">
            <span className="metric-tile-val" id="m-f1">{pctFmt(bestModel?.f1)}</span>
            <span className="metric-tile-key">F1 Score</span>
          </div>
          <div className="metric-tile">
            <span className="metric-tile-val" id="m-train">26k</span>
            <span className="metric-tile-key">Train Rows</span>
          </div>
        </div>
        {/* Accuracy progress bar */}
        <div className="acc-bar-wrap">
          <div className="acc-bar-label">
            <span>Accuracy</span>
            <span id="acc-bar-pct">
              {bestModel ? (bestModel.accuracy * 100).toFixed(1) + "%" : "—"}
            </span>
          </div>
          <div className="acc-bar-track">
            <div
              className="acc-bar-fill"
              id="acc-bar-fill"
              style={{ width: bestModel ? (bestModel.accuracy * 100).toFixed(1) + "%" : "0%" }}
            ></div>
          </div>
        </div>
      </div>
    </aside>
  );
}
