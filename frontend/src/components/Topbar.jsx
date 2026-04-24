/**
 * Topbar.jsx — PoliceSight
 * Shows page title, forecast date label, risk badge counts, and loading spinner.
 */
export default function Topbar({ forecastLabel, predictions, loading }) {
  const high   = predictions.filter(p => p.risk_level === "High").length;
  const medium = predictions.filter(p => p.risk_level === "Medium").length;
  const low    = predictions.filter(p => p.risk_level === "Low").length;

  return (
    <header className="topbar">
      <div className="topbar-left">
        <h1 className="page-title">Crime Risk Map</h1>
        <span className="page-sub" id="forecast-date-label">{forecastLabel}</span>
      </div>
      <div className="topbar-right">
        <div className="badge-group">
          <span className="badge badge-high" id="badge-high">
            {predictions.length ? `${high} High` : "— High"}
          </span>
          <span className="badge badge-medium" id="badge-medium">
            {predictions.length ? `${medium} Med` : "— Medium"}
          </span>
          <span className="badge badge-low" id="badge-low">
            {predictions.length ? `${low} Low` : "— Low"}
          </span>
        </div>
        <div className={`loading-spinner${loading ? " active" : ""}`} id="global-spinner"></div>
      </div>
    </header>
  );
}
