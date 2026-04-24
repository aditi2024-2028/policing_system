/**
 * App.jsx — PoliceSight Dashboard Root
 * Manages global state and API calls, coordinates all components.
 */
import { useState, useEffect, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import Topbar from "./components/Topbar";
import KpiRow from "./components/KpiRow";
import MapView from "./components/MapView";
import DetailPanel from "./components/DetailPanel";
import BottomRow from "./components/BottomRow";

const API = "/api";

function formatDate(d) {
  return d.toISOString().split("T")[0];
}

export function riskColor(level) {
  if (level === "High") return "#ef4444";
  if (level === "Medium") return "#f59e0b";
  return "#22c55e";
}

export default function App() {
  // ── State ─────────────────────────────────────────────────────────────────
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);

  const [date, setDate] = useState(formatDate(yesterday));
  const [hour, setHour] = useState("");
  const [selectedZone, setSelectedZone] = useState("");
  const [zones, setZones] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [zoneStats, setZoneStats] = useState({});
  const [metrics, setMetrics] = useState([]);
  const [featureImportance, setFeatureImportance] = useState([]);
  const [hotspots, setHotspots] = useState([]);
  const [modelStatus, setModelStatus] = useState("loading"); // "ok" | "error" | "loading"
  const [modelStatusText, setModelStatusText] = useState("Connecting…");
  const [statZones, setStatZones] = useState("—");
  const [statAuc, setStatAuc] = useState("—");
  const [bestModel, setBestModel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [forecastLabel, setForecastLabel] = useState("Select a date to forecast");
  const [tableDateLabel, setTableDateLabel] = useState("All Time");
  const [selectedDivision, setSelectedDivision] = useState(null); // detail panel
  const [trendData, setTrendData] = useState(null);
  const [riskFilters, setRiskFilters] = useState({ High: true, Medium: true, Low: true });

  // ── Health check ──────────────────────────────────────────────────────────
  const checkHealth = useCallback(async () => {
    try {
      const r = await fetch(`${API}/health`);
      const d = await r.json();
      setModelStatus(d.tree_model ? "ok" : "error");
      setModelStatusText(d.tree_model ? "Model Ready" : "Model Not Loaded");
    } catch {
      setModelStatus("error");
      setModelStatusText("API Offline");
    }
  }, []);

  // ── Load zone list ─────────────────────────────────────────────────────────
  const loadZoneList = useCallback(async () => {
    try {
      const r = await fetch(`${API}/zones`);
      const d = await r.json();
      setZones(d.zones || []);
      setStatZones(d.count ?? "—");
    } catch (e) {
      console.warn("Zone list unavailable:", e);
    }
  }, []);

  // ── Load metrics ──────────────────────────────────────────────────────────
  const loadMetrics = useCallback(async () => {
    try {
      const r = await fetch(`${API}/stats/metrics`);
      const data = await r.json();
      data.sort((a, b) => b.roc_auc - a.roc_auc);
      setMetrics(data);
      setBestModel(data[0]);
      setStatAuc(data[0]?.roc_auc?.toFixed(3) ?? "—");
    } catch (e) {
      console.warn("Metrics unavailable:", e);
    }
  }, []);

  // ── Load feature importance ────────────────────────────────────────────────
  const loadFeatureImportance = useCallback(async () => {
    try {
      const r = await fetch(`${API}/stats/feature_importance`);
      const data = await r.json();
      setFeatureImportance(data);
    } catch (e) {
      console.warn("Feature importance unavailable:", e);
    }
  }, []);

  // ── Load zone stats ────────────────────────────────────────────────────────
  const loadZoneStats = useCallback(async () => {
    try {
      const r = await fetch(`${API}/stats/zones`);
      const data = await r.json();
      const map = {};
      data.forEach(d => { map[d.zone] = d; });
      setZoneStats(map);
    } catch (e) {
      console.warn("Zone stats unavailable:", e);
    }
  }, []);

  // ── Load hotspots ─────────────────────────────────────────────────────────
  const loadHotspots = useCallback(async () => {
    try {
      const r = await fetch(`${API}/hotspots?top=40`);
      const data = await r.json();
      setHotspots(data);
    } catch (e) {
      console.warn("Hotspots unavailable:", e);
    }
  }, []);

  // ── Load trend for a division ─────────────────────────────────────────────
  const loadTrend = useCallback(async (zone) => {
    try {
      const r = await fetch(`${API}/stats/trend?zone=${encodeURIComponent(zone)}`);
      const data = await r.json();
      setTrendData(data.slice(-60));
    } catch (e) {
      console.warn("Trend error:", e);
    }
  }, []);

  // ── Run forecast (all zones) ───────────────────────────────────────────────
  const runForecast = useCallback(async () => {
    if (!date) { alert("Please select a date."); return; }
    setLoading(true);

    const timeStr = hour ? `${String(hour).padStart(2, "0")}:00:00` : "12:00:00";
    setForecastLabel(
      `Forecast for ${new Date(date + "T" + timeStr).toLocaleDateString("en-US", {
        weekday: "long", year: "numeric", month: "long", day: "numeric",
        hour: "numeric", minute: "2-digit",
      })}`
    );
    setTableDateLabel(date + (hour ? ` @ ${hour}:00` : ""));

    try {
      let url = `${API}/predict/all?date=${date}`;
      if (hour) url += `&hour=${hour}`;
      const r = await fetch(url);
      if (!r.ok) throw new Error(await r.text());
      const preds = await r.json();
      setPredictions(preds);
    } catch (e) {
      console.error("Forecast error:", e);
      alert("Forecast failed. Is the backend running?\n\n" + e.message);
    } finally {
      setLoading(false);
    }
  }, [date, hour]);

  // ── Forecast single zone ───────────────────────────────────────────────────
  const forecastZone = useCallback(async () => {
    if (!selectedZone || !date) { alert("Choose both a division and a date."); return; }
    setLoading(true);
    try {
      const body = { zone: selectedZone, date };
      if (hour) body.hour = parseInt(hour, 10);
      const r = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const d = await r.json();
      setSelectedDivision(d);
      loadTrend(selectedZone);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [selectedZone, date, hour, loadTrend]);

  // ── Handle map/table division click ───────────────────────────────────────
  const handleDivisionClick = useCallback((pred) => {
    setSelectedDivision(pred);
    loadTrend(pred.zone);
  }, [loadTrend]);

  // ── Init ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    checkHealth();
    loadZoneList();
    loadMetrics();
    loadFeatureImportance();
    loadZoneStats();
    loadHotspots();
  }, [checkHealth, loadZoneList, loadMetrics, loadFeatureImportance, loadZoneStats, loadHotspots]);

  return (
    <>
      <Sidebar
        date={date}
        setDate={setDate}
        hour={hour}
        setHour={setHour}
        zones={zones}
        selectedZone={selectedZone}
        setSelectedZone={setSelectedZone}
        riskFilters={riskFilters}
        setRiskFilters={setRiskFilters}
        modelStatus={modelStatus}
        modelStatusText={modelStatusText}
        statZones={statZones}
        statAuc={statAuc}
        bestModel={bestModel}
        onRunForecast={runForecast}
        onForecastZone={forecastZone}
      />

      <main className="main-content">
        <Topbar
          forecastLabel={forecastLabel}
          predictions={predictions}
          loading={loading}
        />

        <KpiRow predictions={predictions} bestModel={bestModel} />

        <div className="map-row">
          <MapView
            predictions={predictions}
            hotspots={hotspots}
            riskFilters={riskFilters}
            date={date}
            onDivisionClick={handleDivisionClick}
          />
          <DetailPanel
            division={selectedDivision}
            zoneStats={zoneStats}
            trendData={trendData}
          />
        </div>

        <BottomRow
          metrics={metrics}
          featureImportance={featureImportance}
          predictions={predictions}
          zoneStats={zoneStats}
          tableDateLabel={tableDateLabel}
          onDivisionClick={handleDivisionClick}
        />
      </main>
    </>
  );
}
