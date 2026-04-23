/**
 * app.js – PoliceSight Dashboard
 * Powers: Leaflet map, Chart.js charts, FastAPI REST calls
 */

const API = "http://localhost:8000/api";

// ── State ─────────────────────────────────────────────────────────────────
let state = {
  date:           null,
  predictions:    [],   // [{zone, risk_score, risk_level, avg_lat, avg_lon}…]
  zoneStats:      {},   // zone → stats row
  trendChart:     null,
  metricsChart:   null,
  importanceChart:null,
  divisionLayers: {},   // zone → Leaflet circle marker
  hotspotLayer:   null,
};

// ── Leaflet Map ──────────────────────────────────────────────────────────
const map = L.map("map", {
  center:  [34.05, -118.25],
  zoom:    11,
  zoomControl: true,
}).setView([34.05, -118.25], 10);

L.tileLayer(
  "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
  {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
    subdomains:  "abcd",
    maxZoom:     19,
  }
).addTo(map);

// ── Risk colour helpers ───────────────────────────────────────────────────
function riskColor(level, score) {
  if (level === "High")   return "#ef4444";
  if (level === "Medium") return "#f59e0b";
  return "#22c55e";
}
function fillOpacity(score) { return 0.25 + score * 0.5; }

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Default date = yesterday
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  document.getElementById("date-picker").value = formatDate(yesterday);

  checkHealth();
  loadZoneList();
  loadMetrics();
  loadFeatureImportance();
  loadZoneStats();
  loadHotspots();
});

// ── Health Check ──────────────────────────────────────────────────────────
async function checkHealth() {
  const badge = document.getElementById("model-status");
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    badge.className = "status-badge " + (d.model_loaded ? "ok" : "error");
    badge.querySelector("span").textContent = d.model_loaded
      ? "Model Ready" : "Model Not Loaded";
  } catch {
    badge.className = "status-badge error";
    badge.querySelector("span").textContent = "API Offline";
  }
}

// ── Zone Dropdown ─────────────────────────────────────────────────────────
async function loadZoneList() {
  try {
    const r = await fetch(`${API}/zones`);
    const d = await r.json();
    const sel = document.getElementById("zone-select");
    d.zones.forEach(z => {
      const opt = document.createElement("option");
      opt.value = z; opt.textContent = z;
      sel.appendChild(opt);
    });
    document.getElementById("stat-zones").textContent = d.count;
  } catch (e) { console.warn("Zone list unavailable:", e); }
}

// ── Forecast (all zones) ──────────────────────────────────────────────────
async function runForecast() {
  const dateVal = document.getElementById("date-picker").value;
  const hourVal = document.getElementById("hour-picker").value;
  if (!dateVal) { alert("Please select a date."); return; }
  state.date = dateVal;

  setLoading(true);
  let timeStr = "12:00:00";
  if (hourVal) {
    const hh = parseInt(hourVal, 10);
    timeStr = `${hh.toString().padStart(2, "0")}:00:00`;
  }
  
  document.getElementById("forecast-date-label").textContent =
    `Forecast for ${new Date(dateVal + "T" + timeStr).toLocaleDateString("en-US", { weekday:"long", year:"numeric", month:"long", day:"numeric", hour:"numeric", minute:"2-digit" })}`;
  document.getElementById("table-date-label").textContent = dateVal + (hourVal ? ` @ ${hourVal}:00` : "");

  try {
    let url = `${API}/predict/all?date=${dateVal}`;
    if (hourVal) url += `&hour=${hourVal}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(await r.text());
    state.predictions = await r.json();

    updateKPIs(state.predictions);
    renderDivisionMarkers(state.predictions);
    populateRiskTable(state.predictions);
    updateBadgeCounts(state.predictions);
  } catch (e) {
    console.error("Forecast error:", e);
    alert("Forecast failed. Is the backend running?\n\n" + e.message);
  } finally {
    setLoading(false);
  }
}

// ── Forecast single zone ───────────────────────────────────────────────────
async function forecastZone() {
  const zone = document.getElementById("zone-select").value;
  const dateVal = document.getElementById("date-picker").value;
  const hourVal = document.getElementById("hour-picker").value;
  if (!zone || !dateVal) { alert("Choose both a division and a date."); return; }

  setLoading(true);
  try {
    let body = { zone, date: dateVal };
    if (hourVal) body.hour = parseInt(hourVal, 10);
    
    const r = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    showDivisionDetail(d);
    loadTrend(zone);
  } catch (e) {
    console.error(e);
  } finally {
    setLoading(false);
  }
}

// ── KPI Cards ─────────────────────────────────────────────────────────────
function updateKPIs(preds) {
  const high   = preds.filter(p => p.risk_level === "High").length;
  const medium = preds.filter(p => p.risk_level === "Medium").length;
  const low    = preds.filter(p => p.risk_level === "Low").length;
  const avg    = preds.length
    ? (preds.reduce((s, p) => s + p.risk_score, 0) / preds.length)
    : 0;

  document.getElementById("kpi-high").textContent   = high;
  document.getElementById("kpi-medium").textContent = medium;
  document.getElementById("kpi-low").textContent    = low;
  document.getElementById("kpi-avg-risk").textContent = avg.toFixed(3);
}

function updateBadgeCounts(preds) {
  const high   = preds.filter(p => p.risk_level === "High").length;
  const medium = preds.filter(p => p.risk_level === "Medium").length;
  const low    = preds.filter(p => p.risk_level === "Low").length;
  document.getElementById("badge-high").textContent   = `${high} High`;
  document.getElementById("badge-medium").textContent = `${medium} Med`;
  document.getElementById("badge-low").textContent    = `${low} Low`;
}

// ── Map Markers ───────────────────────────────────────────────────────────
function renderDivisionMarkers(preds) {
  // Clear old markers
  Object.values(state.divisionLayers).forEach(l => map.removeLayer(l));
  state.divisionLayers = {};

  preds.forEach(p => {
    if (!p.avg_lat || !p.avg_lon) return;

    const color   = riskColor(p.risk_level, p.risk_score);
    const radius  = 1200 + p.risk_score * 1800;
    const opacity = fillOpacity(p.risk_score);

    const circle = L.circle([p.avg_lat, p.avg_lon], {
      radius,
      color,
      fillColor: color,
      fillOpacity: opacity,
      weight:    p.risk_level === "High" ? 2.5 : 1.5,
    }).addTo(map);

    const popupContent = `
      <div style="min-width:160px">
        <div style="font-weight:700;font-size:1rem;margin-bottom:0.3rem">${p.zone}</div>
        <div style="color:#8099c4;font-size:0.75rem;margin-bottom:0.5rem">${state.date}</div>
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:0.8rem;color:#c9d1d9">Risk Score</span>
          <span style="font-family:monospace;font-weight:700;color:${color}">${p.risk_score.toFixed(3)}</span>
        </div>
        <div style="margin-top:0.5rem;text-align:center">
          <span style="padding:0.2rem 0.8rem;border-radius:12px;font-size:0.75rem;font-weight:700;
            background:${color}22;color:${color};border:1px solid ${color}">
            ${p.risk_level} Risk
          </span>
        </div>
      </div>`;

    circle.bindPopup(popupContent);
    circle.on("click", () => {
      showDivisionDetail(p);
      loadTrend(p.zone);
    });

    state.divisionLayers[p.zone] = circle;

    // Pulse animation for high-risk
    if (p.risk_level === "High") {
      const pulse = L.circle([p.avg_lat, p.avg_lon], {
        radius: radius * 1.4, color, fillColor: "transparent",
        weight: 1.5, opacity: 0.4,
      }).addTo(map);
      state.divisionLayers[p.zone + "_pulse"] = pulse;
    }
  });
}

// ── Hotspot Markers ───────────────────────────────────────────────────────
async function loadHotspots() {
  try {
    const r = await fetch(`${API}/hotspots?top=40`);
    const clusters = await r.json();

    if (state.hotspotLayer) map.removeLayer(state.hotspotLayer);
    const group = L.layerGroup();

    clusters.forEach(c => {
      L.circleMarker([c.lat, c.lon], {
        radius:      5,
        color:       "#a78bfa",
        fillColor:   "#a78bfa",
        fillOpacity: 0.7,
        weight:      1,
      }).bindPopup(`<b>Hotspot #${c.cluster}</b><br>Size: ${c.size} incidents<br>Method: ${c.method}`)
        .addTo(group);
    });

    group.addTo(map);
    state.hotspotLayer = group;
  } catch (e) { console.warn("Hotspots unavailable:", e); }
}

// ── Zone Detail Panel ──────────────────────────────────────────────────────
function showDivisionDetail(pred) {
  document.getElementById("panel-placeholder").classList.add("hidden");
  document.getElementById("division-detail").classList.remove("hidden");

  document.getElementById("detail-zone-name").textContent = pred.zone;
  
  let dateText = pred.date || state.date || "";
  if (pred.hour !== undefined && pred.hour !== null) {
      dateText += ` at ${pred.hour}:00`;
  }
  document.getElementById("detail-zone-date").textContent = dateText;

  const score = pred.risk_score;
  document.getElementById("detail-risk-score").textContent = score.toFixed(4);

  const badge = document.getElementById("detail-risk-badge");
  badge.textContent = pred.risk_level + " Risk";
  badge.className = "risk-badge-large " + pred.risk_level;

  // Risk meter
  document.getElementById("detail-risk-fill").style.width = (score * 100) + "%";

  // Zone stats from already-loaded data
  const zs = state.zoneStats[pred.zone];
  if (zs) {
    document.getElementById("ds-high-risk-days").textContent = zs.high_risk_days_pct + "%";
    document.getElementById("ds-avg-daily").textContent      = zs.avg_daily.toFixed(1);
    document.getElementById("ds-total").textContent          = zs.total_crimes.toLocaleString();
    document.getElementById("ds-max").textContent            = zs.max_daily;
  }
  
  // AI Explanation fields
  const bProb = pred.base_risk_score ?? pred.risk_score;
  const tProb = pred.tree_risk_score ?? "-";
  const lProb = pred.lstm_high_risk_prob ?? "-";
  const mult = pred.tod_multiplier ?? 1.0;

  document.getElementById("expl-base-prob").textContent  = bProb.toFixed(4);
  document.getElementById("expl-tree-prob").textContent  = typeof tProb === "number" ? tProb.toFixed(4) : tProb;
  document.getElementById("expl-lstm-prob").textContent  = typeof lProb === "number" ? lProb.toFixed(4) : lProb;
  document.getElementById("expl-multiplier").textContent = mult.toFixed(2) + "x";
  document.getElementById("expl-multiplier").style.color = mult > 1.05 ? "#ef4444" : (mult < 0.95 ? "#22c55e" : "#8099c4");
  document.getElementById("expl-final-prob").textContent = score.toFixed(4);
  document.getElementById("expl-final-prob").style.color = riskColor(pred.risk_level, score);
}

// ── Trend Chart ────────────────────────────────────────────────────────────
async function loadTrend(zone) {
  try {
    const r = await fetch(`${API}/stats/trend?zone=${encodeURIComponent(zone)}`);
    const data = await r.json();

    const labels = data.map(d => d.date).slice(-60);
    const counts = data.map(d => d.count).slice(-60);

    const ctx = document.getElementById("trend-chart").getContext("2d");
    if (state.trendChart) state.trendChart.destroy();

    state.trendChart = new Chart(ctx, {
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
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
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
      }
    });
  } catch (e) { console.warn("Trend error:", e); }
}

// ── Model Metrics Chart + Accuracy Display ────────────────────────────────
async function loadMetrics() {
  try {
    const r = await fetch(`${API}/stats/metrics`);
    const data = await r.json();

    // Sort by roc_auc desc — first row = best model
    data.sort((a, b) => b.roc_auc - a.roc_auc);
    const best = data[0];

    const models  = data.map(d => d.model.replace(/_/g," ").replace(/\b\w/g, c=>c.toUpperCase()));
    const aucs    = data.map(d => d.roc_auc);
    const f1s     = data.map(d => d.f1);
    const precs   = data.map(d => d.precision);
    const recalls = data.map(d => d.recall);
    const accs    = data.map(d => d.accuracy);

    // ── Sidebar: SYSTEM stat ─────────────────────────────────────────────
    document.getElementById("stat-auc").textContent = best.roc_auc.toFixed(3);

    // ── Sidebar: BEST MODEL ACCURACY panel ──────────────────────────────
    const modelLabel = best.model.replace(/_/g," ").replace(/\b\w/g, c=>c.toUpperCase());
    document.getElementById("m-name-badge").textContent = modelLabel;

    const pctFmt = v => (v * 100).toFixed(1) + "%";

    document.getElementById("m-acc").textContent    = pctFmt(best.accuracy);
    document.getElementById("m-auc2").textContent   = best.roc_auc.toFixed(3);
    document.getElementById("m-prec").textContent   = pctFmt(best.precision);
    document.getElementById("m-recall").textContent = pctFmt(best.recall);
    document.getElementById("m-f1").textContent     = pctFmt(best.f1);

    // Accuracy progress bar
    const accPct = (best.accuracy * 100).toFixed(1);
    document.getElementById("acc-bar-pct").textContent  = accPct + "%";
    document.getElementById("acc-bar-fill").style.width = accPct + "%";

    // ── KPI row: Model Accuracy card ────────────────────────────────────
    document.getElementById("kpi-model-name").textContent = modelLabel;
    document.getElementById("kpi-m-acc").textContent      = pctFmt(best.accuracy);
    document.getElementById("kpi-m-auc").textContent      = best.roc_auc.toFixed(3);
    document.getElementById("kpi-m-f1").textContent       = pctFmt(best.f1);
    document.getElementById("kpi-m-prec").textContent     = pctFmt(best.precision);

    // ── Bar Chart: all models ────────────────────────────────────────────
    const ctx = document.getElementById("metrics-chart").getContext("2d");
    if (state.metricsChart) state.metricsChart.destroy();

    state.metricsChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          { label: "Accuracy",  data: accs,    backgroundColor: "#06b6d4cc" },
          { label: "ROC-AUC",   data: aucs,    backgroundColor: "#3b82f6cc" },
          { label: "F1",        data: f1s,     backgroundColor: "#22c55ecc" },
          { label: "Precision", data: precs,   backgroundColor: "#f59e0bcc" },
          { label: "Recall",    data: recalls, backgroundColor: "#ef4444cc" },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#8099c4", font: { size: 10 } } },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(1)}%`
            }
          }
        },
        scales: {
          x: { ticks: { color: "#4a6080", font: { size: 9 } }, grid: { color: "#1e2d47" } },
          y: {
            min: 0, max: 1,
            ticks: {
              color: "#4a6080", font: { size: 10 },
              callback: v => (v * 100).toFixed(0) + "%"
            },
            grid: { color: "#1e2d47" }
          },
        },
      }
    });

  } catch (e) { console.warn("Metrics unavailable:", e); }
}

// ── Feature Importance Chart ───────────────────────────────────────────────
async function loadFeatureImportance() {
  try {
    const r = await fetch(`${API}/stats/feature_importance`);
    const data = await r.json();

    const col = data[0]?.mean_abs_shap !== undefined ? "mean_abs_shap" : "importance";
    const top = data.slice(0, 10).reverse();
    const labels = top.map(d => d.feature.replace(/_/g, " "));
    const vals   = top.map(d => d[col]);

    const ctx = document.getElementById("importance-chart").getContext("2d");
    if (state.importanceChart) state.importanceChart.destroy();

    state.importanceChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          data: vals,
          backgroundColor: vals.map((_, i) =>
            `hsl(${140 + i * 18}, 60%, ${45 + i * 2}%)`
          ),
          borderRadius: 4,
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: "#4a6080", font: { size: 9 } }, grid: { color: "#1e2d47" } },
          y: { ticks: { color: "#8099c4", font: { size: 9 } }, grid: { display: false } },
        },
      }
    });
  } catch (e) { console.warn("Feature importance unavailable:", e); }
}

// ── Zone Stats (for detail panel) ──────────────────────────────────────────
async function loadZoneStats() {
  try {
    const r = await fetch(`${API}/stats/zones`);
    const data = await r.json();
    data.forEach(d => { state.zoneStats[d.zone] = d; });
  } catch (e) { console.warn("Zone stats unavailable:", e); }
}

// ── Risk Table ─────────────────────────────────────────────────────────────
function populateRiskTable(preds) {
  const sorted = [...preds].sort((a, b) => b.risk_score - a.risk_score);
  const tbody = document.getElementById("risk-table-body");
  tbody.innerHTML = "";

  sorted.forEach((p, i) => {
    const zs = state.zoneStats[p.zone] || {};
    const tr = document.createElement("tr");
    tr.className = `highlight-${p.risk_level.toLowerCase()}`;
    tr.innerHTML = `
      <td style="color:var(--text-muted);font-family:monospace">${i+1}</td>
      <td style="font-weight:600;color:var(--text-primary)">${p.zone}</td>
      <td style="font-family:monospace;color:${riskColor(p.risk_level)}">${p.risk_score.toFixed(4)}</td>
      <td><span class="level-pill ${p.risk_level}">${p.risk_level}</span></td>
      <td style="color:var(--text-sub)">${zs.total_crimes?.toLocaleString() ?? "—"}</td>
      <td style="color:var(--text-sub)">${zs.high_risk_days_pct ?? "—"}%</td>
    `;
    tr.addEventListener("click", () => {
      showDivisionDetail(p);
      loadTrend(p.zone);
    });
    tbody.appendChild(tr);
  });
}

// ── Utilities ─────────────────────────────────────────────────────────────
function setLoading(on) {
  document.getElementById("global-spinner").classList.toggle("active", on);
}

function formatDate(d) {
  return d.toISOString().split("T")[0];
}
