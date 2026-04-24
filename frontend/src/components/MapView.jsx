/**
 * MapView.jsx — PoliceSight
 * Leaflet map with division circle markers, hotspot cluster markers, and map legend.
 * Uses react-leaflet.
 */
import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

function riskColor(level) {
  if (level === "High")   return "#ef4444";
  if (level === "Medium") return "#f59e0b";
  return "#22c55e";
}

function fillOpacity(score) {
  return 0.25 + score * 0.5;
}

export default function MapView({ predictions, hotspots, riskFilters, date, onDivisionClick }) {
  const mapRef       = useRef(null);
  const leafletRef   = useRef(null);
  const divLayersRef = useRef({});
  const hotLayerRef  = useRef(null);

  // ── Init map ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (leafletRef.current) return; // already initialised

    leafletRef.current = L.map(mapRef.current, {
      center: [34.05, -118.25],
      zoom: 11,
      zoomControl: true,
    }).setView([34.05, -118.25], 10);

    L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
        subdomains: "abcd",
        maxZoom: 19,
      }
    ).addTo(leafletRef.current);

    return () => {
      if (leafletRef.current) {
        leafletRef.current.remove();
        leafletRef.current = null;
      }
    };
  }, []);

  // ── Render division markers ────────────────────────────────────────────────
  useEffect(() => {
    const map = leafletRef.current;
    if (!map) return;

    // Clear old markers
    Object.values(divLayersRef.current).forEach(l => map.removeLayer(l));
    divLayersRef.current = {};

    // Filter by risk level
    const filtered = predictions.filter(p => riskFilters[p.risk_level]);

    filtered.forEach(p => {
      if (!p.avg_lat || !p.avg_lon) return;

      const color   = riskColor(p.risk_level);
      const radius  = 1200 + p.risk_score * 1800;
      const opacity = fillOpacity(p.risk_score);

      const circle = L.circle([p.avg_lat, p.avg_lon], {
        radius,
        color,
        fillColor: color,
        fillOpacity: opacity,
        weight: p.risk_level === "High" ? 2.5 : 1.5,
      }).addTo(map);

      const popupContent = `
        <div style="min-width:160px">
          <div style="font-weight:700;font-size:1rem;margin-bottom:0.3rem">${p.zone}</div>
          <div style="color:#8099c4;font-size:0.75rem;margin-bottom:0.5rem">${date || ""}</div>
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
      circle.on("click", () => onDivisionClick(p));

      divLayersRef.current[p.zone] = circle;

      // Pulse ring for high-risk
      if (p.risk_level === "High") {
        const pulse = L.circle([p.avg_lat, p.avg_lon], {
          radius: radius * 1.4,
          color,
          fillColor: "transparent",
          weight: 1.5,
          opacity: 0.4,
        }).addTo(map);
        divLayersRef.current[p.zone + "_pulse"] = pulse;
      }
    });
  }, [predictions, riskFilters, date, onDivisionClick]);

  // ── Render hotspot markers ─────────────────────────────────────────────────
  useEffect(() => {
    const map = leafletRef.current;
    if (!map) return;

    if (hotLayerRef.current) map.removeLayer(hotLayerRef.current);
    const group = L.layerGroup();

    hotspots.forEach(c => {
      L.circleMarker([c.lat, c.lon], {
        radius: 5,
        color: "#a78bfa",
        fillColor: "#a78bfa",
        fillOpacity: 0.7,
        weight: 1,
      })
        .bindPopup(`<b>Hotspot #${c.cluster}</b><br>Size: ${c.size} incidents<br>Method: ${c.method}`)
        .addTo(group);
    });

    group.addTo(map);
    hotLayerRef.current = group;
  }, [hotspots]);

  return (
    <div className="map-container">
      <div id="map" ref={mapRef} style={{ width: "100%", height: "100%", minHeight: "420px" }}></div>
      <div className="map-legend">
        <div className="legend-title">Risk Level</div>
        <div className="legend-item"><span className="legend-dot high"></span>High (&gt;0.60)</div>
        <div className="legend-item"><span className="legend-dot medium"></span>Medium (0.30–0.60)</div>
        <div className="legend-item"><span className="legend-dot low"></span>Low (&lt;0.30)</div>
        <div className="legend-item"><span className="legend-dot hotspot"></span>Hotspot cluster</div>
      </div>
    </div>
  );
}
