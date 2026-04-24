/**
 * server/index.js — Express.js proxy server for PoliceSight
 *
 * - In development: Vite handles the frontend (npm run dev in /frontend)
 *   This server proxies /api/* → FastAPI on port 8000
 *
 * - In production: This server serves the built Vite app (frontend/dist)
 *   AND proxies /api/* → FastAPI on port 8000
 *
 * Usage:
 *   Development:  node index.js           (port 3000)
 *   Production:   NODE_ENV=production node index.js
 */

const express = require("express");
const { createProxyMiddleware } = require("http-proxy-middleware");
const cors = require("cors");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = process.env.PORT || 3000;
const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";
const isProd = process.env.NODE_ENV === "production";

// ── CORS ──────────────────────────────────────────────────────────────────────
app.use(cors());

// ── Proxy /api/* → FastAPI Python backend ─────────────────────────────────────
app.use(
  "/api",
  createProxyMiddleware({
    target: FASTAPI_URL,
    changeOrigin: true,
    on: {
      error: (err, req, res) => {
        console.error("[Proxy] Error:", err.message);
        res.status(502).json({
          error: "Backend unavailable",
          detail: "FastAPI server is not running. Start it with: uvicorn backend.main:app --reload --port 8000",
        });
      },
    },
  })
);

// ── Serve built React app (production only) ───────────────────────────────────
if (isProd) {
  const distPath = path.join(__dirname, "..", "frontend", "dist");
  if (fs.existsSync(distPath)) {
    app.use(express.static(distPath));
    // SPA fallback — all non-API routes serve index.html
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
    console.log(`[Server] Serving built React app from: ${distPath}`);
  } else {
    console.warn("[Server] Production build not found. Run: cd frontend && npm run build");
  }
} else {
  // Dev mode — just show a helpful message at root
  app.get("/", (req, res) => {
    res.json({
      message: "PoliceSight Express proxy running in DEV mode",
      info: "Open http://localhost:5173 for the React app (run: cd frontend && npm run dev)",
      proxy: `All /api/* requests proxied to ${FASTAPI_URL}`,
    });
  });
}

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🚔 PoliceSight Express server running on http://localhost:${PORT}`);
  console.log(`   Proxying /api/* → ${FASTAPI_URL}`);
  if (isProd) {
    console.log(`   Serving React app from frontend/dist`);
  } else {
    console.log(`   Dev mode: React app at http://localhost:5173 (run: cd frontend && npm run dev)`);
  }
  console.log("");
});
