#!/usr/bin/env python3
"""
build_frontend.py  —  Convenience script to build the React frontend.

Run this once after cloning (or after changing frontend code) to generate
the frontend/dist/ folder that FastAPI serves.

Usage:
    python build_frontend.py
"""
import subprocess
import sys
from pathlib import Path

frontend_dir = Path(__file__).resolve().parent / "frontend"

if not frontend_dir.exists():
    print("[ERROR] frontend/ directory not found. Did you run the migration?")
    sys.exit(1)

print("[Build] Installing frontend dependencies...")
subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)

print("[Build] Building React app...")
subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True)

dist = frontend_dir / "dist"
if dist.exists():
    print(f"\n✅ React build complete! Output: {dist}")
    print("\nNow start the backend:")
    print("    uvicorn backend.main:app --reload --port 8000")
    print("\nOpen: http://localhost:8000")
else:
    print("[ERROR] Build failed — dist/ not found.")
    sys.exit(1)
