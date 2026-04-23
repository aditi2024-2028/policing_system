import urllib.request, json, sys

base = "http://localhost:8000/api"

try:
    # Test 1: Health
    with urllib.request.urlopen(base + "/health") as r:
        h = json.load(r)
    print("Health:", h)

    # Test 2: Zones list
    with urllib.request.urlopen(base + "/zones") as r:
        d = json.load(r)
    print(f"Zones: {d['count']}  | First 5: {d['zones'][:5]}")

    # Test 3: The previously failing predict/all endpoint
    with urllib.request.urlopen(base + "/predict/all?date=2023-10-01") as r:
        preds = json.load(r)
    print(f"\nPredictions: {len(preds)} zones")
    for p in preds[:5]:
        z     = p["zone"]
        score = p["risk_score"]
        level = p["risk_level"]
        lat   = p["avg_lat"]
        lon   = p["avg_lon"]
        print(f"  {z:<14} score={score:.4f}  level={level:<6}  lat={lat:.4f}  lon={lon:.4f}")

    high   = sum(1 for p in preds if p["risk_level"] == "High")
    medium = sum(1 for p in preds if p["risk_level"] == "Medium")
    low    = sum(1 for p in preds if p["risk_level"] == "Low")
    print(f"\n  High={high}  Medium={medium}  Low={low}")
    print("\nAll API endpoints working correctly!")

except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
