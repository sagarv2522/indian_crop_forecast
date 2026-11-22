"""Test FastAPI endpoints against live Azure Blob Storage layer."""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
from dotenv import load_dotenv

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"

# Verify environment points to Azure Blob Storage
azure_url = os.getenv("AZURE_BLOB_BASE_URL", "")
data_source = os.getenv("DATA_SOURCE", "")
print("=" * 70)
print("  FastAPI End-to-End Smoke Test")
print(f"  DATA_SOURCE       : {data_source}")
print(f"  Target Storage URL: {azure_url}")
print("=" * 70)

endpoints = [
    {
        "name": "1. /predict (Crop price forecast for Tomato in Tamil Nadu)",
        "url": f"{BASE_URL}/predict?crop=Tomato&state=Tamil+Nadu&model=xgb&steps=14",
    },
    {
        "name": "2. /explain (SHAP feature contributions)",
        "url": f"{BASE_URL}/explain?crop=Tomato&state=Tamil+Nadu&model=xgb&top_n=5",
    },
    {
        "name": "3. /advisory (Market advisory & farmer intelligence)",
        "url": f"{BASE_URL}/advisory?crop=Tomato&state=Tamil+Nadu&model=xgb&steps=14&top_n=5",
    },
    {
        "name": "4. /history (Stored prediction log from DuckDB)",
        "url": f"{BASE_URL}/history?limit=5",
    },
    {
        "name": "5. /monitoring (PSI drift metrics from Azure data series)",
        "url": f"{BASE_URL}/monitoring?crop=Tomato&state=Tamil+Nadu&history_limit=5",
    },
]

results = []

for ep in endpoints:
    print(f"\nCalling {ep['name']}...")
    t0 = time.time()
    try:
        req = urllib.request.Request(ep["url"], headers={"User-Agent": "SmokeTestClient"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            elapsed = time.time() - t0
            status = resp.status
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)

            print(f"  HTTP Status Code : {status} OK ({elapsed:.2f}s)")

            if "forecast" in data:
                f_count = len(data.get("forecast", []))
                first_pred = data["forecast"][0] if f_count > 0 else {}
                print(f"  Forecast Horizon : {f_count} days")
                print(f"  Predicted Price  : Rs. {first_pred.get('forecast', 'N/A')} on {first_pred.get('date', 'N/A')}")
                if "advisory" in data:
                    print(f"  Advisory Action  : {data['advisory'].get('action', 'N/A')} (Risk: {data['advisory'].get('risk_level', 'N/A')})")

            elif "explanation" in data:
                top_features = data["explanation"].get("top_contributions", [])
                top_names = [f"{item.get('feature')}: {item.get('contribution', 0):+.2f}" for item in top_features[:3]]
                print(f"  Model Type       : {data.get('model')}")
                print(f"  Top SHAP Drivers : {', '.join(top_names)}")

            elif "advisory" in data and isinstance(data["advisory"], dict):
                adv = data["advisory"]
                print(f"  Action           : {adv.get('action', 'N/A')} (Risk: {adv.get('risk_level', 'N/A')})")
                print(f"  Recommendation   : {adv.get('recommendation', 'N/A')}")
                print(f"  Summary Points   : {adv.get('summary', 'N/A')}")

            elif "records" in data:
                rec_count = len(data.get("records", []))
                print(f"  History Records  : {rec_count} logged forecast records retrieved")

            elif "drift" in data:
                drift = data.get("drift", {})
                print(f"  PSI Value        : {drift.get('psi', 'N/A')}")
                print(f"  Mean Shift       : {drift.get('mean_shift_pct', 'N/A')}% (Baseline: {drift.get('baseline_days', 0)}d, Recent: {drift.get('recent_days', 0)}d)")
                print(f"  Baseline Mean    : Rs. {drift.get('baseline_mean', 'N/A')}, Recent Mean: Rs. {drift.get('recent_mean', 'N/A')}")

            results.append((ep["name"], status, True))

    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        print(f"  HTTP Status Code : {e.code} Error: {err_body}")
        results.append((ep["name"], e.code, False))
    except Exception as e:
        print(f"  Request Failed   : {e}")
        results.append((ep["name"], 0, False))

print("\n" + "=" * 70)
print("  Smoke Test Summary:")
for name, status, ok in results:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name} -> HTTP {status}")
print("=" * 70)
