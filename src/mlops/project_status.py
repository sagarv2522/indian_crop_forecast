"""Project-wide status check for the crop price forecasting platform."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _exists(path: Path) -> CheckResult:
    return CheckResult(
        name=str(path),
        ok=path.exists(),
        detail="present" if path.exists() else "missing",
    )


def _any_exists(paths: Iterable[Path], label: str) -> CheckResult:
    paths = list(paths)
    found = [p for p in paths if p.exists()]
    if found:
        return CheckResult(name=label, ok=True, detail=f"found {len(found)} item(s)")
    return CheckResult(name=label, ok=False, detail="none found")


def collect_status() -> list[CheckResult]:
    checks = [
        _exists(ROOT / "README.md"),
        _exists(ROOT / "PROJECT_RUN_AND_SETUP.txt"),
        _exists(ROOT / "QUICK_START.txt"),
        _exists(ROOT / "SETUP_CHECKLIST.txt"),
        _exists(ROOT / "RUN_FULL_PROJECT.txt"),
        _exists(ROOT / "requirements.txt"),
        _exists(ROOT / ".github" / "workflows" / "weekly_retrain.yml"),
        _exists(ROOT / "database" / "schema.sql"),
        _any_exists([ROOT / "data" / "processed" / "clean_prices.csv"], "processed data"),
        _any_exists((ROOT / "data" / "parquet").glob("year=*/data.parquet"), "parquet partitions"),
        _any_exists((ROOT / "models").glob("arima_*.pkl"), "ARIMA artifacts"),
        _any_exists((ROOT / "models").glob("xgb_*.pkl"), "XGBoost artifacts"),
        _any_exists((ROOT / "models").glob("lstm_*.pt"), "LSTM artifacts"),
        _any_exists((ROOT / "models").glob("ensemble_*.pkl"), "ensemble artifacts"),
        _any_exists((ROOT / "data").glob("prediction_logs.duckdb"), "prediction log db"),
    ]
    return checks


def build_summary(checks: list[CheckResult]) -> dict:
    total = len(checks)
    ok = sum(1 for item in checks if item.ok)
    missing = [item.name for item in checks if not item.ok]
    return {
        "total_checks": total,
        "passed": ok,
        "failed": total - ok,
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the crop forecasting project status")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    checks = collect_status()
    summary = build_summary(checks)

    if args.json:
        payload = {
            "summary": summary,
            "checks": [asdict(item) for item in checks],
        }
        print(json.dumps(payload, indent=2))
        return

    print("Crop Price Forecasting Project Status")
    print("=" * 42)
    for item in checks:
        mark = "OK" if item.ok else "MISSING"
        print(f"{mark:8} {item.name} - {item.detail}")
    print("-" * 42)
    print(
        f"Passed {summary['passed']} / {summary['total_checks']} checks"
        f" | Failed: {summary['failed']}"
    )
    if summary["missing"]:
        print("Missing:")
        for item in summary["missing"]:
            print(f"  - {item}")


if __name__ == "__main__":
    main()

