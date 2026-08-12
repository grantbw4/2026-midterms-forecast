#!/usr/bin/env python3
"""Score rolling-origin forecast snapshots and publish calibration evidence.

The input is intentionally a prediction contract rather than a notebook-only
object, allowing historical model runs to be reproduced and audited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.backtesting import evaluate_predictions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate rolling-origin forecast predictions")
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "data" / "backtests" / "predictions.csv",
        help="Rows for v3, v2, fundamentals, and polls-only models",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs" / "backtest_metrics.json",
    )
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Missing {args.input}. Generate leak-free forecast snapshots at 120/90/60/30/14/7 days first."
        )
    report = evaluate_predictions(pd.read_csv(args.input))
    provenance_path = args.input.parent / "provenance.json"
    if provenance_path.exists():
        report["provenance"] = json.loads(provenance_path.read_text())
    report["scope"] = "Senate candidate-race polling layer"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(args.output)
    print(f"Backtest status: {report['race_polling_gate']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
