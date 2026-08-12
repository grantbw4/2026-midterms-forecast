"""Leak-resistant rolling-origin evaluation utilities for forecast v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


HORIZONS = (120, 90, 60, 30, 14, 7)
REQUIRED_COLUMNS = {
    "year", "horizon", "model", "race_id", "actual_margin",
    "pred_mean", "pred_std", "prob_dem",
}


def _log_loss(actual: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return float(-np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))


def calibration_bins(actual: np.ndarray, probability: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    edges = np.linspace(0, 1, bins + 1)
    output = []
    for idx in range(bins):
        mask = (probability >= edges[idx]) & (probability < edges[idx + 1] if idx < bins - 1 else probability <= 1)
        if not np.any(mask):
            continue
        output.append({
            "bin_low": round(float(edges[idx]), 2),
            "bin_high": round(float(edges[idx + 1]), 2),
            "mean_forecast": round(float(np.mean(probability[mask])), 4),
            "observed_rate": round(float(np.mean(actual[mask])), 4),
            "n": int(np.sum(mask)),
        })
    return output


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Backtest predictions missing columns: {sorted(missing)}")
    frame = predictions.copy()
    numeric = ["year", "horizon", "actual_margin", "pred_mean", "pred_std", "prob_dem"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame[frame["horizon"].isin(HORIZONS)]
    frame["actual_dem"] = (frame["actual_margin"] > 0).astype(float)
    rows = []
    for (model, horizon), group in frame.groupby(["model", "horizon"]):
        error = group["pred_mean"].to_numpy() - group["actual_margin"].to_numpy()
        actual = group["actual_dem"].to_numpy()
        probability = group["prob_dem"].to_numpy()
        std = np.maximum(group["pred_std"].to_numpy(), 1e-6)
        coverage = {}
        for label, z in (("50", 0.67448975), ("80", 1.28155157), ("95", 1.95996398)):
            coverage[label] = round(float(np.mean(np.abs(error) <= z * std)), 4)
        rows.append({
            "model": str(model),
            "horizon": int(horizon),
            "n": int(len(group)),
            "rmse": round(float(np.sqrt(np.mean(error**2))), 4),
            "brier": round(float(np.mean((probability - actual) ** 2)), 5),
            "log_loss": round(_log_loss(actual, probability), 5),
            "coverage": coverage,
        })

    final = frame[frame["horizon"] <= 60]
    aggregate = []
    for model, group in final.groupby("model"):
        actual = group["actual_dem"].to_numpy()
        probability = group["prob_dem"].to_numpy()
        aggregate.append({
            "model": str(model),
            "n": int(len(group)),
            "brier": round(float(np.mean((probability - actual) ** 2)), 5),
            "log_loss": round(_log_loss(actual, probability), 5),
            "calibration": calibration_bins(actual, probability),
        })

    comparison_name = "v2" if "v2" in set(final["model"]) else "fundamentals"
    v3_frame = final[final["model"] == "v3"]
    baseline_frame = final[final["model"] == comparison_name]
    keys = ["year", "horizon", "race_id"]
    matched = v3_frame.merge(baseline_frame, on=keys, suffixes=("_v3", "_baseline"))
    if not matched.empty:
        actual = (matched["actual_margin_v3"].to_numpy() > 0).astype(float)
        v3_brier = float(np.mean((matched["prob_dem_v3"].to_numpy() - actual) ** 2))
        baseline_brier = float(np.mean((matched["prob_dem_baseline"].to_numpy() - actual) ** 2))
        v3_log = _log_loss(actual, matched["prob_dem_v3"].to_numpy())
        baseline_log = _log_loss(actual, matched["prob_dem_baseline"].to_numpy())
        brier_delta = v3_brier - baseline_brier
        log_delta = v3_log - baseline_log
        accepted = brier_delta <= 0.005 and log_delta <= 0.005 and (brier_delta < 0 or log_delta < 0)
        gate = {
            "status": "production" if accepted else "experimental",
            "comparison": comparison_name,
            "matched_n": int(len(matched)),
            "v3_brier": round(v3_brier, 5),
            "baseline_brier": round(baseline_brier, 5),
            "v3_log_loss": round(v3_log, 5),
            "baseline_log_loss": round(baseline_log, 5),
            "brier_delta": round(brier_delta, 5),
            "log_loss_delta": round(log_delta, 5),
            "rule": "Neither score may worsen by >0.005 and at least one must improve.",
        }
    else:
        gate = {"status": "insufficient_data", "rule": "v3 and a baseline are required"}
    return {
        "status": "complete",
        "horizons": list(HORIZONS),
        "metrics": rows,
        "final_60_day_aggregate": aggregate,
        "race_polling_gate": gate,
    }


def rolling_origin_predictions(
    years: Iterable[int],
    forecast: Callable[[int, int, tuple[int, ...]], pd.DataFrame],
) -> pd.DataFrame:
    """Run a callback with only cycles earlier than each holdout year."""
    outputs = []
    ordered = tuple(sorted(set(int(year) for year in years)))
    for holdout in ordered:
        training_years = tuple(year for year in ordered if year < holdout)
        if not training_years:
            continue
        for horizon in HORIZONS:
            result = forecast(holdout, horizon, training_years).copy()
            result["year"] = holdout
            result["horizon"] = horizon
            result["training_years"] = ",".join(map(str, training_years))
            outputs.append(result)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
