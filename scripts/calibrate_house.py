#!/usr/bin/env python3
"""Build authoritative House inputs, fit v4, and run structural holdouts.

The holdout evaluates the district-to-seat layer conditional on the realized
national House margin.  It intentionally does not claim to validate Silver
Bulletin's proprietary maintained averages, for which no public historical
archive comparable to the production feed is available.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.forecast_v3 import STATE_TO_REGION  # noqa: E402
from models.house_model import (  # noqa: E402
    HouseCalibration,
    design_matrix,
    fit_house_calibration,
    posterior_mean_margin,
)
from models.race_polling import normalize_name  # noqa: E402
from scripts.process_historical_data import (  # noqa: E402
    process_538_partisan_lean_2018,
    process_538_partisan_lean_2022,
    process_mit_house_results,
    process_wikipedia_pvi_2026,
)

DATA = PROJECT_ROOT / "data"
PROCESSED = DATA / "processed"
HISTORICAL = DATA / "historical"
OUTPUTS = PROJECT_ROOT / "outputs"
MIT_PATH = DATA / "MIT_Election_files" / "1976-2024-house.tab"
MAP_CHANGED_2024 = {"AL", "GA", "LA", "NC", "NY"}
N_SIMULATIONS = 10_000


def _national_margin(results: pd.DataFrame) -> float:
    dem = float(results["dem_votes"].sum())
    rep = float(results["rep_votes"].sum())
    return 100.0 * (dem - rep) / (dem + rep)


def _district_id(state: str, district: int) -> str:
    return f"{state}-{1 if int(district) == 0 else int(district):02d}"


def derive_incumbency(year: int) -> pd.DataFrame:
    """Infer incumbency by matching the prior winner to the current ballot."""

    raw = pd.read_csv(MIT_PATH, low_memory=False)
    current = raw[(raw["year"] == year) & (raw["stage"] == "GEN") & (raw["state_po"] != "DC")].copy()
    previous = raw[(raw["year"] == year - 2) & (raw["stage"] == "GEN") & (raw["state_po"] != "DC")].copy()
    for frame in (current, previous):
        frame["district_id"] = [_district_id(state, district)
                                for state, district in zip(frame["state_po"], frame["district"])]
        frame["candidate_key"] = frame["candidate"].fillna("").map(normalize_name)
        frame["major_party"] = frame["party"].fillna("").astype(str).str.upper().map(
            lambda value: "D" if "DEMOCRAT" in value else ("R" if "REPUBLICAN" in value else "")
        )

    prior_candidates = (
        previous.groupby(["district_id", "candidate_key", "major_party"], as_index=False)["candidatevotes"].sum()
        .sort_values("candidatevotes")
        .groupby("district_id", as_index=False)
        .tail(1)
    )
    current_names = current.groupby("district_id")["candidate_key"].apply(set).to_dict()
    records = []
    for _, winner in prior_candidates.iterrows():
        race_id = str(winner["district_id"])
        party = str(winner["major_party"])
        incumbent = bool(winner["candidate_key"] and winner["candidate_key"] in current_names.get(race_id, set()))
        if not incumbent or party not in {"D", "R"}:
            party, code = "Open", 0
        else:
            code = 1 if party == "D" else -1
        records.append({"district_id": race_id, "incumbent_party": party,
                        "incumbency_code": code, "year": year})
    result = pd.DataFrame(records).drop_duplicates("district_id")
    if result["district_id"].nunique() != 435:
        raise ValueError(f"Derived {year} incumbency has {result['district_id'].nunique()} districts")
    return result.sort_values("district_id").reset_index(drop=True)


def historical_frame(year: int) -> pd.DataFrame:
    results = process_mit_house_results(year)
    results = results[results["district_id"].str.match(r"^[A-Z]{2}-\d{2}$")].copy()
    if year == 2018:
        lean = process_538_partisan_lean_2018()
        incumbency_path = HISTORICAL / "incumbency_2018.csv"
        incumbency = pd.read_csv(incumbency_path) if incumbency_path.exists() else derive_incumbency(year)
    elif year == 2022:
        lean = process_538_partisan_lean_2022()
        incumbency_path = HISTORICAL / "incumbency_2022.csv"
        incumbency = pd.read_csv(incumbency_path) if incumbency_path.exists() else derive_incumbency(year)
    elif year == 2024:
        # The 2022 and 2024 elections mostly share district boundaries.  States
        # with intervening court/legislative maps are excluded from this holdout.
        lean = process_538_partisan_lean_2022()
        incumbency = derive_incumbency(year)
    else:
        raise ValueError(f"Unsupported calibration year: {year}")
    frame = (
        results.merge(lean[["district_id", "pvi_numeric"]], on="district_id", how="inner")
        .merge(incumbency[["district_id", "incumbency_code"]], on="district_id", how="left")
    )
    frame["incumbency_code"] = frame["incumbency_code"].fillna(0)
    frame["state"] = frame["district_id"].str[:2]
    frame["region"] = frame["state"].map(STATE_TO_REGION)
    frame["year"] = year
    frame["national_margin"] = _national_margin(results)
    frame["map_comparable"] = True
    if year == 2024:
        frame["map_comparable"] = ~frame["state"].isin(MAP_CHANGED_2024)
    frame["contested"] = (frame["dem_votes"] > 100) & (frame["rep_votes"] > 100)
    return frame


def build_current_fundamentals() -> pd.DataFrame:
    cook_path = DATA / "cook" / "house_fundamentals.csv"
    if not cook_path.exists():
        raise FileNotFoundError(
            "Missing current-map House fundamentals; run the Cook race-table fetch first"
        )
    frame = pd.read_csv(cook_path)
    current_results = process_mit_house_results(2024)[["district_id", "margin"]].rename(
        columns={"margin": "margin_2024"}
    )
    frame["state"] = frame["district_id"].str[:2]
    frame["pvi"] = frame["cook_pvi"]
    frame["pvi_source"] = "Cook PVI current 2026 district map"
    frame["pvi_source_url"] = frame["source_url"]
    frame["pvi_effective_date"] = "2026-08-12"
    frame["lean_quality"] = "partisan_lean"
    frame["district_number"] = frame["district_id"].str[-2:].astype(int)
    frame["region"] = frame["state"].map(STATE_TO_REGION)
    frame["incumbent"] = frame["incumbent"].fillna("Unknown")
    frame["incumbent_party"] = frame["incumbent_party"].fillna("")
    frame["open_seat"] = frame["is_open"].fillna(False).astype(bool)
    frame = frame.merge(current_results, on="district_id", how="left", validate="one_to_one")
    columns = [
        "district_id", "state", "district_number", "region", "pvi", "incumbent",
        "incumbent_party", "open_seat", "margin_2024", "pvi_source", "pvi_source_url",
        "pvi_effective_date", "lean_quality", "cook_rating", "cook_pvi_string",
    ]
    result = frame[columns].sort_values("district_id").reset_index(drop=True)
    if len(result) != 435 or result["district_id"].nunique() != 435:
        raise ValueError("Current House fundamentals must contain 435 unique districts")
    if result[["pvi", "pvi_source", "pvi_source_url", "pvi_effective_date"]].isna().any().any():
        raise ValueError("Current House fundamentals contain missing lean provenance")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    result.to_csv(PROCESSED / "districts.csv", index=False)
    return result


def _prediction_rows(model: str, year: int, frame: pd.DataFrame, mean: np.ndarray,
                     std: np.ndarray) -> pd.DataFrame:
    probability = student_t.cdf(mean / np.maximum(std, 1e-6), df=5.0)
    return pd.DataFrame({
        "year": year, "model": model, "race_id": frame["district_id"],
        "actual_margin": frame["margin"], "pred_mean": mean, "pred_std": std,
        "prob_dem": probability,
    })


def _scores(frame: pd.DataFrame) -> dict[str, float]:
    actual = (frame["actual_margin"].to_numpy() > 0).astype(float)
    probability = np.clip(frame["prob_dem"].to_numpy(float), 1e-6, 1 - 1e-6)
    error = frame["pred_mean"].to_numpy(float) - frame["actual_margin"].to_numpy(float)
    return {
        "n": int(len(frame)), "rmse": round(float(np.sqrt(np.mean(error**2))), 4),
        "signed_error": round(float(np.mean(error)), 4),
        "brier": round(float(np.mean((probability - actual)**2)), 5),
        "log_loss": round(float(-np.mean(actual * np.log(probability)
                                           + (1 - actual) * np.log(1 - probability))), 5),
        "coverage_90": round(float(np.mean(np.abs(error) <= 1.64485363 * frame["pred_std"])), 4),
    }


def _seat_holdout(calibration: HouseCalibration, frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    x = design_matrix(frame)
    coefficients = rng.multivariate_normal(
        calibration.posterior_mean, calibration.posterior_covariance, N_SIMULATIONS,
        check_valid="raise",
    )
    mean = coefficients @ x.T
    national = rng.normal(0.0, calibration.sigma_national, (N_SIMULATIONS, 1))
    regions = sorted(frame["region"].unique())
    region_index = frame["region"].map({name: idx for idx, name in enumerate(regions)}).to_numpy()
    regional = rng.normal(0.0, calibration.sigma_regional, (N_SIMULATIONS, len(regions)))
    scale = calibration.sigma_district * math.sqrt((calibration.student_df - 2) / calibration.student_df)
    local = rng.standard_t(calibration.student_df, (N_SIMULATIONS, len(frame))) * scale
    seats = np.sum(mean + national + regional[:, region_index] + local > 0, axis=1)
    actual_seats = int(np.sum(frame["margin"] > 0))
    return {
        "actual_dem_seats": actual_seats,
        "median_dem_seats": int(np.median(seats)),
        "mean_dem_seats": round(float(np.mean(seats)), 2),
        "ci_90": [int(np.percentile(seats, 5)), int(np.percentile(seats, 95))],
        "prob_dem_majority": round(float(np.mean(seats >= 218)), 4),
        "covered_90": bool(np.percentile(seats, 5) <= actual_seats <= np.percentile(seats, 95)),
    }


def run_backtest(frames: dict[int, pd.DataFrame]) -> dict[str, Any]:
    predictions = []
    seat_results = []
    legacy_values = json.loads((PROCESSED / "learned_params.json").read_text())
    for holdout, training_years in ((2022, (2018,)), (2024, (2018, 2022))):
        calibration = fit_house_calibration([frames[year] for year in training_years])
        full_test = frames[holdout].copy()
        test = full_test[full_test["contested"] & full_test["map_comparable"]].copy()
        candidate_mean = posterior_mean_margin(calibration, test)
        x = design_matrix(test)
        parameter_var = np.einsum("ij,jk,ik->i", x, calibration.posterior_covariance, x)
        candidate_std = np.sqrt(parameter_var + calibration.sigma_national**2
                                + calibration.sigma_regional**2 + calibration.sigma_district**2)
        predictions.append(_prediction_rows("house_v4", holdout, test, candidate_mean, candidate_std))
        uniform_mean = test["pvi_numeric"].to_numpy(float) + test["national_margin"].to_numpy(float)
        predictions.append(_prediction_rows("uniform_swing", holdout, test, uniform_mean,
                                           np.full(len(test), 8.0)))
        legacy_mean = 2.0 * (
            float(legacy_values["beta_pvi_mean"]) * test["pvi_numeric"].to_numpy(float)
            + float(legacy_values["beta_inc_mean"]) * test["incumbency_code"].to_numpy(float)
            + float(legacy_values["beta_national_mean"]) * test["national_margin"].to_numpy(float)
        )
        predictions.append(_prediction_rows("legacy_v3", holdout, test, legacy_mean,
                                           np.full(len(test), 2 * float(legacy_values["sigma_district"]))))
        seat_result = _seat_holdout(calibration, full_test, seed=holdout)
        # Official chamber totals are the scoring truth.  District returns are
        # still used for the race layer; this avoids treating DC or reporting
        # quirks in the returns file as a voting House seat.
        official_dem_seats = {2022: 213, 2024: 215}[holdout]
        seat_result["actual_dem_seats"] = official_dem_seats
        low, high = seat_result["ci_90"]
        seat_result["covered_90"] = bool(low <= official_dem_seats <= high)
        seat_results.append({"year": holdout, "trained_on": list(training_years), **seat_result})
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(PROCESSED / "house_backtest_predictions.csv", index=False)
    metrics = {model: _scores(group) for model, group in all_predictions.groupby("model")}
    candidate, legacy, uniform = metrics["house_v4"], metrics["legacy_v3"], metrics["uniform_swing"]
    passed = (
        candidate["brier"] <= min(legacy["brier"], uniform["brier"]) + 0.005
        and candidate["log_loss"] <= min(legacy["log_loss"], uniform["log_loss"]) + 0.01
        and abs(candidate["signed_error"]) <= 1.5
        and all(result["covered_90"] for result in seat_results)
    )
    report = {
        "status": "complete",
        "scope": "House structural model conditional on realized national House margin",
        "holdout_years": [2022, 2024],
        "excluded_2024_map_change_states": sorted(MAP_CHANGED_2024),
        "metrics": metrics,
        "seat_results": seat_results,
        "house_structural_gate": {
            "status": "production" if passed else "experimental",
            "rule": "Candidate must match the best baseline within 0.005 Brier/0.01 log loss, have <=1.5-point signed error, and cover both seat outcomes at 90%.",
        },
        "limitation": "This validates the House margin-to-seat layer, not Silver Bulletin's maintained-average calibration.",
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "house_backtest_metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    build_current_fundamentals()
    incumbency_2024 = derive_incumbency(2024)
    incumbency_2024.to_csv(HISTORICAL / "incumbency_2024.csv", index=False)
    frames = {year: historical_frame(year) for year in (2018, 2022, 2024)}
    backtest = run_backtest(frames)
    if backtest["house_structural_gate"]["status"] != "production":
        raise RuntimeError("House candidate failed the predeclared structural promotion gate")
    fitting_frames = [
        frames[2018], frames[2022],
        frames[2024][frames[2024]["map_comparable"]].copy(),
    ]
    calibration = fit_house_calibration(fitting_frames)
    (PROCESSED / "house_calibration.json").write_text(json.dumps(calibration.to_dict(), indent=2))
    print(json.dumps({
        "house_structural_gate": backtest["house_structural_gate"],
        "coefficients": dict(zip(calibration.coefficient_names, calibration.posterior_mean.round(4))),
        "sigma_national": round(calibration.sigma_national, 4),
        "sigma_regional": round(calibration.sigma_regional, 4),
        "sigma_district": round(calibration.sigma_district, 4),
        "n": calibration.n_districts_fitted,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
