"""Posterior-predictive House and Senate simulation for the production forecast.

This module deliberately contains no PyMC calls.  Historical parameters are
fit offline; daily production composes their posterior draws with the national
state-space posterior and race-poll likelihood.  That is both faster and more
honest than running MCMC over unobserved forecast-year vote shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .dynamic_polling import DynamicPollingResult
from .house_model import HouseCalibration, design_matrix
from .race_polling import CandidateRegistry, update_race_draws


REGIONS = [
    "New_England", "Mid_Atlantic_Northeast", "Rust_Belt", "Southeast",
    "Deep_South", "Texas_Region", "Plains", "Mountain", "Southwest", "Pacific",
]
STATE_TO_REGION = {
    **{s: "New_England" for s in ["ME", "NH", "VT", "MA"]},
    **{s: "Mid_Atlantic_Northeast" for s in ["NY", "NJ", "DE", "MD", "RI", "CT"]},
    **{s: "Rust_Belt" for s in ["IL", "IN", "OH", "MI", "WI", "PA", "MN", "IA"]},
    **{s: "Southeast" for s in ["FL", "GA", "NC", "VA"]},
    **{s: "Deep_South" for s in ["SC", "AL", "MS", "AR", "TN", "KY", "WV", "MO"]},
    **{s: "Texas_Region" for s in ["TX", "OK", "LA"]},
    **{s: "Plains" for s in ["ND", "SD", "NE", "KS"]},
    **{s: "Mountain" for s in ["ID", "MT", "WY", "UT", "AK"]},
    **{s: "Southwest" for s in ["AZ", "NV", "NM", "CO"]},
    **{s: "Pacific" for s in ["CA", "OR", "WA", "HI"]},
}


SENATE_RACES = [
    # state, incumbent, party, state lean (positive is Democratic), open, special
    ("GA", "Jon Ossoff", "D", 0, False, False),
    ("MI", "Open seat", "", 1, True, False),
    ("NH", "Open seat", "", 2, True, False),
    ("VA", "Mark Warner", "D", 3, False, False),
    ("CO", "John Hickenlooper", "D", 5, False, False),
    ("IL", "Open seat", "", 8, True, False),
    ("OR", "Jeff Merkley", "D", 6, False, False),
    ("NM", "Martin Heinrich", "D", 5, False, False),
    ("MN", "Open seat", "", 3, True, False),
    ("DE", "Open seat", "", 7, True, False),
    ("MA", "Ed Markey", "D", 20, False, False),
    ("RI", "Sheldon Whitehouse", "D", 12, False, False),
    ("NJ", "Cory Booker", "D", 7, False, False),
    ("NC", "Open seat", "", -2, True, False),
    ("ME", "Susan Collins", "R", 3, False, False),
    ("IA", "Open seat", "", -6, True, False),
    ("TX", "Open seat", "", -6, True, False),
    ("AK", "Dan Sullivan", "R", -8, False, False),
    ("SC", "Lindsey Graham", "R", -8, False, False),
    ("LA", "Bill Cassidy", "R", -12, False, False),
    ("KY", "Open seat", "", -16, True, False),
    ("AR", "Tom Cotton", "R", -17, False, False),
    ("OK", "Markwayne Mullin", "R", -20, False, False),
    ("SD", "Mike Rounds", "R", -18, False, False),
    ("KS", "Roger Marshall", "R", -10, False, False),
    ("ID", "Jim Risch", "R", -22, False, False),
    ("AL", "Tommy Tuberville", "R", -15, False, False),
    ("MS", "Cindy Hyde-Smith", "R", -10, False, False),
    ("TN", "Bill Hagerty", "R", -15, False, False),
    ("NE", "Pete Ricketts", "R", -14, False, False),
    ("WV", "Shelley Moore Capito", "R", -28, False, False),
    ("WY", "John Barrasso", "R", -40, False, False),
    ("MT", "Steve Daines", "R", -12, False, False),
    ("FL", "Ashley Moody", "R", -6, False, True),
    ("OH", "Jon Husted", "R", -6, False, True),
]


@dataclass(frozen=True)
class ChamberParameters:
    beta_lean_mean: float
    beta_lean_std: float
    beta_inc_mean: float
    beta_inc_std: float
    beta_national_mean: float
    beta_national_std: float
    sigma_regional: float
    sigma_race: float
    source: str


def _house_parameters(data_dir: Path) -> HouseCalibration:
    path = Path(data_dir) / "processed" / "house_calibration.json"
    if not path.exists():
        raise FileNotFoundError("Missing fitted House margin posterior: house_calibration.json")
    return HouseCalibration.load(path)


def _senate_parameters(data_dir: Path) -> ChamberParameters:
    path = Path(data_dir) / "processed" / "learned_params_senate.json"
    if path.exists():
        values = json.loads(path.read_text())
        return ChamberParameters(**values, source="senate_historical_posterior")
    # Separate, deliberately wider regularized Senate prior.  It is not a reuse
    # of House residual variance and is identified as prior-only in metadata.
    return ChamberParameters(0.50, 0.08, 2.5, 0.8, 0.50, 0.10, 1.2, 5.5,
                             "senate_regularized_prior_pending_backtest")


def validate_house_fundamentals(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "district_id", "state", "district_number", "pvi", "incumbent_party", "open_seat",
        "pvi_source", "pvi_source_url", "pvi_effective_date", "lean_quality",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"House fundamentals missing columns: {sorted(missing)}")
    if len(frame) != 435 or frame["district_id"].nunique() != 435:
        raise ValueError("House fundamentals must contain 435 unique districts")
    work = frame.copy()
    work["pvi"] = pd.to_numeric(work["pvi"], errors="raise")
    if not work["pvi"].between(-50, 50).all():
        raise ValueError("House partisan lean outside [-50, 50]")
    provenance_columns = ["pvi_source", "pvi_source_url", "pvi_effective_date", "lean_quality"]
    if work[provenance_columns].isna().any().any():
        raise ValueError("House fundamentals contain missing partisan-lean provenance")
    if not work["pvi_source_url"].astype(str).str.startswith(("http://", "https://")).all():
        raise ValueError("House fundamentals contain invalid partisan-lean source URLs")
    if "region" not in work:
        work["region"] = work["state"].map(STATE_TO_REGION)
    work["fundamentals_source"] = work.get("pvi_source", pd.Series(index=work.index, dtype=object)).fillna(
        "2024 presidential district lean"
    )
    work["fundamentals_effective_date"] = date.today().isoformat()
    placeholders = int(work.get("incumbent", pd.Series(dtype=str)).astype(str).str.startswith("Rep. ").sum())
    return work, {
        "status": "warning" if placeholders else "passed",
        "rows": 435,
        "placeholder_incumbent_names": placeholders,
        "pvi_missing": int(work["pvi"].isna().sum()),
        "lean_provenance_complete": True,
        "rating_proxy_districts": int((work["lean_quality"] == "rating_proxy").sum()),
    }


def senate_fundamentals() -> pd.DataFrame:
    open_seat_holders = {
        "MI": "D", "NH": "D", "IL": "D", "MN": "D", "DE": "D",
        "NC": "R", "IA": "R", "TX": "R", "KY": "R",
    }
    rows = []
    for state, incumbent, party, lean, open_seat, special in SENATE_RACES:
        rows.append({
            "race_id": state, "state": state, "district_number": 0,
            "pvi": float(lean), "incumbent": incumbent, "incumbent_party": party,
            "seat_held_by": party or open_seat_holders[state],
            "open_seat": open_seat, "special": special,
            "region": STATE_TO_REGION[state],
            "fundamentals_source": "2024 state partisan lean + FEC incumbency",
            "fundamentals_effective_date": date.today().isoformat(),
        })
    return pd.DataFrame(rows)


def _category(probability: float) -> str:
    if probability >= 0.85:
        return "safe_d"
    if probability >= 0.70:
        return "likely_d"
    if probability >= 0.55:
        return "lean_d"
    if probability >= 0.45:
        return "toss_up"
    if probability >= 0.30:
        return "lean_r"
    if probability >= 0.15:
        return "likely_r"
    return "safe_r"


def _simulate_chamber(
    fundamentals: pd.DataFrame,
    id_column: str,
    params: Any,
    national: DynamicPollingResult,
    race_polls: pd.DataFrame,
    race_status: dict[str, Any],
    n_simulations: int,
    random_seed: int,
    chamber: str,
    correlated_error_floor: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(random_seed)
    n_races = len(fundamentals)
    ids = fundamentals[id_column].astype(str).tolist()
    lean = fundamentals["pvi"].to_numpy(float)
    incumbency = fundamentals["incumbent_party"].map({"D": 1, "R": -1}).fillna(0).to_numpy(float, copy=True)
    incumbency[fundamentals["open_seat"].astype(bool).to_numpy()] = 0
    region_index = fundamentals["region"].map({name: idx for idx, name in enumerate(REGIONS)}).to_numpy(int)

    national_samples = np.resize(national.election_samples, n_simulations)
    if chamber == "house":
        if not isinstance(params, HouseCalibration):
            raise TypeError("House simulation requires HouseCalibration")
        parameter_draws = rng.multivariate_normal(
            params.posterior_mean, params.posterior_covariance, n_simulations,
            check_valid="raise",
        )
        design_frame = pd.DataFrame({
            "pvi_numeric": lean,
            "incumbency_code": incumbency,
            # National margin varies by simulation and is added below.
            "national_margin": np.zeros(n_races),
            "region": fundamentals["region"].astype(str).to_numpy(),
        })
        base_margin = parameter_draws @ design_matrix(design_frame).T
        national_coefficient = parameter_draws[:, params.coefficient_names.index("national")]
        national_error = rng.normal(0.0, params.sigma_national, n_simulations)
        regional_error = rng.normal(0.0, params.sigma_regional,
                                    (n_simulations, len(REGIONS)))
        df = params.student_df
        race_scale = params.sigma_district * np.sqrt((df - 2.0) / df)
        local = rng.standard_t(df, (n_simulations, n_races)) * race_scale
        # Rating-category proxies are less informative than a district PVI.
        proxy = fundamentals.get("lean_quality", pd.Series("partisan_lean", index=fundamentals.index))
        proxy_scale = np.where(proxy.astype(str).to_numpy() == "rating_proxy", 2.5, 0.0)
        proxy_error = rng.normal(0.0, proxy_scale[None, :], (n_simulations, n_races))
        prior_margins = (
            base_margin
            + national_coefficient[:, None] * national_samples[:, None]
            + national_error[:, None]
            + regional_error[:, region_index]
            + local
            + proxy_error
        )
        prior_votes = 50.0 + prior_margins / 2.0
    else:
        beta_lean = rng.normal(params.beta_lean_mean, params.beta_lean_std, n_simulations)
        beta_inc = rng.normal(params.beta_inc_mean, params.beta_inc_std, n_simulations)
        beta_national = rng.normal(params.beta_national_mean, params.beta_national_std, n_simulations)
        regional = rng.normal(0.0, params.sigma_regional, (n_simulations, len(REGIONS)))
        df = 5.0
        race_scale = params.sigma_race * np.sqrt((df - 2.0) / df)
        local = rng.standard_t(df, (n_simulations, n_races)) * race_scale
        prior_votes = (
            50.0
            + beta_lean[:, None] * lean[None, :]
            + beta_inc[:, None] * incumbency[None, :]
            + beta_national[:, None] * national_samples[:, None]
            + regional[:, region_index]
            + local
        )
    posterior_votes = prior_votes.copy()
    output: list[dict[str, Any]] = []

    for idx, race_id in enumerate(ids):
        polls = race_polls[race_polls["race_id"] == race_id] if not race_polls.empty else pd.DataFrame()
        if not polls.empty:
            polls = polls.copy()
            polls["pollster_effect"] = polls["pollster"].map(national.pollster_effects).fillna(0.0)
            polls["pollster_std"] = 0.8
            prior_margins = 2.0 * (prior_votes[:, idx] - 50.0)
            updated_margins = update_race_draws(
                prior_margins, polls, rng,
                correlated_error_floor=correlated_error_floor,
            )
            posterior_votes[:, idx] = 50.0 + updated_margins / 2.0
        prior_margin = 2.0 * (float(np.mean(prior_votes[:, idx])) - 50.0)
        posterior_margin_draws = 2.0 * (posterior_votes[:, idx] - 50.0)
        posterior_margin = float(np.mean(posterior_margin_draws))
        prob_dem = float(np.mean(posterior_votes[:, idx] > 50.0))
        status = race_status.get(race_id, {})
        data_quality = status.get("status", "fundamentals_only")
        row = fundamentals.iloc[idx]
        item = {
            "id": race_id,
            "state": str(row["state"]),
            "district_number": int(row.get("district_number", 0)),
            "incumbent": {
                "name": str(row.get("incumbent", "Unknown")),
                "party": str(row.get("incumbent_party", "")),
            },
            "pvi": float(row["pvi"]),
            "pvi_source": str(row.get("pvi_source", row.get("fundamentals_source", "unknown"))),
            "pvi_source_url": str(row.get("pvi_source_url", "")) or None,
            "pvi_effective_date": str(row.get("pvi_effective_date", row.get("fundamentals_effective_date", "unknown"))),
            "lean_quality": str(row.get("lean_quality", "partisan_lean")),
            "region": str(row["region"]),
            "open_seat": bool(row["open_seat"]),
            "fundamentals_source": str(row.get("fundamentals_source", "unknown")),
            "fundamentals_effective_date": str(row.get("fundamentals_effective_date", "unknown")),
            "prior_margin": round(prior_margin, 2),
            "posterior_margin": round(posterior_margin, 2),
            "credible_interval_90": [
                round(float(np.percentile(posterior_margin_draws, 5)), 2),
                round(float(np.percentile(posterior_margin_draws, 95)), 2),
            ],
            "prob_dem": round(prob_dem, 4),
            "polling_adjustment": round(posterior_margin - prior_margin, 2),
            "polls_used": int(len(polls)),
            "polling_likelihoods_used": int(len(polls)),
            "polling_input_type": "silver_maintained_average" if not polls.empty else None,
            "latest_poll_date": (
                pd.to_datetime(polls["date"]).max().strftime("%Y-%m-%d") if not polls.empty else None
            ),
            "poll_sources": sorted({
                str(url) for url in polls.get("source_url", pd.Series(dtype=str)).dropna()
                if str(url).startswith(("http://", "https://"))
            }),
            "data_quality": data_quality,
            "category": _category(prob_dem),
        }
        if chamber == "house":
            item.update({
                "mean_vote_share": round(float(np.mean(posterior_votes[:, idx])), 2),
                "std_vote_share": round(float(np.std(posterior_votes[:, idx])), 2),
                "ci_90_low": round(float(np.percentile(posterior_votes[:, idx], 5)), 2),
                "ci_90_high": round(float(np.percentile(posterior_votes[:, idx], 95)), 2),
            })
        else:
            item.update({
                "incumbent": str(row.get("incumbent", "Unknown")),
                "incumbent_party": str(row.get("incumbent_party", "")),
                "special": bool(row.get("special", False)),
            })
        output.append(item)

    posterior_votes = np.clip(posterior_votes, 0.0, 100.0)
    output.sort(key=lambda item: abs(item["prob_dem"] - 0.5))
    return posterior_votes, output


def _seat_distribution(seats: np.ndarray) -> dict[str, list[Any]]:
    values, counts = np.unique(seats, return_counts=True)
    return {"dem_seats": values.astype(int).tolist(), "probabilities": (counts / len(seats)).tolist()}


def _categories(races: list[dict[str, Any]], nested: bool) -> dict[str, Any]:
    counts = {name: sum(race["category"] == name for race in races) for name in
              ["safe_d", "likely_d", "lean_d", "toss_up", "lean_r", "likely_r", "safe_r"]}
    if not nested:
        return counts
    return {
        "dem": {"safe": counts["safe_d"], "likely": counts["likely_d"], "lean": counts["lean_d"]},
        "toss_up": counts["toss_up"],
        "rep": {"safe": counts["safe_r"], "likely": counts["likely_r"], "lean": counts["lean_r"]},
    }


def run_chamber_forecast(
    data_dir: Path,
    national: DynamicPollingResult,
    senate_national: Optional[DynamicPollingResult] = None,
    race_polls: Optional[pd.DataFrame] = None,
    race_poll_status: Optional[dict[str, Any]] = None,
    registry: Optional[CandidateRegistry] = None,
    n_simulations: int = 10_000,
    random_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    data_dir = Path(data_dir)
    senate_environment = senate_national or national
    polls = race_polls if race_polls is not None else pd.DataFrame()
    statuses = (race_poll_status or {}).get("races", {})

    house, fundamentals_diagnostics = validate_house_fundamentals(
        pd.read_csv(data_dir / "processed" / "districts.csv")
    )
    senate = senate_fundamentals()
    if registry is not None:
        house = registry.update_fundamentals(house, "district_id")
        senate = registry.update_fundamentals(senate, "race_id")
        unresolved = int(house["incumbent"].astype(str).str.startswith("Rep. ").sum())
        fundamentals_diagnostics.update({
            "status": "passed" if unresolved == 0 else "warning",
            "placeholder_incumbent_names": unresolved,
            "open_seats": int(house["open_seat"].sum()),
            "incumbent_source": "Clerk of the House + FEC candidate master",
        })
    calibration_path = data_dir / "processed" / "race_poll_calibration.json"
    if calibration_path.exists():
        race_poll_calibration = json.loads(calibration_path.read_text())
        correlated_error_floor = float(race_poll_calibration["correlated_error_floor"])
        calibration_source = str(race_poll_calibration.get("method", "historical calibration"))
    else:
        correlated_error_floor = 5.0
        calibration_source = "regularized default"

    house_votes, house_races = _simulate_chamber(
        house, "district_id", _house_parameters(data_dir), national, polls, statuses,
        n_simulations, random_seed, "house", correlated_error_floor,
    )
    senate_votes, senate_races = _simulate_chamber(
        senate, "race_id", _senate_parameters(data_dir), senate_environment, polls, statuses,
        n_simulations, random_seed + 1, "senate", correlated_error_floor,
    )
    house_seats = np.sum(house_votes > 50.0, axis=1)
    senate_seats = 34 + np.sum(senate_votes > 50.0, axis=1)

    house_summary = {
        "prob_dem_majority": round(float(np.mean(house_seats >= 218)), 4),
        "prob_rep_majority": round(float(np.mean(house_seats < 218)), 4),
        "median_dem_seats": int(np.median(house_seats)),
        "median_rep_seats": 435 - int(np.median(house_seats)),
        "mean_dem_seats": round(float(np.mean(house_seats)), 2),
        "ci_90_low": int(np.percentile(house_seats, 5)),
        "ci_90_high": int(np.percentile(house_seats, 95)),
        "ci_50_low": int(np.percentile(house_seats, 25)),
        "ci_50_high": int(np.percentile(house_seats, 75)),
        "election_day_national_margin": round(national.election_mean, 2),
        "national_likelihood_margin": round(float(
            national.diagnostics.get("generic_average_calibration", {}).get(
                "weighted_average_margin", national.current_mean
            )
        ), 2),
        "poll_updated_current_margin": round(national.current_mean, 2),
        "model_type": "house_margin_hierarchical_bayesian",
    }
    senate_summary = {
        "prob_dem_control": round(float(np.mean(senate_seats >= 51)), 4),
        "prob_rep_control": round(float(np.mean(senate_seats < 51)), 4),
        "median_dem_seats": int(np.median(senate_seats)),
        "mean_dem_seats": round(float(np.mean(senate_seats)), 2),
        "ci_90_low": int(np.percentile(senate_seats, 5)),
        "ci_90_high": int(np.percentile(senate_seats, 95)),
        "seats_up": len(senate),
        "dem_defending": int((senate["seat_held_by"] == "D").sum()),
        "rep_defending": int((senate["seat_held_by"] == "R").sum()),
        "election_day_national_margin": round(senate_environment.election_mean, 2),
        "national_uncertainty": round(senate_environment.election_std, 2),
        "national_likelihood_margin": round(senate_environment.current_mean, 2),
        "model_type": "senate_bayesian_external_average",
        "dem_not_up": 34,
        "rep_not_up": 31,
    }
    house_output = {
        "summary": house_summary,
        "categories": _categories(house_races, nested=True),
        "seat_distribution": _seat_distribution(house_seats),
        "districts": house_races,
    }
    senate_output = {
        "summary": senate_summary,
        "categories": _categories(senate_races, nested=False),
        "seat_distribution": _seat_distribution(senate_seats),
        "races": senate_races,
    }
    diagnostics = {
        "fundamentals": fundamentals_diagnostics,
        "house_parameter_source": _house_parameters(data_dir).source,
        "house_parameter_scale": "democratic_two_party_margin_points",
        "house_parameter_years": list(_house_parameters(data_dir).years_used),
        "senate_parameter_source": _senate_parameters(data_dir).source,
        "race_polling": (race_poll_status or {}).get("summary", {}),
        "current_polling_layer_status": "external_unvalidated",
        "race_poll_correlated_error_floor": round(correlated_error_floor, 3),
        "race_poll_calibration_source": calibration_source,
    }
    return house_output, senate_output, diagnostics
