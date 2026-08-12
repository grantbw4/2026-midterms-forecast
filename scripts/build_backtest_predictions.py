#!/usr/bin/env python3
"""Build leakage-resistant 2018–2024 Senate race backtest predictions.

Historical poll availability is much better for Senate than House. This script
therefore validates the candidate-race update on Senate races, while the
synthetic suite validates the same update algebra used by both chambers.
Fundamentals are refit before every holdout using earlier cycles only.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.backtesting import HORIZONS  # noqa: E402
from models.race_polling import poll_observation_variance, update_race_draws  # noqa: E402


POLL_REPOSITORY = "Jack-Whitcomb/All-US-Senate-polls-2006-2024"
POLL_COMMIT = "5aea8c5c4572ddc71f4491fa84478e70d88a5898"
POLL_URL = (
    f"https://raw.githubusercontent.com/{POLL_REPOSITORY}/{POLL_COMMIT}/"
    "2006_to_2024_senate_polls_with_actuals.csv"
)
DECISION_LABS_ELECTION_URL = "https://www.decisionlabs.ai/api/elections/{year}"
FEC_URL = "https://www.fec.gov/files/bulk-downloads/{year}/cn{yy}.zip"
TARGET_YEARS = (2018, 2020, 2022, 2024)
ALL_YEARS = tuple(range(2006, 2026, 2))
PRESIDENTIAL_BASELINE = {
    2006: 2004, 2008: 2004, 2010: 2008, 2012: 2008, 2014: 2012,
    2016: 2012, 2018: 2016, 2020: 2016, 2022: 2020, 2024: 2020,
}
ELECTION_DATES = {
    2018: date(2018, 11, 6), 2020: date(2020, 11, 3),
    2022: date(2022, 11, 8), 2024: date(2024, 11, 5),
}
STATE_TO_ABBR = {
    "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA",
    "colorado":"CO","connecticut":"CT","delaware":"DE","florida":"FL","georgia":"GA",
    "hawaii":"HI","idaho":"ID","illinois":"IL","indiana":"IN","iowa":"IA",
    "kansas":"KS","kentucky":"KY","louisiana":"LA","maine":"ME","maryland":"MD",
    "massachusetts":"MA","michigan":"MI","minnesota":"MN","mississippi":"MS","missouri":"MO",
    "montana":"MT","nebraska":"NE","nevada":"NV","new hampshire":"NH","new jersey":"NJ",
    "new mexico":"NM","new york":"NY","north carolina":"NC","north dakota":"ND","ohio":"OH",
    "oklahoma":"OK","oregon":"OR","pennsylvania":"PA","rhode island":"RI","south carolina":"SC",
    "south dakota":"SD","tennessee":"TN","texas":"TX","utah":"UT","vermont":"VT",
    "virginia":"VA","washington":"WA","west virginia":"WV","wisconsin":"WI","wyoming":"WY",
}


def _get_bytes(session: requests.Session, url: str) -> bytes:
    response = session.get(url, timeout=90)
    response.raise_for_status()
    return response.content


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_poll_archive(content: bytes) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(content))
    frame = frame[frame["year"].isin(ALL_YEARS)].copy()
    frame["date"] = pd.to_datetime(frame["end_date"], errors="coerce")
    for column in ("dem", "rep", "dem_actual", "rep_actual", "sample_size"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "dem", "rep", "dem_actual", "rep_actual"])
    frame["state"] = frame["state"].str.lower().map(STATE_TO_ABBR)
    frame = frame.dropna(subset=["state"])
    frame["sample_size"] = frame["sample_size"].fillna(frame["sample_size"].median()).clip(lower=100)
    frame["population"] = frame["sample_type"].str.lower().map({
        "lv": "likely_voters", "rv": "registered_voters", "a": "adults",
    }).fillna("adults")
    frame["margin"] = frame["dem"] - frame["rep"]
    frame["actual_margin"] = frame["dem_actual"] - frame["rep_actual"]
    frame["partisan"] = None
    frame["margin_of_error"] = np.nan
    # Exclude rows explicitly marked as overlapping special elections. Their
    # source archive does not provide a D-v-R general-election actual.
    frame = frame[frame["overlapping_special_election"].isna()]
    return frame.reset_index(drop=True)


def state_leans(session: requests.Session) -> tuple[dict[tuple[int, str], float], dict[str, str]]:
    output: dict[tuple[int, str], float] = {}
    checksums: dict[str, str] = {}
    for year in sorted(set(PRESIDENTIAL_BASELINE.values())):
        raw = _get_bytes(session, DECISION_LABS_ELECTION_URL.format(year=year))
        checksums[str(year)] = _sha(raw)
        payload = json.loads(raw)
        national = payload["national"]["popular_vote_pct"]
        national_margin = float(national["democrat"]) - float(national["republican"])
        for state in payload["states"]:
            output[(year, state["abbr"])] = float(state["margin"]) - national_margin
    return output, checksums


def historical_incumbency(session: requests.Session) -> tuple[dict[tuple[int, str], int], dict[str, str]]:
    output: dict[tuple[int, str], int] = {}
    checksums: dict[str, str] = {}
    columns = [
        "candidate_id", "name", "party", "election_year", "state", "office", "district",
        "incumbent_challenge", "status", "committee_id", "street_1", "street_2", "city",
        "mailing_state", "zip_code",
    ]
    for year in ALL_YEARS:
        url = FEC_URL.format(year=year, yy=str(year)[-2:])
        raw = _get_bytes(session, url)
        checksums[str(year)] = _sha(raw)
        with ZipFile(BytesIO(raw)) as archive:
            text = archive.read(archive.namelist()[0]).decode("latin-1")
        frame = pd.read_csv(StringIO(text), sep="|", names=columns, dtype=str)
        frame = frame[(frame["office"] == "S") & (frame["incumbent_challenge"] == "I")]
        for state, group in frame.groupby("state"):
            parties = group["party"].str.upper().map({"DEM": 1, "DFL": 1, "REP": -1, "GOP": -1}).dropna().unique()
            output[(year, state)] = int(parties[0]) if len(parties) == 1 else 0
    return output, checksums


def race_panel(polls: pd.DataFrame, leans: dict[tuple[int, str], float], incumbency: dict[tuple[int, str], int]) -> pd.DataFrame:
    races = polls.groupby(["year", "state"], as_index=False).agg(actual_margin=("actual_margin", "first"))
    races["state_lean"] = [leans[(PRESIDENTIAL_BASELINE[int(y)], s)] for y, s in zip(races.year, races.state)]
    races["incumbency"] = [incumbency.get((int(y), s), 0) for y, s in zip(races.year, races.state)]
    races["race_id"] = races["state"] + "-SEN"
    return races


def fit_robust_fundamentals(training: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Student-t IRLS approximation with regularized Bayesian priors."""
    x = np.column_stack([np.ones(len(training)), training["state_lean"], training["incumbency"]])
    y = training["actual_margin"].to_numpy(float)
    prior_mean = np.array([0.0, 1.0, 0.0])
    prior_std = np.array([8.0, 0.5, 5.0])
    beta = prior_mean.copy()
    sigma = 8.0
    precision = np.diag(1.0 / prior_std**2)
    for _ in range(20):
        residual = y - x @ beta
        weight = (5.0 + 1.0) / (5.0 + (residual / max(sigma, 1.0)) ** 2)
        posterior_precision = (x.T * weight) @ x / sigma**2 + precision
        posterior_cov = np.linalg.inv(posterior_precision)
        beta = posterior_cov @ ((x.T @ (weight * y)) / sigma**2 + precision @ prior_mean)
        sigma = max(2.5, float(1.4826 * np.median(np.abs(residual - np.median(residual)))))
    return beta, posterior_cov, sigma


def pollster_effects(training_polls: pd.DataFrame) -> dict[str, tuple[float, float]]:
    work = training_polls.copy()
    work["error"] = work["margin"] - work["actual_margin"]
    effects: dict[str, tuple[float, float]] = {}
    for pollster, group in work.groupby("pollster"):
        n = len(group)
        effect = n / (n + 5.0) * float(group["error"].mean())
        std = max(0.5, float(group["error"].std(ddof=1) if n > 1 else 2.5) / np.sqrt(n + 2.0))
        effects[str(pollster)] = (effect, std)
    if effects:
        center = float(np.mean([effect for effect, _ in effects.values()]))
        effects = {name: (effect - center, std) for name, (effect, std) in effects.items()}
    return effects


def general_election_day(year: int) -> date:
    """First Tuesday after the first Monday in November."""
    candidate = date(year, 11, 2)
    while candidate.weekday() != 1:
        candidate = date.fromordinal(candidate.toordinal() + 1)
    return candidate


def estimate_correlated_error_floor(training_polls: pd.DataFrame) -> float:
    """Estimate irreducible race-poll error using prior cycles only."""
    errors: list[float] = []
    for (year, _state), group in training_polls.groupby(["year", "state"]):
        election_day = pd.Timestamp(general_election_day(int(year)))
        recent = group[(group["date"] <= election_day) & (group["date"] >= election_day - pd.Timedelta(days=60))]
        if recent.empty:
            continue
        # Cap each pollster's representation so prolific houses do not define
        # the calibration target.
        house_means = recent.groupby("pollster", as_index=False)["margin"].mean()
        aggregate = float(house_means["margin"].median())
        errors.append(aggregate - float(group["actual_margin"].iloc[0]))
    if len(errors) < 10:
        return 5.0
    values = np.asarray(errors)
    robust_sd = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    return float(np.clip(robust_sd, 3.5, 7.0))


def prior_draws(row: pd.Series, beta: np.ndarray, covariance: np.ndarray, sigma: float, rng: np.random.Generator, n: int) -> np.ndarray:
    coefficients = rng.multivariate_normal(beta, covariance, size=n)
    x = np.array([1.0, float(row["state_lean"]), float(row["incumbency"])])
    structural = rng.standard_t(5.0, n) * sigma * np.sqrt(3.0 / 5.0)
    return coefficients @ x + structural


def polls_only_summary(polls: pd.DataFrame, election_day: date) -> tuple[float, float, float]:
    observations = polls["margin"].to_numpy(float) - polls["pollster_effect"].to_numpy(float)
    variances = np.array([poll_observation_variance(row, election_day) for _, row in polls.iterrows()])
    mean = float(np.average(observations, weights=1.0 / variances))
    for _ in range(5):
        standardized = (observations - mean) / np.sqrt(variances + 4.0)
        robust = (5.0 + 1.0) / (5.0 + standardized**2)
        weights = robust / variances
        mean = float(np.average(observations, weights=weights))
    std = float(np.sqrt(1.0 / weights.sum() + 2.0**2))
    # Logistic-normal approximation is sufficiently accurate for scoring.
    probability = float(1.0 / (1.0 + np.exp(-mean / max(std * 1.7, 1e-6))))
    return mean, std, probability


def build_predictions(polls: pd.DataFrame, races: pd.DataFrame, n_draws: int = 6000) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for holdout in TARGET_YEARS:
        training = races[races["year"] < holdout]
        beta, covariance, sigma = fit_robust_fundamentals(training)
        training_polls = polls[polls["year"] < holdout]
        effects = pollster_effects(training_polls)
        correlated_floor = estimate_correlated_error_floor(training_polls)
        for _, race in races[races["year"] == holdout].iterrows():
            race_polls = polls[(polls["year"] == holdout) & (polls["state"] == race["state"])].copy()
            for horizon in HORIZONS:
                cutoff = pd.Timestamp(ELECTION_DATES[holdout]) - pd.Timedelta(days=horizon)
                available = race_polls[(race_polls["date"] <= cutoff) & (race_polls["date"] >= cutoff - pd.Timedelta(days=240))].copy()
                seed = holdout * 100000 + horizon * 100 + sum(map(ord, race["state"]))
                rng = np.random.default_rng(seed)
                prior = prior_draws(race, beta, covariance, sigma, rng, n_draws)
                for model in ("fundamentals", "v2"):
                    rows.append({
                        "year": holdout, "horizon": horizon, "model": model,
                        "race_id": race["race_id"], "actual_margin": race["actual_margin"],
                        "pred_mean": float(np.mean(prior)), "pred_std": float(np.std(prior)),
                        "prob_dem": float(np.mean(prior > 0)), "polls_available": int(len(available)),
                        "correlated_error_floor": correlated_floor,
                    })
                if available.empty:
                    continue
                available["pollster_effect"] = available["pollster"].map(lambda name: effects.get(str(name), (0.0, 1.0))[0])
                available["pollster_std"] = available["pollster"].map(lambda name: effects.get(str(name), (0.0, 1.0))[1])
                posterior = update_race_draws(
                    prior, available, rng, election_date=ELECTION_DATES[holdout],
                    correlated_error_floor=correlated_floor,
                )
                rows.append({
                    "year": holdout, "horizon": horizon, "model": "v3",
                    "race_id": race["race_id"], "actual_margin": race["actual_margin"],
                    "pred_mean": float(np.mean(posterior)), "pred_std": float(np.std(posterior)),
                    "prob_dem": float(np.mean(posterior > 0)), "polls_available": int(len(available)),
                    "correlated_error_floor": correlated_floor,
                })
                mean, std, probability = polls_only_summary(available, ELECTION_DATES[holdout])
                rows.append({
                    "year": holdout, "horizon": horizon, "model": "polls_only",
                    "race_id": race["race_id"], "actual_margin": race["actual_margin"],
                    "pred_mean": mean, "pred_std": std, "prob_dem": probability,
                    "polls_available": int(len(available)),
                    "correlated_error_floor": correlated_floor,
                })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/backtests/predictions.csv")
    parser.add_argument("--draws", type=int, default=6000)
    args = parser.parse_args()
    session = requests.Session()
    poll_bytes = _get_bytes(session, POLL_URL)
    polls = load_poll_archive(poll_bytes)
    leans, election_checksums = state_leans(session)
    incumbency, fec_checksums = historical_incumbency(session)
    races = race_panel(polls, leans, incumbency)
    predictions = build_predictions(polls, races, n_draws=args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    production_floor = estimate_correlated_error_floor(polls)
    calibration_path = PROJECT_ROOT / "data/processed/race_poll_calibration.json"
    calibration_path.write_text(json.dumps({
        "correlated_error_floor": round(production_floor, 4),
        "estimated_from_cycles": list(ALL_YEARS),
        "method": "robust MAD of final-60-day pollster-balanced race-average errors",
        "source_commit": POLL_COMMIT,
    }, indent=2))
    provenance = {
        "scope": "Senate candidate-race polling layer",
        "evaluated_years": list(TARGET_YEARS),
        "horizons": list(HORIZONS),
        "poll_source": POLL_URL,
        "poll_source_commit": POLL_COMMIT,
        "poll_source_sha256": _sha(poll_bytes),
        "poll_source_license": "No repository license declared; raw archive is not redistributed",
        "state_lean_source": "Decision Labs presidential election API (CC BY 4.0)",
        "state_lean_sha256": election_checksums,
        "incumbency_source": "FEC candidate master bulk archives (U.S. government data)",
        "incumbency_sha256": fec_checksums,
        "method": "rolling-origin; fundamentals fit only on cycles earlier than holdout",
        "rows": int(len(predictions)),
        "production_correlated_error_floor": round(production_floor, 4),
    }
    (args.output.parent / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"Wrote {len(predictions)} predictions across {predictions.race_id.nunique()} races")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
