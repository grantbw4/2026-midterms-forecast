#!/usr/bin/env python3
"""Generate the v3 Bayesian House and Senate forecast.

All inputs are validated before either public forecast is replaced.  Network
failure can use a recent cache, but stale or invalid data never becomes a fake
zero-valued forecast.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.dynamic_polling import (  # noqa: E402
    DynamicNationalModel,
    build_fundamentals_prior,
    summarize_approval,
)
from models.economic_fundamentals import EconomicFundamentals  # noqa: E402
from models.forecast_v3 import run_v3_forecast  # noqa: E402
from models.race_polling import (  # noqa: E402
    load_cached_candidate_registry,
    refresh_candidate_registry,
)
from models.silver_bulletin import (  # noqa: E402
    SilverBulletinClient,
    load_silver_cache,
    prepare_silver_averages,
    save_silver_cache,
)
from scripts.fetch_votehub import VoteHubFetcher  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WEBSITE_DIR = PROJECT_ROOT / "website"
ELECTION_DATE = date(2026, 11, 3)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_cached_approval() -> pd.DataFrame:
    polling = DATA_DIR / "raw" / "polling"
    approval_path = polling / "trump_approval.csv"
    return pd.read_csv(approval_path) if approval_path.exists() else pd.DataFrame()


def _fetch_approval(skip_fetch: bool) -> tuple[pd.DataFrame, list[str]]:
    fallbacks: list[str] = []
    if skip_fetch:
        return _load_cached_approval(), ["approval_cache_requested"]

    fetcher = VoteHubFetcher()
    approval = fetcher.fetch_trump_approval(days_back=240)
    if approval.empty:
        approval = _load_cached_approval()
        fallbacks.append("votehub_approval_cache")
    fetcher.save_polls(pd.DataFrame(), approval)
    return approval, fallbacks


def _fetch_registry(skip_fetch: bool) -> tuple[Any, list[str]]:
    if skip_fetch:
        registry = load_cached_candidate_registry(DATA_DIR)
        if registry is None:
            raise FileNotFoundError("No cached official candidate registry")
        return registry, ["candidate_registry_cache_requested"]
    try:
        registry = refresh_candidate_registry(
            DATA_DIR, fec_api_key=os.getenv("FEC_API_KEY", "DEMO_KEY")
        )
        return registry, []
    except Exception as exc:
        logger.warning("Candidate refresh failed; checking last valid cache: %s", exc)
        registry = load_cached_candidate_registry(DATA_DIR)
        if registry is None:
            raise
        return registry, ["candidate_registry_cache_after_fetch_failure"]


def _fetch_silver_averages(skip_fetch: bool, registry: Any) -> tuple[Any, list[str]]:
    if skip_fetch:
        return load_silver_cache(DATA_DIR), ["silver_average_cache_requested"]
    try:
        data = prepare_silver_averages(SilverBulletinClient().fetch(), registry)
        save_silver_cache(DATA_DIR, data)
        return data, []
    except Exception as exc:
        logger.warning("Silver Bulletin average refresh failed; checking cache: %s", exc)
        return load_silver_cache(DATA_DIR), ["silver_average_cache_after_fetch_failure"]


def _economic_index() -> tuple[float, list[str]]:
    try:
        model = EconomicFundamentals()
        model.load_data()
        result = model.calculate_index()
        return float(result["normalized_index"]), []
    except Exception as exc:
        logger.warning("Economic prior input unavailable: %s", exc)
        return 0.0, ["economic_prior_unavailable"]


def format_polling_data(generic: pd.DataFrame, approval: pd.DataFrame, limit: int = 50) -> dict[str, Any]:
    output: dict[str, Any] = {"generic_ballot": [], "approval": []}
    for _, row in generic.sort_values("date", ascending=False).head(limit).iterrows():
        sample_size = pd.to_numeric(row.get("sample_size"), errors="coerce")
        output["generic_ballot"].append({
            "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
            "pollster": str(row["pollster"]),
            "sample_size": int(sample_size) if pd.notna(sample_size) else None,
            "population": str(row.get("population", "a")),
            "dem_pct": round(float(row.get("dem_pct", np.nan)), 1),
            "rep_pct": round(float(row.get("rep_pct", np.nan)), 1),
            "margin": round(float(row["margin"]), 1),
        })
    if approval is not None and not approval.empty:
        for _, row in approval.sort_values("date", ascending=False).head(limit).iterrows():
            output["approval"].append({
                "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "pollster": str(row["pollster"]),
                "sample_size": int(row["sample_size"]),
                "population": str(row.get("population", "a")),
                "approve": round(float(row["approve"]), 1),
                "disapprove": round(float(row["disapprove"]), 1),
                "net_approval": round(float(row["net_approval"]), 1),
            })
    return output


def _metadata(
    national: Any,
    diagnostics: dict[str, Any],
    fallbacks: list[str],
    n_simulations: int,
    race_polls: pd.DataFrame,
    allow_stale: bool,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    data_date = date.fromisoformat(national.data_through)
    age = (now.date() - data_date).days
    stale = age > 21
    warnings = []
    if stale:
        warnings.append(f"Generic-ballot feed is {age} days old")
    if diagnostics["fundamentals"].get("status") != "passed":
        warnings.append("Some incumbent names remain unresolved")
    if not race_polls.empty:
        race_date = pd.to_datetime(race_polls["date"]).max().date()
        race_age = (now.date() - race_date).days
    else:
        race_date, race_age = None, None
        warnings.append("No validated candidate-race polls available")
    status = "degraded" if warnings or fallbacks else "healthy"
    digest = hashlib.sha256(
        f"{national.data_through}|{national.election_mean:.5f}|{n_simulations}".encode()
    ).hexdigest()[:12]
    return {
        "updated_at": now.isoformat(),
        "model_version": "3.0.0",
        "schema_version": "3.0.0",
        "model_type": "bayesian_external_average",
        "model_status": status,
        "run_id": f"v3-{now.strftime('%Y%m%dT%H%M%SZ')}-{digest}",
        "election_date": ELECTION_DATE.isoformat(),
        "days_until_election": max((ELECTION_DATE - now.date()).days, 0),
        "n_simulations": n_simulations,
        "districts_total": 435,
        "data_through": national.data_through,
        "inference_method": "external_average_bayesian_update + posterior_predictive_draws",
        "fallback_used": bool(fallbacks),
        "fallbacks": fallbacks,
        "warnings": warnings,
        "source_freshness": {
            "generic_ballot": {"latest": national.data_through, "age_days": age, "stale": stale},
            "race_polls": {
                "latest": race_date.isoformat() if race_date else None,
                "age_days": race_age,
                "stale": race_age is None or race_age > 30,
            },
        },
        "stale_override": bool(stale and allow_stale),
        "diagnostics": {**national.diagnostics, **diagnostics},
    }


def _load_backtest_summary() -> dict[str, Any]:
    path = OUTPUT_DIR / "backtest_metrics.json"
    payload = json.loads(path.read_text()) if path.exists() else {
        "status": "not_run",
        "message": "Run scripts/backtest_v3.py after preparing historical forecast snapshots.",
    }
    payload["current_likelihood_applicability"] = {
        "status": "supporting_evidence_only",
        "validated_component": "robust race update and correlated error floor",
        "not_directly_validated": "Silver Bulletin maintained averages",
        "reason": "No comparable public historical archive of race averages",
    }
    return payload


def _change_decomposition(previous: dict[str, Any], current: dict[str, Any], national: Any) -> dict[str, Any]:
    old_summary = previous.get("summary", {}) if previous else {}
    return {
        "probability_change": round(
            current["summary"]["prob_dem_majority"] - float(old_summary.get("prob_dem_majority", current["summary"]["prob_dem_majority"])), 4
        ),
        "median_seat_change": int(
            current["summary"]["median_dem_seats"] - int(old_summary.get("median_dem_seats", current["summary"]["median_dem_seats"]))
        ),
        "national_update": {
            "fundamentals_prior": round(national.prior.mean, 2),
            "poll_updated_current": round(national.current_mean, 2),
            "election_day_mean": round(national.election_mean, 2),
            "polling_contribution": round(national.current_mean - national.prior.mean, 2),
            "future_uncertainty_std": round(national.election_std, 2),
        },
    }


def update_timeline(summary: dict[str, Any], chamber: str) -> None:
    path = OUTPUT_DIR / ("timeline.csv" if chamber == "house" else "senate_timeline.csv")
    probability = "prob_dem_majority" if chamber == "house" else "prob_dem_control"
    row = {
        "date": date.today().isoformat(),
        probability: summary[probability],
        "median_dem_seats": summary["median_dem_seats"],
        "mean_dem_seats": summary["mean_dem_seats"],
        "ci_90_low": summary["ci_90_low"],
        "ci_90_high": summary["ci_90_high"],
        "national_env": summary["national_environment"],
        "approval": summary.get("approval_rating", 0),
        "generic_ballot": summary.get("generic_ballot_margin", 0),
        "model_version": "3.0.0",
        "methodology_break": True,
    }
    history = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not history.empty and "date" in history:
        history = history[history["date"] != row["date"]]
    _atomic_csv(path, pd.concat([history, pd.DataFrame([row])], ignore_index=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the forecast v3 public artifacts")
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-race-fetch", action="store_true")
    parser.add_argument("--skip-timeline", action="store_true")
    parser.add_argument("--allow-stale", action="store_true", help="Development only: publish stale cached polls")
    args = parser.parse_args()

    registry, registry_fallbacks = _fetch_registry(args.skip_race_fetch)
    silver, silver_fallbacks = _fetch_silver_averages(args.skip_fetch, registry)
    approval, fallbacks = _fetch_approval(args.skip_fetch)
    fallbacks.extend(registry_fallbacks)
    fallbacks.extend(silver_fallbacks)
    generic = silver.generic_history
    race_polls = silver.race_likelihoods
    race_status = silver.status
    approval_mean, approval_std = summarize_approval(approval)
    economic_index, economic_fallbacks = _economic_index()
    fallbacks.extend(economic_fallbacks)

    prior = build_fundamentals_prior(
        approval_mean=approval_mean,
        approval_std=approval_std,
        economic_index=economic_index,
    )
    national = DynamicNationalModel(random_seed=42).fit_external_average(
        generic, prior, election_date=ELECTION_DATE, n_draws=args.simulations
    )
    poll_age = (date.today() - date.fromisoformat(national.data_through)).days
    # Refuse catastrophically stale data.  A 21-day freshness warning remains
    # visible in metadata; 45 days is the hard publication cutoff this early in
    # the cycle.
    if poll_age > 45 and not args.allow_stale:
        raise RuntimeError(
            f"Latest generic-ballot poll is {poll_age} days old; existing public forecast was retained. "
            "Use --allow-stale only for local development."
        )

    house, senate, diagnostics = run_v3_forecast(
        DATA_DIR,
        national,
        race_polls=race_polls,
        race_poll_status=race_status,
        registry=registry,
        n_simulations=args.simulations,
    )
    metadata = _metadata(national, diagnostics, fallbacks, args.simulations, race_polls, args.allow_stale)
    previous_path = OUTPUT_DIR / "forecast.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    polling_display = format_polling_data(generic, approval)
    shared = {
        "metadata": metadata,
        "national_model": national.to_public_dict(),
        "polling": polling_display,
        "backtest": _load_backtest_summary(),
        "data_sources": {
            "polling_likelihood": "Silver Bulletin maintained polling averages",
            "approval_prior_input": "VoteHub",
            "candidate_registry": "Federal Election Commission",
            "polling_source_url": "https://www.natesilver.net/p/nate-silver-2026-midterm-election-polls-model",
        },
    }
    house = {**shared, **house}
    senate = {**shared, **senate}
    house["change_decomposition"] = _change_decomposition(previous, house, national)
    senate["change_decomposition"] = house["change_decomposition"]

    _atomic_json(OUTPUT_DIR / "forecast.json", house)
    _atomic_json(OUTPUT_DIR / "senate_forecast.json", senate)
    if not args.skip_timeline:
        update_timeline(house["summary"], "house")
        update_timeline(senate["summary"], "senate")

    for filename in ("forecast.json", "senate_forecast.json", "timeline.csv", "senate_timeline.csv"):
        source = OUTPUT_DIR / filename
        if source.exists():
            shutil.copy2(source, WEBSITE_DIR / filename)

    logger.info(
        "Published v3 forecast %s: House D majority %.1f%%, Senate D control %.1f%%",
        metadata["run_id"], house["summary"]["prob_dem_majority"] * 100,
        senate["summary"]["prob_dem_control"] * 100,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
