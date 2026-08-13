#!/usr/bin/env python3
"""Generate the House and Senate forecast from validated local inputs."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.dynamic_polling import (  # noqa: E402
    DynamicNationalModel,
    build_fundamentals_prior,
)
from models.economic_fundamentals import EconomicFundamentals  # noqa: E402
from models.chamber_forecast import run_chamber_forecast  # noqa: E402
from models.race_polling import load_cached_candidate_registry  # noqa: E402
from models.silver_bulletin import (  # noqa: E402
    calibrate_generic_average_uncertainty,
    load_silver_cache,
    prepare_generic_poll_likelihood,
)


DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
WEBSITE_DIR = PROJECT_ROOT / "website"
ELECTION_DATE = date(2026, 11, 3)
FORECAST_EPOCH = date(2026, 8, 12)
FORECAST_TIMEZONE = ZoneInfo("America/Los_Angeles")
MODEL_VERSION = "5.0.0"
SCHEMA_VERSION = "5.0.0"
MANIFEST_PATH = DATA_DIR / "processed" / "input_manifest.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def _load_manifest(allow_blocked: bool) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Missing validated input manifest; run scripts/fetch_inputs.py first")
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("forecast_epoch") != FORECAST_EPOCH.isoformat():
        raise ValueError("Input manifest forecast epoch does not match the production forecast")
    if manifest.get("blocked_sources") and not allow_blocked:
        raise RuntimeError(
            "Refusing to publish from blocked inputs: " + ", ".join(manifest["blocked_sources"])
        )
    return manifest


def _load_cached_inputs() -> tuple[Any, Any, pd.DataFrame, dict[str, Any]]:
    registry = load_cached_candidate_registry(DATA_DIR)
    if registry is None:
        raise FileNotFoundError("No cached official candidate registry")
    silver = load_silver_cache(DATA_DIR)
    polls_path = DATA_DIR / "processed" / "silver_generic_polls.csv"
    if not polls_path.exists():
        raise FileNotFoundError("No cached Silver Bulletin generic poll universe")
    generic_polls = pd.read_csv(polls_path)
    calibration = calibrate_generic_average_uncertainty(silver.generic_history, generic_polls)
    generic_likelihood = prepare_generic_poll_likelihood(
        generic_polls, float(calibration["observation_std"])
    )
    return registry, silver, generic_likelihood, calibration


def _economic_calculation() -> dict[str, Any]:
    model = EconomicFundamentals()
    model.load_data()
    if len(model.data) != 5:
        raise ValueError("All five economic series are required")
    return model.calculate_index()


def format_polling_data(generic: pd.DataFrame, limit: int = 50) -> dict[str, Any]:
    output: dict[str, Any] = {"generic_ballot": []}
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
    return output


def _metadata(
    national: Any,
    diagnostics: dict[str, Any],
    manifest: dict[str, Any],
    n_simulations: int,
    allow_blocked: bool,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    local_today = now.astimezone(FORECAST_TIMEZONE).date()
    warnings = [f"{name} is {manifest['sources'][name]['state']}"
                for name in [*manifest.get("degraded_sources", []), *manifest.get("blocked_sources", [])]]
    if diagnostics["fundamentals"].get("status") != "passed":
        warnings.append("Some incumbent names remain unresolved")
    status = "blocked" if manifest.get("blocked_sources") else (
        "degraded" if warnings or manifest.get("fallback_used") else "healthy"
    )
    digest = hashlib.sha256(
        f"{national.polling_input_date}|{national.election_mean:.5f}|{n_simulations}".encode()
    ).hexdigest()[:12]
    return {
        "updated_at": now.isoformat(),
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "forecast_epoch": FORECAST_EPOCH.isoformat(),
        "forecast_timezone": str(FORECAST_TIMEZONE),
        "model_type": "house_hierarchical_bayesian_bulletin_polls",
        "model_status": status,
        "run_id": f"forecast-{now.strftime('%Y%m%dT%H%M%SZ')}-{digest}",
        "election_date": ELECTION_DATE.isoformat(),
        "days_until_election": max((ELECTION_DATE - local_today).days, 0),
        "n_simulations": n_simulations,
        "districts_total": 435,
        "inference_method": "bulletin_adjusted_poll_aggregate_bayesian_update + hierarchical_posterior_predictive_draws",
        "fallback_used": bool(manifest.get("fallback_used")),
        "fallbacks": [name for name, source in manifest["sources"].items() if source.get("fallback_used")],
        "warnings": warnings,
        "source_freshness": manifest["sources"],
        "input_manifest_generated_at": manifest.get("generated_at"),
        "stale_override": bool(manifest.get("blocked_sources") and allow_blocked),
        "diagnostics": {**national.diagnostics, **diagnostics},
    }


def _load_backtest_summary() -> dict[str, Any]:
    path = OUTPUT_DIR / "backtest_metrics.json"
    payload = json.loads(path.read_text()) if path.exists() else {
        "status": "not_run",
        "message": "Run scripts/backtest_forecast.py after preparing historical forecast snapshots.",
    }
    payload["current_likelihood_applicability"] = {
        "status": "supporting_evidence_only",
        "validated_component": "robust race update and correlated error floor",
        "not_directly_validated": "Silver Bulletin maintained averages",
        "reason": "No comparable public historical archive of race averages",
    }
    house_path = OUTPUT_DIR / "house_backtest_metrics.json"
    if house_path.exists():
        house = json.loads(house_path.read_text())
        payload["house_structural_validation"] = {
            "scope": house.get("scope"),
            "metrics": house.get("metrics", {}),
            "seat_results": house.get("seat_results", []),
            "gate": house.get("house_structural_gate", {}),
            "limitation": house.get("limitation"),
        }
    return payload


def _change_decomposition(
    previous: dict[str, Any], current: dict[str, Any], national: Any, chamber: str
) -> dict[str, Any] | None:
    local_today = datetime.now(timezone.utc).astimezone(FORECAST_TIMEZONE).date()
    if local_today == FORECAST_EPOCH:
        return None
    previous_metadata = previous.get("metadata", {}) if previous else {}
    if (
        previous_metadata.get("schema_version") != SCHEMA_VERSION
        or previous_metadata.get("forecast_epoch") != FORECAST_EPOCH.isoformat()
    ):
        return None
    old_summary = previous.get("summary", {}) if previous else {}
    probability = "prob_dem_majority" if chamber == "house" else "prob_dem_control"
    return {
        "probability_change": round(
            current["summary"][probability] - float(old_summary.get(probability, current["summary"][probability])), 4
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


def _timeline_frame(path: Path, summary: dict[str, Any], chamber: str) -> pd.DataFrame:
    probability = "prob_dem_majority" if chamber == "house" else "prob_dem_control"
    local_today = datetime.now(timezone.utc).astimezone(FORECAST_TIMEZONE).date()
    row = {
        "date": local_today.isoformat(),
        probability: summary[probability],
        "median_dem_seats": summary["median_dem_seats"],
        "mean_dem_seats": summary["mean_dem_seats"],
        "ci_90_low": summary["ci_90_low"],
        "ci_90_high": summary["ci_90_high"],
        "election_day_national_margin": summary["election_day_national_margin"],
        "published_generic_ballot": summary["published_generic_ballot_margin"],
        "national_likelihood": summary["national_likelihood_margin"],
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "forecast_epoch": FORECAST_EPOCH.isoformat(),
    }
    try:
        history = pd.read_csv(path) if path.exists() else pd.DataFrame()
    except (OSError, pd.errors.ParserError):
        history = pd.DataFrame()
    if not history.empty and "date" in history:
        history["date"] = history["date"].astype(str)
        history = history[
            (history["date"] >= FORECAST_EPOCH.isoformat())
            & (history.get("model_version", "") == MODEL_VERSION)
            & (history.get("schema_version", "") == SCHEMA_VERSION)
            & (history.get("forecast_epoch", "") == FORECAST_EPOCH.isoformat())
        ]
        history = history[history["date"] != row["date"]]
        if history.empty:
            history = pd.DataFrame()
    return pd.concat([history, pd.DataFrame([row])], ignore_index=True)


def _publish_staged_bundle(stage: Path, destination: Path, filenames: list[str]) -> None:
    """Validate the complete bundle before replacing any public artifact."""
    for filename in filenames:
        source = stage / filename
        if filename.endswith(".json"):
            json.loads(source.read_text())
        else:
            frame = pd.read_csv(source)
            if frame.empty:
                raise ValueError(f"Staged timeline is empty: {filename}")
    destination.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        shutil.copy2(stage / filename, destination / f"{filename}.tmp")
    for filename in filenames:
        (destination / f"{filename}.tmp").replace(destination / filename)


def _load_previous_forecast(path: Path) -> dict[str, Any]:
    """Treat a missing or damaged prior artifact as non-comparable history."""
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _interval(mean: float, std: float) -> list[float]:
    return [round(mean - 1.64485363 * std, 3), round(mean + 1.64485363 * std, 3)]


def _national_environment(
    national: Any,
    published: dict[str, Any],
    polling_input: dict[str, Any],
    economy: dict[str, Any],
) -> dict[str, Any]:
    """Build the explicit, chamber-specific national model explanation."""
    return {
        "fundamentals_prior": {
            "description": "Economy-only pre-poll expectation for the Democratic two-party margin",
            "mean": round(national.prior.mean, 3),
            "std": round(national.prior.std, 3),
            "ci_90": _interval(national.prior.mean, national.prior.std),
            "economic_coefficient": national.prior.components["economy"],
        },
        "published_sentiment": {
            "description": "Silver Bulletin published likely-voter maintained average",
            "margin": published["margin"],
            "date": published["date"],
            "source_url": published["source_url"],
        },
        "polling_input": polling_input,
        "poll_updated_current": {
            "description": "Posterior national sentiment after one polling update",
            "mean": round(national.current_mean, 3),
            "std": round(national.current_std, 3),
            "ci_90": _interval(national.current_mean, national.current_std),
            "date": national.polling_input_date,
        },
        "election_day": {
            "description": "Posterior after future movement and election-error uncertainty",
            "mean": round(national.election_mean, 3),
            "std": round(national.election_std, 3),
            "ci_90": [
                round(float(np.percentile(national.election_samples, 5)), 3),
                round(float(np.percentile(national.election_samples, 95)), 3),
            ],
            "date": ELECTION_DATE.isoformat(),
        },
        "economy": {
            "description": "Five-series economic composite; positive values favor the incumbent party",
            "calculation_date": economy["date"],
            "raw_index": economy["raw_index"],
            "standardized_index": economy["normalized_index"],
            "interpretation": economy["interpretation"],
            "components": economy["components"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the public forecast artifacts")
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--skip-timeline", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--website-dir", type=Path, default=WEBSITE_DIR)
    parser.add_argument("--skip-website-copy", action="store_true")
    parser.add_argument(
        "--allow-blocked-inputs",
        action="store_true",
        help="Development/bootstrap only: generate from a manifest containing blocked inputs",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.allow_blocked_inputs)
    registry, silver, generic_likelihood, generic_calibration = _load_cached_inputs()
    generic = silver.generic_history
    race_polls = silver.race_likelihoods
    race_status = silver.status
    economy = _economic_calculation()
    prior = build_fundamentals_prior(economic_index=float(economy["normalized_index"]))
    process_calibration = DynamicNationalModel.calibrate_process_std(generic)
    national = DynamicNationalModel(
        process_std_per_day=float(process_calibration["process_std_per_day"]), random_seed=42
    ).fit_external_average(
        generic_likelihood, prior, election_date=ELECTION_DATE, n_draws=args.simulations
    )
    national.diagnostics["generic_average_calibration"] = generic_calibration
    national.diagnostics["process_calibration"] = process_calibration
    # The House and Senate intentionally retain distinct national polling
    # inputs. Any future Senate recalibration requires its own promotion gate.
    senate_national = DynamicNationalModel(random_seed=42).fit_external_average(
        generic, prior, election_date=ELECTION_DATE, n_draws=args.simulations
    )
    house, senate, diagnostics = run_chamber_forecast(
        DATA_DIR,
        national,
        senate_national=senate_national,
        race_polls=race_polls,
        race_poll_status=race_status,
        registry=registry,
        n_simulations=args.simulations,
    )
    metadata = _metadata(national, diagnostics, manifest, args.simulations, args.allow_blocked_inputs)
    senate_metadata = {
        **metadata,
        "model_type": "senate_bayesian_external_average",
        "races_total": 35,
    }
    senate_metadata.pop("districts_total", None)
    previous_house_path = args.output_dir / "forecast.json"
    previous_senate_path = args.output_dir / "senate_forecast.json"
    previous_house = _load_previous_forecast(previous_house_path)
    previous_senate = _load_previous_forecast(previous_senate_path)
    polling_display = format_polling_data(generic)
    published_row = generic.sort_values("date").iloc[-1]
    published_average = {
        "date": pd.to_datetime(published_row["date"]).strftime("%Y-%m-%d"),
        "provider": "Silver Bulletin",
        "input": "published likely-voter-adjusted maintained average",
        "dem_pct": round(float(published_row["dem_pct"]), 2),
        "rep_pct": round(float(published_row["rep_pct"]), 2),
        "margin": round(float(published_row["margin"]), 2),
        "source_url": str(published_row["source_url"]),
    }
    likelihood_row = generic_likelihood.iloc[-1]
    polling_display["national_likelihood"] = {
        "date": pd.to_datetime(likelihood_row["date"]).strftime("%Y-%m-%d"),
        "provider": "Silver Bulletin",
        "input": "influence-weighted adjusted poll universe",
        "margin": round(float(likelihood_row["margin"]), 2),
        "observation_std": round(float(likelihood_row["observation_std"]), 3),
        "poll_rows": int(likelihood_row["poll_rows"]),
        "house_effects": "provider-adjusted; not re-estimated",
    }
    polling_display["published_average"] = published_average
    polling_display["generic_ballot_series_role"] = (
        "Silver forecast-page likely-voter average shown for context; not an additional likelihood"
    )
    shared = {
        "national_model": national.to_public_dict(),
        "polling": polling_display,
        "backtest": _load_backtest_summary(),
        "data_sources": {
            "national_polling_likelihood": "Silver Bulletin adjusted generic-ballot polls and influence weights",
            "race_polling_likelihood": "Silver Bulletin maintained candidate-race averages",
            "house_partisan_lean": "Cook Political Report current-map Cook PVI",
            "fundamentals_prior_input": "FRED economic composite",
            "candidate_registry": "Federal Election Commission",
            "national_polling_source_url": "https://www.natesilver.net/p/generic-ballot-average-2026-nate-silver-bulletin-congress-polls",
            "race_polling_source_url": "https://www.natesilver.net/p/nate-silver-2026-midterm-election-polls-model",
        },
    }
    house = {**shared, **house, "metadata": metadata}
    senate = {**shared, **senate, "metadata": senate_metadata}
    senate["national_model"] = senate_national.to_public_dict()
    for output in (house, senate):
        output["summary"]["published_generic_ballot_margin"] = published_average["margin"]
        output["summary"]["published_generic_ballot_date"] = published_average["date"]
    house["summary"]["national_likelihood_date"] = polling_display["national_likelihood"]["date"]
    senate["summary"]["national_likelihood_margin"] = published_average["margin"]
    senate["summary"]["national_likelihood_date"] = published_average["date"]
    senate["summary"]["poll_updated_current_margin"] = round(senate_national.current_mean, 2)
    house_polling_input = {
        "description": "Influence-weighted Silver adjusted-poll universe used once by the House model",
        "margin": polling_display["national_likelihood"]["margin"],
        "date": polling_display["national_likelihood"]["date"],
        "std": polling_display["national_likelihood"]["observation_std"],
        "poll_rows": polling_display["national_likelihood"]["poll_rows"],
        "provenance": "Silver Bulletin adjusted generic-ballot polls and provider influence weights",
    }
    senate_poll_std = float(generic.sort_values("date").iloc[-1].get("observation_std", 1.5))
    if not np.isfinite(senate_poll_std):
        senate_poll_std = 1.5
    senate_polling_input = {
        "description": "Silver published likely-voter maintained average used once by the Senate model",
        "margin": published_average["margin"],
        "date": published_average["date"],
        "std": round(senate_poll_std, 3),
        "poll_rows": 1,
        "provenance": "Silver Bulletin published likely-voter maintained average",
    }
    house["national_environment"] = _national_environment(
        national, published_average, house_polling_input, economy
    )
    senate["national_environment"] = _national_environment(
        senate_national, published_average, senate_polling_input, economy
    )
    house["change_decomposition"] = _change_decomposition(previous_house, house, national, "house")
    senate["change_decomposition"] = _change_decomposition(previous_senate, senate, senate_national, "senate")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="forecast-stage-", dir=args.output_dir) as stage_name:
        stage = Path(stage_name)
        (stage / "forecast.json").write_text(json.dumps(house, indent=2, allow_nan=False))
        (stage / "senate_forecast.json").write_text(json.dumps(senate, indent=2, allow_nan=False))
        if not args.skip_timeline:
            _timeline_frame(args.output_dir / "timeline.csv", house["summary"], "house").to_csv(
                stage / "timeline.csv", index=False
            )
            _timeline_frame(args.output_dir / "senate_timeline.csv", senate["summary"], "senate").to_csv(
                stage / "senate_timeline.csv", index=False
            )
        filenames = ["forecast.json", "senate_forecast.json"]
        if not args.skip_timeline:
            filenames.extend(["timeline.csv", "senate_timeline.csv"])
        _publish_staged_bundle(stage, args.output_dir, filenames)

    if not args.skip_website_copy:
        args.website_dir.mkdir(parents=True, exist_ok=True)
        filenames = ["forecast.json", "senate_forecast.json"]
        if not args.skip_timeline:
            filenames.extend(["timeline.csv", "senate_timeline.csv"])
        for filename in filenames:
            temporary = args.website_dir / f"{filename}.tmp"
            shutil.copy2(args.output_dir / filename, temporary)
        for filename in filenames:
            (args.website_dir / f"{filename}.tmp").replace(args.website_dir / filename)

    logger.info(
        "Published forecast %s: House D majority %.1f%%, Senate D control %.1f%%",
        metadata["run_id"], house["summary"]["prob_dem_majority"] * 100,
        senate["summary"]["prob_dem_control"] * 100,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
