#!/usr/bin/env python3
"""
Generate forecast.json output from the House forecast model.

This script:
1. Loads all data sources
2. Runs the Bayesian hierarchical model
3. Outputs forecast.json with all predictions
4. Updates timeline.csv with historical tracking
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.forecast import run_forecast, HouseForecastModel
from models.senate_forecast import SenateForecastModel

# Setup paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_forecast_json(model: HouseForecastModel) -> dict:
    """Generate the complete forecast.json structure."""

    now = datetime.utcnow()
    election_date = datetime(2026, 11, 3)
    days_until = (election_date - now).days

    summary = model.get_summary()
    categories = model.get_category_counts()

    forecast = {
        "metadata": {
            "updated_at": now.isoformat() + "Z",
            "model_version": "1.0.0",
            "election_date": "2026-11-03",
            "days_until_election": days_until,
            "n_simulations": model.n_simulations,
            "districts_total": 435,
        },
        "summary": {
            "prob_dem_majority": round(summary["prob_dem_majority"], 3),
            "prob_rep_majority": round(summary["prob_rep_majority"], 3),
            "median_dem_seats": summary["median_dem_seats"],
            "median_rep_seats": summary["median_rep_seats"],
            "mean_dem_seats": round(summary["mean_dem_seats"], 1),
            "ci_90_low": summary["ci_90_low"],
            "ci_90_high": summary["ci_90_high"],
            "ci_50_low": summary["ci_50_low"],
            "ci_50_high": summary["ci_50_high"],
            "national_environment": round(summary["national_environment"], 1),
            "generic_ballot_margin": round(summary["generic_ballot_margin"], 1),
            "approval_rating": round(summary["approval_rating"], 1),
            "net_approval": round(summary["net_approval"], 1),
        },
        "categories": {
            "dem": {
                "safe": categories["safe_d"],
                "likely": categories["likely_d"],
                "lean": categories["lean_d"],
            },
            "toss_up": categories["toss_up"],
            "rep": {
                "safe": categories["safe_r"],
                "likely": categories["likely_r"],
                "lean": categories["lean_r"],
            },
        },
        "seat_distribution": model.get_seat_distribution(),
        "districts": model.get_district_forecasts(),
    }

    return forecast


def update_timeline(summary: dict) -> None:
    """Update the timeline.csv with current forecast."""
    timeline_path = OUTPUT_DIR / "timeline.csv"

    today = datetime.now().strftime("%Y-%m-%d")

    new_row = {
        "date": today,
        "prob_dem_majority": summary["prob_dem_majority"],
        "median_dem_seats": summary["median_dem_seats"],
        "mean_dem_seats": summary["mean_dem_seats"],
        "ci_90_low": summary["ci_90_low"],
        "ci_90_high": summary["ci_90_high"],
        "national_env": summary["national_environment"],
        "approval": summary["approval_rating"],
        "generic_ballot": summary["generic_ballot_margin"],
    }

    if timeline_path.exists():
        timeline = pd.read_csv(timeline_path)
        # Remove today's entry if it exists (for re-runs)
        timeline = timeline[timeline["date"] != today]
        timeline = pd.concat([timeline, pd.DataFrame([new_row])], ignore_index=True)
    else:
        timeline = pd.DataFrame([new_row])

    timeline.to_csv(timeline_path, index=False)
    logger.info(f"Updated timeline: {timeline_path}")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("2026 House Forecast - Generating Predictions")
    logger.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run the forecast model
    logger.info("\n--- Running Forecast Model ---")
    model = run_forecast(DATA_DIR, n_simulations=10000)

    # Generate forecast JSON
    logger.info("\n--- Generating Output ---")
    forecast = generate_forecast_json(model)

    # Save forecast.json
    forecast_path = OUTPUT_DIR / "forecast.json"
    with open(forecast_path, "w") as f:
        json.dump(forecast, f, indent=2)
    logger.info(f"Saved forecast: {forecast_path}")

    # Update timeline
    update_timeline(forecast["summary"])

    # --- Senate Forecast ---
    logger.info("\n--- Running Senate Forecast ---")
    national_env = forecast["summary"]["national_environment"]
    senate_model = SenateForecastModel(national_environment=national_env, n_simulations=10000)
    senate_model.simulate_elections()

    senate_forecast = {
        "metadata": forecast["metadata"].copy(),
        "summary": senate_model.get_summary(),
        "categories": senate_model.get_category_counts(),
        "seat_distribution": senate_model.get_seat_distribution(),
        "races": senate_model.get_race_forecasts(),
    }

    senate_path = OUTPUT_DIR / "senate_forecast.json"
    with open(senate_path, "w") as f:
        json.dump(senate_forecast, f, indent=2)
    logger.info(f"Saved Senate forecast: {senate_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("FORECAST SUMMARY")
    logger.info("=" * 60)

    s = forecast["summary"]
    c = forecast["categories"]

    logger.info(f"\nDemocratic Majority Probability: {s['prob_dem_majority']:.1%}")
    logger.info(f"Median Democratic Seats: {s['median_dem_seats']}")
    logger.info(f"90% Confidence Interval: [{s['ci_90_low']}, {s['ci_90_high']}]")

    logger.info(f"\nNational Environment: D+{s['national_environment']:.1f}")
    logger.info(f"Generic Ballot: D+{s['generic_ballot_margin']:.1f}")
    logger.info(f"Approval Rating: {s['approval_rating']:.1f}%")

    logger.info("\nSeat Categories:")
    logger.info(f"  Safe D: {c['dem']['safe']}  |  Safe R: {c['rep']['safe']}")
    logger.info(f"  Likely D: {c['dem']['likely']}  |  Likely R: {c['rep']['likely']}")
    logger.info(f"  Lean D: {c['dem']['lean']}  |  Lean R: {c['rep']['lean']}")
    logger.info(f"  Toss-up: {c['toss_up']}")

    # Show most competitive races
    logger.info("\nTop 10 Most Competitive Races:")
    for dist in forecast["districts"][:10]:
        prob = dist["prob_dem"]
        inc = dist["incumbent"]
        logger.info(
            f"  {dist['id']}: {prob:.0%} D "
            f"({inc['name']}, {inc['party']}) - {dist['category'].replace('_', ' ').title()}"
        )

    logger.info(f"\nForecast saved to: {forecast_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
