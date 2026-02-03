#!/usr/bin/env python3
"""
Generate forecast.json output from the 2026 Election Forecast Model.

Pipeline:
1. Fetch latest polling data from VoteHub API
2. Run Bayesian model to infer national environment from polls
3. Run hierarchical House forecast with learned parameters
4. Run Senate simulations with national environment
5. Output forecast.json files for the website
6. Update timeline.csv with historical tracking

The national environment (inferred from polls) is the SINGLE driving variable.
Model uses learned parameters from 2018/2022 historical fitting.
"""

# IMPORTANT: Set PyTensor flags BEFORE any imports that might trigger PyMC/PyTensor
import os
os.environ.setdefault("PYTENSOR_FLAGS", "device=cpu,floatX=float64,optimizer=fast_compile,cxx=")

import argparse
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
from models.national_environment import NationalEnvironmentModel
from models.economic_fundamentals import EconomicFundamentals
from scripts.fetch_votehub import VoteHubFetcher
from scripts.fetch_cook_ratings import CookPoliticalScraper, adjust_pvi_with_cook_ratings

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


def format_polling_data(gb_polls: pd.DataFrame, approval_polls: pd.DataFrame, max_polls: int = 50) -> dict:
    """Format polling data for website display.

    Args:
        gb_polls: Generic ballot polls DataFrame
        approval_polls: Trump approval polls DataFrame
        max_polls: Maximum number of polls to include (most recent)

    Returns:
        Dictionary with formatted polling data
    """
    polling_data = {
        "generic_ballot": [],
        "approval": [],
    }

    if gb_polls is not None and not gb_polls.empty:
        # Sort by date descending and take most recent
        gb_sorted = gb_polls.sort_values("date", ascending=False).head(max_polls)
        for _, row in gb_sorted.iterrows():
            polling_data["generic_ballot"].append({
                "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10],
                "pollster": row["pollster"],
                "sample_size": int(row["sample_size"]),
                "population": row.get("population", "a"),
                "dem_pct": round(float(row["dem_pct"]), 1),
                "rep_pct": round(float(row["rep_pct"]), 1),
                "margin": round(float(row["margin"]), 1),
            })

    if approval_polls is not None and not approval_polls.empty:
        # Sort by date descending and take most recent
        app_sorted = approval_polls.sort_values("date", ascending=False).head(max_polls)
        for _, row in app_sorted.iterrows():
            polling_data["approval"].append({
                "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10],
                "pollster": row["pollster"],
                "sample_size": int(row["sample_size"]),
                "population": row.get("population", "a"),
                "approve": round(float(row["approve"]), 1),
                "disapprove": round(float(row["disapprove"]), 1),
                "net_approval": round(float(row["net_approval"]), 1),
            })

    return polling_data


def generate_forecast_json(model: HouseForecastModel, env_result: dict = None, polling_data: dict = None) -> dict:
    """Generate the complete forecast.json structure.

    Args:
        model: The HouseForecastModel with simulation results
        env_result: Optional dict from NationalEnvironmentModel with approval data
        polling_data: Optional dict with formatted polling data for website display
    """

    now = datetime.utcnow()
    election_date = datetime(2026, 11, 3)
    days_until = (election_date - now).days

    summary = model.get_summary()
    categories = model.get_category_counts()

    # Override data from env_result if available
    if env_result:
        summary["approval_rating"] = env_result.get("approval_mean", 0)
        summary["net_approval"] = env_result.get("approval_mean", 0)  # net_approval is the same as approval_mean
        summary["economic_adjustment"] = env_result.get("economic_adjustment", 0)
        summary["generic_ballot_margin"] = env_result.get("polling_national_env", summary["national_environment"])

    # Determine model version based on type
    model_type = summary.get("model_type", "legacy")
    model_version = "2.0.0" if model_type == "hierarchical_bayesian" else "1.0.0"

    forecast = {
        "metadata": {
            "updated_at": now.isoformat() + "Z",
            "model_version": model_version,
            "model_type": model_type,
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
            "economic_adjustment": round(summary.get("economic_adjustment", 0), 1),
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

    # Add polling data if provided
    if polling_data:
        forecast["polling"] = polling_data

    return forecast


def update_timeline(summary: dict, chamber: str = "house") -> None:
    """Update the timeline CSV with current forecast."""
    filename = "timeline.csv" if chamber == "house" else "senate_timeline.csv"
    timeline_path = OUTPUT_DIR / filename

    today = datetime.now().strftime("%Y-%m-%d")
    prob_key = "prob_dem_majority" if chamber == "house" else "prob_dem_control"

    new_row = {
        "date": today,
        prob_key: summary.get(prob_key, summary.get("prob_dem_majority", 0)),
        "median_dem_seats": summary["median_dem_seats"],
        "mean_dem_seats": summary["mean_dem_seats"],
        "ci_90_low": summary["ci_90_low"],
        "ci_90_high": summary["ci_90_high"],
        "national_env": summary["national_environment"],
        "approval": summary.get("approval_rating", 0),
        "generic_ballot": summary.get("generic_ballot_margin", summary["national_environment"]),
    }

    if timeline_path.exists():
        timeline = pd.read_csv(timeline_path)
        timeline = timeline[timeline["date"] != today]
        timeline = pd.concat([timeline, pd.DataFrame([new_row])], ignore_index=True)
    else:
        timeline = pd.DataFrame([new_row])

    timeline.to_csv(timeline_path, index=False)
    logger.info(f"Updated {chamber} timeline: {timeline_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate 2026 election forecast")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy flat simulation instead of hierarchical model"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations (default: 10000)"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching new polling data (use cached)"
    )
    parser.add_argument(
        "--skip-timeline",
        action="store_true",
        help="Skip updating timeline (for manual reruns)"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("2026 Election Forecast - Generating Predictions")
    logger.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model: {'Legacy' if args.legacy else 'Hierarchical Bayesian'}")
    logger.info("Poll-anchored: Yes (no midterm/economic stacking)")
    logger.info("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch latest polling data from VoteHub
    if not args.skip_fetch:
        logger.info("\n--- Fetching VoteHub Polling Data ---")
        fetcher = VoteHubFetcher()
        gb_polls = fetcher.fetch_generic_ballot(days_back=180)
        approval_polls = fetcher.fetch_trump_approval(days_back=180)
        fetcher.save_polls(gb_polls, approval_polls)
    else:
        logger.info("\n--- Loading Cached Polling Data ---")
        gb_polls = pd.read_csv(DATA_DIR / "raw" / "polling" / "generic_ballot.csv")
        approval_polls = pd.read_csv(DATA_DIR / "raw" / "polling" / "trump_approval.csv")

    # Step 1b: Fetch Cook Political ratings and adjust PVI for redistricted districts
    logger.info("\n--- Fetching Cook Political Ratings ---")
    cook_scraper = CookPoliticalScraper()
    cook_ratings = cook_scraper.fetch_house_ratings()
    senate_ratings = cook_scraper.fetch_senate_ratings()
    redistricting_df = cook_scraper.fetch_redistricting_status()
    cook_scraper.save_data(cook_ratings, redistricting_df, senate_ratings)

    # Get list of redistricted states
    redistricted_states = cook_scraper.get_redistricted_states(redistricting_df)
    if redistricted_states:
        logger.info(f"Redistricted states: {', '.join(redistricted_states)}")

    # Load and adjust districts data with Cook ratings
    districts_path = DATA_DIR / "processed" / "districts.csv"
    if districts_path.exists() and not cook_ratings.empty:
        districts_df = pd.read_csv(districts_path)
        original_count = len(districts_df)

        # Adjust PVI for redistricted districts
        districts_df = adjust_pvi_with_cook_ratings(
            districts_df,
            cook_ratings,
            redistricted_states
        )

        # Save adjusted districts
        districts_df.to_csv(districts_path, index=False)
        logger.info(f"Updated districts.csv with Cook rating adjustments")

    # ==========================================================================
    # NATIONAL ENVIRONMENT COMPUTATION
    # ==========================================================================
    # The national environment is computed in THREE sequential steps:
    #
    # Step A: PyMC Bayesian inference on generic ballot polls only
    #         → Produces mu_posterior (latent national environment from GB polls)
    #
    # Step B: Post-hoc approval adjustment (OUTSIDE PyMC model)
    #         → polling_national_env = mu_posterior + (-0.3 × approval_weight × net_approval)
    #         → This is a heuristic adjustment, not a fitted relationship
    #
    # Step C: Economic adjustment (using fitted coefficients)
    #         → national_env = β_polls × polling_national_env + β_econ × econ_index
    #         → β_polls ≈ 1.0, β_econ ≈ 0.34 (but with large uncertainty)
    #
    # IMPORTANT: Uncertainty only accounts for Step A (GB poll variance).
    # Steps B and C add deterministic adjustments that do not propagate uncertainty.
    # ==========================================================================

    # Step 2A: Infer national environment from generic ballot polls
    logger.info("\n--- Inferring National Environment (Step A: GB Polls) ---")
    env_model = NationalEnvironmentModel(gb_polls=gb_polls, approval_polls=approval_polls)
    env_result = env_model.fit(use_pymc=True)  # Full Bayesian inference on GB polls only
    env_model.save_results(env_result)

    # NOTE: env_result["national_environment"] already includes the post-hoc
    # approval adjustment (Step B) applied inside NationalEnvironmentModel.fit_pymc()
    polling_national_env = env_result["national_environment"]
    approval_mean = env_result.get("approval_mean", 0)
    logger.info(f"Polling-based National Environment: D{polling_national_env:+.1f} ± {env_result['uncertainty']:.1f}")
    if approval_mean != 0:
        logger.info(f"Trump Net Approval: {approval_mean:+.1f}%")
        logger.info(f"  (Approval adjustment already applied in Step B)")

    # Step 2C: Apply empirically-fitted coefficients for economic adjustment
    logger.info("\n--- National Environment (Step C: Economic Adjustment) ---")
    coeffs_path = DATA_DIR / "processed" / "national_env_coefficients.json"
    if coeffs_path.exists():
        with open(coeffs_path) as f:
            coeffs = json.load(f)
        beta_polls = coeffs.get("beta_polls", 1.0)
        beta_econ = coeffs.get("beta_econ", 0.0)
        logger.info(f"Loaded fitted coefficients: β_polls={beta_polls:.2f}, β_econ={beta_econ:.2f}")
    else:
        logger.warning("No fitted coefficients found, using defaults (β_polls=1, β_econ=0)")
        beta_polls = 1.0
        beta_econ = 0.0

    # Calculate economic index for adjustment
    econ = EconomicFundamentals()
    econ.load_data()
    econ_result = econ.calculate_index()

    # Adjust economic index for president's party (R president: negate index)
    raw_econ = econ_result["normalized_index"]
    adjusted_econ = -raw_econ  # R president: bad economy helps Dems

    # Apply fitted model: national_env = β_polls × polling + β_econ × econ
    national_env = beta_polls * polling_national_env + beta_econ * adjusted_econ
    econ_adjustment = beta_econ * adjusted_econ

    logger.info(f"Economic Index: {raw_econ:.2f} ({econ_result['interpretation']})")
    logger.info(f"Economic Adjustment: {econ_adjustment:+.1f} points (β_econ × adjusted_index)")
    logger.info(f"Final National Environment: D{national_env:+.1f}")

    # Store in env_result for downstream use
    env_result["economic_adjustment"] = econ_adjustment
    env_result["economic_index"] = raw_econ
    env_result["polling_national_env"] = polling_national_env
    env_result["beta_polls"] = beta_polls
    env_result["beta_econ"] = beta_econ

    # Step 3: Run the House forecast model
    model_type = "legacy" if args.legacy else "hierarchical Bayesian"
    logger.info(f"\n--- Running House Forecast Model ({model_type}) ---")

    # Load districts
    districts_df = pd.read_csv(DATA_DIR / "processed" / "districts.csv")

    # Initialize model (poll-anchored - no midterm/economic adjustments)
    model = HouseForecastModel(
        districts_df=districts_df,
        national_environment=national_env,
        n_simulations=args.simulations,
        use_hierarchical=not args.legacy,
    )
    model.simulate_elections()

    # Format polling data for website
    polling_data = format_polling_data(gb_polls, approval_polls)
    logger.info(f"Formatted {len(polling_data['generic_ballot'])} GB polls and {len(polling_data['approval'])} approval polls for website")

    # Generate forecast JSON
    logger.info("\n--- Generating Output ---")
    forecast = generate_forecast_json(model, env_result, polling_data)

    # Save forecast.json
    forecast_path = OUTPUT_DIR / "forecast.json"
    with open(forecast_path, "w") as f:
        json.dump(forecast, f, indent=2)
    logger.info(f"Saved forecast: {forecast_path}")

    # Update timeline
    if not args.skip_timeline:
        update_timeline(forecast["summary"], chamber="house")

    # --- Senate Forecast ---
    logger.info("\n--- Running Senate Forecast ---")
    # Poll-anchored: use generic ballot directly, no stacking adjustments
    # Uses same hierarchical Bayesian approach as House model with PyMC
    senate_model = SenateForecastModel(
        national_environment=national_env,
        n_simulations=args.simulations,
    )
    senate_model.run(use_pymc=not args.legacy)  # Use PyMC by default (same as House)

    senate_summary = senate_model.get_summary()
    # Add approval and economic data from env_result (same as House forecast)
    if env_result:
        senate_summary["generic_ballot_margin"] = env_result.get("polling_national_env", national_env)
        senate_summary["approval_rating"] = env_result.get("approval_mean", 0)
        senate_summary["net_approval"] = env_result.get("approval_mean", 0)
        senate_summary["economic_adjustment"] = env_result.get("economic_adjustment", 0)

    senate_forecast = {
        "metadata": forecast["metadata"].copy(),
        "summary": senate_summary,
        "categories": senate_model.get_category_counts(),
        "seat_distribution": senate_model.get_seat_distribution(),
        "races": senate_model.get_race_forecasts(),
        "polling": polling_data,
    }

    senate_path = OUTPUT_DIR / "senate_forecast.json"
    with open(senate_path, "w") as f:
        json.dump(senate_forecast, f, indent=2)
    logger.info(f"Saved Senate forecast: {senate_path}")

    # Update Senate timeline
    if not args.skip_timeline:
        update_timeline(senate_summary, chamber="senate")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("FORECAST SUMMARY")
    logger.info("=" * 60)

    s = forecast["summary"]
    c = forecast["categories"]

    logger.info(f"\nDemocratic Majority Probability: {s['prob_dem_majority']:.1%}")
    logger.info(f"Median Democratic Seats: {s['median_dem_seats']}")
    logger.info(f"90% Confidence Interval: [{s['ci_90_low']}, {s['ci_90_high']}]")

    logger.info(f"\nNational Environment: D{s['national_environment']:+.1f}")
    logger.info(f"  Generic Ballot (polling): D{s['generic_ballot_margin']:+.1f}")
    logger.info(f"  Economic Adjustment: {s['economic_adjustment']:+.1f}")
    logger.info(f"Trump Net Approval: {s['net_approval']:+.1f}%")

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
