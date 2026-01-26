#!/usr/bin/env python3
"""
Bayesian Hierarchical Forecasting Model for 2026 House Elections.

Three-layer structure:
1. National Layer: Combines fundamentals with polling to estimate national environment
2. Regional Layer: Models how regions respond to national environment
3. District Layer: Predicts vote share for each of 435 districts

Uses Monte Carlo simulation for uncertainty quantification.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class NationalEnvironment:
    """National political environment estimates."""
    generic_ballot_margin: float  # D-R margin in generic ballot
    approval_rating: float  # Presidential approval
    net_approval: float  # Approve - Disapprove
    midterm_penalty: float  # Historical penalty for president's party
    economic_index: float  # Combined economic indicator
    national_swing: float  # Expected national swing from 2024


@dataclass
class DistrictForecast:
    """Forecast for a single district."""
    district_id: str
    state: str
    prob_dem: float
    mean_vote_share: float
    std_vote_share: float
    ci_90_low: float
    ci_90_high: float
    category: str
    simulated_outcomes: np.ndarray


class HouseForecastModel:
    """
    Bayesian hierarchical model for House forecasting.

    Model structure:
    - National environment from fundamentals + polling
    - Regional effects (Northeast, South, Midwest, West)
    - District-level predictions with proper uncertainty
    """

    # Model parameters (calibrated from historical data)
    PARAMS = {
        # Fundamentals weights
        "generic_ballot_weight": 0.6,
        "approval_weight": 0.25,
        "economic_weight": 0.15,

        # Midterm penalty (for president's party, in midterm years)
        "midterm_penalty": -3.5,  # Points against president's party

        # Incumbency advantage
        "incumbency_advantage": 3.0,  # Points for incumbent party

        # Regional effects (relative to national)
        "regional_effects": {
            "Northeast": -1.5,  # More Democratic
            "Midwest": 0.5,
            "South": 2.0,  # More Republican
            "West": -0.5,
        },

        # Uncertainty parameters
        "national_uncertainty": 3.0,  # SD of national environment
        "regional_uncertainty": 1.5,  # SD of regional effects
        "district_uncertainty": 4.5,  # SD of district-level variation

        # Correlation parameters
        "national_correlation": 0.8,  # How much districts move together
        "regional_correlation": 0.6,  # How much same-region districts correlate
    }

    def __init__(
        self,
        districts_df: pd.DataFrame,
        generic_ballot_df: pd.DataFrame,
        approval_df: pd.DataFrame,
        n_simulations: int = 10000,
        random_seed: int = 42,
    ):
        """
        Initialize the forecast model.

        Args:
            districts_df: DataFrame with district fundamentals
            generic_ballot_df: DataFrame with generic ballot polls
            approval_df: DataFrame with presidential approval polls
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        self.districts = districts_df.copy()
        self.generic_ballot = generic_ballot_df.copy()
        self.approval = approval_df.copy()
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)

        # Initialize results storage
        self.national_env: Optional[NationalEnvironment] = None
        self.district_forecasts: dict[str, DistrictForecast] = {}
        self.seat_simulations: Optional[np.ndarray] = None

    def calculate_national_environment(self) -> NationalEnvironment:
        """Calculate the national political environment."""
        logger.info("Calculating national environment...")

        # Generic ballot average (weighted by recency)
        gb = self.generic_ballot.copy()
        gb["date"] = pd.to_datetime(gb["date"])
        gb = gb.sort_values("date", ascending=False)

        # Exponential decay weighting (more recent polls count more)
        days_old = (gb["date"].max() - gb["date"]).dt.days
        weights = np.exp(-days_old / 14)  # 14-day half-life
        gb_margin = np.average(gb["margin"], weights=weights)

        # Approval rating average
        app = self.approval.copy()
        app["date"] = pd.to_datetime(app["date"])
        app = app.sort_values("date", ascending=False)
        days_old = (app["date"].max() - app["date"]).dt.days
        weights = np.exp(-days_old / 14)

        approval_rating = np.average(app["approve"], weights=weights)
        net_approval = np.average(app["net"], weights=weights)

        # Midterm bonus for out-party (assuming Republican president in 2026)
        # Democrats (out-party) get a bonus in midterms - historically ~3-4 points
        midterm_bonus = abs(self.PARAMS["midterm_penalty"])  # +3.5 for Dems

        # Economic index (placeholder - would use actual economic data)
        economic_index = 0.0  # Neutral for now

        # Calculate national swing (positive = favors Democrats)
        # Components:
        # - Generic ballot: direct D-R margin
        # - Approval: negative approval helps out-party (Dems)
        # - Economic: good economy helps incumbent party (hurts Dems)
        # - Midterm bonus: out-party historically gains
        national_swing = (
            self.PARAMS["generic_ballot_weight"] * gb_margin +
            self.PARAMS["approval_weight"] * (-net_approval / 10) +  # Bad approval helps Dems
            self.PARAMS["economic_weight"] * (-economic_index) +
            midterm_bonus  # Out-party bonus
        )
        midterm_penalty = -midterm_bonus  # For display (negative for president's party)

        self.national_env = NationalEnvironment(
            generic_ballot_margin=gb_margin,
            approval_rating=approval_rating,
            net_approval=net_approval,
            midterm_penalty=midterm_penalty,
            economic_index=economic_index,
            national_swing=national_swing,
        )

        logger.info(f"  Generic ballot: D+{gb_margin:.1f}")
        logger.info(f"  Approval: {approval_rating:.1f}% (net: {net_approval:.1f})")
        logger.info(f"  National swing: D+{national_swing:.1f}")

        return self.national_env

    def simulate_elections(self) -> np.ndarray:
        """
        Run Monte Carlo simulations of the election.

        Returns:
            Array of shape (n_simulations,) with Democratic seat counts
        """
        logger.info(f"Running {self.n_simulations:,} simulations...")

        if self.national_env is None:
            self.calculate_national_environment()

        n_districts = len(self.districts)
        n_sims = self.n_simulations

        # Pre-compute district-level constants (vectorized)
        baselines = 50 - self.districts["pvi"].values / 2

        # Incumbency effects
        incumbency = np.zeros(n_districts)
        incumbency[self.districts["incumbent_party"].values == "D"] = self.PARAMS["incumbency_advantage"] / 2
        incumbency[self.districts["incumbent_party"].values == "R"] = -self.PARAMS["incumbency_advantage"] / 2

        # Regional effects mapping
        regions = self.districts["region"].values
        region_list = list(self.PARAMS["regional_effects"].keys())
        region_effects = np.array([
            self.PARAMS["regional_effects"].get(r, 0) for r in regions
        ])

        # Region indices for each district
        region_indices = np.array([
            region_list.index(r) if r in region_list else 0 for r in regions
        ])

        # Generate all random samples at once (vectorized)
        national_shocks = self.rng.normal(0, self.PARAMS["national_uncertainty"], n_sims)
        regional_shocks = self.rng.normal(0, self.PARAMS["regional_uncertainty"], (n_sims, len(region_list)))
        district_shocks = self.rng.normal(0, self.PARAMS["district_uncertainty"], (n_sims, n_districts))

        # Calculate vote shares for all simulations at once
        # Shape: (n_sims, n_districts)
        vote_shares = (
            baselines[np.newaxis, :] +  # (1, n_districts)
            (self.national_env.national_swing + national_shocks[:, np.newaxis]) +  # (n_sims, 1)
            incumbency[np.newaxis, :] -  # (1, n_districts)
            region_effects[np.newaxis, :] -  # (1, n_districts)
            regional_shocks[:, region_indices] +  # (n_sims, n_districts)
            district_shocks  # (n_sims, n_districts)
        )

        # Clip to reasonable range
        vote_shares = np.clip(vote_shares, 5, 95)

        # Store results for each district
        for i, (_, district) in enumerate(self.districts.iterrows()):
            district_votes = vote_shares[:, i]

            self.district_forecasts[district["district_id"]] = DistrictForecast(
                district_id=district["district_id"],
                state=district["state"],
                prob_dem=np.mean(district_votes > 50),
                mean_vote_share=np.mean(district_votes),
                std_vote_share=np.std(district_votes),
                ci_90_low=np.percentile(district_votes, 5),
                ci_90_high=np.percentile(district_votes, 95),
                category=self._categorize_prob(np.mean(district_votes > 50)),
                simulated_outcomes=district_votes,
            )

        # Calculate seat totals
        dem_wins = vote_shares > 50
        self.seat_simulations = dem_wins.sum(axis=1)

        logger.info(f"  Median Dem seats: {np.median(self.seat_simulations):.0f}")
        logger.info(f"  90% CI: [{np.percentile(self.seat_simulations, 5):.0f}, "
                   f"{np.percentile(self.seat_simulations, 95):.0f}]")
        logger.info(f"  P(Dem majority): {np.mean(self.seat_simulations >= 218):.1%}")

        return self.seat_simulations

    def _categorize_prob(self, prob: float) -> str:
        """Categorize probability into rating."""
        if prob >= 0.85:
            return "safe_d"
        elif prob >= 0.70:
            return "likely_d"
        elif prob >= 0.55:
            return "lean_d"
        elif prob >= 0.45:
            return "toss_up"
        elif prob >= 0.30:
            return "lean_r"
        elif prob >= 0.15:
            return "likely_r"
        else:
            return "safe_r"

    def get_summary(self) -> dict:
        """Get summary statistics for the forecast."""
        if self.seat_simulations is None:
            self.simulate_elections()

        return {
            "prob_dem_majority": float(np.mean(self.seat_simulations >= 218)),
            "prob_rep_majority": float(np.mean(self.seat_simulations < 218)),
            "median_dem_seats": int(np.median(self.seat_simulations)),
            "median_rep_seats": 435 - int(np.median(self.seat_simulations)),
            "mean_dem_seats": float(np.mean(self.seat_simulations)),
            "ci_90_low": int(np.percentile(self.seat_simulations, 5)),
            "ci_90_high": int(np.percentile(self.seat_simulations, 95)),
            "ci_50_low": int(np.percentile(self.seat_simulations, 25)),
            "ci_50_high": int(np.percentile(self.seat_simulations, 75)),
            "national_environment": self.national_env.national_swing if self.national_env else 0,
            "generic_ballot_margin": self.national_env.generic_ballot_margin if self.national_env else 0,
            "approval_rating": self.national_env.approval_rating if self.national_env else 0,
            "net_approval": self.national_env.net_approval if self.national_env else 0,
        }

    def get_seat_distribution(self) -> dict:
        """Get probability distribution of seat outcomes."""
        if self.seat_simulations is None:
            self.simulate_elections()

        # Count occurrences of each seat count
        unique, counts = np.unique(self.seat_simulations, return_counts=True)
        probs = counts / len(self.seat_simulations)

        return {
            "dem_seats": unique.tolist(),
            "probabilities": probs.tolist(),
        }

    def get_district_forecasts(self) -> list[dict]:
        """Get forecasts for all districts."""
        if not self.district_forecasts:
            self.simulate_elections()

        forecasts = []
        for district_id, forecast in self.district_forecasts.items():
            # Get district info
            district_info = self.districts[
                self.districts["district_id"] == district_id
            ].iloc[0]

            forecasts.append({
                "id": district_id,
                "state": forecast.state,
                "district_number": int(district_info["district_number"]),
                "incumbent": {
                    "name": district_info["incumbent"],
                    "party": district_info["incumbent_party"],
                },
                "prob_dem": round(forecast.prob_dem, 3),
                "mean_vote_share": round(forecast.mean_vote_share, 1),
                "std_vote_share": round(forecast.std_vote_share, 1),
                "ci_90_low": round(forecast.ci_90_low, 1),
                "ci_90_high": round(forecast.ci_90_high, 1),
                "category": forecast.category,
                "pvi": float(district_info["pvi"]),
                "region": district_info["region"],
                "open_seat": bool(district_info["open_seat"]),
            })

        # Sort by competitiveness (closest to 50%)
        forecasts.sort(key=lambda x: abs(x["prob_dem"] - 0.5))

        return forecasts

    def get_category_counts(self) -> dict:
        """Get count of districts in each category."""
        if not self.district_forecasts:
            self.simulate_elections()

        categories = {
            "safe_d": 0,
            "likely_d": 0,
            "lean_d": 0,
            "toss_up": 0,
            "lean_r": 0,
            "likely_r": 0,
            "safe_r": 0,
        }

        for forecast in self.district_forecasts.values():
            categories[forecast.category] += 1

        return categories


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required data files."""
    districts = pd.read_csv(data_dir / "processed" / "districts.csv")
    generic_ballot = pd.read_csv(data_dir / "raw" / "generic_ballot.csv")
    approval = pd.read_csv(data_dir / "raw" / "approval.csv")

    return districts, generic_ballot, approval


def run_forecast(data_dir: Path, n_simulations: int = 10000) -> HouseForecastModel:
    """Run the full forecast pipeline."""
    logger.info("Loading data...")
    districts, generic_ballot, approval = load_data(data_dir)

    logger.info("Initializing model...")
    model = HouseForecastModel(
        districts_df=districts,
        generic_ballot_df=generic_ballot,
        approval_df=approval,
        n_simulations=n_simulations,
    )

    logger.info("Running simulations...")
    model.simulate_elections()

    return model
