#!/usr/bin/env python3
"""
Bayesian Hierarchical Forecasting Model for 2026 House Elections.

ACTIVE PIPELINE (default):
==========================
1. National environment inferred from VoteHub polls (NationalEnvironmentModel)
2. Parameters loaded from historical fitting (2018/2022 via ParameterFitter)
3. HierarchicalForecastModel runs posterior predictive Monte Carlo
4. If PyMC available: full MCMC sampling of parameters
5. If PyMC unavailable: parameters sampled from stored Gaussian posteriors

ALTERNATIVE PATHS:
==================
- Legacy mode (--legacy flag): Uses hardcoded parameters in flat simulation
- OLS fitting: Fast fallback when PyMC unavailable for parameter fitting
- Simple poll average: Fallback when PyMC unavailable for national environment

PVI SIGN CONVENTION: positive = D-leaning, negative = R-leaning
(e.g., D+10 = +10, R+10 = -10)

The national environment is the SINGLE driving variable - everything else flows from it.

UNUSED/EXPERIMENTAL:
====================
- District Poll Updates (Layer 3 in docstrings): NOT IMPLEMENTED
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# Import national environment model
try:
    from .national_environment import NationalEnvironmentModel
    from .hierarchical_model import (
        HierarchicalForecastModel,
        HierarchicalForecastResult,
        NationalPosterior,
    )
    from .parameter_fitting import ParameterFitter, LearnedParameters
    from .economic_fundamentals import EconomicFundamentals
except ImportError:
    from national_environment import NationalEnvironmentModel
    from hierarchical_model import (
        HierarchicalForecastModel,
        HierarchicalForecastResult,
        NationalPosterior,
    )
    from parameter_fitting import ParameterFitter, LearnedParameters
    from economic_fundamentals import EconomicFundamentals

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


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

        # Regional effects (relative to national) - FiveThirtyEight 10 regions
        # See: https://abcnews.go.com/538/538s-2024-presidential-election-forecast-works/story?id=113068753
        "regional_effects": {
            "New_England": -2.0,  # More Democratic
            "Mid_Atlantic_Northeast": -1.5,
            "Rust_Belt": 0.5,
            "Southeast": 1.0,
            "Deep_South": 3.0,  # More Republican
            "Texas_Region": 2.5,
            "Plains": 2.0,
            "Mountain": 1.5,
            "Southwest": 0.0,
            "Pacific": -1.0,
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
        national_environment: Optional[float] = None,
        national_uncertainty: Optional[float] = None,
        n_simulations: int = 10000,
        random_seed: int = 42,
        use_hierarchical: bool = True,
        learned_params: Optional[LearnedParameters] = None,
        include_economic: bool = True,
    ):
        """
        Initialize the forecast model.

        Args:
            districts_df: DataFrame with district fundamentals
            national_environment: Pre-computed national environment (D margin).
                                  If None, will be inferred from VoteHub polling.
            national_uncertainty: Uncertainty in national environment.
                                  If None, uses default or inferred value.
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
            use_hierarchical: If True, use hierarchical model with learned params.
                             If False, use legacy flat simulation.
            learned_params: Pre-loaded learned parameters (optional).
            include_economic: If True, adjust prior based on economic fundamentals.
        """
        self.districts = districts_df.copy()
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)
        self.use_hierarchical = use_hierarchical
        self.include_economic = include_economic
        self._learned_params = learned_params

        # National environment (the single driving variable)
        self._national_environment = national_environment
        self._national_uncertainty = national_uncertainty

        # Initialize results storage
        self.national_env: Optional[NationalEnvironment] = None
        self.district_forecasts: dict[str, DistrictForecast] = {}
        self.seat_simulations: Optional[np.ndarray] = None
        self._hierarchical_result: Optional[HierarchicalForecastResult] = None

    def calculate_national_environment(self) -> NationalEnvironment:
        """
        Calculate the national political environment.

        POLL-ANCHORED APPROACH (Option A):
        The generic ballot already reflects voter sentiment including:
        - Economic conditions
        - Midterm context
        - Presidential approval
        - Candidate/party brand

        We use polls directly as the national environment with appropriate
        uncertainty (σ ≈ 2.5-3.0 for generic ballot error). No additional
        midterm or economic adjustments - that would be double-counting.

        Fundamentals (midterm penalty, economic index) are NOT added on top.
        They are implicitly captured in the polls already.
        """
        logger.info("Calculating national environment from VoteHub polling...")

        # If national environment was provided, use it directly
        if self._national_environment is not None:
            national_swing = self._national_environment
            uncertainty = self._national_uncertainty or self.PARAMS["national_uncertainty"]
            logger.info(f"  Using provided national environment: D{national_swing:+.1f}")
        else:
            # Infer from VoteHub polling data using Bayesian model
            env_model = NationalEnvironmentModel()
            env_model.load_polls()
            result = env_model.fit(use_pymc=True)  # Full Bayesian inference

            national_swing = result["national_environment"]
            uncertainty = result["uncertainty"]

            # Update the parameter for simulations
            self.PARAMS["national_uncertainty"] = uncertainty

            logger.info(f"  Inferred national environment: D{national_swing:+.1f} ± {uncertainty:.1f}")

        # IMPORTANT: No midterm bonus or economic adjustment added here!
        # The generic ballot already captures these effects.
        # Adding them would be double-counting (post-treatment bias).
        total_swing = national_swing

        logger.info(f"  Total national swing: D{total_swing:+.1f}")
        logger.info(f"  (No midterm/economic adjustments - polls already capture these)")

        self.national_env = NationalEnvironment(
            generic_ballot_margin=national_swing,  # Raw polling margin
            approval_rating=0,  # Will be loaded from polls if available
            net_approval=0,
            midterm_penalty=0,  # Not used - polls capture this
            economic_index=0,   # Not used - polls capture this
            national_swing=total_swing,
        )

        return self.national_env

    def _run_hierarchical_simulation(self) -> np.ndarray:
        """
        Run hierarchical Bayesian simulation.

        Uses learned parameters and proper uncertainty propagation.
        """
        logger.info("Running hierarchical Bayesian simulation...")

        # Get national environment
        if self.national_env is None:
            self.calculate_national_environment()

        # Create national posterior
        national_posterior = NationalPosterior(
            mean=self.national_env.national_swing,
            std=self._national_uncertainty or self.PARAMS["national_uncertainty"],
        )

        # Load learned parameters if not provided
        learned_params = self._learned_params
        if learned_params is None:
            try:
                learned_params = ParameterFitter.load_parameters()
                logger.info("  Loaded learned parameters from historical fitting")
            except FileNotFoundError:
                logger.warning("  No learned parameters found, using defaults")
                learned_params = None

        # Create hierarchical model
        hierarchical_model = HierarchicalForecastModel(
            districts_df=self.districts,
            national_posterior=national_posterior,
            learned_params=learned_params,
            n_simulations=self.n_simulations,
        )

        # Run simulation
        result = hierarchical_model.run(use_pymc=True)
        self._hierarchical_result = result

        # Convert results to legacy format for compatibility
        for i, district_id in enumerate(result.district_ids):
            district_info = self.districts[self.districts["district_id"] == district_id].iloc[0]

            self.district_forecasts[district_id] = DistrictForecast(
                district_id=district_id,
                state=district_info["state"],
                prob_dem=result.prob_dem[i],
                mean_vote_share=result.mean_vote_share[i],
                std_vote_share=result.std_vote_share[i],
                ci_90_low=result.ci_90_low[i],
                ci_90_high=result.ci_90_high[i],
                category=self._categorize_prob(result.prob_dem[i]),
                simulated_outcomes=np.array([]),  # Not stored in hierarchical
            )

        self.seat_simulations = result.seat_simulations
        return self.seat_simulations

    def simulate_elections(self) -> np.ndarray:
        """
        Run Monte Carlo simulations of the election.

        Returns:
            Array of shape (n_simulations,) with Democratic seat counts
        """
        # Use hierarchical model if enabled
        if self.use_hierarchical:
            return self._run_hierarchical_simulation()

        logger.info(f"Running {self.n_simulations:,} legacy simulations...")

        if self.national_env is None:
            self.calculate_national_environment()

        n_districts = len(self.districts)
        n_sims = self.n_simulations

        # Pre-compute district-level constants (vectorized)
        # PVI SIGN CONVENTION: positive = D-leaning, negative = R-leaning
        # Adding PVI increases Dem vote share (e.g., D+10 district → +5 points to Dem baseline)
        baselines = 50 + self.districts["pvi"].values / 2

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
        vote_shares = np.clip(vote_shares, 0, 100)

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

        summary = {
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
            "model_type": "hierarchical_bayesian" if self.use_hierarchical else "legacy",
        }

        # Add economic index if available
        if self.national_env and self.national_env.economic_index != 0:
            summary["economic_adjustment"] = self.national_env.economic_index

        return summary

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


def load_districts(data_dir: Path) -> pd.DataFrame:
    """Load district fundamentals data."""
    return pd.read_csv(data_dir / "processed" / "districts.csv")


def run_forecast(
    data_dir: Path,
    n_simulations: int = 10000,
    national_environment: Optional[float] = None,
) -> HouseForecastModel:
    """
    Run the full forecast pipeline.

    Args:
        data_dir: Path to data directory
        n_simulations: Number of Monte Carlo simulations
        national_environment: Pre-computed national environment (optional).
                              If None, will be inferred from VoteHub polling.
    """
    logger.info("Loading district data...")
    districts = load_districts(data_dir)

    logger.info("Initializing model...")
    model = HouseForecastModel(
        districts_df=districts,
        national_environment=national_environment,
        n_simulations=n_simulations,
    )

    logger.info("Running simulations...")
    model.simulate_elections()

    return model
