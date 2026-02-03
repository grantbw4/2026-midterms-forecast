#!/usr/bin/env python3
"""
Bayesian Hierarchical Model for Election Forecasting.

ACTIVE STRUCTURE (what is actually implemented):
================================================
1. National Environment - Posterior from NationalEnvironmentModel (poll aggregation)
2. District Fundamentals - PVI + Incumbency + Regional effects (from districts.csv)
3. Election Simulation - Posterior predictive Monte Carlo OR PyMC MCMC

UNIMPLEMENTED:
==============
- District Poll Updates (Layer 3 in some docstrings): NOT IMPLEMENTED
  District-level polls are not incorporated into predictions.

PVI SIGN CONVENTION: positive = D-leaning, negative = R-leaning
(e.g., D+10 = +10, R+10 = -10)

Uses learned parameters from historical fitting (parameter_fitting.py).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Configure PyTensor to use Python mode to avoid C compilation issues on macOS
# Must be set before importing pymc
os.environ.setdefault("PYTENSOR_FLAGS", "device=cpu,floatX=float64,optimizer=fast_compile,cxx=")

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None

from .parameter_fitting import LearnedParameters, ParameterFitter, REGIONS, STATE_TO_REGION

logger = logging.getLogger(__name__)


def compute_pvi_scaled_sigma(
    pvi: np.ndarray,
    sigma_base: float = 4.5,
    sigma_floor: float = 2.0,
    midpoint: float = 15.0,
    steepness: float = 0.15,
) -> np.ndarray:
    """
    Compute PVI-scaled uncertainty using logistic decay.

    Safer districts (higher |PVI|) get less uncertainty, reflecting that
    upsets are historically rarer in non-competitive districts.

    PVI SIGN CONVENTION: positive = D-leaning, negative = R-leaning
    (e.g., D+10 = +10, R+10 = -10)

    Args:
        pvi: Array of PVI values (positive = D-leaning, negative = R-leaning)
        sigma_base: Maximum uncertainty for competitive districts (|PVI| near 0)
        sigma_floor: Minimum uncertainty for safe districts (|PVI| >> midpoint)
        midpoint: PVI value at which sigma is halfway between base and floor
        steepness: How quickly sigma decays (higher = sharper transition)

    Returns:
        Array of sigma values for each district

    The logistic function ensures:
        - |PVI| ≈ 0: sigma ≈ sigma_base (competitive districts, full uncertainty)
        - |PVI| = midpoint: sigma ≈ (sigma_base + sigma_floor) / 2
        - |PVI| >> midpoint: sigma → sigma_floor (safe districts, minimal uncertainty)
    """
    abs_pvi = np.abs(pvi)
    # Logistic decay: high at low PVI, low at high PVI
    logistic = 1 / (1 + np.exp(steepness * (abs_pvi - midpoint)))
    return sigma_floor + (sigma_base - sigma_floor) * logistic

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


@dataclass
class NationalPosterior:
    """Posterior distribution for national environment."""
    mean: float
    std: float
    samples: Optional[np.ndarray] = None

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from the posterior."""
        if self.samples is not None and len(self.samples) >= n:
            return rng.choice(self.samples, size=n, replace=True)
        return rng.normal(self.mean, self.std, size=n)


@dataclass
class HierarchicalForecastResult:
    """Results from hierarchical model forecast."""
    # District-level results
    district_ids: list[str]
    prob_dem: np.ndarray  # Shape: (n_districts,)
    mean_vote_share: np.ndarray  # Shape: (n_districts,)
    std_vote_share: np.ndarray  # Shape: (n_districts,)
    ci_90_low: np.ndarray
    ci_90_high: np.ndarray

    # Aggregate results
    seat_simulations: np.ndarray  # Shape: (n_simulations,)
    prob_dem_majority: float
    median_dem_seats: int
    mean_dem_seats: float

    # National environment
    national_mean: float
    national_std: float

    # Metadata
    n_simulations: int
    n_districts: int


class HierarchicalForecastModel:
    """
    Posterior predictive Monte Carlo simulation for election forecasting.

    Uses PyMC-fitted parameters from historical data (2018, 2022) to run
    posterior predictive simulations. Uncertainty flows through model
    structure via parameter sampling, not additive noise shocks.

    Model Structure:
    ----------------
    PVI SIGN CONVENTION: positive = D-leaning, negative = R-leaning
    (e.g., D+10 = +10, R+10 = -10)

    For each simulation, sample:
        β_pvi ~ Normal(fitted_mean, fitted_std)
        β_inc ~ Normal(fitted_mean, fitted_std)
        μ_region ~ Normal(0, σ_regional)
        μ_national ~ Normal(poll_posterior_mean, poll_posterior_std)

    Then compute:
        fundamentals_d = 50 + β_pvi × PVI_d + β_inc × Incumbency_d
                        + μ_region[region_d] + β_national × μ_national

        vote_share_d = fundamentals_d + ε_d,  where ε_d ~ Normal(0, σ_district)

    Since PVI is positive for D-leaning districts and β_pvi > 0, adding
    β_pvi × PVI increases Dem vote share for D-leaning districts (correct).

    Parameters fitted from historical data:
        - β_pvi: PVI coefficient
        - β_inc: Incumbency advantage
        - σ_regional: Regional effect magnitude
        - σ_district: District-level noise
    """

    # Default uncertainty parameters (can be overridden by learned params)
    DEFAULT_SIGMA_NATIONAL = 3.0
    DEFAULT_SIGMA_REGIONAL = 1.5
    DEFAULT_SIGMA_DISTRICT = 4.5

    def __init__(
        self,
        districts_df: pd.DataFrame,
        national_posterior: Optional[NationalPosterior] = None,
        learned_params: Optional[LearnedParameters] = None,
        n_simulations: int = 10000,
        random_seed: int = 42,
    ):
        """
        Initialize the hierarchical forecast model.

        Args:
            districts_df: DataFrame with district fundamentals
            national_posterior: Posterior from national environment model
            learned_params: Parameters learned from historical fitting
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        self.districts = districts_df.copy()
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)

        # Load or set learned parameters
        if learned_params is None:
            try:
                self.params = ParameterFitter.load_parameters()
                logger.info("Loaded learned parameters from file")
            except FileNotFoundError:
                logger.warning("No learned parameters found, using defaults")
                self.params = self._default_parameters()
        else:
            self.params = learned_params

        # National environment posterior
        if national_posterior is None:
            self.national = NationalPosterior(mean=0.0, std=self.DEFAULT_SIGMA_NATIONAL)
        else:
            self.national = national_posterior

        # Prepare district data
        self._prepare_districts()

    def _default_parameters(self) -> LearnedParameters:
        """Return default parameters if no learned params available."""
        return LearnedParameters(
            beta_pvi_mean=0.5,
            beta_pvi_std=0.1,
            beta_inc_mean=3.0,
            beta_inc_std=1.0,
            beta_national_mean=0.4,
            beta_national_std=0.05,
            regional_effects={r: 0.0 for r in REGIONS},
            regional_effects_std={r: 1.0 for r in REGIONS},
            sigma_national=self.DEFAULT_SIGMA_NATIONAL,
            sigma_regional=self.DEFAULT_SIGMA_REGIONAL,
            sigma_district=self.DEFAULT_SIGMA_DISTRICT,
            n_districts_fitted=0,
            years_used=[],
            rmse=5.0,
            r_squared=0.9,
        )

    def _prepare_districts(self) -> None:
        """Prepare district data for simulation."""
        # Ensure region column exists
        if "region" not in self.districts.columns:
            self.districts["region"] = self.districts["state"].map(STATE_TO_REGION)

        # Create region index
        self.region_list = REGIONS
        self.districts["region_idx"] = self.districts["region"].map(
            {r: i for i, r in enumerate(self.region_list)}
        )

        # Encode incumbency
        # D incumbent = +1, R incumbent = -1, Open = 0
        inc_map = {"D": 1, "R": -1}
        if "incumbency_code" not in self.districts.columns:
            self.districts["incumbency_code"] = self.districts["incumbent_party"].map(inc_map).fillna(0)

        # Handle open seats
        if "open_seat" in self.districts.columns:
            self.districts.loc[self.districts["open_seat"] == True, "incumbency_code"] = 0

        self.n_districts = len(self.districts)
        logger.info(f"Prepared {self.n_districts} districts for simulation")

    def calculate_fundamentals(self, national_shift: float = 0.0) -> np.ndarray:
        """
        Calculate district fundamentals.

        fundamentals_d = 50 + β_pvi × PVI_d + β_inc × Incumbency_d
                        + μ_region[region_d] + national_shift

        Args:
            national_shift: Shift from baseline (national_env - 50)

        Returns:
            Array of fundamental vote shares for each district
        """
        pvi = self.districts["pvi"].values
        inc = self.districts["incumbency_code"].values
        region_idx = self.districts["region_idx"].values.astype(int)

        # Get regional effects as array
        regional_effects = np.array([
            self.params.regional_effects.get(r, 0.0) for r in self.region_list
        ])

        # Calculate fundamentals
        fundamentals = (
            50.0
            + self.params.beta_pvi_mean * pvi
            + self.params.beta_inc_mean * inc
            + regional_effects[region_idx]
            + national_shift
        )

        return fundamentals

    def simulate_elections(self) -> HierarchicalForecastResult:
        """
        Posterior predictive Monte Carlo simulation.

        For each simulation:
        1. Sample national environment from posterior
        2. Sample parameters (beta_pvi, beta_inc, regional_effects) from fitted posteriors
        3. Calculate fundamentals with sampled parameters
        4. Add district-level noise only
        5. Determine winners and count seats

        This is a true posterior predictive simulation - uncertainty flows
        through the model structure, not through added noise shocks.

        Returns:
            HierarchicalForecastResult with full results
        """
        logger.info(f"Running {self.n_simulations:,} posterior predictive simulations...")

        n_sims = self.n_simulations
        n_districts = self.n_districts
        n_regions = len(self.region_list)

        # Get district data
        pvi = self.districts["pvi"].values
        inc = self.districts["incumbency_code"].values
        region_idx = self.districts["region_idx"].values.astype(int)

        # Compute PVI-scaled uncertainty for each district
        # Safer districts get less noise, reflecting historical upset rates
        sigma_per_district = compute_pvi_scaled_sigma(
            pvi,
            sigma_base=self.DEFAULT_SIGMA_DISTRICT,
            sigma_floor=2.0,
            midpoint=15.0,
            steepness=0.15,
        )
        logger.info(f"  PVI-scaled σ range: [{sigma_per_district.min():.2f}, {sigma_per_district.max():.2f}]")

        # Sample national environment for each simulation
        national_samples = self.national.sample(n_sims, self.rng)

        # Sample parameters for each simulation (posterior predictive)
        beta_pvi_samples = self.rng.normal(
            self.params.beta_pvi_mean, self.params.beta_pvi_std, size=n_sims
        )
        beta_inc_samples = self.rng.normal(
            self.params.beta_inc_mean, self.params.beta_inc_std, size=n_sims
        )
        beta_national_samples = self.rng.normal(
            self.params.beta_national_mean, self.params.beta_national_std, size=n_sims
        )
        # NOTE: Regional effects are sampled from Normal(0, σ_regional), NOT from
        # the fitted regional_effects posteriors. The stored regional_effects dict
        # in LearnedParameters is NOT used here - only sigma_regional is used.
        # This treats regional effects as exchangeable random effects rather than
        # using the specific fitted values for each region.
        regional_effects_samples = self.rng.normal(
            0, self.params.sigma_regional, size=(n_sims, n_regions)
        )

        # District shocks with PVI-scaled uncertainty (independent per district)
        # Each district gets its own sigma based on how competitive it is
        district_shocks = self.rng.normal(0, 1, size=(n_sims, n_districts)) * sigma_per_district

        # Calculate vote shares for all simulations
        # Shape: (n_sims, n_districts)
        vote_shares = np.zeros((n_sims, n_districts))

        for sim in range(n_sims):
            # Sampled parameters for this simulation
            beta_pvi = beta_pvi_samples[sim]
            beta_inc = beta_inc_samples[sim]
            beta_national = beta_national_samples[sim]
            regional_effects = regional_effects_samples[sim]
            national_shift = national_samples[sim]

            # Calculate fundamentals with sampled parameters
            # Scale national environment by beta_national (typically ~0.4)
            fundamentals = (
                50.0
                + beta_pvi * pvi
                + beta_inc * inc
                + regional_effects[region_idx]
                + beta_national * national_shift
            )

            # Add only district-level noise
            vote_shares[sim, :] = fundamentals + district_shocks[sim, :]

        # Clip to reasonable range
        vote_shares = np.clip(vote_shares, 0, 100)

        # Calculate district-level statistics
        prob_dem = np.mean(vote_shares > 50, axis=0)
        mean_vote_share = np.mean(vote_shares, axis=0)
        std_vote_share = np.std(vote_shares, axis=0)
        ci_90_low = np.percentile(vote_shares, 5, axis=0)
        ci_90_high = np.percentile(vote_shares, 95, axis=0)

        # Calculate seat totals
        dem_wins = vote_shares > 50
        seat_simulations = dem_wins.sum(axis=1)

        # Aggregate statistics
        prob_dem_majority = np.mean(seat_simulations >= 218)
        median_dem_seats = int(np.median(seat_simulations))
        mean_dem_seats = float(np.mean(seat_simulations))

        logger.info(f"  National environment: D{self.national.mean:+.1f} ± {self.national.std:.1f}")
        logger.info(f"  Median Dem seats: {median_dem_seats}")
        logger.info(f"  90% CI: [{np.percentile(seat_simulations, 5):.0f}, {np.percentile(seat_simulations, 95):.0f}]")
        logger.info(f"  P(Dem majority): {prob_dem_majority:.1%}")

        return HierarchicalForecastResult(
            district_ids=self.districts["district_id"].tolist(),
            prob_dem=prob_dem,
            mean_vote_share=mean_vote_share,
            std_vote_share=std_vote_share,
            ci_90_low=ci_90_low,
            ci_90_high=ci_90_high,
            seat_simulations=seat_simulations,
            prob_dem_majority=prob_dem_majority,
            median_dem_seats=median_dem_seats,
            mean_dem_seats=mean_dem_seats,
            national_mean=self.national.mean,
            national_std=self.national.std,
            n_simulations=n_sims,
            n_districts=n_districts,
        )

    def fit_with_pymc(self, draws: int = 1000, tune: int = 1000) -> HierarchicalForecastResult:
        """
        Full Bayesian fit using PyMC with posterior predictive sampling.

        This samples the hierarchical parameters (national, regional, coefficients)
        and then generates vote share predictions via posterior predictive sampling.
        This is the correct approach for forecasting when we have no observed vote data.
        """
        if not PYMC_AVAILABLE:
            logger.warning("PyMC not available, falling back to Monte Carlo simulation")
            return self.simulate_elections()

        logger.info("Fitting hierarchical model with PyMC (posterior predictive)...")

        # Prepare data
        pvi = self.districts["pvi"].values
        inc = self.districts["incumbency_code"].values
        region_idx = self.districts["region_idx"].values.astype(int)
        n_districts = self.n_districts
        n_regions = len(self.region_list)

        # Compute PVI-scaled sigma for each district
        sigma_per_district = compute_pvi_scaled_sigma(
            pvi,
            sigma_base=self.DEFAULT_SIGMA_DISTRICT,
            sigma_floor=2.0,
            midpoint=15.0,
            steepness=0.15,
        )

        with pm.Model() as model:
            # Level 1: National environment (prior from poll aggregation)
            mu_national = pm.Normal(
                "mu_national",
                mu=self.national.mean,
                sigma=self.national.std
            )

            # Parameters with informative priors from historical fitting
            beta_pvi = pm.Normal(
                "beta_pvi",
                mu=self.params.beta_pvi_mean,
                sigma=self.params.beta_pvi_std
            )
            beta_inc = pm.Normal(
                "beta_inc",
                mu=self.params.beta_inc_mean,
                sigma=self.params.beta_inc_std
            )
            beta_national = pm.Normal(
                "beta_national",
                mu=self.params.beta_national_mean,
                sigma=self.params.beta_national_std
            )

            # Regional effects (use non-centered parameterization for better sampling)
            sigma_region = pm.HalfNormal("sigma_region", sigma=self.params.sigma_regional)
            regional_effects_raw = pm.Normal("regional_effects_raw", mu=0, sigma=1, shape=n_regions)
            regional_effects = pm.Deterministic("regional_effects", regional_effects_raw * sigma_region)

            # District fundamentals (deterministic given parameters)
            fundamentals = pm.Deterministic(
                "fundamentals",
                50.0
                + beta_pvi * pvi
                + beta_inc * inc
                + regional_effects[region_idx]
                + beta_national * mu_national
            )

            # Sample only the parameters (not vote shares - those come from posterior predictive)
            # Use higher target_accept for better exploration
            try:
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    cores=4,
                    chains=4,
                    init="jitter+adapt_diag",
                    target_accept=0.9,
                    return_inferencedata=True,
                    progressbar=True,
                    random_seed=42,
                    nuts_sampler="numpyro",
                )
            except (ImportError, Exception) as e:
                logger.info(f"NumPyro not available ({e}), using default sampler")
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    cores=4,
                    chains=4,
                    init="jitter+adapt_diag",
                    target_accept=0.9,
                    return_inferencedata=True,
                    progressbar=True,
                    random_seed=42,
                )

        # Extract fundamentals from posterior
        fundamentals_samples = trace.posterior["fundamentals"].values
        # Shape: (chains, draws, n_districts) -> (n_samples, n_districts)
        fundamentals_flat = fundamentals_samples.reshape(-1, n_districts)
        n_samples = len(fundamentals_flat)

        # Generate vote shares via posterior predictive sampling
        # Add district-level noise with PVI-scaled uncertainty
        rng = np.random.default_rng(42)
        district_noise = rng.normal(0, 1, size=(n_samples, n_districts)) * sigma_per_district
        vote_share_flat = np.clip(fundamentals_flat + district_noise, 0, 100)

        # Calculate statistics
        prob_dem = np.mean(vote_share_flat > 50, axis=0)
        mean_vote_share = np.mean(vote_share_flat, axis=0)
        std_vote_share = np.std(vote_share_flat, axis=0)
        ci_90_low = np.percentile(vote_share_flat, 5, axis=0)
        ci_90_high = np.percentile(vote_share_flat, 95, axis=0)

        # Seat counts
        dem_wins = vote_share_flat > 50
        seat_simulations = dem_wins.sum(axis=1)

        prob_dem_majority = np.mean(seat_simulations >= 218)
        median_dem_seats = int(np.median(seat_simulations))
        mean_dem_seats = float(np.mean(seat_simulations))

        logger.info(f"  PyMC fit complete with {n_samples} posterior samples")
        logger.info(f"  Median Dem seats: {median_dem_seats}")
        logger.info(f"  P(Dem majority): {prob_dem_majority:.1%}")

        return HierarchicalForecastResult(
            district_ids=self.districts["district_id"].tolist(),
            prob_dem=prob_dem,
            mean_vote_share=mean_vote_share,
            std_vote_share=std_vote_share,
            ci_90_low=ci_90_low,
            ci_90_high=ci_90_high,
            seat_simulations=seat_simulations,
            prob_dem_majority=prob_dem_majority,
            median_dem_seats=median_dem_seats,
            mean_dem_seats=mean_dem_seats,
            national_mean=float(trace.posterior["mu_national"].mean()),
            national_std=float(trace.posterior["mu_national"].std()),
            n_simulations=n_samples,
            n_districts=n_districts,
        )

    def run(self, use_pymc: bool = False) -> HierarchicalForecastResult:
        """
        Run the hierarchical forecast.

        Args:
            use_pymc: Whether to use full PyMC model (slower)

        Returns:
            HierarchicalForecastResult
        """
        if use_pymc and PYMC_AVAILABLE:
            return self.fit_with_pymc()
        else:
            return self.simulate_elections()


def load_districts() -> pd.DataFrame:
    """Load district data."""
    return pd.read_csv(DATA_PROCESSED / "districts.csv")


def main():
    """Test the hierarchical model."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Hierarchical Forecast Model Test")
    logger.info("=" * 60)

    # Load districts
    districts = load_districts()

    # Create national posterior (example: D+5 with uncertainty)
    national = NationalPosterior(mean=5.0, std=3.0)

    # Initialize model
    model = HierarchicalForecastModel(
        districts_df=districts,
        national_posterior=national,
        n_simulations=10000,
    )

    # Run simulation
    result = model.run(use_pymc=False)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"P(Dem majority): {result.prob_dem_majority:.1%}")
    logger.info(f"Median Dem seats: {result.median_dem_seats}")
    logger.info(f"Mean Dem seats: {result.mean_dem_seats:.1f}")
    logger.info(f"90% CI: [{np.percentile(result.seat_simulations, 5):.0f}, "
                f"{np.percentile(result.seat_simulations, 95):.0f}]")

    # Most competitive districts
    logger.info("\nMost competitive districts:")
    competitive = sorted(
        zip(result.district_ids, result.prob_dem),
        key=lambda x: abs(x[1] - 0.5)
    )[:10]
    for district_id, prob in competitive:
        logger.info(f"  {district_id}: {prob:.1%}")

    return result


if __name__ == "__main__":
    main()
