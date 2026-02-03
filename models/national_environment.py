#!/usr/bin/env python3
"""
Bayesian National Environment Model using PyMC.

This model infers the latent national political environment from polling data.
Each poll is treated as a noisy observation of the true underlying sentiment.

The inferred national environment drives all downstream forecasts:
- House district predictions
- Senate race predictions
- Regional adjustments
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None

PROJECT_ROOT = Path(__file__).parent.parent
DATA_POLLING = PROJECT_ROOT / "data" / "raw" / "polling"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class NationalEnvironmentModel:
    """
    Bayesian model to infer latent national political environment.

    Model structure:
    - mu: latent national environment (D margin on generic ballot scale)
    - sigma_pollster: pollster-specific house effects
    - sigma_poll: individual poll sampling error

    Each poll observation:
        y_i ~ Normal(mu + house_effect_pollster[i], sigma_i)

    where sigma_i depends on sample size and pollster quality.
    """

    def __init__(
        self,
        gb_polls: Optional[pd.DataFrame] = None,
        approval_polls: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize with polling data.

        Args:
            gb_polls: Generic ballot polls DataFrame
            approval_polls: Presidential approval polls DataFrame
        """
        self.gb_polls = gb_polls
        self.approval_polls = approval_polls
        self.trace = None
        self.national_environment = None
        self.uncertainty = None

        # Model parameters
        self.approval_weight = 0.3  # How much approval affects environment

    def load_polls(self) -> None:
        """Load polling data from disk."""
        gb_path = DATA_POLLING / "generic_ballot.csv"
        app_path = DATA_POLLING / "trump_approval.csv"

        if gb_path.exists():
            self.gb_polls = pd.read_csv(gb_path)
            self.gb_polls["date"] = pd.to_datetime(self.gb_polls["date"])
            logger.info(f"Loaded {len(self.gb_polls)} generic ballot polls")
        else:
            logger.warning("No generic ballot data found")
            self.gb_polls = pd.DataFrame()

        if app_path.exists():
            self.approval_polls = pd.read_csv(app_path)
            self.approval_polls["date"] = pd.to_datetime(self.approval_polls["date"])
            logger.info(f"Loaded {len(self.approval_polls)} approval polls")
        else:
            logger.warning("No approval data found")
            self.approval_polls = pd.DataFrame()

    def _calculate_poll_variance(self, sample_size: int, weight: float) -> float:
        """
        Calculate observation variance for a poll.

        Combines:
        - Sampling error: ~1/sqrt(n)
        - Pollster quality adjustment
        - Additional noise for uncertainty
        """
        # Base sampling error (margin has variance ~1/n for proportions)
        sampling_var = 4 * 50 * 50 / sample_size  # Assuming p=0.5 worst case

        # Adjust by pollster quality (higher weight = lower variance)
        quality_factor = max(0.5, weight) ** 2

        return sampling_var / quality_factor + 4.0  # Add base uncertainty

    def fit_simple(self) -> dict:
        """
        Simple weighted average fallback (no PyMC required).

        Returns estimated national environment.
        """
        logger.info("Using simple weighted average model")

        result = {
            "national_environment": 0.0,
            "uncertainty": 3.0,
            "method": "weighted_average",
        }

        # Generic ballot contribution
        if self.gb_polls is not None and not self.gb_polls.empty:
            weights = self.gb_polls["weight"].values
            margins = self.gb_polls["margin"].values
            gb_mean = (margins * weights).sum() / weights.sum()

            # Uncertainty from poll variance
            gb_var = np.average((margins - gb_mean) ** 2, weights=weights)
            gb_std = np.sqrt(gb_var / len(margins) + 2.0)  # Add irreducible uncertainty

            result["generic_ballot_mean"] = round(gb_mean, 2)
            result["generic_ballot_std"] = round(gb_std, 2)
        else:
            gb_mean = 0.0
            gb_std = 5.0

        # Approval contribution (convert to generic ballot scale)
        # Rule of thumb: each point of net approval ≈ 0.3 points on generic ballot
        if self.approval_polls is not None and not self.approval_polls.empty:
            weights = self.approval_polls["weight"].values
            net_approvals = self.approval_polls["net_approval"].values
            app_mean = (net_approvals * weights).sum() / weights.sum()

            # Convert to GB scale: negative approval helps Dems
            # Historical relationship: GB_margin ≈ -0.3 * net_approval + baseline
            app_contribution = -0.3 * app_mean

            result["approval_mean"] = round(app_mean, 2)
            result["approval_contribution"] = round(app_contribution, 2)
        else:
            app_contribution = 0.0

        # Combined estimate
        # Weight generic ballot more heavily since it's more direct
        combined = (1 - self.approval_weight) * gb_mean + self.approval_weight * app_contribution
        combined_std = np.sqrt(
            (1 - self.approval_weight) ** 2 * gb_std ** 2 +
            self.approval_weight ** 2 * 4.0  # Uncertainty in approval-to-GB conversion
        )

        result["national_environment"] = round(combined, 2)
        result["uncertainty"] = round(combined_std, 2)

        self.national_environment = result["national_environment"]
        self.uncertainty = result["uncertainty"]

        return result

    def fit_pymc(self, draws: int = 500, tune: int = 500) -> dict:
        """
        Full Bayesian model using PyMC.

        Treats each poll as a noisy observation of latent national environment.
        """
        if not PYMC_AVAILABLE:
            logger.warning("PyMC not available, falling back to simple model")
            return self.fit_simple()

        if self.gb_polls is None or self.gb_polls.empty:
            logger.warning("No polling data, using prior")
            return self.fit_simple()

        logger.info("Fitting PyMC Bayesian model...")

        # Prepare data
        margins = self.gb_polls["margin"].values
        sample_sizes = self.gb_polls["sample_size"].values
        weights = self.gb_polls["weight"].values
        pollsters = self.gb_polls["pollster"].values

        # Get unique pollsters and encode
        unique_pollsters = list(set(pollsters))
        pollster_idx = np.array([unique_pollsters.index(p) for p in pollsters])
        n_pollsters = len(unique_pollsters)

        # Calculate observation standard deviations
        obs_sigma = np.array([
            np.sqrt(self._calculate_poll_variance(n, w))
            for n, w in zip(sample_sizes, weights)
        ])

        with pm.Model() as model:
            # Priors
            # National environment: weakly informative, centered at 0
            mu = pm.Normal("mu", mu=0, sigma=5)

            # Pollster house effects (deviations from true value)
            # Hierarchical: each pollster has a house effect drawn from common distribution
            sigma_house = pm.HalfNormal("sigma_house", sigma=2)
            house_effects = pm.Normal(
                "house_effects",
                mu=0,
                sigma=sigma_house,
                shape=n_pollsters
            )

            # Additional polling error beyond sampling
            sigma_extra = pm.HalfNormal("sigma_extra", sigma=2)

            # Likelihood
            # Each poll is mu + pollster's house effect + noise
            poll_mean = mu + house_effects[pollster_idx]
            poll_sigma = pm.math.sqrt(obs_sigma ** 2 + sigma_extra ** 2)

            y = pm.Normal("y", mu=poll_mean, sigma=poll_sigma, observed=margins)

            # Sample with optimized settings
            # Try NumPyro backend for 2-10x speedup (requires JAX)
            try:
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    cores=4,
                    chains=4,
                    init="advi",
                    return_inferencedata=True,
                    progressbar=True,
                    nuts_sampler="numpyro",
                )
            except (ImportError, Exception) as e:
                logger.info(f"NumPyro not available ({e}), using default sampler")
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    cores=4,
                    chains=4,
                    init="advi",
                    return_inferencedata=True,
                    progressbar=True,
                )

        # Extract results
        mu_samples = self.trace.posterior["mu"].values.flatten()
        mu_mean = float(np.mean(mu_samples))
        mu_std = float(np.std(mu_samples))
        mu_hdi = az.hdi(self.trace, var_names=["mu"], hdi_prob=0.9)

        result = {
            "national_environment": round(mu_mean, 2),
            "uncertainty": round(mu_std, 2),
            "hdi_90_low": round(float(mu_hdi["mu"].values[0]), 2),
            "hdi_90_high": round(float(mu_hdi["mu"].values[1]), 2),
            "method": "pymc_bayesian",
            "n_polls": len(margins),
            "n_pollsters": n_pollsters,
            "sigma_house": round(float(self.trace.posterior["sigma_house"].mean()), 2),
            "sigma_extra": round(float(self.trace.posterior["sigma_extra"].mean()), 2),
        }

        # POST-HOC APPROVAL ADJUSTMENT (applied outside PyMC model)
        # This is NOT part of the Bayesian model - it's a deterministic adjustment
        # added after MCMC sampling completes. The uncertainty does not account
        # for uncertainty in the approval-to-generic-ballot relationship.
        if self.approval_polls is not None and not self.approval_polls.empty:
            weights = self.approval_polls["weight"].values
            net_approvals = self.approval_polls["net_approval"].values
            app_mean = (net_approvals * weights).sum() / weights.sum()

            # Store approval mean in result
            result["approval_mean"] = round(app_mean, 2)

            # Adjust national environment by approval
            # Formula: each point of net approval ≈ -0.3 × approval_weight points on GB
            # This is a heuristic, not a fitted relationship
            app_adjustment = -0.3 * app_mean * self.approval_weight
            result["national_environment"] = round(mu_mean + app_adjustment, 2)
            result["approval_adjustment"] = round(app_adjustment, 2)

        self.national_environment = result["national_environment"]
        self.uncertainty = result["uncertainty"]

        logger.info(f"National environment: D{result['national_environment']:+.1f} ± {result['uncertainty']:.1f}")

        return result

    def fit(self, use_pymc: bool = True) -> dict:
        """
        Fit the national environment model.

        Args:
            use_pymc: Whether to use full PyMC model (slower but more accurate)

        Returns:
            Dictionary with national environment estimate and uncertainty
        """
        if self.gb_polls is None:
            self.load_polls()

        if use_pymc and PYMC_AVAILABLE:
            return self.fit_pymc()
        else:
            return self.fit_simple()

    def save_results(self, result: dict) -> None:
        """Save model results to disk."""
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

        result["updated_at"] = datetime.now().isoformat()

        output_path = DATA_PROCESSED / "national_environment.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved national environment to {output_path}")

    def get_environment(self) -> tuple[float, float]:
        """
        Get current national environment estimate.

        Returns:
            (mean, std) of national environment
        """
        if self.national_environment is None:
            self.fit(use_pymc=False)

        return self.national_environment, self.uncertainty


def main():
    """Run national environment model."""
    logger.info("=" * 60)
    logger.info("National Environment Model")
    logger.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    model = NationalEnvironmentModel()
    model.load_polls()

    # Fit model (use simple model for speed, PyMC for production)
    result = model.fit(use_pymc=PYMC_AVAILABLE)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"National Environment: D{result['national_environment']:+.1f}")
    logger.info(f"Uncertainty (σ): {result['uncertainty']:.1f}")
    if "hdi_90_low" in result:
        logger.info(f"90% HDI: [{result['hdi_90_low']:.1f}, {result['hdi_90_high']:.1f}]")

    model.save_results(result)

    return result


if __name__ == "__main__":
    main()
