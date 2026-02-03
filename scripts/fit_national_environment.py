#!/usr/bin/env python3
"""
Fit national environment model coefficients from historical midterm data.

Uses Bayesian shrinkage to combine:
1. Academic literature priors for economic effects
2. Limited historical data (2018, 2022 midterms)

This approach prevents overfitting to just 2 data points while incorporating
domain knowledge from political science research.

Key references for priors:
- Abramowitz (2018): Economy affects midterms ~1-2 pts per SD
- Hibbs "Bread and Peace" model: ~0.5-1 pt per economic unit
- Fair model: Economic growth coefficient ~0.5-1.0
- Erikson & Wlezien (2012): Polls dominate close to election

Model:
    actual_margin = β_polls × polling_margin + β_econ × economic_index + ε

With Bayesian shrinkage:
    β_econ ~ Normal(prior_mean, prior_std)
    posterior = prior weighted by data likelihood
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.economic_fundamentals import EconomicFundamentals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_HISTORICAL = PROJECT_ROOT / "data" / "historical"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Academic literature priors for economic effect
# Based on: Abramowitz, Hibbs, Fair, Campbell et al.
# Each SD of economic conditions affects House vote by ~0.5-1.5 points
PRIOR_BETA_ECON_MEAN = 0.5  # Conservative estimate (reduced from literature's 0.8)
PRIOR_BETA_ECON_STD = 0.4   # Uncertainty in the prior

# Prior for polling coefficient (should be close to 1.0)
PRIOR_BETA_POLLS_MEAN = 1.0
PRIOR_BETA_POLLS_STD = 0.1


def load_historical_data() -> list[dict]:
    """Load historical national environment data for midterm years."""
    years = [2018, 2022]
    data = []

    for year in years:
        path = DATA_HISTORICAL / f"national_environment_{year}.json"
        if path.exists():
            with open(path) as f:
                record = json.load(f)
                data.append(record)
                logger.info(f"Loaded {year} data")
        else:
            logger.warning(f"Missing data for {year}")

    return data


def calculate_historical_economic_indices(years: list[int]) -> dict[int, float]:
    """Calculate economic index for each historical year."""
    econ = EconomicFundamentals()
    econ.load_data()

    indices = {}
    for year in years:
        # Calculate index for October of election year (pre-election)
        target_date = datetime(year, 10, 1)
        result = econ.calculate_index(target_date, normalize=False)
        indices[year] = result["raw_index"]
        logger.info(f"  {year} economic index: {result['raw_index']:.2f}")

    return indices


def fit_bayesian_shrinkage(
    historical_data: list[dict],
    economic_indices: dict[int, float],
    prior_beta_econ_mean: float = PRIOR_BETA_ECON_MEAN,
    prior_beta_econ_std: float = PRIOR_BETA_ECON_STD,
) -> dict:
    """
    Fit β_econ using Bayesian shrinkage toward literature prior.

    With limited data (n=2), pure OLS would overfit. Instead, we use
    a Bayesian approach that shrinks the estimate toward the prior.

    The posterior is a weighted average of prior and data:
        posterior_mean = (prior_precision × prior_mean + data_precision × data_mean) /
                        (prior_precision + data_precision)

    Where precision = 1/variance.

    This gives us a defensible estimate that:
    - Uses domain knowledge when data is limited
    - Converges to data estimate as more elections are observed
    """
    n = len(historical_data)

    if n < 1:
        logger.error("Need at least 1 historical election")
        return None

    # Build arrays
    actual_margins = []
    polling_margins = []
    econ_indices = []

    for record in historical_data:
        year = record["year"]

        # Actual Democratic margin from results
        actual_dem_margin = record["dem_pct"] - record["rep_pct"]
        actual_margins.append(actual_dem_margin)

        # Polling margin (generic ballot)
        polling_margin = record["generic_ballot_final"]
        polling_margins.append(polling_margin)

        # Economic index, adjusted for president's party
        raw_econ = economic_indices.get(year, 0)
        if record["president_party"] == "R":
            adjusted_econ = -raw_econ  # Bad economy under R helps D
        else:
            adjusted_econ = raw_econ   # Bad economy under D hurts D
        econ_indices.append(adjusted_econ)

        logger.info(f"{year}: actual={actual_dem_margin:+.1f}, polls={polling_margin:+.1f}, "
                   f"econ_adj={adjusted_econ:+.2f} (pres={record['president_party']})")

    y = np.array(actual_margins)
    X_polls = np.array(polling_margins)
    X_econ = np.array(econ_indices)

    # Step 1: Estimate β_polls (assume it's close to 1.0)
    # Residual after accounting for polls tells us about economic effect
    beta_polls = 1.0  # Fix at 1.0 (polls predict outcome 1:1)

    # Residuals after polls
    residuals_after_polls = y - beta_polls * X_polls

    # Step 2: OLS estimate for β_econ from residuals
    # residual = β_econ × X_econ + ε
    if np.var(X_econ) > 0:
        # OLS: β = Cov(X,Y) / Var(X)
        beta_econ_ols = np.cov(X_econ, residuals_after_polls)[0, 1] / np.var(X_econ)

        # Estimate variance of OLS estimator
        # With only 2 points, we can't estimate σ² from residuals (df=0)
        # Instead, use a reasonable estimate of election-level noise
        # Historical generic ballot polling error is ~2-3 points
        sigma_election = 2.5
        var_beta_ols = (sigma_election ** 2) / np.sum(X_econ ** 2)

        # With n=2 and high leverage, uncertainty is substantial
        # Ensure variance reflects our limited information
        var_beta_ols = max(var_beta_ols, 0.3 ** 2)  # Floor at std=0.3
    else:
        beta_econ_ols = 0.0
        var_beta_ols = 10.0

    logger.info(f"\nOLS estimate (from data only):")
    logger.info(f"  β_econ_ols = {beta_econ_ols:.3f} ± {np.sqrt(var_beta_ols):.3f}")

    # Step 3: Bayesian shrinkage
    # Posterior = weighted average of prior and data
    prior_precision = 1.0 / (prior_beta_econ_std ** 2)
    data_precision = 1.0 / var_beta_ols

    posterior_precision = prior_precision + data_precision
    posterior_mean = (prior_precision * prior_beta_econ_mean + data_precision * beta_econ_ols) / posterior_precision
    posterior_std = np.sqrt(1.0 / posterior_precision)

    # Calculate shrinkage factor (how much we moved from OLS toward prior)
    shrinkage = prior_precision / posterior_precision  # 0 = all data, 1 = all prior

    logger.info(f"\nBayesian shrinkage:")
    logger.info(f"  Prior: β_econ ~ N({prior_beta_econ_mean:.2f}, {prior_beta_econ_std:.2f})")
    logger.info(f"  Data:  β_econ_ols = {beta_econ_ols:.3f}")
    logger.info(f"  Shrinkage factor: {shrinkage:.1%} toward prior")
    logger.info(f"  Posterior: β_econ = {posterior_mean:.3f} ± {posterior_std:.3f}")

    # Verify with historical data
    logger.info(f"\nVerification:")
    for i, record in enumerate(historical_data):
        year = record["year"]
        actual = y[i]
        predicted = beta_polls * X_polls[i] + posterior_mean * X_econ[i]
        error = actual - predicted
        logger.info(f"  {year}: predicted={predicted:+.1f}, actual={actual:+.1f}, error={error:+.1f}")

    # Calculate fit statistics with posterior estimate
    y_pred = beta_polls * X_polls + posterior_mean * X_econ
    residuals = y - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))

    return {
        "beta_polls": float(beta_polls),
        "beta_econ": float(posterior_mean),
        "beta_econ_std": float(posterior_std),
        "beta_econ_ols": float(beta_econ_ols),
        "shrinkage_factor": float(shrinkage),
        "prior_mean": float(prior_beta_econ_mean),
        "prior_std": float(prior_beta_econ_std),
        "rmse": float(rmse),
        "n_elections": n,
        "years_used": [r["year"] for r in historical_data],
        "method": "bayesian_shrinkage",
        "fitted_at": datetime.now().isoformat(),
    }


def fit_coefficients(historical_data: list[dict], economic_indices: dict[int, float]) -> dict:
    """
    Fit β_polls and β_econ using Bayesian shrinkage.

    This replaces the pure OLS approach with one that incorporates
    academic literature priors, preventing overfitting with limited data.
    """
    return fit_bayesian_shrinkage(historical_data, economic_indices)


def save_coefficients(coeffs: dict) -> None:
    """Save fitted coefficients to file."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    path = DATA_PROCESSED / "national_env_coefficients.json"
    with open(path, "w") as f:
        json.dump(coeffs, f, indent=2)

    logger.info(f"\nSaved coefficients to {path}")


def main():
    """Fit national environment coefficients from historical data."""
    logger.info("=" * 60)
    logger.info("Fitting National Environment Coefficients")
    logger.info("=" * 60)

    # Load historical election data
    logger.info("\nLoading historical data...")
    historical_data = load_historical_data()

    if len(historical_data) < 2:
        logger.error("Insufficient historical data")
        return 1

    # Calculate economic indices for those years
    logger.info("\nCalculating historical economic indices...")
    years = [r["year"] for r in historical_data]
    economic_indices = calculate_historical_economic_indices(years)

    # Fit coefficients
    logger.info("\nFitting coefficients...")
    coeffs = fit_coefficients(historical_data, economic_indices)

    if coeffs is None:
        return 1

    # Save
    save_coefficients(coeffs)

    # Verify with historical data
    logger.info("\n" + "=" * 60)
    logger.info("Verification (fitted vs actual)")
    logger.info("=" * 60)

    for record in historical_data:
        year = record["year"]
        actual = record["dem_pct"] - record["rep_pct"]
        polls = record["generic_ballot_final"]

        raw_econ = economic_indices.get(year, 0)
        if record["president_party"] == "R":
            adj_econ = -raw_econ
        else:
            adj_econ = raw_econ

        predicted = coeffs["beta_polls"] * polls + coeffs["beta_econ"] * adj_econ
        error = actual - predicted

        logger.info(f"{year}: predicted={predicted:+.1f}, actual={actual:+.1f}, error={error:+.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
