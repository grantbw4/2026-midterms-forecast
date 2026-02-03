#!/usr/bin/env python3
"""
CLI script to fit Bayesian model parameters on historical data.

Usage:
    python scripts/fit_historical.py [--use-pymc] [--draws 2000] [--cross-validate]

Outputs:
    data/processed/learned_params.json
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.parameter_fitting import ParameterFitter, PYMC_AVAILABLE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fit Bayesian model parameters on 2018/2022 historical data"
    )
    parser.add_argument(
        "--use-pymc",
        action="store_true",
        default=False,
        help="Use full PyMC Bayesian model (slower but more accurate)"
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=2000,
        help="Number of MCMC draws for PyMC (default: 2000)"
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        default=False,
        help="Run leave-one-year-out cross-validation"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2018, 2022],
        help="Years to use for fitting (default: 2018 2022)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for learned parameters (default: data/processed/learned_params.json)"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Historical Parameter Fitting")
    logger.info("=" * 60)
    logger.info(f"Years: {args.years}")
    logger.info(f"Method: {'PyMC Bayesian' if args.use_pymc else 'OLS'}")

    if args.use_pymc and not PYMC_AVAILABLE:
        logger.warning("PyMC not available, falling back to OLS")
        args.use_pymc = False

    # Initialize fitter
    fitter = ParameterFitter(years=args.years)

    # Load data
    logger.info("\nLoading historical data...")
    try:
        fitter.load_historical_data()
    except FileNotFoundError as e:
        logger.error(f"Missing data file: {e}")
        sys.exit(1)

    # Cross-validation (optional)
    if args.cross_validate:
        logger.info("\n" + "-" * 60)
        logger.info("Cross-Validation")
        logger.info("-" * 60)
        cv_results = fitter.cross_validate()
        logger.info(f"\nCross-validation results:")
        for year, metrics in cv_results.items():
            if year == "average":
                logger.info(f"  AVERAGE - RMSE: {metrics['rmse']:.2f}, Accuracy: {metrics['accuracy']:.1%}")
            else:
                logger.info(f"  {year} - RMSE: {metrics['rmse']:.2f}, Accuracy: {metrics['accuracy']:.1%}")

    # Fit on all data
    logger.info("\n" + "-" * 60)
    logger.info("Fitting on All Data")
    logger.info("-" * 60)

    if args.use_pymc:
        params = fitter.fit_pymc(draws=args.draws)
    else:
        params = fitter.fit_ols()

    # Save parameters
    output_path = Path(args.output) if args.output else None
    fitter.save_parameters(output_path)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("LEARNED PARAMETERS")
    logger.info("=" * 60)
    logger.info(f"β_pvi (PVI coefficient):     {params.beta_pvi_mean:.4f} ± {params.beta_pvi_std:.4f}")
    logger.info(f"β_inc (Incumbency advantage): {params.beta_inc_mean:.2f} ± {params.beta_inc_std:.2f}")
    logger.info("")
    logger.info("Regional effects (relative to South baseline):")
    for region, effect in sorted(params.regional_effects.items()):
        std = params.regional_effects_std.get(region, 0)
        logger.info(f"  {region:12s}: {effect:+.2f} ± {std:.2f}")
    logger.info("")
    logger.info("Uncertainty parameters:")
    logger.info(f"  σ_national:  {params.sigma_national:.2f}")
    logger.info(f"  σ_regional:  {params.sigma_regional:.2f}")
    logger.info(f"  σ_district:  {params.sigma_district:.2f}")
    logger.info("")
    logger.info("Fit statistics:")
    logger.info(f"  RMSE: {params.rmse:.2f}")
    logger.info(f"  R²:   {params.r_squared:.3f}")
    logger.info(f"  N districts: {params.n_districts_fitted}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
