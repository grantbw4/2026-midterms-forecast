#!/usr/bin/env python3
"""
Parameter Fitting Module for Bayesian Hierarchical Model.

Fits model parameters (β_pvi, β_inc, regional effects, uncertainty terms)
using 2018 and 2022 historical midterm election data.

The learned parameters are then used as informed priors for 2026 forecasting.
"""

import json
import logging
from dataclasses import dataclass
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

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_HISTORICAL = PROJECT_ROOT / "data" / "historical"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Region mapping using FiveThirtyEight's 10 political regions
# See: https://abcnews.go.com/538/538s-2024-presidential-election-forecast-works/story?id=113068753
REGIONS = [
    "New_England",
    "Mid_Atlantic_Northeast",
    "Rust_Belt",
    "Southeast",
    "Deep_South",
    "Texas_Region",
    "Plains",
    "Mountain",
    "Southwest",
    "Pacific",
]

# State to region mapping (FiveThirtyEight 10 regions)
STATE_TO_REGION = {
    # New England
    "ME": "New_England", "NH": "New_England", "VT": "New_England", "MA": "New_England",
    # Mid-Atlantic/Northeast
    "NY": "Mid_Atlantic_Northeast", "NJ": "Mid_Atlantic_Northeast",
    "DE": "Mid_Atlantic_Northeast", "MD": "Mid_Atlantic_Northeast",
    "RI": "Mid_Atlantic_Northeast", "CT": "Mid_Atlantic_Northeast",
    # Rust Belt
    "IL": "Rust_Belt", "IN": "Rust_Belt", "OH": "Rust_Belt", "MI": "Rust_Belt",
    "WI": "Rust_Belt", "PA": "Rust_Belt", "MN": "Rust_Belt", "IA": "Rust_Belt",
    # Southeast
    "FL": "Southeast", "GA": "Southeast", "NC": "Southeast", "VA": "Southeast",
    # Deep South
    "SC": "Deep_South", "AL": "Deep_South", "MS": "Deep_South", "AR": "Deep_South",
    "TN": "Deep_South", "KY": "Deep_South", "WV": "Deep_South", "MO": "Deep_South",
    # Texas Region
    "TX": "Texas_Region", "OK": "Texas_Region", "LA": "Texas_Region",
    # Plains
    "ND": "Plains", "SD": "Plains", "NE": "Plains", "KS": "Plains",
    # Mountain
    "ID": "Mountain", "MT": "Mountain", "WY": "Mountain", "UT": "Mountain", "AK": "Mountain",
    # Southwest
    "AZ": "Southwest", "NV": "Southwest", "NM": "Southwest", "CO": "Southwest",
    # Pacific
    "CA": "Pacific", "OR": "Pacific", "WA": "Pacific", "HI": "Pacific",
}


@dataclass
class LearnedParameters:
    """Container for learned model parameters."""
    # Coefficients
    beta_pvi_mean: float
    beta_pvi_std: float
    beta_inc_mean: float
    beta_inc_std: float
    beta_national_mean: float
    beta_national_std: float

    # Regional effects (relative to baseline)
    regional_effects: dict[str, float]
    regional_effects_std: dict[str, float]

    # Uncertainty parameters
    sigma_national: float
    sigma_regional: float
    sigma_district: float

    # Metadata
    n_districts_fitted: int
    years_used: list[int]
    rmse: float
    r_squared: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "beta_pvi_mean": self.beta_pvi_mean,
            "beta_pvi_std": self.beta_pvi_std,
            "beta_inc_mean": self.beta_inc_mean,
            "beta_inc_std": self.beta_inc_std,
            "beta_national_mean": self.beta_national_mean,
            "beta_national_std": self.beta_national_std,
            "regional_effects": self.regional_effects,
            "regional_effects_std": self.regional_effects_std,
            "sigma_national": self.sigma_national,
            "sigma_regional": self.sigma_regional,
            "sigma_district": self.sigma_district,
            "n_districts_fitted": self.n_districts_fitted,
            "years_used": self.years_used,
            "rmse": self.rmse,
            "r_squared": self.r_squared,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LearnedParameters":
        """Create from dictionary."""
        # Handle older saved params without beta_national
        if "beta_national_mean" not in d:
            d["beta_national_mean"] = 0.4
            d["beta_national_std"] = 0.05
        return cls(**d)


class ParameterFitter:
    """
    Fit Bayesian model parameters from historical election data.

    Uses 2018 and 2022 midterm results to learn:
    - β_pvi: How much PVI affects vote share
    - β_inc: Incumbency advantage
    - Regional effects: How regions deviate from national trend
    - Uncertainty parameters: σ_national, σ_regional, σ_district
    """

    def __init__(self, years: list[int] = None):
        """
        Initialize the parameter fitter.

        Args:
            years: Years to use for fitting (default: [2018, 2022])
        """
        self.years = years or [2018, 2022]
        self.data: dict[int, pd.DataFrame] = {}
        self.combined_data: Optional[pd.DataFrame] = None
        self.trace = None
        self.learned_params: Optional[LearnedParameters] = None

    def load_historical_data(self) -> None:
        """Load and merge historical data for all years."""
        logger.info(f"Loading historical data for years: {self.years}")

        all_data = []

        for year in self.years:
            # Load election results (ground truth)
            val_path = DATA_HISTORICAL / f"validation_{year}.csv"
            if not val_path.exists():
                raise FileNotFoundError(f"Missing validation data: {val_path}")
            validation = pd.read_csv(val_path)

            # Load PVI
            pvi_path = DATA_HISTORICAL / f"partisan_lean_{year}.csv"
            if not pvi_path.exists():
                raise FileNotFoundError(f"Missing PVI data: {pvi_path}")
            pvi = pd.read_csv(pvi_path)

            # Load incumbency
            inc_path = DATA_HISTORICAL / f"incumbency_{year}.csv"
            if not inc_path.exists():
                raise FileNotFoundError(f"Missing incumbency data: {inc_path}")
            incumbency = pd.read_csv(inc_path)

            # Load national environment
            env_path = DATA_HISTORICAL / f"national_environment_{year}.json"
            if env_path.exists():
                with open(env_path) as f:
                    nat_env = json.load(f)
                national_margin = nat_env.get("generic_ballot_final", 0)
            else:
                logger.warning(f"No national environment for {year}, using 0")
                national_margin = 0

            # Merge data
            df = validation[["district_id", "dem_pct", "margin"]].copy()
            df = df.merge(
                pvi[["district_id", "pvi_numeric"]],
                on="district_id",
                how="left"
            )
            df = df.merge(
                incumbency[["district_id", "incumbent_party", "incumbency_code"]],
                on="district_id",
                how="left"
            )

            # Add year and national environment
            df["year"] = year
            df["national_margin"] = national_margin

            # Add state and region
            df["state"] = df["district_id"].str.split("-").str[0]
            df["region"] = df["state"].map(STATE_TO_REGION)

            # Filter out uncontested races (dem_pct == 0 or 100)
            n_before = len(df)
            df = df[(df["dem_pct"] > 0) & (df["dem_pct"] < 100)]
            n_after = len(df)
            logger.info(f"  {year}: {n_after} contested districts (removed {n_before - n_after} uncontested)")

            # Drop rows with missing data
            df = df.dropna(subset=["pvi_numeric", "incumbency_code", "region"])

            all_data.append(df)
            self.data[year] = df

        # Combine all years
        self.combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total observations: {len(self.combined_data)}")

    def fit_ols(self) -> LearnedParameters:
        """
        Fit parameters using ordinary least squares (fast fallback).

        Model:
            dem_vote_share = 50 + β_pvi * PVI + β_inc * Inc + regional_effect + national_swing + ε
        """
        logger.info("Fitting parameters using OLS...")

        if self.combined_data is None:
            self.load_historical_data()

        df = self.combined_data.copy()

        # Create design matrix
        # Target: dem_pct (actual Democratic vote share)
        y = df["dem_pct"].values

        # Features
        pvi = df["pvi_numeric"].values
        inc = df["incumbency_code"].values
        national = df["national_margin"].values

        # Region dummies (baseline = South, the largest region)
        region_dummies = pd.get_dummies(df["region"], drop_first=False)
        # Use South as baseline
        region_features = region_dummies.drop(columns=["South"], errors="ignore").values
        region_names = [c for c in region_dummies.columns if c != "South"]

        # Stack features: [intercept, pvi, inc, national, region_dummies]
        X = np.column_stack([
            np.ones(len(y)),  # Intercept (should be ~50)
            pvi,
            inc,
            national,
            region_features
        ])

        # OLS fit: β = (X'X)^(-1) X'y
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        # Extract coefficients
        intercept = beta[0]
        beta_pvi = beta[1]
        beta_inc = beta[2]
        beta_national = beta[3]
        beta_regions = beta[4:]

        # Calculate residuals and uncertainty
        y_pred = X @ beta
        residuals = y - y_pred
        rmse = np.sqrt(np.mean(residuals ** 2))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        # Estimate standard errors (assuming homoscedastic errors)
        n, p = X.shape
        mse = ss_res / (n - p)
        var_beta = mse * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(var_beta))

        # Regional effects dictionary
        regional_effects = {"South": 0.0}  # Baseline
        regional_effects_std = {"South": 0.0}
        for i, region in enumerate(region_names):
            regional_effects[region] = float(beta_regions[i])
            regional_effects_std[region] = float(se_beta[4 + i])

        # Estimate uncertainty components from residual structure
        # Group residuals by year to estimate national uncertainty
        year_residuals = df.groupby("year").apply(
            lambda x: (x["dem_pct"] - X[x.index, :] @ beta).mean()
        )
        sigma_national = float(year_residuals.std()) if len(year_residuals) > 1 else 3.0

        # Group by region to estimate regional uncertainty
        region_residuals = df.groupby(["year", "region"]).apply(
            lambda x: (x["dem_pct"] - X[x.index, :] @ beta).mean()
        )
        sigma_regional = float(region_residuals.std()) if len(region_residuals) > 1 else 1.5

        # Remaining is district-level
        sigma_district = rmse

        logger.info(f"  Intercept: {intercept:.2f} (should be ~50)")
        logger.info(f"  β_pvi: {beta_pvi:.3f} ± {se_beta[1]:.3f}")
        logger.info(f"  β_inc: {beta_inc:.2f} ± {se_beta[2]:.2f}")
        logger.info(f"  β_national: {beta_national:.3f} ± {se_beta[3]:.3f}")
        logger.info(f"  Regional effects: {regional_effects}")
        logger.info(f"  RMSE: {rmse:.2f}, R²: {r_squared:.3f}")

        self.learned_params = LearnedParameters(
            beta_pvi_mean=float(beta_pvi),
            beta_pvi_std=float(se_beta[1]),
            beta_inc_mean=float(beta_inc),
            beta_inc_std=float(se_beta[2]),
            beta_national_mean=float(beta_national),
            beta_national_std=float(se_beta[3]),
            regional_effects=regional_effects,
            regional_effects_std=regional_effects_std,
            sigma_national=max(sigma_national, 1.0),  # Floor at 1.0
            sigma_regional=max(sigma_regional, 0.5),
            sigma_district=sigma_district,
            n_districts_fitted=len(df),
            years_used=self.years,
            rmse=float(rmse),
            r_squared=float(r_squared),
        )

        return self.learned_params

    def fit_pymc(self, draws: int = 2000, tune: int = 1000) -> LearnedParameters:
        """
        Fit parameters using PyMC Bayesian model.

        Hierarchical structure:
        - β_pvi, β_inc: Shared across all districts
        - regional_effects: Drawn from common distribution
        - national_swing: Year-specific effect
        - ε: District-level noise
        """
        if not PYMC_AVAILABLE:
            logger.warning("PyMC not available, falling back to OLS")
            return self.fit_ols()

        logger.info("Fitting parameters using PyMC Bayesian model...")

        if self.combined_data is None:
            self.load_historical_data()

        df = self.combined_data.copy()

        # Prepare data
        y = df["dem_pct"].values
        pvi = df["pvi_numeric"].values
        inc = df["incumbency_code"].values
        national = df["national_margin"].values

        # Region indices
        region_idx = df["region"].map({r: i for i, r in enumerate(REGIONS)}).values

        # Year indices
        years_list = sorted(df["year"].unique())
        year_idx = df["year"].map({y: i for i, y in enumerate(years_list)}).values

        n_regions = len(REGIONS)
        n_years = len(years_list)

        with pm.Model() as model:
            # Priors on structural parameters
            # PVI coefficient: expect ~0.5 (each point of PVI -> 0.5 point vote share)
            beta_pvi = pm.Normal("beta_pvi", mu=0.5, sigma=0.2)

            # Incumbency advantage: expect ~3-5 points
            beta_inc = pm.Normal("beta_inc", mu=3.0, sigma=2.0)

            # National swing coefficient: expect ~1 (1:1 with generic ballot)
            beta_national = pm.Normal("beta_national", mu=1.0, sigma=0.3)

            # Regional effects (hierarchical)
            sigma_region = pm.HalfNormal("sigma_region", sigma=3.0)
            regional_raw = pm.Normal("regional_raw", mu=0, sigma=1, shape=n_regions)
            regional_effects = pm.Deterministic(
                "regional_effects",
                regional_raw * sigma_region
            )

            # Year-specific national shock (captures unmeasured national factors)
            sigma_year = pm.HalfNormal("sigma_year", sigma=3.0)
            year_effects = pm.Normal("year_effects", mu=0, sigma=sigma_year, shape=n_years)

            # District-level noise
            sigma_district = pm.HalfNormal("sigma_district", sigma=10.0)

            # Expected vote share
            mu = (
                50.0  # Baseline
                + beta_pvi * pvi
                + beta_inc * inc
                + beta_national * national
                + regional_effects[region_idx]
                + year_effects[year_idx]
            )

            # Likelihood
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_district, observed=y)

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
                    random_seed=42,
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
                    random_seed=42,
                )

        # Extract posterior summaries
        summary = az.summary(self.trace, var_names=[
            "beta_pvi", "beta_inc", "beta_national",
            "regional_effects", "sigma_region", "sigma_district", "sigma_year"
        ])

        # Extract means and stds
        beta_pvi_mean = float(self.trace.posterior["beta_pvi"].mean())
        beta_pvi_std = float(self.trace.posterior["beta_pvi"].std())
        beta_inc_mean = float(self.trace.posterior["beta_inc"].mean())
        beta_inc_std = float(self.trace.posterior["beta_inc"].std())
        beta_national_mean = float(self.trace.posterior["beta_national"].mean())
        beta_national_std = float(self.trace.posterior["beta_national"].std())

        regional_means = self.trace.posterior["regional_effects"].mean(dim=["chain", "draw"]).values
        regional_stds = self.trace.posterior["regional_effects"].std(dim=["chain", "draw"]).values

        regional_effects_dict = {REGIONS[i]: float(regional_means[i]) for i in range(n_regions)}
        regional_effects_std_dict = {REGIONS[i]: float(regional_stds[i]) for i in range(n_regions)}

        sigma_district_mean = float(self.trace.posterior["sigma_district"].mean())
        sigma_region_mean = float(self.trace.posterior["sigma_region"].mean())
        sigma_year_mean = float(self.trace.posterior["sigma_year"].mean())

        # Calculate fit statistics
        y_pred = (
            50.0
            + beta_pvi_mean * pvi
            + beta_inc_mean * inc
            + float(self.trace.posterior["beta_national"].mean()) * national
            + np.array([regional_means[r] for r in region_idx])
        )
        residuals = y - y_pred
        rmse = np.sqrt(np.mean(residuals ** 2))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        logger.info(f"  β_pvi: {beta_pvi_mean:.3f} ± {beta_pvi_std:.3f}")
        logger.info(f"  β_inc: {beta_inc_mean:.2f} ± {beta_inc_std:.2f}")
        logger.info(f"  σ_district: {sigma_district_mean:.2f}")
        logger.info(f"  σ_region: {sigma_region_mean:.2f}")
        logger.info(f"  σ_year (national): {sigma_year_mean:.2f}")
        logger.info(f"  Regional effects: {regional_effects_dict}")
        logger.info(f"  RMSE: {rmse:.2f}, R²: {r_squared:.3f}")

        self.learned_params = LearnedParameters(
            beta_pvi_mean=beta_pvi_mean,
            beta_pvi_std=beta_pvi_std,
            beta_inc_mean=beta_inc_mean,
            beta_inc_std=beta_inc_std,
            beta_national_mean=beta_national_mean,
            beta_national_std=beta_national_std,
            regional_effects=regional_effects_dict,
            regional_effects_std=regional_effects_std_dict,
            sigma_national=sigma_year_mean,
            sigma_regional=sigma_region_mean,
            sigma_district=sigma_district_mean,
            n_districts_fitted=len(df),
            years_used=self.years,
            rmse=float(rmse),
            r_squared=float(r_squared),
        )

        return self.learned_params

    def fit(self, use_pymc: bool = True) -> LearnedParameters:
        """
        Fit model parameters.

        Args:
            use_pymc: Whether to use PyMC (slower but more accurate)

        Returns:
            LearnedParameters object
        """
        if self.combined_data is None:
            self.load_historical_data()

        if use_pymc and PYMC_AVAILABLE:
            return self.fit_pymc()
        else:
            return self.fit_ols()

    def cross_validate(self) -> dict:
        """
        Perform leave-one-year-out cross-validation.

        Returns:
            Dictionary with cross-validation results
        """
        logger.info("Running leave-one-year-out cross-validation...")

        results = {}

        for holdout_year in self.years:
            train_years = [y for y in self.years if y != holdout_year]
            logger.info(f"  Training on {train_years}, testing on {holdout_year}")

            # Create fitter with training years only
            fitter = ParameterFitter(years=train_years)
            fitter.load_historical_data()
            params = fitter.fit_ols()  # Use OLS for speed in CV

            # Predict on holdout year
            holdout_data = self.data[holdout_year].copy()

            # Make predictions
            predictions = (
                50.0
                + params.beta_pvi_mean * holdout_data["pvi_numeric"]
                + params.beta_inc_mean * holdout_data["incumbency_code"]
                + holdout_data["national_margin"]  # Use actual national environment
                + holdout_data["region"].map(params.regional_effects)
            )

            # Calculate error metrics
            actual = holdout_data["dem_pct"]
            residuals = actual - predictions
            rmse = np.sqrt(np.mean(residuals ** 2))
            mae = np.mean(np.abs(residuals))

            # Classification accuracy (predict winner)
            pred_dem_win = predictions > 50
            actual_dem_win = actual > 50
            accuracy = np.mean(pred_dem_win == actual_dem_win)

            results[holdout_year] = {
                "rmse": float(rmse),
                "mae": float(mae),
                "accuracy": float(accuracy),
                "n_districts": len(holdout_data),
            }

            logger.info(f"    RMSE: {rmse:.2f}, MAE: {mae:.2f}, Accuracy: {accuracy:.1%}")

        # Average across years
        avg_rmse = np.mean([r["rmse"] for r in results.values()])
        avg_accuracy = np.mean([r["accuracy"] for r in results.values()])

        results["average"] = {
            "rmse": float(avg_rmse),
            "accuracy": float(avg_accuracy),
        }

        logger.info(f"  Average RMSE: {avg_rmse:.2f}, Average Accuracy: {avg_accuracy:.1%}")

        return results

    def save_parameters(self, path: Optional[Path] = None) -> None:
        """Save learned parameters to JSON file."""
        if self.learned_params is None:
            raise ValueError("No parameters to save. Call fit() first.")

        if path is None:
            path = DATA_PROCESSED / "learned_params.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        output = self.learned_params.to_dict()
        output["fitted_at"] = datetime.now().isoformat()
        output["method"] = "pymc" if self.trace is not None else "ols"

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved learned parameters to {path}")

    @staticmethod
    def load_parameters(path: Optional[Path] = None) -> LearnedParameters:
        """Load learned parameters from JSON file."""
        if path is None:
            path = DATA_PROCESSED / "learned_params.json"

        if not path.exists():
            raise FileNotFoundError(f"No learned parameters found at {path}")

        with open(path) as f:
            data = json.load(f)

        # Remove metadata fields
        data.pop("fitted_at", None)
        data.pop("method", None)

        return LearnedParameters.from_dict(data)


def main():
    """Run parameter fitting."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Bayesian Parameter Fitting")
    logger.info("=" * 60)

    fitter = ParameterFitter(years=[2018, 2022])
    fitter.load_historical_data()

    # Run cross-validation first
    cv_results = fitter.cross_validate()

    # Fit on all data
    logger.info("\nFitting on all data...")
    params = fitter.fit(use_pymc=PYMC_AVAILABLE)

    # Save results
    fitter.save_parameters()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("LEARNED PARAMETERS")
    logger.info("=" * 60)
    logger.info(f"β_pvi: {params.beta_pvi_mean:.3f} ± {params.beta_pvi_std:.3f}")
    logger.info(f"β_inc: {params.beta_inc_mean:.2f} ± {params.beta_inc_std:.2f}")
    logger.info(f"Regional effects: {params.regional_effects}")
    logger.info(f"σ_national: {params.sigma_national:.2f}")
    logger.info(f"σ_regional: {params.sigma_regional:.2f}")
    logger.info(f"σ_district: {params.sigma_district:.2f}")
    logger.info(f"RMSE: {params.rmse:.2f}")
    logger.info(f"R²: {params.r_squared:.3f}")

    return params


if __name__ == "__main__":
    main()
