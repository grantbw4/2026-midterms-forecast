"""House-specific Bayesian calibration on Democratic two-party margin.

The legacy model was fit on vote share and then converted to margin during
simulation.  That made coefficient scale and validation hard to audit.  This
module keeps every quantity on the same scale: positive Democratic margin
points.  Coefficients are estimated with a regularized Bayesian linear model
and a Student-t iteratively reweighted likelihood.  The resulting joint
posterior covariance is retained so production draws preserve parameter
dependence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REGIONS = (
    "New_England", "Mid_Atlantic_Northeast", "Rust_Belt", "Southeast",
    "Deep_South", "Texas_Region", "Plains", "Mountain", "Southwest", "Pacific",
)

BASE_COEFFICIENTS = ("intercept", "lean", "incumbency", "national")
COEFFICIENT_NAMES = BASE_COEFFICIENTS + tuple(f"region:{region}" for region in REGIONS)


@dataclass(frozen=True)
class HouseCalibration:
    coefficient_names: tuple[str, ...]
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    sigma_national: float
    sigma_regional: float
    sigma_district: float
    student_df: float
    years_used: tuple[int, ...]
    n_districts_fitted: int
    signed_error: float
    rmse: float
    source: str = "house_margin_bayesian_robust_regression"

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficient_names": list(self.coefficient_names),
            "posterior_mean": self.posterior_mean.tolist(),
            "posterior_covariance": self.posterior_covariance.tolist(),
            "sigma_national": self.sigma_national,
            "sigma_regional": self.sigma_regional,
            "sigma_district": self.sigma_district,
            "student_df": self.student_df,
            "years_used": list(self.years_used),
            "n_districts_fitted": self.n_districts_fitted,
            "signed_error": self.signed_error,
            "rmse": self.rmse,
            "source": self.source,
            "scale": "democratic_two_party_margin_points",
            "inference": "regularized_bayesian_regression_with_student_t_reweighting",
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "HouseCalibration":
        return cls(
            coefficient_names=tuple(values["coefficient_names"]),
            posterior_mean=np.asarray(values["posterior_mean"], dtype=float),
            posterior_covariance=np.asarray(values["posterior_covariance"], dtype=float),
            sigma_national=float(values["sigma_national"]),
            sigma_regional=float(values["sigma_regional"]),
            sigma_district=float(values["sigma_district"]),
            student_df=float(values.get("student_df", 5.0)),
            years_used=tuple(int(year) for year in values["years_used"]),
            n_districts_fitted=int(values["n_districts_fitted"]),
            signed_error=float(values["signed_error"]),
            rmse=float(values["rmse"]),
            source=str(values.get("source", "house_margin_bayesian_robust_regression")),
        )

    @classmethod
    def load(cls, path: Path) -> "HouseCalibration":
        return cls.from_dict(json.loads(Path(path).read_text()))


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Create the production/training design matrix on the margin scale."""

    required = {"pvi_numeric", "incumbency_code", "national_margin", "region"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"House calibration frame missing columns: {sorted(missing)}")
    region = pd.Categorical(frame["region"], categories=REGIONS)
    if region.isna().any():
        unknown = sorted(set(frame.loc[region.isna(), "region"].astype(str)))
        raise ValueError(f"Unknown House regions: {unknown}")
    # Centering makes regional coefficients deviations from the national mean.
    one_hot = np.column_stack([(region == name).astype(float) for name in REGIONS])
    one_hot -= one_hot.mean(axis=0, keepdims=True)
    return np.column_stack([
        np.ones(len(frame)),
        frame["pvi_numeric"].to_numpy(float),
        frame["incumbency_code"].to_numpy(float),
        frame["national_margin"].to_numpy(float),
        one_hot,
    ])


def _regularized_scale(values: Iterable[float], prior_scale: float, prior_df: float = 4.0) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) <= 1:
        return float(prior_scale)
    observed = float(np.var(array, ddof=1))
    return float(np.sqrt((observed * (len(array) - 1) + prior_df * prior_scale**2)
                         / (len(array) - 1 + prior_df)))


def fit_house_calibration(frames: Iterable[pd.DataFrame], student_df: float = 5.0) -> HouseCalibration:
    """Fit a robust, regularized Bayesian House margin model.

    Priors are intentionally weak but identify the national coefficient when
    only a few whole election cycles are available.  They are stated directly
    on the margin scale: lean and national effects center on one-for-one,
    incumbency centers on 3.5 points, and regional offsets center on zero.
    """

    combined = pd.concat([frame.copy() for frame in frames], ignore_index=True)
    required = {"year", "district_id", "margin", "pvi_numeric", "incumbency_code",
                "national_margin", "region"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"House training data missing columns: {sorted(missing)}")
    for column in ("margin", "pvi_numeric", "incumbency_code", "national_margin"):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined = combined.dropna(subset=list(required))
    # Uncontested races do not reveal a two-party preference margin.
    combined = combined[combined["margin"].between(-98.0, 98.0)]
    if len(combined) < 300:
        raise ValueError("Too few contested House districts for calibration")

    x = design_matrix(combined)
    y = combined["margin"].to_numpy(float)
    prior_mean = np.asarray([0.0, 1.0, 3.5, 1.0] + [0.0] * len(REGIONS))
    prior_std = np.asarray([3.0, 0.25, 2.0, 0.35] + [1.5] * len(REGIONS))
    prior_precision = np.diag(1.0 / prior_std**2)
    weights = np.ones(len(y))
    sigma = 7.5
    posterior_mean = prior_mean.copy()
    posterior_covariance = np.diag(prior_std**2)

    for _ in range(12):
        weighted_x = x * weights[:, None]
        precision = prior_precision + x.T @ weighted_x / sigma**2
        posterior_covariance = np.linalg.inv(precision)
        rhs = prior_precision @ prior_mean + x.T @ (weights * y) / sigma**2
        posterior_mean = posterior_covariance @ rhs
        residual = y - x @ posterior_mean
        standardized = residual / max(sigma, 1e-6)
        weights = (student_df + 1.0) / (student_df + standardized**2)
        robust_sigma = np.sqrt(np.sum(weights * residual**2) / np.sum(weights))
        sigma = float(np.clip(robust_sigma, 4.0, 12.0))

    # District rows identify lean and local residuals well, but they do not
    # create hundreds of independent observations of the national coefficient,
    # intercept, or a region-wide cycle shock.  Retain explicit uncertainty
    # floors for those cluster-level quantities instead of allowing the row
    # count to manufacture false precision.
    posterior_std_floor = np.asarray(
        [1.5, 0.08, 0.5, 0.25] + [0.5] * len(REGIONS), dtype=float
    )
    diagonal_addition = np.maximum(
        posterior_std_floor**2 - np.diag(posterior_covariance), 0.0
    )
    posterior_covariance = posterior_covariance + np.diag(diagonal_addition)

    residual = y - x @ posterior_mean
    work = combined[["year", "region"]].copy()
    work["residual"] = residual
    national_residual = work.groupby("year")["residual"].mean()
    work["national_residual"] = work["year"].map(national_residual)
    work["after_national"] = work["residual"] - work["national_residual"]
    regional_residual = work.groupby(["year", "region"])["after_national"].mean()
    regional_lookup = regional_residual.to_dict()
    work["regional_residual"] = [regional_lookup[(year, region)]
                                  for year, region in zip(work["year"], work["region"])]
    local_residual = work["after_national"] - work["regional_residual"]

    # Only three modern cycles are available.  A broad regularizing prior is
    # therefore more honest than the tiny empirical SD of two or three means.
    sigma_national = _regularized_scale(national_residual, prior_scale=3.0, prior_df=8.0)
    sigma_regional = _regularized_scale(regional_residual, prior_scale=1.25)
    local_mad = 1.4826 * float(np.median(np.abs(local_residual - np.median(local_residual))))
    sigma_district = float(np.sqrt((local_mad**2 * len(local_residual) + 6.5**2 * 80)
                                   / (len(local_residual) + 80)))

    return HouseCalibration(
        coefficient_names=COEFFICIENT_NAMES,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
        sigma_national=sigma_national,
        sigma_regional=sigma_regional,
        sigma_district=sigma_district,
        student_df=float(student_df),
        years_used=tuple(sorted(int(year) for year in combined["year"].unique())),
        n_districts_fitted=int(len(combined)),
        signed_error=float(np.mean(-residual)),
        rmse=float(np.sqrt(np.mean(residual**2))),
    )


def posterior_mean_margin(calibration: HouseCalibration, frame: pd.DataFrame) -> np.ndarray:
    return design_matrix(frame) @ calibration.posterior_mean
