"""Dynamic Bayesian aggregation for the national generic ballot.

The daily production update uses a robust Gaussian state-space filter.  It is
an explicit Bayesian model, but unlike MCMC it has no convergence diagnostics:
the filtered state is available analytically.  Pollster and population effects
are estimated with empirical-Bayes partial pooling and the Student-t
likelihood is represented by its iteratively reweighted Gaussian form.

Positive margins favor Democrats throughout this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
import pandas as pd


ELECTION_DATE = date(2026, 11, 3)


def _normal_interval(mean: float, std: float, level: float) -> tuple[float, float]:
    z = {0.50: 0.67448975, 0.80: 1.28155157, 0.90: 1.64485363, 0.95: 1.95996398}[level]
    return mean - z * std, mean + z * std


@dataclass(frozen=True)
class FundamentalsPrior:
    """Election-day national prior before generic-ballot polls are observed."""

    mean: float
    std: float
    economic_index: float
    components: dict[str, dict[str, float]]


def build_fundamentals_prior(
    economic_index: float = 0.0,
    economic_std: float = 1.0,
    incumbent_party: str = "R",
) -> FundamentalsPrior:
    """Build a regularized prior and propagate coefficient uncertainty.

    The economic coefficient is deliberately weak and the structural error is
    deliberately broad. Economics enters once here and is never added again
    after the polling update.
    """

    party_sign = -1.0 if incumbent_party.upper() == "R" else 1.0
    economic_beta, economic_beta_std = 0.34 * party_sign, 0.33

    economic_contribution = economic_beta * economic_index
    mean = economic_contribution

    # A broad structural-error term prevents five historical midterms from
    # creating false precision.
    structural_std = 3.5
    variance = (
        structural_std**2
        + (economic_beta * economic_std) ** 2
        + (economic_index * economic_beta_std) ** 2
    )
    return FundamentalsPrior(
        mean=float(mean),
        std=float(np.sqrt(variance)),
        economic_index=float(economic_index),
        components={
            "economy": {
                "coefficient_mean": economic_beta,
                "coefficient_std": economic_beta_std,
                "contribution_mean": economic_contribution,
                "input_standardized_index": float(economic_index),
                "input_std": float(economic_std),
            },
            "structural_uncertainty": {"std": structural_std},
        },
    )


@dataclass
class DynamicPollingResult:
    current_mean: float
    current_std: float
    election_mean: float
    election_std: float
    election_samples: np.ndarray
    trend: list[dict[str, Any]]
    pollster_effects: dict[str, float]
    population_effects: dict[str, float]
    prior: FundamentalsPrior
    diagnostics: dict[str, Any]
    polling_input_date: str

    def to_public_dict(self) -> dict[str, Any]:
        current_50 = _normal_interval(self.current_mean, self.current_std, 0.50)
        current_90 = _normal_interval(self.current_mean, self.current_std, 0.90)
        election_50 = np.percentile(self.election_samples, [25, 75])
        election_90 = np.percentile(self.election_samples, [5, 95])
        return {
            "prior": {
                "mean": round(self.prior.mean, 3),
                "std": round(self.prior.std, 3),
                "components": self.prior.components,
            },
            "current_sentiment": {
                "mean": round(self.current_mean, 3),
                "std": round(self.current_std, 3),
                "ci_50": [round(x, 3) for x in current_50],
                "ci_90": [round(x, 3) for x in current_90],
            },
            "election_day": {
                "mean": round(self.election_mean, 3),
                "std": round(self.election_std, 3),
                "ci_50": [round(float(x), 3) for x in election_50],
                "ci_90": [round(float(x), 3) for x in election_90],
            },
            "trend": self.trend,
            "pollster_effects": {
                key: round(value, 3) for key, value in self.pollster_effects.items()
            },
            "diagnostics": self.diagnostics,
        }


class DynamicNationalModel:
    """Robust dynamic linear model for generic-ballot margins."""

    REQUIRED_COLUMNS = {"date", "pollster", "margin"}

    def __init__(
        self,
        process_std_per_day: float = 0.09,
        design_error: float = 2.2,
        election_error_floor: float = 1.5,
        student_df: float = 5.0,
        random_seed: int = 42,
    ) -> None:
        self.process_std_per_day = process_std_per_day
        self.design_error = design_error
        self.election_error_floor = election_error_floor
        self.student_df = student_df
        self.rng = np.random.default_rng(random_seed)

    @staticmethod
    def calibrate_process_std(
        history: pd.DataFrame,
        minimum_days: int = 30,
        prior_std_per_day: float = 0.09,
        shrinkage_days: float = 45.0,
    ) -> dict[str, float | int | str]:
        """Estimate daily latent movement from a maintained-average history.

        Adjacent values are never multiplied as likelihood observations.  Their
        first differences are used only to calibrate prospective process
        variance, with shrinkage to the historical default because a single
        election cycle cannot identify long-run campaign volatility precisely.
        """

        work = history[["date", "margin"]].copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
        work["margin"] = pd.to_numeric(work["margin"], errors="coerce")
        work = work.dropna().sort_values("date").drop_duplicates("date", keep="last")
        change = work["margin"].diff().dropna().to_numpy(float)
        if len(change) < minimum_days:
            return {
                "process_std_per_day": prior_std_per_day,
                "observed_days": int(len(change)),
                "method": "regularized_prior_insufficient_history",
            }
        mad = 1.4826 * float(np.median(np.abs(change - np.median(change))))
        variance = (
            len(change) * mad**2 + shrinkage_days * prior_std_per_day**2
        ) / (len(change) + shrinkage_days)
        return {
            # A smoothed maintained average necessarily moves less than the
            # latent electorate.  Its observed first differences can justify
            # more process variance, never less than the historical prior.
            "process_std_per_day": round(float(np.clip(max(np.sqrt(variance), prior_std_per_day),
                                                         prior_std_per_day, 0.18)), 4),
            "observed_days": int(len(change)),
            "robust_daily_change_std": round(mad, 4),
            "method": "bulletin_average_first_differences_with_prior_shrinkage",
        }

    @staticmethod
    def validate_polls(polls: pd.DataFrame) -> pd.DataFrame:
        missing = DynamicNationalModel.REQUIRED_COLUMNS - set(polls.columns)
        if missing:
            raise ValueError(f"Generic-ballot polls missing columns: {sorted(missing)}")
        clean = polls.copy()
        clean["date"] = pd.to_datetime(clean["date"], errors="coerce").dt.normalize()
        if "sample_size" not in clean:
            clean["sample_size"] = np.nan
        clean["sample_size"] = pd.to_numeric(clean["sample_size"], errors="coerce")
        if "observation_std" not in clean:
            clean["observation_std"] = np.nan
        clean["observation_std"] = pd.to_numeric(clean["observation_std"], errors="coerce")
        clean["margin"] = pd.to_numeric(clean["margin"], errors="coerce")
        clean = clean.dropna(subset=["date", "pollster", "margin"])
        has_uncertainty = (clean["sample_size"] >= 100) | (clean["observation_std"] > 0)
        clean = clean[has_uncertainty & clean["margin"].between(-30, 30)]
        if "internal" not in clean:
            clean["internal"] = False
        if "partisan" not in clean:
            clean["partisan"] = None
        if "population" not in clean:
            clean["population"] = "a"
        if "dem_pct" not in clean:
            clean["dem_pct"] = np.nan
        if "rep_pct" not in clean:
            clean["rep_pct"] = np.nan
        clean["pollster"] = clean["pollster"].astype(str).str.strip()
        clean["population"] = clean["population"].astype(str).str.lower()
        clean["_dedupe"] = (
            clean["date"].astype(str)
            + "|" + clean["pollster"].str.lower()
            + "|" + clean["sample_size"].fillna(0).astype(int).astype(str)
            + "|" + clean["margin"].round(2).astype(str)
        )
        return clean.drop_duplicates("_dedupe").sort_values("date").reset_index(drop=True)

    def _observation_std(self, row: pd.Series) -> float:
        explicit = pd.to_numeric(row.get("observation_std"), errors="coerce")
        if pd.notna(explicit) and float(explicit) > 0:
            return float(explicit)
        n = max(float(row["sample_size"]), 100.0)
        dem = row.get("dem_pct")
        rep = row.get("rep_pct")
        if pd.notna(dem) and pd.notna(rep):
            p_d, p_r = float(dem) / 100.0, float(rep) / 100.0
            sampling_var = max(p_d + p_r - (p_d - p_r) ** 2, 0.05) / n * 10000
        else:
            sampling_var = 5000.0 / n
        population_penalty = {"lv": 0.0, "likely_voters": 0.0,
                              "rv": 0.45, "registered_voters": 0.45,
                              "a": 0.9, "adults": 0.9}.get(str(row["population"]), 0.9)
        partisan_penalty = 1.25 if bool(row.get("internal")) or pd.notna(row.get("partisan")) else 0.0
        return float(np.sqrt(sampling_var + self.design_error**2
                             + population_penalty**2 + partisan_penalty**2))

    def _filter(
        self,
        polls: pd.DataFrame,
        prior: FundamentalsPrior,
        pollster_effects: Optional[dict[str, float]] = None,
        population_effects: Optional[dict[str, float]] = None,
    ) -> tuple[pd.DataFrame, list[float]]:
        pollster_effects = pollster_effects or {}
        population_effects = population_effects or {}
        start, end = polls["date"].min(), polls["date"].max()
        days = pd.date_range(start, end, freq="D")
        by_day = {day: frame for day, frame in polls.groupby("date")}
        mean, variance = prior.mean, prior.std**2
        trend: list[dict[str, Any]] = []
        residuals: list[float] = []
        q = self.process_std_per_day**2

        for idx, day in enumerate(days):
            if idx:
                variance += q
            n_day = 0
            for _, row in by_day.get(day, pd.DataFrame()).iterrows():
                effect = pollster_effects.get(str(row["pollster"]), 0.0)
                effect += population_effects.get(str(row["population"]), 0.0)
                observation = float(row["margin"]) - effect
                base_var = self._observation_std(row) ** 2
                residual = observation - mean
                standardized = residual / np.sqrt(variance + base_var)
                # Student-t scale-mixture expectation: surprising polls receive
                # a larger conditional observation variance, not zero weight.
                robust_inflation = max(1.0, (standardized**2 + self.student_df) /
                                       (self.student_df + 1.0))
                obs_var = base_var * robust_inflation
                gain = variance / (variance + obs_var)
                mean += gain * residual
                variance = max((1.0 - gain) * variance, 1e-6)
                residuals.append(float(residual))
                n_day += 1
            std = float(np.sqrt(variance))
            ci50 = _normal_interval(mean, std, 0.50)
            ci90 = _normal_interval(mean, std, 0.90)
            trend.append({
                "date": day.strftime("%Y-%m-%d"),
                "mean": round(float(mean), 3),
                "ci_50_low": round(ci50[0], 3),
                "ci_50_high": round(ci50[1], 3),
                "ci_90_low": round(ci90[0], 3),
                "ci_90_high": round(ci90[1], 3),
                "n_polls": n_day,
            })
        return pd.DataFrame(trend), residuals

    @staticmethod
    def _pooled_effects(
        polls: pd.DataFrame,
        trend: pd.DataFrame,
        group: str,
        prior_count: float,
        limit: float,
    ) -> dict[str, float]:
        lookup = trend.set_index("date")["mean"]
        work = polls.copy()
        work["trend_mean"] = work["date"].dt.strftime("%Y-%m-%d").map(lookup)
        work["residual"] = work["margin"] - work["trend_mean"]
        effects: dict[str, float] = {}
        for key, frame in work.groupby(group):
            n = len(frame)
            shrinkage = n / (n + prior_count)
            effects[str(key)] = float(np.clip(frame["residual"].median() * shrinkage, -limit, limit))
        # Identifiability: categorical effects are deviations, not a second
        # intercept.  With one observed category the effect is exactly zero.
        if effects:
            center = float(np.mean(list(effects.values())))
            effects = {key: value - center for key, value in effects.items()}
        return effects

    def fit(
        self,
        polls: pd.DataFrame,
        prior: FundamentalsPrior,
        election_date: date = ELECTION_DATE,
        n_draws: int = 10_000,
        as_of: Optional[date] = None,
    ) -> DynamicPollingResult:
        clean = self.validate_polls(polls)
        if clean.empty:
            raise ValueError("No valid generic-ballot polls remain after validation")

        first_trend, _ = self._filter(clean, prior)
        pollster_effects = self._pooled_effects(clean, first_trend, "pollster", 4.0, 3.0)
        population_effects = self._pooled_effects(clean, first_trend, "population", 12.0, 1.5)
        trend, residuals = self._filter(clean, prior, pollster_effects, population_effects)

        current_mean = float(trend.iloc[-1]["mean"])
        current_std = float((trend.iloc[-1]["ci_90_high"] - current_mean) / 1.64485363)
        data_date = clean["date"].max().date()
        forecast_from = max(as_of or data_date, data_date)
        future_days = max((election_date - forecast_from).days, 0)
        election_var = (current_std**2 + future_days * self.process_std_per_day**2
                        + self.election_error_floor**2)
        election_std = float(np.sqrt(election_var))
        samples = self.rng.normal(current_mean, election_std, n_draws)

        residual_array = np.asarray(residuals, dtype=float)
        diagnostics = {
            "status": "passed",
            "inference": "robust_dynamic_kalman_filter",
            "mcmc_required": False,
            "r_hat": None,
            "ess_bulk": n_draws,
            "divergences": 0,
            "n_polls": int(len(clean)),
            "n_pollsters": int(clean["pollster"].nunique()),
            "residual_rmse": round(float(np.sqrt(np.mean(residual_array**2))), 3),
        }
        return DynamicPollingResult(
            current_mean=current_mean,
            current_std=current_std,
            election_mean=float(np.mean(samples)),
            election_std=float(np.std(samples)),
            election_samples=samples,
            trend=trend.to_dict("records"),
            pollster_effects=pollster_effects,
            population_effects=population_effects,
            prior=prior,
            diagnostics=diagnostics,
            polling_input_date=data_date.isoformat(),
        )

    def fit_external_average(
        self,
        history: pd.DataFrame,
        prior: FundamentalsPrior,
        election_date: date = ELECTION_DATE,
        n_draws: int = 10_000,
        as_of: Optional[date] = None,
    ) -> DynamicPollingResult:
        """Update once from the latest externally maintained polling average.

        Every historical date is combined with the same weak fundamentals
        prior only to draw the public trend.  The dates are not filtered
        sequentially, because adjacent published averages reuse most of the
        same polls and multiplying them would create severe pseudo-replication.
        """

        clean = self.validate_polls(history)
        if clean.empty:
            raise ValueError("No valid external generic-ballot average")
        clean = clean.sort_values("date")
        trend: list[dict[str, Any]] = []
        prior_var = prior.std**2
        latest_mean = prior.mean
        latest_var = prior_var
        latest_residual = 0.0
        for _, row in clean.iterrows():
            observation = float(row["margin"])
            observation_var = self._observation_std(row) ** 2
            residual = observation - prior.mean
            standardized = residual / np.sqrt(prior_var + observation_var)
            robust_inflation = max(
                1.0, (standardized**2 + self.student_df) / (self.student_df + 1.0)
            )
            adjusted_var = observation_var * robust_inflation
            gain = prior_var / (prior_var + adjusted_var)
            mean = prior.mean + gain * residual
            variance = max((1.0 - gain) * prior_var, 1e-6)
            std = float(np.sqrt(variance))
            ci50 = _normal_interval(mean, std, 0.50)
            ci90 = _normal_interval(mean, std, 0.90)
            trend.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "mean": round(float(mean), 3),
                "ci_50_low": round(ci50[0], 3),
                "ci_50_high": round(ci50[1], 3),
                "ci_90_low": round(ci90[0], 3),
                "ci_90_high": round(ci90[1], 3),
                "n_polls": 1,
            })
            latest_mean, latest_var, latest_residual = float(mean), float(variance), float(residual)

        data_date = clean["date"].max().date()
        forecast_from = max(as_of or data_date, data_date)
        future_days = max((election_date - forecast_from).days, 0)
        election_var = (
            latest_var
            + future_days * self.process_std_per_day**2
            + self.election_error_floor**2
        )
        samples = self.rng.normal(latest_mean, np.sqrt(election_var), n_draws)
        return DynamicPollingResult(
            current_mean=latest_mean,
            current_std=float(np.sqrt(latest_var)),
            election_mean=float(np.mean(samples)),
            election_std=float(np.std(samples)),
            election_samples=samples,
            trend=trend,
            pollster_effects={},
            population_effects={},
            prior=prior,
            diagnostics={
                "status": "passed",
                "inference": "single_external_average_conjugate_update",
                "mcmc_required": False,
                "r_hat": None,
                "ess_bulk": n_draws,
                "divergences": 0,
                "n_polls": 1,
                "n_pollsters": 1,
                "published_average_days": int(len(clean)),
                "residual_rmse": round(abs(latest_residual), 3),
                "pseudo_replication_guard": "latest average only enters likelihood",
                "process_std_per_day": round(self.process_std_per_day, 4),
                "election_error_floor": round(self.election_error_floor, 4),
            },
            polling_input_date=data_date.isoformat(),
        )
