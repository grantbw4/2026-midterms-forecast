#!/usr/bin/env python3
"""
Senate Forecast Model for 2026 Elections.

2026 Senate races: 35 seats (Class 2 + special elections)
Democrats defending: 13 seats
Republicans defending: 22 seats
Control requires 51 seats (no VP tiebreaker with Republican president)

ACTIVE PIPELINE:
================
1. National Environment - Shared with House (from NationalEnvironmentModel)
2. State Fundamentals - PVI + Incumbency + Regional effects
3. Election Simulation - PyMC MCMC (default) or Monte Carlo fallback

PVI SIGN CONVENTION: positive = D-leaning, negative = R-leaning
(e.g., D+10 = +10, R+10 = -10)

The national environment (inferred from VoteHub polling) drives all predictions.
Uses House-fitted parameters (from parameter_fitting.py) - no Senate-specific fitting.
"""

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Configure PyTensor to use Python mode to avoid C compilation issues on macOS
os.environ.setdefault("PYTENSOR_FLAGS", "device=cpu,floatX=float64,optimizer=fast_compile,cxx=")

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None

# Import models
try:
    from .national_environment import NationalEnvironmentModel
    from .parameter_fitting import ParameterFitter, LearnedParameters, REGIONS, STATE_TO_REGION
    from .economic_fundamentals import EconomicFundamentals
    from .hierarchical_model import NationalPosterior
except ImportError:
    from national_environment import NationalEnvironmentModel
    from parameter_fitting import ParameterFitter, LearnedParameters, REGIONS, STATE_TO_REGION
    from economic_fundamentals import EconomicFundamentals
    from hierarchical_model import NationalPosterior

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_COOK = PROJECT_ROOT / "data" / "cook"

# Cook rating to internal rating mapping
COOK_RATING_MAP = {
    "Solid D": "safe_d",
    "Likely D": "likely_d",
    "Lean D": "lean_d",
    "Toss-up": "toss_up",
    "Lean R": "lean_r",
    "Likely R": "likely_r",
    "Solid R": "safe_r",
}

# Cook rating to PVI approximation
# PVI SIGN CONVENTION: positive = D-leaning, negative = R-leaning
# (e.g., D+12 = +12, R+12 = -12)
COOK_TO_PVI = {
    "Solid D": 12.0,
    "Likely D": 7.0,
    "Lean D": 3.0,
    "Toss-up": 0.0,
    "Lean R": -3.0,
    "Likely R": -7.0,
    "Solid R": -12.0,
}


def compute_pvi_scaled_sigma(
    pvi: np.ndarray,
    sigma_base: float = 5.0,
    sigma_floor: float = 2.5,
    midpoint: float = 12.0,
    steepness: float = 0.2,
) -> np.ndarray:
    """
    Compute PVI-scaled uncertainty using logistic decay.

    Safer races (higher |PVI|) get less uncertainty, reflecting that
    upsets are historically rarer in non-competitive states.

    PVI SIGN CONVENTION: positive = D-leaning, negative = R-leaning
    (e.g., D+10 = +10, R+10 = -10)

    Args:
        pvi: Array of PVI values (positive = D-leaning, negative = R-leaning)
        sigma_base: Maximum uncertainty for competitive races (|PVI| near 0)
        sigma_floor: Minimum uncertainty for safe races (|PVI| >> midpoint)
        midpoint: PVI value at which sigma is halfway between base and floor
        steepness: How quickly sigma decays (higher = sharper transition)

    Returns:
        Array of sigma values for each race

    The logistic function ensures:
        - |PVI| ≈ 0: sigma ≈ sigma_base (competitive races, full uncertainty)
        - |PVI| = midpoint: sigma ≈ (sigma_base + sigma_floor) / 2
        - |PVI| >> midpoint: sigma → sigma_floor (safe races, minimal uncertainty)
    """
    abs_pvi = np.abs(pvi)
    # Logistic decay: high at low PVI, low at high PVI
    logistic = 1 / (1 + np.exp(steepness * (abs_pvi - midpoint)))
    return sigma_floor + (sigma_base - sigma_floor) * logistic


@dataclass
class SenateRace:
    """Individual Senate race data."""
    state: str
    seat_class: int  # 1, 2, or 3
    incumbent: str
    incumbent_party: str
    up_in_2026: bool
    pvi: float  # State partisan lean
    rating: str  # safe_d, likely_d, lean_d, toss_up, lean_r, likely_r, safe_r
    open_seat: bool = False
    special: bool = False


@dataclass
class SenateRaceForecast:
    """Forecast for a single Senate race."""
    state: str
    incumbent: str
    incumbent_party: str
    prob_dem: float
    category: str
    pvi: float
    open_seat: bool
    special: bool


class SenateForecastModel:
    """
    Senate forecast model using the same Bayesian hierarchical methodology as House.

    Uses the HierarchicalForecastModel approach:
    - Samples parameters (β_pvi, β_inc, regional effects) from posteriors
    - Proper uncertainty propagation through model structure
    - Learned parameters from historical fitting

    Key differences from House model:
    - State-level races (not district)
    - Fewer races = more uncertainty per seat
    - Incumbency advantage slightly stronger in Senate
    - Uses same national environment but with state-level PVI
    """

    # Default uncertainty parameters (matches hierarchical_model.py)
    DEFAULT_SIGMA_NATIONAL = 3.0
    DEFAULT_SIGMA_REGIONAL = 1.5
    DEFAULT_SIGMA_DISTRICT = 5.0  # Slightly higher for Senate (fewer races, more variance)

    # 2026 Senate races - Class 2 seats + special elections
    # Current composition: 53R - 47D (including independents caucusing with Dems)
    # 35 seats up: 13 D defending, 22 R defending
    #
    # PVI SIGN CONVENTION: positive = D-leaning, negative = R-leaning
    # (e.g., D+10 = +10, R+10 = -10)
    RACES_2026 = [
        # Democrats defending (13 seats) - positive PVI = D-leaning
        SenateRace("GA", 2, "Jon Ossoff", "D", True, 0, "toss_up"),
        SenateRace("MI", 2, "Gary Peters", "D", True, 1, "lean_d"),
        SenateRace("NH", 2, "Jeanne Shaheen", "D", True, 2, "likely_d"),
        SenateRace("VA", 2, "Mark Warner", "D", True, 3, "likely_d"),
        SenateRace("CO", 2, "John Hickenlooper", "D", True, 5, "likely_d"),
        SenateRace("IL", 2, "Dick Durbin", "D", True, 8, "safe_d"),
        SenateRace("OR", 2, "Jeff Merkley", "D", True, 6, "safe_d"),
        SenateRace("NM", 2, "Martin Heinrich", "D", True, 5, "likely_d"),
        SenateRace("MN", 2, "Tina Smith", "D", True, 3, "likely_d"),
        SenateRace("DE", 2, "Vacant", "D", True, 7, "safe_d", open_seat=True),
        SenateRace("MA", 2, "Ed Markey", "D", True, 20, "safe_d"),
        SenateRace("RI", 2, "Sheldon Whitehouse", "D", True, 12, "safe_d"),
        SenateRace("NJ", 2, "Cory Booker", "D", True, 7, "safe_d"),

        # Republicans defending (22 seats) - negative PVI = R-leaning
        SenateRace("NC", 2, "Thom Tillis", "R", True, -2, "toss_up"),
        SenateRace("ME", 2, "Susan Collins", "R", True, 3, "lean_r"),
        SenateRace("IA", 2, "Joni Ernst", "R", True, -6, "likely_r"),
        SenateRace("TX", 2, "John Cornyn", "R", True, -6, "likely_r"),
        SenateRace("AK", 2, "Dan Sullivan", "R", True, -8, "likely_r"),
        SenateRace("SC", 2, "Lindsey Graham", "R", True, -8, "safe_r"),
        SenateRace("LA", 2, "Bill Cassidy", "R", True, -12, "safe_r"),
        SenateRace("KY", 2, "Mitch McConnell", "R", True, -16, "safe_r"),
        SenateRace("AR", 2, "Tom Cotton", "R", True, -17, "safe_r"),
        SenateRace("OK", 2, "Markwayne Mullin", "R", True, -20, "safe_r"),
        SenateRace("SD", 2, "Mike Rounds", "R", True, -18, "safe_r"),
        SenateRace("KS", 2, "Roger Marshall", "R", True, -10, "safe_r"),
        SenateRace("ID", 2, "Jim Risch", "R", True, -22, "safe_r"),
        SenateRace("AL", 2, "Tommy Tuberville", "R", True, -15, "safe_r"),
        SenateRace("MS", 2, "Cindy Hyde-Smith", "R", True, -10, "safe_r"),
        SenateRace("TN", 2, "Bill Hagerty", "R", True, -15, "safe_r"),
        SenateRace("NE", 2, "Pete Ricketts", "R", True, -14, "safe_r"),
        SenateRace("WV", 2, "Shelley Moore Capito", "R", True, -28, "safe_r"),
        SenateRace("WY", 2, "John Barrasso", "R", True, -40, "safe_r"),
        SenateRace("MT", 2, "Steve Daines", "R", True, -12, "likely_r"),
        SenateRace("FL", 2, "Ashley Moody", "R", True, -6, "lean_r", special=True),
        SenateRace("OH", 2, "Jon Husted", "R", True, -6, "lean_r", special=True),
    ]

    # Current Senate composition (before 2026 election)
    # Democrats: 47 (including independents caucusing with Dems)
    # Republicans: 53
    # Dems defending: 13 seats -> Dems NOT up: 47 - 13 = 34
    DEM_SEATS_NOT_UP = 34

    @classmethod
    def load_races_from_cook(cls) -> list["SenateRace"]:
        """
        Load Senate races from Cook Political ratings CSV.

        Updates race data with latest Cook ratings, incumbent info, and open seat status.
        Falls back to hardcoded RACES_2026 if Cook data unavailable.
        """
        cook_path = DATA_COOK / "senate_ratings.csv"
        if not cook_path.exists():
            logger.warning("Cook Senate ratings not found, using hardcoded races")
            return cls.RACES_2026

        try:
            cook_df = pd.read_csv(cook_path)
            logger.info(f"Loaded {len(cook_df)} Senate races from Cook Political")
        except Exception as e:
            logger.warning(f"Could not load Cook Senate ratings: {e}")
            return cls.RACES_2026

        # Build lookup from hardcoded races for PVI values we trust
        hardcoded_lookup = {r.state: r for r in cls.RACES_2026}

        races = []
        for _, row in cook_df.iterrows():
            state = row["state"]
            cook_rating = row["cook_rating"]
            incumbent = row["incumbent"]
            incumbent_party = row["incumbent_party"]
            is_open = row["is_open"]

            # Get PVI from hardcoded data if available, otherwise use Cook-derived
            if state in hardcoded_lookup:
                pvi = hardcoded_lookup[state].pvi
                special = hardcoded_lookup[state].special
            else:
                # Use Cook rating to approximate PVI
                pvi = COOK_TO_PVI.get(cook_rating, 0.0)
                special = False

            # Map Cook rating to internal rating format
            rating = COOK_RATING_MAP.get(cook_rating, "toss_up")

            races.append(SenateRace(
                state=state,
                seat_class=2,  # 2026 = Class 2
                incumbent=incumbent,
                incumbent_party=incumbent_party,
                up_in_2026=True,
                pvi=pvi,
                rating=rating,
                open_seat=is_open,
                special=special,
            ))

        logger.info(f"  Open seats: {sum(1 for r in races if r.open_seat)}")
        logger.info(f"  D defending: {sum(1 for r in races if r.incumbent_party == 'D')}")
        logger.info(f"  R defending: {sum(1 for r in races if r.incumbent_party == 'R')}")

        return races

    def _races_to_dataframe(self) -> pd.DataFrame:
        """
        Convert Senate races to DataFrame format compatible with hierarchical model.

        PVI SIGN CONVENTION: positive = D-leaning, negative = R-leaning
        (e.g., D+10 = +10, R+10 = -10)

        Returns DataFrame with columns matching House district format:
            - district_id: State abbreviation (e.g., "GA", "TX")
            - state: State abbreviation
            - pvi: State partisan lean (positive = D-leaning, negative = R-leaning)
            - incumbent_party: "D", "R", or None
            - open_seat: Boolean
            - region: Geographic region
            - incumbency_code: +1 for D incumbent, -1 for R incumbent, 0 for open
        """
        records = []
        for race in self.races:
            # Encode incumbency: D=+1, R=-1, Open=0
            if race.open_seat:
                inc_code = 0
            elif race.incumbent_party == "D":
                inc_code = 1
            elif race.incumbent_party == "R":
                inc_code = -1
            else:
                inc_code = 0

            records.append({
                "district_id": race.state,
                "state": race.state,
                "pvi": race.pvi,
                "incumbent": race.incumbent,
                "incumbent_party": race.incumbent_party,
                "open_seat": race.open_seat,
                "region": STATE_TO_REGION.get(race.state, "South"),
                "incumbency_code": inc_code,
                "special": race.special,
            })

        return pd.DataFrame(records)

    def _default_parameters(self) -> LearnedParameters:
        """Return default parameters if no learned params available."""
        return LearnedParameters(
            beta_pvi_mean=0.5,
            beta_pvi_std=0.1,
            beta_inc_mean=3.0,  # Incumbency advantage
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

    def __init__(
        self,
        national_environment: Optional[float] = None,
        n_simulations: int = 10000,
        random_seed: int = 42,
        learned_params: Optional[LearnedParameters] = None,
        use_cook_data: bool = True,
    ):
        """
        Initialize Senate forecast model.

        Args:
            national_environment: National swing (positive = D advantage).
                                  If None, will be inferred from VoteHub polling.
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
            learned_params: Pre-loaded learned parameters (optional)
            use_cook_data: If True, load race data from Cook Political ratings
        """
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)

        # Load learned parameters
        if learned_params is not None:
            self.params = learned_params
        else:
            try:
                self.params = ParameterFitter.load_parameters()
                logger.info("Loaded learned parameters from historical fitting")
            except FileNotFoundError:
                logger.warning("No learned parameters found, using defaults")
                self.params = self._default_parameters()

        # Load national environment from Bayesian model if not provided
        if national_environment is None:
            logger.info("Loading national environment from VoteHub polling...")
            env_model = NationalEnvironmentModel()
            env_model.load_polls()
            result = env_model.fit(use_pymc=False)
            self.national_env = result["national_environment"]
            self.national_uncertainty = result["uncertainty"]
            logger.info(f"National environment: D{self.national_env:+.1f} ± {self.national_uncertainty:.1f}")
        else:
            self.national_env = national_environment
            self.national_uncertainty = self.DEFAULT_SIGMA_NATIONAL

        # Create national posterior for hierarchical model
        self.national_posterior = NationalPosterior(
            mean=self.national_env,
            std=self.national_uncertainty,
        )

        logger.info(f"Total national environment: D{self.national_env:+.1f}")
        logger.info(f"  (No midterm/economic adjustments - polls already capture these)")

        # Load races - use Cook data if available and enabled
        if use_cook_data:
            all_races = self.load_races_from_cook()
        else:
            all_races = self.RACES_2026

        self.races = [r for r in all_races if r.up_in_2026]
        self.races_df = self._races_to_dataframe()
        self.n_races = len(self.races)

        # Prepare region indexing
        self.region_list = REGIONS
        self.races_df["region_idx"] = self.races_df["region"].map(
            {r: i for i, r in enumerate(self.region_list)}
        )

        self.forecasts: list[SenateRaceForecast] = []
        self.seat_simulations: Optional[np.ndarray] = None

    def simulate_elections(self) -> np.ndarray:
        """
        Posterior predictive Monte Carlo simulation for Senate races.

        Uses the same methodology as HierarchicalForecastModel for House:
        - Sample parameters (beta_pvi, beta_inc, regional_effects) from posteriors
        - Sample national environment from posterior
        - Calculate fundamentals with sampled parameters
        - Add state-level noise

        This is a true posterior predictive simulation - uncertainty flows
        through the model structure, not through added noise shocks.
        """
        logger.info(f"Running {self.n_simulations:,} Senate posterior predictive simulations...")

        n_sims = self.n_simulations
        n_races = self.n_races
        n_regions = len(self.region_list)

        # Get race data from DataFrame
        pvi = self.races_df["pvi"].values
        inc = self.races_df["incumbency_code"].values
        region_idx = self.races_df["region_idx"].values.astype(int)

        # Compute PVI-scaled uncertainty for each race
        # Safer races get less noise, reflecting historical upset rates
        sigma_per_race = compute_pvi_scaled_sigma(
            pvi,
            sigma_base=self.DEFAULT_SIGMA_DISTRICT,
            sigma_floor=2.5,
            midpoint=12.0,
            steepness=0.2,
        )
        logger.info(f"  PVI-scaled σ range: [{sigma_per_race.min():.2f}, {sigma_per_race.max():.2f}]")

        # Sample national environment for each simulation
        national_samples = self.national_posterior.sample(n_sims, self.rng)

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
        # NOTE: Regional effects sampled from Normal(0, σ_regional), NOT from
        # fitted regional_effects posteriors. Only sigma_regional is used.
        regional_effects_samples = self.rng.normal(
            0, self.params.sigma_regional, size=(n_sims, n_regions)
        )

        # State shocks with PVI-scaled uncertainty (independent per state)
        # Each race gets its own sigma based on how competitive it is
        state_shocks = self.rng.normal(0, 1, size=(n_sims, n_races)) * sigma_per_race

        # Calculate vote shares for all simulations
        vote_shares = np.zeros((n_sims, n_races))

        for sim in range(n_sims):
            # Sampled parameters for this simulation
            beta_pvi = beta_pvi_samples[sim]
            beta_inc = beta_inc_samples[sim]
            beta_national = beta_national_samples[sim]
            regional_effects = regional_effects_samples[sim]
            national_shift = national_samples[sim]

            # Calculate fundamentals with sampled parameters
            # PVI SIGN CONVENTION: positive = D-leaning, negative = R-leaning
            # Adding β_pvi × PVI increases Dem vote share for D-leaning states (correct)
            fundamentals = (
                50.0
                + beta_pvi * pvi  # Add because positive PVI = D-leaning
                + beta_inc * inc  # +1 for D incumbent, -1 for R incumbent
                + regional_effects[region_idx]
                + beta_national * national_shift
            )

            # Add only state-level noise
            vote_shares[sim, :] = fundamentals + state_shocks[sim, :]

        # Clip to reasonable range
        vote_shares = np.clip(vote_shares, 1, 99)

        # Store forecasts
        self.forecasts = []
        for i, race in enumerate(self.races):
            race_votes = vote_shares[:, i]
            prob_dem = np.mean(race_votes > 50)

            self.forecasts.append(SenateRaceForecast(
                state=race.state,
                incumbent=race.incumbent,
                incumbent_party=race.incumbent_party,
                prob_dem=prob_dem,
                category=self._categorize_prob(prob_dem),
                pvi=race.pvi,
                open_seat=race.open_seat,
                special=race.special,
            ))

        # Calculate seat outcomes
        # Current: 47D (13 defending) + 53R (22 defending) = 35 seats up
        # Dems NOT up: 47 - 13 = 34 seats
        # To control Senate: need 51 seats (no VP tiebreaker with R president)
        dem_wins = vote_shares > 50
        dem_seats_won = dem_wins.sum(axis=1)
        self.seat_simulations = self.DEM_SEATS_NOT_UP + dem_seats_won

        logger.info(f"  National environment: D{self.national_env:+.1f} ± {self.national_uncertainty:.1f}")
        logger.info(f"  Median Dem seats: {np.median(self.seat_simulations):.0f}")
        logger.info(f"  90% CI: [{np.percentile(self.seat_simulations, 5):.0f}, "
                   f"{np.percentile(self.seat_simulations, 95):.0f}]")
        logger.info(f"  P(Dem control): {np.mean(self.seat_simulations >= 51):.1%}")

        return self.seat_simulations

    def fit_with_pymc(self, draws: int = 1000, tune: int = 1000) -> np.ndarray:
        """
        Full Bayesian fit using PyMC with posterior predictive sampling.

        This samples the hierarchical parameters (national, regional, coefficients)
        and then generates vote share predictions via posterior predictive sampling.
        Uses the same methodology as the House hierarchical model.
        """
        if not PYMC_AVAILABLE:
            logger.warning("PyMC not available, falling back to Monte Carlo simulation")
            return self.simulate_elections()

        logger.info("Fitting Senate model with PyMC (posterior predictive)...")

        # Prepare data
        pvi = self.races_df["pvi"].values
        inc = self.races_df["incumbency_code"].values
        region_idx = self.races_df["region_idx"].values.astype(int)
        n_races = self.n_races
        n_regions = len(self.region_list)

        # Compute PVI-scaled sigma for each race
        sigma_per_race = compute_pvi_scaled_sigma(
            pvi,
            sigma_base=self.DEFAULT_SIGMA_DISTRICT,
            sigma_floor=2.5,
            midpoint=12.0,
            steepness=0.2,
        )

        with pm.Model() as model:
            # Level 1: National environment (prior from poll aggregation)
            mu_national = pm.Normal(
                "mu_national",
                mu=self.national_env,
                sigma=self.national_uncertainty
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

            # State fundamentals (deterministic given parameters)
            # PVI SIGN CONVENTION: positive = D-leaning, negative = R-leaning
            # Adding β_pvi × PVI increases Dem vote share for D-leaning states (correct)
            fundamentals = pm.Deterministic(
                "fundamentals",
                50.0
                + beta_pvi * pvi  # Add because positive PVI = D-leaning
                + beta_inc * inc
                + regional_effects[region_idx]
                + beta_national * mu_national
            )

            # Sample only the parameters (not vote shares - those come from posterior predictive)
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
        # Shape: (chains, draws, n_races) -> (n_samples, n_races)
        fundamentals_flat = fundamentals_samples.reshape(-1, n_races)
        n_samples = len(fundamentals_flat)

        # Generate vote shares via posterior predictive sampling
        # Add state-level noise with PVI-scaled uncertainty
        rng = np.random.default_rng(42)
        state_noise = rng.normal(0, 1, size=(n_samples, n_races)) * sigma_per_race
        vote_shares = np.clip(fundamentals_flat + state_noise, 1, 99)

        # Store forecasts
        self.forecasts = []
        for i, race in enumerate(self.races):
            race_votes = vote_shares[:, i]
            prob_dem = np.mean(race_votes > 50)

            self.forecasts.append(SenateRaceForecast(
                state=race.state,
                incumbent=race.incumbent,
                incumbent_party=race.incumbent_party,
                prob_dem=prob_dem,
                category=self._categorize_prob(prob_dem),
                pvi=race.pvi,
                open_seat=race.open_seat,
                special=race.special,
            ))

        # Calculate seat outcomes
        dem_wins = vote_shares > 50
        dem_seats_won = dem_wins.sum(axis=1)
        self.seat_simulations = self.DEM_SEATS_NOT_UP + dem_seats_won

        logger.info(f"  PyMC fit complete with {n_samples} posterior samples")
        logger.info(f"  Median Dem seats: {np.median(self.seat_simulations):.0f}")
        logger.info(f"  90% CI: [{np.percentile(self.seat_simulations, 5):.0f}, "
                   f"{np.percentile(self.seat_simulations, 95):.0f}]")
        logger.info(f"  P(Dem control): {np.mean(self.seat_simulations >= 51):.1%}")

        return self.seat_simulations

    def run(self, use_pymc: bool = True) -> np.ndarray:
        """
        Run the Senate forecast.

        Args:
            use_pymc: Whether to use full PyMC model (default True)

        Returns:
            Array of seat simulations
        """
        if use_pymc and PYMC_AVAILABLE:
            return self.fit_with_pymc()
        else:
            return self.simulate_elections()

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
        """Get summary statistics."""
        if self.seat_simulations is None:
            self.simulate_elections()

        summary = {
            "prob_dem_control": float(np.mean(self.seat_simulations >= 51)),
            "prob_rep_control": float(np.mean(self.seat_simulations < 51)),
            "median_dem_seats": int(np.median(self.seat_simulations)),
            "mean_dem_seats": float(np.mean(self.seat_simulations)),
            "ci_90_low": int(np.percentile(self.seat_simulations, 5)),
            "ci_90_high": int(np.percentile(self.seat_simulations, 95)),
            "seats_up": len(self.races),
            "dem_defending": sum(1 for r in self.races if r.incumbent_party == "D"),
            "rep_defending": sum(1 for r in self.races if r.incumbent_party == "R"),
            "national_environment": float(self.national_env),
            "national_uncertainty": float(self.national_uncertainty),
            "model_type": "hierarchical_bayesian",
        }

        return summary

    def get_seat_distribution(self) -> dict:
        """Get probability distribution of seat outcomes."""
        if self.seat_simulations is None:
            self.simulate_elections()

        unique, counts = np.unique(self.seat_simulations, return_counts=True)
        probs = counts / len(self.seat_simulations)

        return {
            "dem_seats": unique.tolist(),
            "probabilities": probs.tolist(),
        }

    def get_race_forecasts(self) -> list[dict]:
        """Get forecasts for all races."""
        if not self.forecasts:
            self.simulate_elections()

        return [
            {
                "state": f.state,
                "incumbent": f.incumbent,
                "incumbent_party": f.incumbent_party,
                "prob_dem": round(f.prob_dem, 3),
                "category": f.category,
                "pvi": f.pvi,
                "open_seat": f.open_seat,
                "special": f.special,
            }
            for f in sorted(self.forecasts, key=lambda x: abs(x.prob_dem - 0.5))
        ]

    def get_category_counts(self) -> dict:
        """Get count of races in each category."""
        if not self.forecasts:
            self.simulate_elections()

        categories = {
            "safe_d": 0, "likely_d": 0, "lean_d": 0,
            "toss_up": 0,
            "lean_r": 0, "likely_r": 0, "safe_r": 0,
        }

        for f in self.forecasts:
            categories[f.category] += 1

        return categories
