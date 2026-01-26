#!/usr/bin/env python3
"""
Senate Forecast Model for 2026 Elections.

2026 Senate races: 33 Class 2 seats + 2 special elections
Democrats defending: 12 seats
Republicans defending: 21 seats (historically favorable map for Dems)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
    Senate forecast model using similar methodology to House model.

    Key differences:
    - State-level races (not district)
    - Fewer races = more uncertainty per seat
    - Incumbent advantage stronger in Senate
    """

    PARAMS = {
        "incumbency_advantage": 4.0,  # Stronger in Senate
        "midterm_bonus": 3.5,  # Out-party bonus
        "national_weight": 0.5,  # How much national environment matters
        "state_uncertainty": 5.0,  # SD for state-level races
    }

    # 2026 Senate races - Class 2 seats
    # Current composition: 53R - 47D (including independents caucusing with Dems)
    RACES_2026 = [
        # Democrats defending (12 seats)
        SenateRace("GA", 2, "Jon Ossoff", "D", True, 0, "toss_up"),
        SenateRace("MI", 2, "Gary Peters", "D", True, -1, "lean_d"),
        SenateRace("NH", 2, "Jeanne Shaheen", "D", True, -2, "likely_d"),
        SenateRace("VA", 2, "Mark Warner", "D", True, -3, "likely_d"),
        SenateRace("CO", 2, "John Hickenlooper", "D", True, -5, "likely_d"),
        SenateRace("IL", 2, "Dick Durbin", "D", True, -8, "safe_d"),
        SenateRace("OR", 2, "Jeff Merkley", "D", True, -6, "safe_d"),
        SenateRace("NM", 2, "Martin Heinrich", "D", True, -5, "likely_d"),
        SenateRace("MN", 2, "Tina Smith", "D", True, -3, "likely_d"),
        SenateRace("DE", 2, "Vacant", "D", True, -7, "safe_d", open_seat=True),
        SenateRace("MA", 2, "Ed Markey", "D", True, -20, "safe_d"),
        SenateRace("RI", 2, "Sheldon Whitehouse", "D", True, -12, "safe_d"),

        # Republicans defending (21 seats)
        SenateRace("NC", 2, "Thom Tillis", "R", True, 2, "toss_up"),
        SenateRace("ME", 2, "Susan Collins", "R", True, -3, "lean_r"),
        SenateRace("IA", 2, "Joni Ernst", "R", True, 6, "likely_r"),
        SenateRace("TX", 2, "John Cornyn", "R", True, 6, "likely_r"),
        SenateRace("AK", 2, "Dan Sullivan", "R", True, 8, "likely_r"),
        SenateRace("SC", 2, "Lindsey Graham", "R", True, 8, "safe_r"),
        SenateRace("LA", 2, "Bill Cassidy", "R", True, 12, "safe_r"),
        SenateRace("KY", 2, "Mitch McConnell", "R", True, 16, "safe_r"),
        SenateRace("AR", 2, "Tom Cotton", "R", True, 17, "safe_r"),
        SenateRace("OK", 2, "Jim Inhofe", "R", True, 20, "safe_r"),
        SenateRace("SD", 2, "Mike Rounds", "R", True, 18, "safe_r"),
        SenateRace("KS", 2, "Jerry Moran", "R", True, 10, "safe_r"),
        SenateRace("ID", 2, "Jim Risch", "R", True, 22, "safe_r"),
        SenateRace("AL", 2, "Katie Britt", "R", True, 15, "safe_r"),
        SenateRace("MS", 2, "Cindy Hyde-Smith", "R", True, 10, "safe_r"),
        SenateRace("TN", 2, "Bill Hagerty", "R", True, 15, "safe_r"),
        SenateRace("NE", 2, "Deb Fischer", "R", True, 14, "safe_r"),
        SenateRace("WV", 2, "Shelley Moore Capito", "R", True, 28, "safe_r"),
        SenateRace("WY", 2, "John Barrasso", "R", True, 40, "safe_r"),
        SenateRace("MT", 2, "Steve Daines", "R", True, 12, "likely_r"),
        SenateRace("AZ", 2, "Vacant", "R", True, 0, "toss_up", open_seat=True),

        # Not up in 2026 (Class 1 and 3) - included for context
        # These are NOT part of 2026 races but shown for current composition
    ]

    def __init__(
        self,
        national_environment: float,
        n_simulations: int = 10000,
        random_seed: int = 42,
    ):
        """
        Initialize Senate forecast model.

        Args:
            national_environment: National swing (positive = D advantage)
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        self.national_env = national_environment
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)

        self.races = [r for r in self.RACES_2026 if r.up_in_2026]
        self.forecasts: list[SenateRaceForecast] = []
        self.seat_simulations: Optional[np.ndarray] = None

    def simulate_elections(self) -> np.ndarray:
        """Run Monte Carlo simulations."""
        logger.info(f"Running {self.n_simulations:,} Senate simulations...")

        n_races = len(self.races)
        vote_shares = np.zeros((self.n_simulations, n_races))

        for sim in range(self.n_simulations):
            # National shock (correlated across all races)
            national_shock = self.rng.normal(0, 2.5)

            for i, race in enumerate(self.races):
                # Baseline from state PVI
                baseline = 50 - race.pvi / 2

                # National environment + shock
                nat_effect = (self.national_env + national_shock) * self.PARAMS["national_weight"]

                # Incumbency advantage
                if race.incumbent_party == "D" and not race.open_seat:
                    incumbency = self.PARAMS["incumbency_advantage"] / 2
                elif race.incumbent_party == "R" and not race.open_seat:
                    incumbency = -self.PARAMS["incumbency_advantage"] / 2
                else:
                    incumbency = 0

                # Midterm bonus for out-party (Democrats)
                midterm = self.PARAMS["midterm_bonus"] / 2

                # State-level uncertainty
                state_shock = self.rng.normal(0, self.PARAMS["state_uncertainty"])

                # Final vote share
                dem_vote = baseline + nat_effect + incumbency + midterm + state_shock
                vote_shares[sim, i] = np.clip(dem_vote, 5, 95)

        # Store forecasts
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
        # Current Dem seats not up: 35 (47 total - 12 up)
        # Need to add simulated wins to get total
        dem_wins = vote_shares > 50
        dem_seats_won = dem_wins.sum(axis=1)

        # Dems have 35 seats not up in 2026
        # To control Senate: need 50 seats (VP breaks tie)
        # Current: 47D (12 up) + 53R (21 up)
        # Seats not up: 35D + 32R = 67 seats
        # 33 seats up for election
        self.seat_simulations = 35 + dem_seats_won  # 35 Dem seats not up

        logger.info(f"  Median Dem seats: {np.median(self.seat_simulations):.0f}")
        logger.info(f"  90% CI: [{np.percentile(self.seat_simulations, 5):.0f}, "
                   f"{np.percentile(self.seat_simulations, 95):.0f}]")
        logger.info(f"  P(Dem control): {np.mean(self.seat_simulations >= 51):.1%}")

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
        """Get summary statistics."""
        if self.seat_simulations is None:
            self.simulate_elections()

        return {
            "prob_dem_control": float(np.mean(self.seat_simulations >= 51)),
            "prob_rep_control": float(np.mean(self.seat_simulations < 51)),
            "median_dem_seats": int(np.median(self.seat_simulations)),
            "mean_dem_seats": float(np.mean(self.seat_simulations)),
            "ci_90_low": int(np.percentile(self.seat_simulations, 5)),
            "ci_90_high": int(np.percentile(self.seat_simulations, 95)),
            "seats_up": len(self.races),
            "dem_defending": sum(1 for r in self.races if r.incumbent_party == "D"),
            "rep_defending": sum(1 for r in self.races if r.incumbent_party == "R"),
        }

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
