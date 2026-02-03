#!/usr/bin/env python3
"""
Economic Fundamentals Module.

Processes economic data to create a composite economic index that can
inform the national environment prior in the Bayesian hierarchical model.

Economic indicators:
- Consumer sentiment (University of Michigan)
- Unemployment rate
- Real disposable income growth
- GDP growth
- CPI inflation

The economic index is used to adjust the national environment prior,
not as a direct predictor. This follows the methodology where economic
conditions affect the president's party performance in midterms.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ECONOMIC = PROJECT_ROOT / "data" / "raw" / "economic"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_HISTORICAL = PROJECT_ROOT / "data" / "historical"

# Economic indicator weights (based on predictive power for midterms)
INDICATOR_WEIGHTS = {
    "consumer_sentiment": 0.35,  # Strong predictor of voter mood
    "unemployment_rate": 0.25,   # Visible economic indicator
    "real_disposable_income": 0.20,  # How people feel about their finances
    "gdp": 0.15,                 # Overall economic health
    "cpi": 0.05,                 # Inflation (inverted - high is bad)
}


class EconomicFundamentals:
    """
    Process economic data to inform national environment priors.

    The economic index represents how favorable economic conditions are
    for the incumbent president's party. Positive = good for incumbent,
    negative = bad for incumbent.
    """

    def __init__(self):
        """Initialize economic fundamentals processor."""
        self.data: dict[str, pd.DataFrame] = {}
        self.current_index: Optional[float] = None
        self.historical_indices: dict[int, float] = {}

    def load_data(self) -> None:
        """Load all economic time series data."""
        logger.info("Loading economic data...")

        indicators = [
            "consumer_sentiment",
            "unemployment_rate",
            "real_disposable_income",
            "gdp",
            "cpi",
        ]

        for indicator in indicators:
            path = DATA_ECONOMIC / f"{indicator}.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")
                self.data[indicator] = df
                logger.info(f"  Loaded {indicator}: {len(df)} observations")
            else:
                logger.warning(f"  Missing {indicator} data")

    def _get_value_at_date(
        self,
        indicator: str,
        target_date: datetime,
        lookback_months: int = 3
    ) -> Optional[float]:
        """
        Get indicator value at or near a target date.

        Uses average of last `lookback_months` months for stability.
        """
        if indicator not in self.data:
            return None

        df = self.data[indicator]
        col = df.columns[1]  # Second column is the value

        # Filter to dates before target
        mask = df["date"] <= target_date
        recent = df[mask].tail(lookback_months)

        if len(recent) == 0:
            return None

        return recent[col].mean()

    def _calculate_yoy_change(
        self,
        indicator: str,
        target_date: datetime
    ) -> Optional[float]:
        """Calculate year-over-year change for an indicator."""
        if indicator not in self.data:
            return None

        df = self.data[indicator]
        col = df.columns[1]

        # Get current value (average of last 3 months)
        current = self._get_value_at_date(indicator, target_date, lookback_months=3)
        if current is None:
            return None

        # Get value from 12 months ago
        year_ago = target_date - pd.DateOffset(months=12)
        past = self._get_value_at_date(indicator, year_ago, lookback_months=3)
        if past is None:
            return None

        # Calculate change
        if indicator in ["unemployment_rate", "cpi"]:
            # For these, higher is worse, so invert
            return past - current  # Positive = improvement
        elif indicator == "consumer_sentiment":
            # Already in "good = high" form
            return current - past
        else:
            # For income/GDP, calculate percentage change
            if past == 0:
                return 0
            return ((current - past) / abs(past)) * 100

    def calculate_index(
        self,
        target_date: Optional[datetime] = None,
        normalize: bool = True
    ) -> dict:
        """
        Calculate composite economic index for a target date.

        Args:
            target_date: Date to calculate index for (default: most recent)
            normalize: Whether to normalize to historical distribution

        Returns:
            Dictionary with index value and component breakdowns
        """
        if not self.data:
            self.load_data()

        if target_date is None:
            # Use most recent data point
            target_date = datetime.now()

        logger.info(f"Calculating economic index for {target_date.strftime('%Y-%m')}")

        components = {}
        weighted_sum = 0
        total_weight = 0

        for indicator, weight in INDICATOR_WEIGHTS.items():
            change = self._calculate_yoy_change(indicator, target_date)

            if change is not None:
                components[indicator] = {
                    "yoy_change": round(change, 2),
                    "weight": weight,
                    "contribution": round(change * weight, 2),
                }
                weighted_sum += change * weight
                total_weight += weight
            else:
                components[indicator] = {"yoy_change": None, "weight": weight}

        # Normalize by actual weights used
        if total_weight > 0:
            raw_index = weighted_sum / total_weight
        else:
            raw_index = 0

        # Optionally normalize to historical distribution
        if normalize:
            # Calculate historical indices for reference
            historical_values = self._calculate_historical_indices()
            if historical_values:
                hist_mean = np.mean(list(historical_values.values()))
                hist_std = np.std(list(historical_values.values()))
                if hist_std > 0:
                    normalized_index = (raw_index - hist_mean) / hist_std
                else:
                    normalized_index = 0
            else:
                normalized_index = raw_index / 10  # Rough normalization
        else:
            normalized_index = raw_index

        result = {
            "date": target_date.strftime("%Y-%m-%d"),
            "raw_index": round(raw_index, 2),
            "normalized_index": round(normalized_index, 2),
            "components": components,
            "interpretation": self._interpret_index(normalized_index),
        }

        self.current_index = normalized_index
        return result

    def _calculate_historical_indices(self) -> dict[int, float]:
        """Calculate economic index for historical midterm years."""
        if self.historical_indices:
            return self.historical_indices

        # Calculate for October of midterm years (pre-election)
        midterm_years = [2006, 2010, 2014, 2018, 2022]

        for year in midterm_years:
            target_date = datetime(year, 10, 1)
            try:
                result = self.calculate_index(target_date, normalize=False)
                self.historical_indices[year] = result["raw_index"]
            except Exception:
                pass

        return self.historical_indices

    def _interpret_index(self, index: float) -> str:
        """Interpret the economic index value."""
        if index > 1.0:
            return "Very favorable for incumbent party"
        elif index > 0.5:
            return "Favorable for incumbent party"
        elif index > -0.5:
            return "Neutral economic conditions"
        elif index > -1.0:
            return "Unfavorable for incumbent party"
        else:
            return "Very unfavorable for incumbent party"

    def get_prior_adjustment(
        self,
        president_party: str = "R",
        target_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate adjustment to national environment prior based on economics.

        For Republican president:
        - Good economy → slight boost for Republicans (reduces Dem advantage)
        - Bad economy → boost for Democrats (increases Dem advantage)

        Args:
            president_party: "R" or "D"
            target_date: Date for economic calculation

        Returns:
            Adjustment to add to Democratic margin (positive = helps Dems)
        """
        if self.current_index is None:
            self.calculate_index(target_date)

        # Economic effect magnitude (conservative estimate)
        # Each standard deviation of economic index ≈ 1 point on generic ballot
        econ_effect = -self.current_index * 1.0

        # If Democratic president, flip the effect
        if president_party == "D":
            econ_effect = -econ_effect

        logger.info(f"Economic prior adjustment: {econ_effect:+.2f} points")
        return econ_effect

    def get_historical_comparison(self, target_date: Optional[datetime] = None) -> dict:
        """
        Compare current economic conditions to historical midterms.

        Returns percentile ranking and similar historical years.
        """
        if self.current_index is None:
            self.calculate_index(target_date)

        historical = self._calculate_historical_indices()

        if not historical:
            return {"percentile": 50, "similar_years": []}

        # Calculate percentile
        values = list(historical.values())
        percentile = (np.sum(np.array(values) < self.current_index) / len(values)) * 100

        # Find most similar years
        similarities = [
            (year, abs(idx - self.current_index))
            for year, idx in historical.items()
        ]
        similarities.sort(key=lambda x: x[1])
        similar_years = [year for year, _ in similarities[:2]]

        return {
            "current_index": round(self.current_index, 2),
            "percentile": round(percentile, 0),
            "similar_years": similar_years,
            "historical_indices": {k: round(v, 2) for k, v in historical.items()},
        }

    def save_current(self, path: Optional[Path] = None) -> None:
        """Save current economic index to file."""
        if path is None:
            path = DATA_PROCESSED / "economic_index.json"

        result = self.calculate_index()
        result["calculated_at"] = datetime.now().isoformat()
        result["historical_comparison"] = self.get_historical_comparison()

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved economic index to {path}")


def main():
    """Run economic fundamentals calculation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Economic Fundamentals")
    logger.info("=" * 60)

    econ = EconomicFundamentals()
    econ.load_data()

    # Calculate current index
    result = econ.calculate_index()

    logger.info("\n" + "-" * 60)
    logger.info("Current Economic Index")
    logger.info("-" * 60)
    logger.info(f"Date: {result['date']}")
    logger.info(f"Raw Index: {result['raw_index']:.2f}")
    logger.info(f"Normalized Index: {result['normalized_index']:.2f}")
    logger.info(f"Interpretation: {result['interpretation']}")

    logger.info("\nComponents:")
    for indicator, data in result["components"].items():
        if data["yoy_change"] is not None:
            logger.info(f"  {indicator:25s}: {data['yoy_change']:+.2f} (weight: {data['weight']})")

    # Historical comparison
    comparison = econ.get_historical_comparison()
    logger.info("\n" + "-" * 60)
    logger.info("Historical Comparison")
    logger.info("-" * 60)
    logger.info(f"Percentile: {comparison['percentile']:.0f}th")
    logger.info(f"Most similar years: {comparison['similar_years']}")

    logger.info("\nHistorical midterm indices:")
    for year, idx in sorted(comparison.get("historical_indices", {}).items()):
        logger.info(f"  {year}: {idx:+.2f}")

    # Prior adjustment for 2026 (Republican president)
    adjustment = econ.get_prior_adjustment(president_party="R")
    logger.info(f"\nPrior adjustment for 2026 (R president): {adjustment:+.2f} points")

    # Save results
    econ.save_current()

    return result


if __name__ == "__main__":
    main()
