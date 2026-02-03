#!/usr/bin/env python3
"""
VoteHub API data fetcher for polling data.

Fetches generic ballot and presidential approval polls from VoteHub API.
API endpoint: https://api.votehub.com/polls
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_POLLING = DATA_RAW / "polling"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class VoteHubFetcher:
    """Fetch polling data from VoteHub API."""

    BASE_URL = "https://api.votehub.com/polls"

    # Pollster quality ratings (538-style grades, higher = better)
    # A+ = 3.0, A = 2.7, A- = 2.3, B+ = 2.0, etc.
    POLLSTER_RATINGS = {
        # Top tier (A/A+)
        "Marist": 3.0,
        "Monmouth": 2.7,
        "Pew Research": 2.7,
        "ABC News": 2.7,
        "CBS News": 2.7,
        "NBC News": 2.7,
        "CNN": 2.7,
        "Fox News": 2.7,
        "Quinnipiac": 2.7,
        "Siena College": 2.7,
        "NYT/Siena": 3.0,
        "Selzer": 3.0,

        # Good tier (A-/B+)
        "YouGov": 2.3,
        "Ipsos": 2.3,
        "Gallup": 2.3,
        "Reuters": 2.3,
        "Morning Consult": 2.0,
        "Emerson": 2.0,
        "SurveyUSA": 2.0,
        "PPP": 2.0,

        # Mid tier (B/B-)
        "Rasmussen": 1.7,
        "Trafalgar": 1.3,
        "TIPP Insights": 1.7,
        "RMG Research": 1.7,
        "HarrisX": 1.7,
        "Data for Progress": 1.7,
        "Echelon Insights": 1.7,

        # Lower tier (C)
        "InsiderAdvantage": 1.0,
        "McLaughlin": 1.0,
    }

    DEFAULT_RATING = 1.5  # For unknown pollsters

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "2026-Forecast-Model/1.0"
        })

    def _get_pollster_weight(self, pollster: str) -> float:
        """Get weight for pollster based on quality rating."""
        # Try exact match first
        if pollster in self.POLLSTER_RATINGS:
            return self.POLLSTER_RATINGS[pollster]

        # Try partial match
        for known_pollster, rating in self.POLLSTER_RATINGS.items():
            if known_pollster.lower() in pollster.lower():
                return rating

        return self.DEFAULT_RATING

    def _get_population_weight(self, population: str) -> float:
        """Weight by population type (LV > RV > A)."""
        weights = {
            "lv": 1.0,   # Likely voters - best
            "rv": 0.85,  # Registered voters
            "a": 0.7,    # Adults
        }
        return weights.get(population, 0.7)

    def fetch_generic_ballot(self, days_back: int = 180) -> pd.DataFrame:
        """
        Fetch generic ballot polls.

        Returns DataFrame with columns:
        - date: poll end date
        - pollster: pollster name
        - sample_size: number of respondents
        - population: lv/rv/a
        - dem_pct: Democratic percentage
        - rep_pct: Republican percentage
        - margin: Dem - Rep (positive = D lead)
        - weight: combined quality weight
        """
        logger.info("Fetching generic ballot polls from VoteHub...")

        try:
            response = self.session.get(
                self.BASE_URL,
                params={"poll": "generic_ballot_2026"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch generic ballot: {e}")
            return pd.DataFrame()

        # Filter for generic ballot polls
        polls = [p for p in data if p.get("poll_type") == "generic-ballot"]
        logger.info(f"Found {len(polls)} generic ballot polls")

        # Parse into structured format
        records = []
        cutoff_date = datetime.now() - timedelta(days=days_back)

        for poll in polls:
            try:
                end_date = datetime.strptime(poll["end_date"], "%Y-%m-%d")
                if end_date < cutoff_date:
                    continue

                # Extract D/R percentages
                answers = {a["choice"]: a["pct"] for a in poll.get("answers", [])}
                dem_pct = answers.get("Dem", answers.get("Democrat", 0))
                rep_pct = answers.get("Rep", answers.get("Republican", 0))

                if dem_pct == 0 or rep_pct == 0:
                    continue

                pollster = poll.get("pollster", "Unknown")
                population = poll.get("population", "a")
                sample_size = poll.get("sample_size") or 500  # Default if None

                # Calculate weight
                pollster_weight = self._get_pollster_weight(pollster)
                pop_weight = self._get_population_weight(population)
                # Weight also by recency (exponential decay, half-life = 30 days)
                days_old = (datetime.now() - end_date).days
                recency_weight = 0.5 ** (days_old / 30)
                # Weight by sample size (log scale)
                sample_weight = min(1.0, (sample_size / 1000) ** 0.5)

                combined_weight = pollster_weight * pop_weight * recency_weight * sample_weight

                records.append({
                    "date": end_date.strftime("%Y-%m-%d"),
                    "pollster": pollster,
                    "sample_size": sample_size,
                    "population": population,
                    "dem_pct": dem_pct,
                    "rep_pct": rep_pct,
                    "margin": dem_pct - rep_pct,
                    "weight": combined_weight,
                    "partisan": poll.get("partisan"),
                    "internal": poll.get("internal", False),
                })
            except (KeyError, ValueError) as e:
                logger.debug(f"Skipping malformed poll: {e}")
                continue

        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date", ascending=False)

        logger.info(f"Processed {len(df)} recent generic ballot polls")
        return df

    def fetch_trump_approval(self, days_back: int = 180) -> pd.DataFrame:
        """
        Fetch Trump approval polls.

        Returns DataFrame with columns:
        - date: poll end date
        - pollster: pollster name
        - sample_size: number of respondents
        - population: lv/rv/a
        - approve: approval percentage
        - disapprove: disapproval percentage
        - net_approval: approve - disapprove
        - weight: combined quality weight
        """
        logger.info("Fetching Trump approval polls from VoteHub...")

        try:
            response = self.session.get(
                self.BASE_URL,
                params={"poll": "trump_approval"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Trump approval: {e}")
            return pd.DataFrame()

        # Filter for Trump approval polls
        polls = [
            p for p in data
            if p.get("poll_type") == "approval"
            and "Trump" in str(p.get("subject", ""))
        ]
        logger.info(f"Found {len(polls)} Trump approval polls")

        records = []
        cutoff_date = datetime.now() - timedelta(days=days_back)

        for poll in polls:
            try:
                end_date = datetime.strptime(poll["end_date"], "%Y-%m-%d")
                if end_date < cutoff_date:
                    continue

                answers = {a["choice"]: a["pct"] for a in poll.get("answers", [])}
                approve = answers.get("Approve", 0)
                disapprove = answers.get("Disapprove", 0)

                if approve == 0 or disapprove == 0:
                    continue

                pollster = poll.get("pollster", "Unknown")
                population = poll.get("population", "a")
                sample_size = poll.get("sample_size") or 500  # Default if None

                pollster_weight = self._get_pollster_weight(pollster)
                pop_weight = self._get_population_weight(population)
                days_old = (datetime.now() - end_date).days
                recency_weight = 0.5 ** (days_old / 30)
                sample_weight = min(1.0, (sample_size / 1000) ** 0.5)

                combined_weight = pollster_weight * pop_weight * recency_weight * sample_weight

                records.append({
                    "date": end_date.strftime("%Y-%m-%d"),
                    "pollster": pollster,
                    "sample_size": sample_size,
                    "population": population,
                    "approve": approve,
                    "disapprove": disapprove,
                    "net_approval": approve - disapprove,
                    "weight": combined_weight,
                })
            except (KeyError, ValueError) as e:
                logger.debug(f"Skipping malformed poll: {e}")
                continue

        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date", ascending=False)

        logger.info(f"Processed {len(df)} recent Trump approval polls")
        return df

    def save_polls(self, gb_df: pd.DataFrame, approval_df: pd.DataFrame) -> None:
        """Save polling data to CSV files."""
        DATA_POLLING.mkdir(parents=True, exist_ok=True)

        if not gb_df.empty:
            gb_path = DATA_POLLING / "generic_ballot.csv"
            gb_df.to_csv(gb_path, index=False)
            logger.info(f"Saved {len(gb_df)} generic ballot polls to {gb_path}")

        if not approval_df.empty:
            app_path = DATA_POLLING / "trump_approval.csv"
            approval_df.to_csv(app_path, index=False)
            logger.info(f"Saved {len(approval_df)} approval polls to {app_path}")

    def get_polling_summary(self, gb_df: pd.DataFrame, approval_df: pd.DataFrame) -> dict:
        """Calculate weighted polling averages."""
        summary = {}

        if not gb_df.empty:
            # Weighted average of generic ballot margin
            weights = gb_df["weight"].values
            margins = gb_df["margin"].values
            weighted_margin = (margins * weights).sum() / weights.sum()

            summary["generic_ballot"] = {
                "weighted_margin": round(weighted_margin, 2),
                "n_polls": len(gb_df),
                "latest_date": gb_df["date"].max().strftime("%Y-%m-%d"),
            }

        if not approval_df.empty:
            weights = approval_df["weight"].values
            net_approvals = approval_df["net_approval"].values
            weighted_net = (net_approvals * weights).sum() / weights.sum()

            approves = approval_df["approve"].values
            weighted_approve = (approves * weights).sum() / weights.sum()

            summary["trump_approval"] = {
                "weighted_approve": round(weighted_approve, 1),
                "weighted_net": round(weighted_net, 2),
                "n_polls": len(approval_df),
                "latest_date": approval_df["date"].max().strftime("%Y-%m-%d"),
            }

        return summary


def main():
    """Fetch and save VoteHub polling data."""
    logger.info("=" * 60)
    logger.info("VoteHub Polling Data Fetch")
    logger.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    fetcher = VoteHubFetcher()

    # Fetch polls
    gb_df = fetcher.fetch_generic_ballot(days_back=180)
    approval_df = fetcher.fetch_trump_approval(days_back=180)

    # Save to disk
    fetcher.save_polls(gb_df, approval_df)

    # Print summary
    summary = fetcher.get_polling_summary(gb_df, approval_df)

    logger.info("\n" + "=" * 60)
    logger.info("POLLING SUMMARY")
    logger.info("=" * 60)

    if "generic_ballot" in summary:
        gb = summary["generic_ballot"]
        margin_str = f"D+{gb['weighted_margin']:.1f}" if gb['weighted_margin'] > 0 else f"R+{-gb['weighted_margin']:.1f}"
        logger.info(f"Generic Ballot: {margin_str} ({gb['n_polls']} polls)")

    if "trump_approval" in summary:
        app = summary["trump_approval"]
        logger.info(f"Trump Approval: {app['weighted_approve']:.1f}% (net: {app['weighted_net']:+.1f}, {app['n_polls']} polls)")

    # Save summary JSON
    summary_path = DATA_POLLING / "polling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary to {summary_path}")

    return summary


if __name__ == "__main__":
    main()
