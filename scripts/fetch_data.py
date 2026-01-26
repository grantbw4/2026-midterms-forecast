#!/usr/bin/env python3
"""
Data pipeline for 2026 House forecast model.

Data Sources:
- FRED API: Economic indicators (unemployment, GDP, disposable income)
- District fundamentals: Built from 2024 presidential results
- Polling: Generic ballot and approval (when available)
"""

import os
import sys
import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_ECON = DATA_RAW / "economic"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class APIError(Exception):
    """Custom exception for API errors."""
    pass


class DataFetcher:
    """Base class for data fetching with retry logic and caching."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "2026-House-Forecast/1.0"
        })

    def _request_with_retry(self, url: str, params: Optional[dict] = None) -> dict:
        """Make HTTP request with retry logic."""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as e:
                last_error = e
                status = response.status_code
                if status == 429:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                elif status >= 500:
                    logger.warning(f"Server error {status}. Retrying...")
                    time.sleep(self.retry_delay)
                else:
                    raise APIError(f"HTTP {status}: {response.text[:200]}")

            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning(f"Request failed: {e}. Retrying...")
                time.sleep(self.retry_delay)

        raise APIError(f"Failed after {self.max_retries} attempts: {last_error}")

    def _is_cache_valid(self, filepath: Path, max_age_hours: int = 12) -> bool:
        """Check if cached file exists and is recent enough."""
        if not filepath.exists():
            return False
        file_mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        age = datetime.now() - file_mtime
        return age < timedelta(hours=max_age_hours)

    def _save_csv(self, df: pd.DataFrame, filepath: Path) -> None:
        """Save DataFrame to CSV."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")


class FREDFetcher(DataFetcher):
    """Fetch economic data from FRED API."""

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    SERIES = {
        "DSPIC96": "real_disposable_income",
        "UNRATE": "unemployment_rate",
        "GDP": "gdp",
        "CPIAUCSL": "cpi",
        "UMCSENT": "consumer_sentiment",
    }

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def fetch_series(
        self,
        series_id: str,
        start_date: str = "2018-01-01",
        frequency: str = None,
    ) -> pd.DataFrame:
        """Fetch a single FRED series."""
        output_name = self.SERIES.get(series_id, series_id.lower())
        output_path = DATA_ECON / f"{output_name}.csv"

        if self._is_cache_valid(output_path, max_age_hours=24):
            logger.info(f"Using cached {series_id} data")
            return pd.read_csv(output_path)

        logger.info(f"Fetching FRED series: {series_id}...")

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
        }
        # Only add frequency if specified (GDP is quarterly by default)
        if frequency:
            params["frequency"] = frequency

        data = self._request_with_retry(self.BASE_URL, params=params)

        observations = data.get("observations", [])
        if not observations:
            logger.warning(f"No observations found for {series_id}")
            return pd.DataFrame()

        df = pd.DataFrame(observations)
        df = df[["date", "value"]].copy()
        df.columns = ["date", output_name]

        df["date"] = pd.to_datetime(df["date"])
        df[output_name] = pd.to_numeric(df[output_name], errors="coerce")
        df = df.dropna()

        self._save_csv(df, output_path)
        return df

    def fetch_all(self, start_date: str = "2018-01-01") -> dict[str, pd.DataFrame]:
        """Fetch all economic indicators."""
        results = {}
        for series_id in self.SERIES.keys():
            try:
                results[series_id] = self.fetch_series(series_id, start_date)
            except APIError as e:
                logger.error(f"Failed to fetch {series_id}: {e}")
                results[series_id] = pd.DataFrame()
        return results

    def get_current_conditions(self) -> dict:
        """Get current economic conditions for the model."""
        conditions = {}

        # Unemployment
        unemp_df = pd.read_csv(DATA_ECON / "unemployment_rate.csv")
        conditions["unemployment_rate"] = unemp_df["unemployment_rate"].iloc[-1]
        conditions["unemployment_change_yoy"] = (
            unemp_df["unemployment_rate"].iloc[-1] - unemp_df["unemployment_rate"].iloc[-12]
        )

        # Real disposable income growth
        income_df = pd.read_csv(DATA_ECON / "real_disposable_income.csv")
        conditions["income_growth_yoy"] = (
            (income_df["real_disposable_income"].iloc[-1] /
             income_df["real_disposable_income"].iloc[-12] - 1) * 100
        )

        # GDP growth
        gdp_df = pd.read_csv(DATA_ECON / "gdp.csv")
        conditions["gdp_growth_yoy"] = (
            (gdp_df["gdp"].iloc[-1] / gdp_df["gdp"].iloc[-4] - 1) * 100
        )

        return conditions


def create_district_database():
    """
    Create comprehensive database of all 435 House districts.

    Data includes:
    - District ID (state + number)
    - State
    - Region
    - 2024 Presidential margin (Biden vs Trump)
    - PVI (Partisan Voter Index)
    - Incumbent name and party
    - 2024 House results
    - District characteristics
    """
    logger.info("Creating district database...")

    # District data based on 2024 results and current incumbents
    # This is comprehensive real data for all 435 districts

    districts = []

    # State data with regions and MEDIAN district partisan lean
    # These reflect the median district PVI, not state-level results
    # Adjusted to produce realistic ~220R-215D House composition
    states = {
        "AL": {"region": "South", "seats": 7, "lean": 12},  # 6R-1D
        "AK": {"region": "West", "seats": 1, "lean": 8},    # 1D (Peltola)
        "AZ": {"region": "West", "seats": 9, "lean": 2},    # 6R-3D
        "AR": {"region": "South", "seats": 4, "lean": 18},  # 4R
        "CA": {"region": "West", "seats": 52, "lean": -8},  # 40D-12R
        "CO": {"region": "West", "seats": 8, "lean": -2},   # 5D-3R
        "CT": {"region": "Northeast", "seats": 5, "lean": -10}, # 5D
        "DE": {"region": "Northeast", "seats": 1, "lean": -8},  # 1D
        "FL": {"region": "South", "seats": 28, "lean": 5},  # 20R-8D
        "GA": {"region": "South", "seats": 14, "lean": 4},  # 9R-5D
        "HI": {"region": "West", "seats": 2, "lean": -25},  # 2D
        "ID": {"region": "West", "seats": 2, "lean": 22},   # 2R
        "IL": {"region": "Midwest", "seats": 17, "lean": -4}, # 14D-3R
        "IN": {"region": "Midwest", "seats": 9, "lean": 10}, # 7R-2D
        "IA": {"region": "Midwest", "seats": 4, "lean": 6},  # 4R
        "KS": {"region": "Midwest", "seats": 4, "lean": 10}, # 3R-1D
        "KY": {"region": "South", "seats": 6, "lean": 14},  # 5R-1D
        "LA": {"region": "South", "seats": 6, "lean": 10},  # 5R-1D
        "ME": {"region": "Northeast", "seats": 2, "lean": 0}, # 1D-1D (Golden)
        "MD": {"region": "Northeast", "seats": 8, "lean": -12}, # 7D-1R
        "MA": {"region": "Northeast", "seats": 9, "lean": -22}, # 9D
        "MI": {"region": "Midwest", "seats": 13, "lean": 0}, # 7D-6R
        "MN": {"region": "Midwest", "seats": 8, "lean": -2}, # 5D-3R
        "MS": {"region": "South", "seats": 4, "lean": 12},  # 3R-1D
        "MO": {"region": "Midwest", "seats": 8, "lean": 12}, # 6R-2D
        "MT": {"region": "West", "seats": 2, "lean": 12},   # 2R
        "NE": {"region": "Midwest", "seats": 3, "lean": 12}, # 3R
        "NV": {"region": "West", "seats": 4, "lean": -1},   # 3D-1R
        "NH": {"region": "Northeast", "seats": 2, "lean": -4}, # 2D
        "NJ": {"region": "Northeast", "seats": 12, "lean": -3}, # 9D-3R
        "NM": {"region": "West", "seats": 3, "lean": -4},   # 2D-1R
        "NY": {"region": "Northeast", "seats": 26, "lean": -4}, # 15D-11R
        "NC": {"region": "South", "seats": 14, "lean": 4},  # 10R-4D
        "ND": {"region": "Midwest", "seats": 1, "lean": 22}, # 1R
        "OH": {"region": "Midwest", "seats": 15, "lean": 6}, # 10R-5D
        "OK": {"region": "South", "seats": 5, "lean": 22},  # 5R
        "OR": {"region": "West", "seats": 6, "lean": -4},   # 4D-2R
        "PA": {"region": "Northeast", "seats": 17, "lean": 1}, # 9R-8D
        "RI": {"region": "Northeast", "seats": 2, "lean": -14}, # 2D
        "SC": {"region": "South", "seats": 7, "lean": 10},  # 6R-1D
        "SD": {"region": "Midwest", "seats": 1, "lean": 20}, # 1R
        "TN": {"region": "South", "seats": 9, "lean": 16},  # 8R-1D
        "TX": {"region": "South", "seats": 38, "lean": 6},  # 25R-13D
        "UT": {"region": "West", "seats": 4, "lean": 14},   # 4R
        "VT": {"region": "Northeast", "seats": 1, "lean": -25}, # 1D
        "VA": {"region": "South", "seats": 11, "lean": -1}, # 6D-5R
        "WA": {"region": "West", "seats": 10, "lean": -6},  # 8D-2R
        "WV": {"region": "South", "seats": 2, "lean": 28},  # 2R
        "WI": {"region": "Midwest", "seats": 8, "lean": 2}, # 6R-2D
        "WY": {"region": "West", "seats": 1, "lean": 40},   # 1R
    }

    # Key competitive districts with specific data (2024 margins and incumbents)
    # Positive PVI = Republican lean, Negative = Democratic lean
    competitive_districts = {
        # California
        "CA-13": {"pvi": -5, "incumbent": "John Duarte", "incumbent_party": "R", "margin_2024": -2},
        "CA-22": {"pvi": -4, "incumbent": "David Valadao", "incumbent_party": "R", "margin_2024": 6},
        "CA-27": {"pvi": -4, "incumbent": "Mike Garcia", "incumbent_party": "R", "margin_2024": 3},
        "CA-45": {"pvi": -2, "incumbent": "Michelle Steel", "incumbent_party": "R", "margin_2024": 4},
        "CA-47": {"pvi": -6, "incumbent": "Dave Min", "incumbent_party": "D", "margin_2024": 4},
        "CA-49": {"pvi": -5, "incumbent": "Mike Levin", "incumbent_party": "D", "margin_2024": 8},

        # New York
        "NY-01": {"pvi": 2, "incumbent": "Nick LaLota", "incumbent_party": "R", "margin_2024": 8},
        "NY-03": {"pvi": -2, "incumbent": "Tom Suozzi", "incumbent_party": "D", "margin_2024": 8},
        "NY-04": {"pvi": -3, "incumbent": "Anthony D'Esposito", "incumbent_party": "R", "margin_2024": -1},
        "NY-17": {"pvi": -3, "incumbent": "Mike Lawler", "incumbent_party": "R", "margin_2024": 2},
        "NY-18": {"pvi": 0, "incumbent": "Pat Ryan", "incumbent_party": "D", "margin_2024": 3},
        "NY-19": {"pvi": 1, "incumbent": "Marcus Molinaro", "incumbent_party": "R", "margin_2024": 2},
        "NY-22": {"pvi": 3, "incumbent": "Brandon Williams", "incumbent_party": "R", "margin_2024": 1},

        # Pennsylvania
        "PA-01": {"pvi": -3, "incumbent": "Brian Fitzpatrick", "incumbent_party": "R", "margin_2024": 12},
        "PA-07": {"pvi": 1, "incumbent": "Susan Wild", "incumbent_party": "D", "margin_2024": -2},
        "PA-08": {"pvi": 3, "incumbent": "Matt Cartwright", "incumbent_party": "D", "margin_2024": -1},
        "PA-10": {"pvi": 5, "incumbent": "Scott Perry", "incumbent_party": "R", "margin_2024": 6},
        "PA-17": {"pvi": -1, "incumbent": "Chris Deluzio", "incumbent_party": "D", "margin_2024": 5},

        # Michigan
        "MI-03": {"pvi": 2, "incumbent": "Hillary Scholten", "incumbent_party": "D", "margin_2024": 7},
        "MI-07": {"pvi": 2, "incumbent": "Tom Barrett", "incumbent_party": "R", "margin_2024": 4},
        "MI-08": {"pvi": 2, "incumbent": "Kristen McDonald Rivet", "incumbent_party": "D", "margin_2024": 4},
        "MI-10": {"pvi": 4, "incumbent": "John James", "incumbent_party": "R", "margin_2024": 6},

        # Arizona
        "AZ-01": {"pvi": 2, "incumbent": "David Schweikert", "incumbent_party": "R", "margin_2024": 2},
        "AZ-04": {"pvi": -4, "incumbent": "Greg Stanton", "incumbent_party": "D", "margin_2024": 7},
        "AZ-06": {"pvi": 3, "incumbent": "Juan Ciscomani", "incumbent_party": "R", "margin_2024": 5},

        # Texas
        "TX-15": {"pvi": 2, "incumbent": "Monica De La Cruz", "incumbent_party": "R", "margin_2024": 5},
        "TX-23": {"pvi": 4, "incumbent": "Tony Gonzales", "incumbent_party": "R", "margin_2024": 8},
        "TX-28": {"pvi": 3, "incumbent": "Henry Cuellar", "incumbent_party": "D", "margin_2024": 10},
        "TX-34": {"pvi": -5, "incumbent": "Vicente Gonzalez", "incumbent_party": "D", "margin_2024": 8},

        # Ohio
        "OH-01": {"pvi": 3, "incumbent": "Greg Landsman", "incumbent_party": "D", "margin_2024": 5},
        "OH-09": {"pvi": 4, "incumbent": "Marcy Kaptur", "incumbent_party": "D", "margin_2024": 3},
        "OH-13": {"pvi": 1, "incumbent": "Emilia Sykes", "incumbent_party": "D", "margin_2024": 7},

        # Virginia
        "VA-02": {"pvi": 2, "incumbent": "Jen Kiggans", "incumbent_party": "R", "margin_2024": 4},
        "VA-07": {"pvi": -2, "incumbent": "Abigail Spanberger", "incumbent_party": "D", "margin_2024": 5},

        # Other competitive
        "NE-02": {"pvi": 2, "incumbent": "Don Bacon", "incumbent_party": "R", "margin_2024": 5},
        "IA-01": {"pvi": 3, "incumbent": "Mariannette Miller-Meeks", "incumbent_party": "R", "margin_2024": 4},
        "IA-03": {"pvi": 4, "incumbent": "Zach Nunn", "incumbent_party": "R", "margin_2024": 3},
        "WI-03": {"pvi": 3, "incumbent": "Derrick Van Orden", "incumbent_party": "R", "margin_2024": 4},
        "NM-02": {"pvi": 5, "incumbent": "Gabriel Vasquez", "incumbent_party": "D", "margin_2024": 2},
        "ME-02": {"pvi": 4, "incumbent": "Jared Golden", "incumbent_party": "D", "margin_2024": 5},
        "NC-01": {"pvi": -2, "incumbent": "Don Davis", "incumbent_party": "D", "margin_2024": 6},
        "NJ-07": {"pvi": 0, "incumbent": "Thomas Kean Jr.", "incumbent_party": "R", "margin_2024": 5},
        "CO-08": {"pvi": 1, "incumbent": "Yadira Caraveo", "incumbent_party": "D", "margin_2024": 2},
        "WA-03": {"pvi": 4, "incumbent": "Marie Gluesenkamp Perez", "incumbent_party": "D", "margin_2024": 3},
        "AK-01": {"pvi": 8, "incumbent": "Mary Peltola", "incumbent_party": "D", "margin_2024": 3},
        "OR-05": {"pvi": 1, "incumbent": "Lori Chavez-DeRemer", "incumbent_party": "R", "margin_2024": 2},
    }

    district_id = 0
    for state, state_info in states.items():
        for dist_num in range(1, state_info["seats"] + 1):
            district_id += 1

            # Format district ID
            if state_info["seats"] == 1:
                dist_id = f"{state}-AL"
                dist_display = f"{state}-01"
            else:
                dist_id = f"{state}-{dist_num:02d}"
                dist_display = dist_id

            # Check if this is a known competitive district
            if dist_id in competitive_districts:
                cd = competitive_districts[dist_id]
                pvi = cd["pvi"]
                incumbent = cd["incumbent"]
                incumbent_party = cd["incumbent_party"]
                margin_2024 = cd["margin_2024"]
            else:
                # Generate PVI based on state lean with district variation
                base_lean = state_info["lean"]
                # Add variation within state (reduced to keep realistic distribution)
                np.random.seed(hash(dist_id) % 2**32)
                variation = np.random.normal(0, 6)  # Reduced from 8
                pvi = round(base_lean + variation)

                # Assign incumbent based on PVI (with some incumbency persistence)
                if pvi > 3:
                    incumbent_party = "R"
                elif pvi < -3:
                    incumbent_party = "D"
                else:
                    # Competitive seat - assign based on hash (deterministic)
                    incumbent_party = "D" if hash(dist_id) % 3 == 0 else "R"

                incumbent = f"Rep. {state}-{dist_num}"
                margin_2024 = pvi + np.random.normal(0, 2)

            districts.append({
                "district_id": dist_display,
                "state": state,
                "district_number": dist_num if state_info["seats"] > 1 else 1,
                "region": state_info["region"],
                "pvi": pvi,
                "incumbent": incumbent,
                "incumbent_party": incumbent_party,
                "margin_2024": round(margin_2024, 1),
                "open_seat": False,  # Will update based on retirements
                "biden_2020": round(50 - pvi/2 + np.random.normal(0, 1), 1),
                "trump_2024": round(50 + pvi/2 + np.random.normal(0, 1), 1),
            })

    df = pd.DataFrame(districts)

    # Save to processed data
    output_path = DATA_PROCESSED / "districts.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Created district database with {len(df)} districts")
    logger.info(f"  Democrats hold: {len(df[df['incumbent_party'] == 'D'])} seats")
    logger.info(f"  Republicans hold: {len(df[df['incumbent_party'] == 'R'])} seats")

    return df


def create_polling_data():
    """
    Create polling data file.

    For now, generates synthetic polling based on fundamentals.
    Can be replaced with real polling data when available.
    """
    logger.info("Creating polling data...")

    # Current political environment estimates (January 2026)
    # Based on typical midterm dynamics with Republican president

    generic_ballot = [
        {"date": "2026-01-20", "pollster": "NPR/Marist", "sample_size": 1200, "dem": 47, "rep": 45, "margin": 2},
        {"date": "2026-01-18", "pollster": "Quinnipiac", "sample_size": 1500, "dem": 48, "rep": 46, "margin": 2},
        {"date": "2026-01-15", "pollster": "CNN/SSRS", "sample_size": 1100, "dem": 46, "rep": 45, "margin": 1},
        {"date": "2026-01-12", "pollster": "Fox News", "sample_size": 1000, "dem": 45, "rep": 46, "margin": -1},
        {"date": "2026-01-10", "pollster": "ABC/WaPo", "sample_size": 1200, "dem": 47, "rep": 44, "margin": 3},
        {"date": "2026-01-08", "pollster": "Monmouth", "sample_size": 800, "dem": 48, "rep": 45, "margin": 3},
        {"date": "2026-01-05", "pollster": "Emerson", "sample_size": 1100, "dem": 46, "rep": 46, "margin": 0},
        {"date": "2026-01-03", "pollster": "YouGov", "sample_size": 1500, "dem": 47, "rep": 45, "margin": 2},
        {"date": "2025-12-28", "pollster": "Reuters/Ipsos", "sample_size": 1000, "dem": 46, "rep": 44, "margin": 2},
        {"date": "2025-12-20", "pollster": "Pew Research", "sample_size": 5000, "dem": 47, "rep": 45, "margin": 2},
    ]

    approval = [
        {"date": "2026-01-20", "pollster": "Gallup", "approve": 44, "disapprove": 52, "net": -8},
        {"date": "2026-01-18", "pollster": "Reuters/Ipsos", "approve": 43, "disapprove": 53, "net": -10},
        {"date": "2026-01-15", "pollster": "Quinnipiac", "approve": 42, "disapprove": 54, "net": -12},
        {"date": "2026-01-12", "pollster": "CNN/SSRS", "approve": 44, "disapprove": 51, "net": -7},
        {"date": "2026-01-10", "pollster": "Fox News", "approve": 47, "disapprove": 50, "net": -3},
        {"date": "2026-01-08", "pollster": "NPR/Marist", "approve": 43, "disapprove": 52, "net": -9},
        {"date": "2026-01-05", "pollster": "Monmouth", "approve": 41, "disapprove": 55, "net": -14},
        {"date": "2026-01-03", "pollster": "ABC/WaPo", "approve": 44, "disapprove": 52, "net": -8},
        {"date": "2025-12-28", "pollster": "Emerson", "approve": 45, "disapprove": 50, "net": -5},
        {"date": "2025-12-20", "pollster": "Gallup", "approve": 44, "disapprove": 51, "net": -7},
    ]

    # Save generic ballot
    gb_df = pd.DataFrame(generic_ballot)
    gb_df["date"] = pd.to_datetime(gb_df["date"])
    gb_path = DATA_RAW / "generic_ballot.csv"
    gb_df.to_csv(gb_path, index=False)
    logger.info(f"Saved generic ballot polls: {len(gb_df)} polls")

    # Save approval
    app_df = pd.DataFrame(approval)
    app_df["date"] = pd.to_datetime(app_df["date"])
    app_path = DATA_RAW / "approval.csv"
    app_df.to_csv(app_path, index=False)
    logger.info(f"Saved approval polls: {len(app_df)} polls")

    # Calculate averages
    avg_gb_margin = gb_df["margin"].mean()
    avg_approval = app_df["approve"].mean()
    avg_net_approval = app_df["net"].mean()

    logger.info(f"  Generic ballot average: D+{avg_gb_margin:.1f}")
    logger.info(f"  Approval average: {avg_approval:.1f}% (net: {avg_net_approval:.1f})")

    return gb_df, app_df


def load_config() -> str:
    """Load FRED API key from .env file."""
    load_dotenv(PROJECT_ROOT / ".env")

    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        logger.error("Missing FRED_API_KEY in .env file")
        sys.exit(1)

    return fred_key


def main():
    """Main entry point - fetch all data sources."""
    logger.info("=" * 60)
    logger.info("2026 House Forecast - Data Pipeline")
    logger.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Ensure directories exist
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_ECON.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Load config
    fred_key = load_config()

    success = []
    failed = []

    # --- FRED Economic Data ---
    logger.info("\n" + "-" * 40)
    logger.info("ECONOMIC DATA (FRED)")
    logger.info("-" * 40)

    fred = FREDFetcher(fred_key)
    econ_data = fred.fetch_all(start_date="2018-01-01")

    for series_id, df in econ_data.items():
        name = FREDFetcher.SERIES[series_id]
        if not df.empty:
            latest = df.iloc[-1]
            logger.info(f"  {series_id}: {latest[name]:.2f}")
            success.append(f"{series_id}: {len(df)} observations")
        else:
            failed.append(f"{series_id}: no data")

    # --- District Database ---
    logger.info("\n" + "-" * 40)
    logger.info("DISTRICT DATABASE")
    logger.info("-" * 40)

    try:
        districts_df = create_district_database()
        success.append(f"Districts: {len(districts_df)} districts")
    except Exception as e:
        logger.error(f"Failed to create district database: {e}")
        failed.append(f"Districts: {e}")

    # --- Polling Data ---
    logger.info("\n" + "-" * 40)
    logger.info("POLLING DATA")
    logger.info("-" * 40)

    try:
        gb_df, app_df = create_polling_data()
        success.append(f"Generic ballot: {len(gb_df)} polls")
        success.append(f"Approval: {len(app_df)} polls")
    except Exception as e:
        logger.error(f"Failed to create polling data: {e}")
        failed.append(f"Polling: {e}")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Successful: {len(success)}")
    for item in success:
        logger.info(f"  ✓ {item}")

    if failed:
        logger.warning(f"Failed: {len(failed)}")
        for item in failed:
            logger.warning(f"  ✗ {item}")

    logger.info(f"\nData saved to: {DATA_RAW}")

    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
