#!/usr/bin/env python3
"""
Process historical election data for model validation.

Processes:
1. MIT Election Lab House results (1976-2024) → 2018, 2022 results
2. FiveThirtyEight partisan lean → standardized PVI
3. Wikipedia PVI 2026 → current district fundamentals
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HISTORICAL_DIR = DATA_DIR / "historical"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# State abbreviations mapping
STATE_ABBREV = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
    'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
    'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
    'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
    'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
    'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
    'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
    'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
    'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
    'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
    'WISCONSIN': 'WI', 'WYOMING': 'WY', 'DISTRICT OF COLUMBIA': 'DC',
}

# At-large states (1 district)
AT_LARGE_STATES = ['AK', 'DE', 'MT', 'ND', 'SD', 'VT', 'WY']


def process_mit_house_results(year: int) -> pd.DataFrame:
    """
    Process MIT Election Lab House results for a given year.

    Returns DataFrame with:
    - district_id: ST-## format
    - dem_votes, rep_votes, other_votes
    - total_votes
    - dem_pct, rep_pct (two-party vote share)
    - winner_party
    """
    logger.info(f"Processing MIT House results for {year}...")

    mit_file = DATA_DIR / "MIT_Election_files" / "1976-2024-house.tab"

    # Read tab-separated file
    df = pd.read_csv(mit_file, sep=',', low_memory=False)

    # Filter to year and general election
    df = df[(df['year'] == year) & (df['stage'] == 'GEN')]

    # Use state_po column which already has state abbreviations
    df['state_abbrev'] = df['state_po']

    # Create district_id
    def make_district_id(row):
        state = row['state_abbrev']
        dist = row['district']
        # At-large districts have district=0
        if dist == 0:
            return f"{state}-01"
        return f"{state}-{int(dist):02d}"

    df['district_id'] = df.apply(make_district_id, axis=1)

    # Normalize party names
    def normalize_party(party):
        if pd.isna(party):
            return 'OTHER'
        party = str(party).upper()
        if 'DEMOCRAT' in party:
            return 'DEM'
        elif 'REPUBLICAN' in party:
            return 'REP'
        else:
            return 'OTHER'

    df['party_normalized'] = df['party'].apply(normalize_party)

    # Aggregate votes by district and party
    votes = df.groupby(['district_id', 'party_normalized'])['candidatevotes'].sum().unstack(fill_value=0)
    votes = votes.reset_index()

    # Ensure all columns exist
    for col in ['DEM', 'REP', 'OTHER']:
        if col not in votes.columns:
            votes[col] = 0

    votes = votes.rename(columns={'DEM': 'dem_votes', 'REP': 'rep_votes', 'OTHER': 'other_votes'})

    # Calculate totals and percentages
    votes['total_votes'] = votes['dem_votes'] + votes['rep_votes'] + votes['other_votes']
    votes['two_party_total'] = votes['dem_votes'] + votes['rep_votes']

    # Two-party vote share (standard for PVI calculations)
    votes['dem_pct'] = (votes['dem_votes'] / votes['two_party_total'] * 100).round(2)
    votes['rep_pct'] = (votes['rep_votes'] / votes['two_party_total'] * 100).round(2)

    # Handle edge cases (uncontested races)
    votes.loc[votes['two_party_total'] == 0, 'dem_pct'] = 50.0
    votes.loc[votes['two_party_total'] == 0, 'rep_pct'] = 50.0

    # Determine winner
    votes['winner_party'] = np.where(votes['dem_votes'] > votes['rep_votes'], 'D', 'R')
    votes.loc[votes['dem_votes'] == votes['rep_votes'], 'winner_party'] = 'TIE'

    # Calculate margin (positive = D win)
    votes['margin'] = votes['dem_pct'] - votes['rep_pct']

    # Select final columns
    result = votes[['district_id', 'dem_votes', 'rep_votes', 'other_votes',
                    'total_votes', 'dem_pct', 'rep_pct', 'margin', 'winner_party']]

    result = result.sort_values('district_id').reset_index(drop=True)

    logger.info(f"  Processed {len(result)} districts")
    logger.info(f"  D wins: {(result['winner_party'] == 'D').sum()}, R wins: {(result['winner_party'] == 'R').sum()}")

    return result


def standardize_district_id(district_id: str) -> str:
    """Standardize district ID to ST-## format (e.g., AL-01, CA-12)."""
    if pd.isna(district_id):
        return None
    # Handle formats like "AL-1" -> "AL-01" or "CA-12" -> "CA-12"
    match = re.match(r'([A-Z]{2})-(\d+)', str(district_id))
    if match:
        state, dist = match.groups()
        return f"{state}-{int(dist):02d}"
    return district_id


def process_538_partisan_lean_2022() -> pd.DataFrame:
    """
    Process FiveThirtyEight partisan lean data for 2022.

    538 format: positive = D lean, negative = R lean
    Output: Standardized to Cook PVI format (D+X or R+X)
    """
    logger.info("Processing 538 partisan lean 2022...")

    df = pd.read_csv(DATA_DIR / "538" / "partisan_lean_districts_2022.csv")

    # Rename and standardize district ID
    df = df.rename(columns={'district': 'district_id_raw', '2022': 'partisan_lean_538'})
    df['district_id'] = df['district_id_raw'].apply(standardize_district_id)

    # Convert to Cook PVI format string
    def to_pvi_string(lean):
        if lean > 0:
            return f"D+{abs(lean):.1f}"
        elif lean < 0:
            return f"R+{abs(lean):.1f}"
        else:
            return "EVEN"

    df['pvi_string'] = df['partisan_lean_538'].apply(to_pvi_string)

    # Numeric PVI (positive = D, negative = R) - same as 538 convention
    df['pvi_numeric'] = df['partisan_lean_538'].round(2)

    logger.info(f"  Processed {len(df)} districts")

    return df


def process_538_partisan_lean_2018() -> pd.DataFrame:
    """
    Process FiveThirtyEight partisan lean data for 2018.

    538 2018 format: "R+15.21" or "D+5.5"
    """
    logger.info("Processing 538 partisan lean 2018...")

    df = pd.read_csv(DATA_DIR / "538" / "partisan_lean_districts_2018.csv", encoding='utf-8-sig')

    # Rename and standardize district ID
    df = df.rename(columns={'district': 'district_id_raw', '2018': 'pvi_string'})
    df['district_id'] = df['district_id_raw'].apply(standardize_district_id)

    # Parse PVI string to numeric
    def parse_pvi(pvi_str):
        if pd.isna(pvi_str) or pvi_str == 'EVEN':
            return 0.0
        match = re.match(r'([DR])\+?([\d.]+)', str(pvi_str))
        if match:
            party, value = match.groups()
            value = float(value)
            return value if party == 'D' else -value
        return 0.0

    df['pvi_numeric'] = df['pvi_string'].apply(parse_pvi)

    logger.info(f"  Processed {len(df)} districts")

    return df


def process_wikipedia_pvi_2026() -> pd.DataFrame:
    """
    Process Wikipedia PVI 2026 data.

    Format: "Alabama 1", "R+27", "Republican"
    """
    logger.info("Processing Wikipedia PVI 2026...")

    df = pd.read_csv(DATA_DIR / "wikipedia" / "PVI_2026_Districts.csv")

    # Parse district name to district_id
    def parse_district(name):
        # Handle "Alabama 1" -> "AL-01"
        parts = str(name).rsplit(' ', 1)
        if len(parts) == 2:
            state_name, dist_num = parts
            # Look up state abbreviation
            state_abbrev = None
            for full_name, abbrev in STATE_ABBREV.items():
                if full_name.upper() == state_name.upper():
                    state_abbrev = abbrev
                    break
            if state_abbrev:
                if dist_num.upper() == 'AT-LARGE' or dist_num == '0':
                    return f"{state_abbrev}-01"
                return f"{state_abbrev}-{int(dist_num):02d}"
        return None

    df['district_id'] = df['District'].apply(parse_district)

    # Parse PVI
    def parse_pvi(pvi_str):
        if pd.isna(pvi_str) or pvi_str == 'EVEN':
            return 0.0
        match = re.match(r'([DR])\+?([\d.]+)', str(pvi_str))
        if match:
            party, value = match.groups()
            value = float(value)
            return value if party == 'D' else -value
        return 0.0

    df['pvi_numeric'] = df['PVI'].apply(parse_pvi)
    df['pvi_string'] = df['PVI']

    # Get incumbent party
    df['incumbent_party'] = df['Party of\nrepresentative'].apply(
        lambda x: 'D' if 'Dem' in str(x) else ('R' if 'Rep' in str(x) else None)
    )

    # Clean up
    result = df[['district_id', 'pvi_string', 'pvi_numeric', 'incumbent_party']].dropna(subset=['district_id'])

    logger.info(f"  Processed {len(result)} districts")

    return result


def create_historical_national_environment():
    """
    Create national environment files for 2018 and 2022.

    Sources: Historical records from RCP/538 archives
    """
    logger.info("Creating historical national environment files...")

    # 2022 National Environment
    # Final generic ballot was roughly R+3 (actual result was R+2.8 in popular vote)
    # Biden approval was around 41%
    env_2022 = {
        "year": 2022,
        "election_date": "2022-11-08",
        "generic_ballot_final": -2.8,  # Negative = R advantage
        "dem_pct": 47.8,
        "rep_pct": 50.6,
        "presidential_approval": 41.0,
        "presidential_disapproval": 54.0,
        "net_approval": -13.0,
        "president_party": "D",
        "is_midterm": True,
        "actual_house_result": {
            "dem_seats": 213,
            "rep_seats": 222,
            "dem_popular_vote_pct": 47.8
        },
        "notes": "Biden midterm. Republicans won House by narrow margin."
    }

    # 2018 National Environment
    # Final generic ballot was D+8.6 (actual result was D+8.4 in popular vote)
    # Trump approval was around 43%
    env_2018 = {
        "year": 2018,
        "election_date": "2018-11-06",
        "generic_ballot_final": 8.4,  # Positive = D advantage
        "dem_pct": 53.4,
        "rep_pct": 45.0,
        "presidential_approval": 43.0,
        "presidential_disapproval": 52.0,
        "net_approval": -9.0,
        "president_party": "R",
        "is_midterm": True,
        "actual_house_result": {
            "dem_seats": 235,
            "rep_seats": 200,
            "dem_popular_vote_pct": 53.4
        },
        "notes": "Trump midterm. Blue wave - Democrats gained 40 seats."
    }

    # Save files
    with open(HISTORICAL_DIR / "national_environment_2022.json", 'w') as f:
        json.dump(env_2022, f, indent=2)

    with open(HISTORICAL_DIR / "national_environment_2018.json", 'w') as f:
        json.dump(env_2018, f, indent=2)

    logger.info("  Created national_environment_2022.json")
    logger.info("  Created national_environment_2018.json")


def main():
    """Process all historical data."""
    logger.info("=" * 60)
    logger.info("Processing Historical Election Data")
    logger.info("=" * 60)

    # Create historical directory
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Process MIT House results
    logger.info("\n--- Processing House Results ---")

    results_2022 = process_mit_house_results(2022)
    results_2022.to_csv(HISTORICAL_DIR / "house_results_2022.csv", index=False)
    logger.info(f"Saved house_results_2022.csv ({len(results_2022)} districts)")

    results_2018 = process_mit_house_results(2018)
    results_2018.to_csv(HISTORICAL_DIR / "house_results_2018.csv", index=False)
    logger.info(f"Saved house_results_2018.csv ({len(results_2018)} districts)")

    # 2. Process partisan lean data
    logger.info("\n--- Processing Partisan Lean ---")

    pvi_2022 = process_538_partisan_lean_2022()
    pvi_2022.to_csv(HISTORICAL_DIR / "partisan_lean_2022.csv", index=False)
    logger.info(f"Saved partisan_lean_2022.csv ({len(pvi_2022)} districts)")

    pvi_2018 = process_538_partisan_lean_2018()
    pvi_2018.to_csv(HISTORICAL_DIR / "partisan_lean_2018.csv", index=False)
    logger.info(f"Saved partisan_lean_2018.csv ({len(pvi_2018)} districts)")

    # 3. Process 2026 PVI
    logger.info("\n--- Processing 2026 PVI ---")

    pvi_2026 = process_wikipedia_pvi_2026()
    pvi_2026.to_csv(HISTORICAL_DIR / "pvi_2026.csv", index=False)
    logger.info(f"Saved pvi_2026.csv ({len(pvi_2026)} districts)")

    # 4. Create national environment files
    logger.info("\n--- Creating National Environment Files ---")
    create_historical_national_environment()

    # 5. Create merged validation datasets
    logger.info("\n--- Creating Merged Validation Datasets ---")

    # 2022 validation dataset
    val_2022 = results_2022.merge(pvi_2022[['district_id', 'pvi_numeric']], on='district_id', how='left')
    val_2022.to_csv(HISTORICAL_DIR / "validation_2022.csv", index=False)
    logger.info(f"Saved validation_2022.csv ({len(val_2022)} districts)")

    # 2018 validation dataset
    val_2018 = results_2018.merge(pvi_2018[['district_id', 'pvi_numeric']], on='district_id', how='left')
    val_2018.to_csv(HISTORICAL_DIR / "validation_2018.csv", index=False)
    logger.info(f"Saved validation_2018.csv ({len(val_2018)} districts)")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Output directory: {HISTORICAL_DIR}")
    logger.info(f"Files created:")
    for f in sorted(HISTORICAL_DIR.glob("*")):
        logger.info(f"  - {f.name}")


if __name__ == "__main__":
    main()
