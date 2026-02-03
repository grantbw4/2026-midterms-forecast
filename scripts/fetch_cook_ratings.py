#!/usr/bin/env python3
"""
Cook Political Rating Scraper.

Scrapes:
1. House race ratings from Cook Political
2. Redistricting tracker to identify states with new maps

Uses cloudscraper to bypass Cloudflare protection.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_COOK = DATA_DIR / "cook"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Cook Rating to PVI mapping (midpoint values)
# PVI SIGN CONVENTION: positive = D-leaning, negative = R-leaning
# (e.g., D+12 = +12.0, R+12 = -12.0)
COOK_TO_PVI = {
    "Solid D": 12.0,      # D+12 (D+10 or more)
    "Solid Democratic": 12.0,
    "Likely D": 7.0,      # D+7 (D+5 to D+9)
    "Likely Democratic": 7.0,
    "Lean D": 3.0,        # D+3 (D+2 to D+4)
    "Lean Democratic": 3.0,
    "Toss-up": 0.0,       # Even
    "Toss Up": 0.0,
    "Tossup": 0.0,
    "Lean R": -3.0,       # R+3 (R+2 to R+4)
    "Lean Republican": -3.0,
    "Likely R": -7.0,     # R+7 (R+5 to R+9)
    "Likely Republican": -7.0,
    "Solid R": -12.0,     # R+12 (R+10 or more)
    "Solid Republican": -12.0,
}

# State abbreviations
STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC',
}


class CookPoliticalScraper:
    """Scrape Cook Political ratings and redistricting data."""

    HOUSE_RATINGS_URL = "https://www.cookpolitical.com/ratings/house-race-ratings"
    SENATE_RATINGS_URL = "https://www.cookpolitical.com/ratings/senate-race-ratings"
    REDISTRICTING_URL = "https://www.cookpolitical.com/redistricting/2025-26-mid-decade-map"

    # Keep legacy alias for backwards compatibility
    RATINGS_URL = HOUSE_RATINGS_URL

    def __init__(self):
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'darwin',
                'desktop': True,
            }
        )

    def _parse_district_id(self, text: str) -> Optional[str]:
        """
        Parse district identifier from various formats.

        Examples:
        - "AL-01" -> "AL-01"
        - "California 12" -> "CA-12"
        - "NY-10" -> "NY-10"
        - "Alaska At-Large" -> "AK-01"
        - "DE-AL" -> "DE-01" (at-large format)
        """
        text = str(text).strip()

        # Handle ST-AL format (at-large districts like DE-AL, VT-AL)
        match_al = re.match(r'([A-Z]{2})-AL$', text)
        if match_al:
            state = match_al.group(1)
            return f"{state}-01"

        # Already in ST-## format
        match = re.match(r'([A-Z]{2})-?(\d+)', text)
        if match:
            state, dist = match.groups()
            return f"{state}-{int(dist):02d}"

        # "State Name ##" format
        for state_name, abbrev in STATE_ABBREV.items():
            if text.lower().startswith(state_name.lower()):
                rest = text[len(state_name):].strip()
                if rest.lower() in ['at-large', 'al', '']:
                    return f"{abbrev}-01"
                try:
                    dist_num = int(re.search(r'\d+', rest).group())
                    return f"{abbrev}-{dist_num:02d}"
                except (AttributeError, ValueError):
                    continue

        return None

    def fetch_house_ratings(self) -> pd.DataFrame:
        """
        Fetch current House race ratings from Cook Political.

        Returns DataFrame with:
        - district_id: ST-## format
        - cook_rating: e.g., "Solid D", "Toss-up", "Lean R"
        - cook_pvi: Numeric PVI derived from rating
        - incumbent: Incumbent name if listed
        - incumbent_party: D or R
        """
        logger.info("Fetching Cook Political House ratings...")

        try:
            response = self.scraper.get(self.RATINGS_URL, timeout=30)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch Cook ratings: {e}")
            return pd.DataFrame()

        soup = BeautifulSoup(response.text, 'html.parser')
        records = []

        # Cook Political uses race-card divs with data-category attribute
        # Categories: solid-d, solid-r, likely-d, likely-r, lean-d, lean-r, tossup
        category_mapping = {
            'solid-d': 'Solid D',
            'solid-r': 'Solid R',
            'likely-d': 'Likely D',
            'likely-r': 'Likely R',
            'lean-d': 'Lean D',
            'lean-r': 'Lean R',
            'tossup': 'Toss-up',
        }

        race_cards = soup.find_all('div', class_='race-card')
        logger.info(f"Found {len(race_cards)} race card sections")

        for card in race_cards:
            # Get the rating category from data-category attribute
            category_code = card.get('data-category', '')
            rating = category_mapping.get(category_code, 'Unknown')

            if rating == 'Unknown':
                # Try to find category from class names
                card_classes = ' '.join(card.get('class', []))
                for code, name in category_mapping.items():
                    if code in card_classes:
                        rating = name
                        break

            # Find all race items in this card
            race_items = card.find_all('li', class_='race-item')

            for item in race_items:
                # Get party from class
                item_classes = item.get('class', [])
                incumbent_party = 'D' if 'democrat' in item_classes else ('R' if 'republican' in item_classes else None)

                # Find the race link
                link = item.find('a', class_='race-link')
                if not link:
                    continue

                # Extract district ID
                district_span = link.find('span', class_='race-district')
                district_text = district_span.get_text(strip=True) if district_span else ''
                district_id = self._parse_district_id(district_text)

                if not district_id:
                    continue

                # Extract incumbent name
                name_span = link.find('span', class_='race-name')
                incumbent = name_span.get_text(strip=True) if name_span else ''

                # Check if it's an open seat
                is_open = 'OPEN' in incumbent or 'VACANT' in incumbent

                records.append({
                    'district_id': district_id,
                    'cook_rating': rating,
                    'cook_pvi': COOK_TO_PVI.get(rating, 0.0),
                    'incumbent': incumbent,
                    'incumbent_party': incumbent_party,
                    'is_open': is_open,
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.drop_duplicates(subset=['district_id'], keep='first')
            df = df.sort_values('district_id').reset_index(drop=True)

        logger.info(f"Found {len(df)} House race ratings")

        # Log summary by category
        if not df.empty:
            for rating in category_mapping.values():
                count = (df['cook_rating'] == rating).sum()
                if count > 0:
                    logger.info(f"  {rating}: {count}")

        return df

    def fetch_senate_ratings(self) -> pd.DataFrame:
        """
        Fetch current Senate race ratings from Cook Political.

        Returns DataFrame with:
        - state: State abbreviation
        - cook_rating: e.g., "Solid D", "Toss-up", "Lean R"
        - cook_pvi: Numeric PVI derived from rating
        - incumbent: Incumbent senator name
        - incumbent_party: D or R (from text color - red=R, blue=D)
        - is_open: Whether it's an open seat
        - seat_class: Senate class (1, 2, or 3)
        """
        logger.info("Fetching Cook Political Senate ratings...")

        try:
            response = self.scraper.get(self.SENATE_RATINGS_URL, timeout=30)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to fetch Senate ratings: {e}")
            return pd.DataFrame()

        soup = BeautifulSoup(response.text, 'html.parser')
        records = []

        # Same category mapping as House
        category_mapping = {
            'solid-d': 'Solid D',
            'solid-r': 'Solid R',
            'likely-d': 'Likely D',
            'likely-r': 'Likely R',
            'lean-d': 'Lean D',
            'lean-r': 'Lean R',
            'tossup': 'Toss-up',
        }

        race_cards = soup.find_all('div', class_='race-card')
        logger.info(f"Found {len(race_cards)} Senate race card sections")

        for card in race_cards:
            # Get the rating category from data-category attribute
            category_code = card.get('data-category', '')
            rating = category_mapping.get(category_code, 'Unknown')

            if rating == 'Unknown':
                card_classes = ' '.join(card.get('class', []))
                for code, name in category_mapping.items():
                    if code in card_classes:
                        rating = name
                        break

            # Find all race items in this card
            race_items = card.find_all('li', class_='race-item')

            for item in race_items:
                # Get party from class (democrat = blue text, republican = red text)
                item_classes = item.get('class', [])
                incumbent_party = 'D' if 'democrat' in item_classes else ('R' if 'republican' in item_classes else None)

                # Find the race link
                link = item.find('a', class_='race-link')
                if not link:
                    continue

                # Extract state from district span (for Senate it's just state name)
                district_span = link.find('span', class_='race-district')
                state_text = district_span.get_text(strip=True) if district_span else ''

                # Convert state name to abbreviation
                state = STATE_ABBREV.get(state_text.strip(), None)
                if not state:
                    # Try direct abbreviation
                    if len(state_text) == 2 and state_text.upper() in STATE_ABBREV.values():
                        state = state_text.upper()
                    else:
                        continue

                # Extract incumbent name
                name_span = link.find('span', class_='race-name')
                incumbent = name_span.get_text(strip=True) if name_span else ''

                # Check if it's an open seat
                is_open = 'OPEN' in incumbent.upper() or 'VACANT' in incumbent.upper()

                records.append({
                    'state': state,
                    'cook_rating': rating,
                    'cook_pvi': COOK_TO_PVI.get(rating, 0.0),
                    'incumbent': incumbent,
                    'incumbent_party': incumbent_party,
                    'is_open': is_open,
                    'seat_class': 2,  # 2026 = Class 2 seats
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.drop_duplicates(subset=['state'], keep='first')
            df = df.sort_values('state').reset_index(drop=True)

        logger.info(f"Found {len(df)} Senate race ratings")

        # Log summary by category
        if not df.empty:
            for rating in category_mapping.values():
                count = (df['cook_rating'] == rating).sum()
                if count > 0:
                    logger.info(f"  {rating}: {count}")

            # Log party breakdown
            d_count = (df['incumbent_party'] == 'D').sum()
            r_count = (df['incumbent_party'] == 'R').sum()
            logger.info(f"  D defending: {d_count}, R defending: {r_count}")

        return df

    # Known redistricted states for 2026 cycle (updated manually based on news)
    # These states have enacted new maps or have pending court-ordered changes
    REDISTRICTED_STATES_2026 = {
        'TX': {'status': 'New Map', 'notes': 'Republican-drawn map enacted 2023'},
        'FL': {'status': 'New Map', 'notes': 'DeSantis map enacted, court challenges ongoing'},
        'NC': {'status': 'New Map', 'notes': 'GOP-controlled legislature new map'},
        'VA': {'status': 'Pending', 'notes': 'Democrats pushing redistricting'},
        'UT': {'status': 'New Map', 'notes': 'Court-ordered map favoring Democrats'},
        'LA': {'status': 'New Map', 'notes': 'Court-ordered second majority-minority district'},
        'NY': {'status': 'Pending', 'notes': 'Democrats may attempt new map'},
        'AL': {'status': 'New Map', 'notes': 'Court-ordered second majority-minority district'},
        'GA': {'status': 'Pending', 'notes': 'Potential court challenges'},
    }

    def fetch_redistricting_status(self) -> pd.DataFrame:
        """
        Get redistricting status for states.

        Uses a manually maintained list of known redistricted states
        since Cook Political's tracker requires JavaScript rendering.

        Returns DataFrame with:
        - state: State abbreviation
        - redistricting_status: e.g., "New Map", "Pending", "No Change"
        - notes: Additional context
        """
        logger.info("Loading redistricting status (known states)...")

        # Use the known redistricted states list
        records = []
        for state, info in self.REDISTRICTED_STATES_2026.items():
            records.append({
                'state': state,
                'redistricting_status': info['status'],
                'notes': info['notes'],
            })

        df = pd.DataFrame(records)
        df = df.sort_values('state').reset_index(drop=True)

        logger.info(f"Found {len(df)} states with redistricting activity")
        return df

    def get_redistricted_states(self, redistricting_df: pd.DataFrame) -> list:
        """Get list of states that have been redistricted."""
        if redistricting_df.empty:
            return []

        # States with new maps or pending changes
        redistricted = redistricting_df[
            redistricting_df['redistricting_status'].str.lower().str.contains(
                'new|enacted|adopted|pending|litigation', na=False
            )
        ]['state'].tolist()

        return redistricted

    def save_data(
        self,
        ratings_df: pd.DataFrame,
        redistricting_df: pd.DataFrame,
        senate_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Save scraped data to disk."""
        DATA_COOK.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d")

        if not ratings_df.empty:
            # Save current ratings
            ratings_path = DATA_COOK / "house_ratings.csv"
            ratings_df.to_csv(ratings_path, index=False)
            logger.info(f"Saved {len(ratings_df)} ratings to {ratings_path}")

            # Also save timestamped version for history
            history_path = DATA_COOK / f"house_ratings_{timestamp}.csv"
            ratings_df.to_csv(history_path, index=False)

        if senate_df is not None and not senate_df.empty:
            # Save Senate ratings
            senate_path = DATA_COOK / "senate_ratings.csv"
            senate_df.to_csv(senate_path, index=False)
            logger.info(f"Saved {len(senate_df)} Senate ratings to {senate_path}")

            # Also save timestamped version for history
            senate_history_path = DATA_COOK / f"senate_ratings_{timestamp}.csv"
            senate_df.to_csv(senate_history_path, index=False)

        if not redistricting_df.empty:
            redistricting_path = DATA_COOK / "redistricting_status.csv"
            redistricting_df.to_csv(redistricting_path, index=False)
            logger.info(f"Saved redistricting status to {redistricting_path}")

        # Save metadata
        metadata = {
            "last_updated": datetime.now().isoformat(),
            "house_ratings_count": len(ratings_df),
            "senate_ratings_count": len(senate_df) if senate_df is not None else 0,
            "redistricted_states": self.get_redistricted_states(redistricting_df),
        }
        with open(DATA_COOK / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def pvi_aligns_with_rating(pvi: float, rating: str) -> bool:
    """
    Check if a PVI value aligns with a Cook rating category.

    Rating categories and expected PVI ranges:
    - Solid D/R: |PVI| >= 10
    - Likely D/R: 5 <= |PVI| < 10
    - Lean D/R: 2 <= |PVI| < 5
    - Toss-up: |PVI| < 2

    Args:
        pvi: PVI value (positive = D lean, negative = R lean)
        rating: Cook rating string (e.g., "Solid D", "Lean R", "Toss-up")

    Returns:
        True if PVI aligns with the rating category
    """
    rating_lower = rating.lower()
    abs_pvi = abs(pvi)

    # Determine expected party direction from rating
    if 'solid d' in rating_lower or 'likely d' in rating_lower or 'lean d' in rating_lower:
        expected_dem = True
    elif 'solid r' in rating_lower or 'likely r' in rating_lower or 'lean r' in rating_lower:
        expected_dem = False
    else:  # Toss-up
        expected_dem = None

    # Check party direction alignment (skip for toss-ups)
    if expected_dem is not None:
        pvi_is_dem = pvi > 0
        if pvi_is_dem != expected_dem:
            return False  # Wrong party direction

    # Check magnitude alignment
    if 'solid' in rating_lower:
        return abs_pvi >= 10
    elif 'likely' in rating_lower:
        return 5 <= abs_pvi < 10
    elif 'lean' in rating_lower:
        return 2 <= abs_pvi < 5
    else:  # Toss-up
        return abs_pvi < 2


def adjust_pvi_with_cook_ratings(
    districts_df: pd.DataFrame,
    cook_ratings_df: pd.DataFrame,
    redistricted_states: list,
) -> pd.DataFrame:
    """
    Adjust PVI values for redistricted districts using Cook ratings.

    For districts in redistricted states where PVI doesn't align with
    the Cook rating category, use Cook-derived PVI instead.

    Rating alignment rules:
    - Solid: |PVI| >= 10
    - Likely: 5 <= |PVI| < 10
    - Lean: 2 <= |PVI| < 5
    - Toss-up: |PVI| < 2

    Args:
        districts_df: DataFrame with district_id and pvi columns
        cook_ratings_df: DataFrame with cook_rating and cook_pvi
        redistricted_states: List of state abbreviations that were redistricted

    Returns:
        Updated districts DataFrame with adjusted PVI values
    """
    if cook_ratings_df.empty:
        logger.warning("No Cook ratings available for PVI adjustment")
        return districts_df

    df = districts_df.copy()

    # Merge with Cook ratings
    df = df.merge(
        cook_ratings_df[['district_id', 'cook_rating', 'cook_pvi']],
        on='district_id',
        how='left'
    )

    # Determine PVI column name (could be 'pvi' or 'pvi_numeric')
    pvi_col = 'pvi' if 'pvi' in df.columns else 'pvi_numeric'

    # For districts in redistricted states, check if adjustment needed
    adjusted_count = 0
    for idx, row in df.iterrows():
        state = row['district_id'][:2]

        # Only adjust redistricted states
        if state not in redistricted_states:
            continue

        # Skip if no Cook rating available
        if pd.isna(row.get('cook_rating')):
            continue

        wiki_pvi = row.get(pvi_col, 0)
        if pd.isna(wiki_pvi):
            wiki_pvi = 0
        cook_pvi = row['cook_pvi']
        cook_rating = row['cook_rating']

        # Check if PVI aligns with the Cook rating category
        if not pvi_aligns_with_rating(wiki_pvi, cook_rating):
            # Use Cook-derived PVI for redistricted districts
            df.at[idx, pvi_col] = cook_pvi
            df.at[idx, 'pvi_adjusted'] = True
            df.at[idx, 'pvi_source'] = 'cook'
            adjusted_count += 1
            logger.info(
                f"  {row['district_id']}: PVI {wiki_pvi:.1f} → {cook_pvi:.1f} "
                f"(Cook: {cook_rating})"
            )
        else:
            df.at[idx, 'pvi_adjusted'] = False
            df.at[idx, 'pvi_source'] = 'wikipedia'

    logger.info(f"Adjusted PVI for {adjusted_count} redistricted districts")

    # Drop the temporary cook columns before returning
    cols_to_drop = ['cook_rating', 'cook_pvi']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df


def main():
    """Fetch Cook Political data."""
    logger.info("=" * 60)
    logger.info("Cook Political Data Fetch")
    logger.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    scraper = CookPoliticalScraper()

    # Fetch House ratings
    ratings_df = scraper.fetch_house_ratings()

    # Fetch Senate ratings
    senate_df = scraper.fetch_senate_ratings()

    # Fetch redistricting status
    redistricting_df = scraper.fetch_redistricting_status()

    # Save data
    scraper.save_data(ratings_df, redistricting_df, senate_df)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if not ratings_df.empty:
        logger.info(f"\nHouse Ratings ({len(ratings_df)} districts):")
        for rating in COOK_TO_PVI.keys():
            count = (ratings_df['cook_rating'] == rating).sum()
            if count > 0:
                logger.info(f"  {rating}: {count}")

    if not senate_df.empty:
        logger.info(f"\nSenate Ratings ({len(senate_df)} races):")
        for rating in COOK_TO_PVI.keys():
            count = (senate_df['cook_rating'] == rating).sum()
            if count > 0:
                logger.info(f"  {rating}: {count}")
        d_count = (senate_df['incumbent_party'] == 'D').sum()
        r_count = (senate_df['incumbent_party'] == 'R').sum()
        logger.info(f"  D defending: {d_count}, R defending: {r_count}")

    redistricted = scraper.get_redistricted_states(redistricting_df)
    if redistricted:
        logger.info(f"\nRedistricted States: {', '.join(redistricted)}")

    return ratings_df, redistricting_df, senate_df


if __name__ == "__main__":
    main()
