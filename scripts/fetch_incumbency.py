#!/usr/bin/env python3
"""
Process historical incumbency data for 2018 and 2022 House elections.

Data sources:
- 2018: data/candidates_2006-2020.csv (from academic dataset)
- 2022: data/2022_House_Inc - Sheet1.csv (from Wikipedia)

This produces clean incumbency files for Bayesian model parameter fitting.
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_HISTORICAL = DATA_DIR / "historical"

# State name to abbreviation mapping (including variants)
STATE_ABBREVS = {
    "Ala.": "AL", "Alaska": "AK", "Ariz.": "AZ", "Ark.": "AR", "Calif.": "CA",
    "Colo.": "CO", "Conn.": "CT", "Del.": "DE", "Fla.": "FL", "Ga.": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Ill.": "IL", "Ind.": "IN", "Iowa": "IA",
    "Kan.": "KS", "Ky.": "KY", "La.": "LA", "Maine": "ME", "Me.": "ME",
    "Md.": "MD", "Mass.": "MA", "Mich.": "MI", "Minn.": "MN", "Miss.": "MS",
    "Mo.": "MO", "Mont.": "MT", "Neb.": "NE", "Nev.": "NV", "N.H.": "NH",
    "N.J.": "NJ", "N.M.": "NM", "N.Y.": "NY", "N.C.": "NC", "N.D.": "ND",
    "Ohio": "OH", "Okla.": "OK", "Ore.": "OR", "Pa.": "PA", "R.I.": "RI",
    "S.C.": "SC", "S.D.": "SD", "Tenn.": "TN", "Texas": "TX", "Tex.": "TX",
    "Utah": "UT", "Vt.": "VT", "Va.": "VA", "Wash.": "WA", "W.Va.": "WV",
    "Wis.": "WI", "Wyo.": "WY",
}

# At-large states (single district)
AT_LARGE_STATES = ["AK", "DE", "MT", "ND", "SD", "VT", "WY"]


def process_2018_incumbency() -> pd.DataFrame:
    """
    Process 2018 incumbency from candidates_2006-2020.csv.

    The 'inc' column indicates if candidate was incumbent (1) or not (0).
    """
    candidates_path = DATA_DIR / "candidates_2006-2020.csv"

    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing {candidates_path}")

    df = pd.read_csv(candidates_path)

    # Filter to 2018 House general elections
    house_2018 = df[
        (df["office"] == "H") &
        (df["type"] == "G") &
        (df["year"] == 2018)
    ].copy()

    # Create district_id (e.g., "AL-01")
    house_2018["district_id"] = (
        house_2018["state"] + "-" +
        house_2018["dist"].astype(int).astype(str).str.zfill(2)
    )

    # For each district, determine incumbent status
    def get_incumbent_party(group):
        incumbents = group[group["inc"] == 1]
        if len(incumbents) == 0:
            return "Open"
        # Map party codes
        party = incumbents.iloc[0]["party"]
        if party == "D":
            return "D"
        elif party == "R":
            return "R"
        else:
            return "Open"  # Third-party incumbents treated as open

    incumbency = house_2018.groupby("district_id").apply(
        get_incumbent_party, include_groups=False
    ).reset_index()
    incumbency.columns = ["district_id", "incumbent_party"]

    # Add numeric encoding: D=+1, R=-1, Open=0
    incumbency["incumbency_code"] = incumbency["incumbent_party"].map({
        "D": 1, "R": -1, "Open": 0
    })

    incumbency["year"] = 2018

    return incumbency


def process_2022_incumbency() -> pd.DataFrame:
    """
    Process 2022 incumbency from Wikipedia data.

    District format: "N.H. 1" or "Alaska" (at-large)
    Incumbency: "Dem.", "Rep.", "Open"
    """
    wiki_path = DATA_DIR / "2022_House_Inc - Sheet1.csv"

    if not wiki_path.exists():
        raise FileNotFoundError(f"Missing {wiki_path}")

    df = pd.read_csv(wiki_path)

    results = []

    for _, row in df.iterrows():
        district_raw = row["District"]
        inc_raw = row["Inc."]

        # Parse district
        district_id = parse_district(district_raw)

        if district_id is None:
            print(f"Warning: Could not parse district '{district_raw}'")
            continue

        # Parse incumbency
        if inc_raw == "Dem.":
            incumbent_party = "D"
            incumbency_code = 1
        elif inc_raw == "Rep.":
            incumbent_party = "R"
            incumbency_code = -1
        else:  # "Open"
            incumbent_party = "Open"
            incumbency_code = 0

        results.append({
            "district_id": district_id,
            "incumbent_party": incumbent_party,
            "incumbency_code": incumbency_code,
            "year": 2022,
        })

    return pd.DataFrame(results)


def parse_district(district_str: str) -> str | None:
    """
    Parse district string to standard format (e.g., "N.H. 1" -> "NH-01").
    """
    district_str = district_str.strip()

    # Check for at-large states (just state name, no number)
    for state_name, abbrev in STATE_ABBREVS.items():
        if district_str == state_name or district_str == abbrev:
            return f"{abbrev}-01"

    # Parse "State N" format
    match = re.match(r"^(.+?)\s+(\d+)$", district_str)
    if match:
        state_part = match.group(1).strip()
        district_num = int(match.group(2))

        # Look up state abbreviation
        abbrev = STATE_ABBREVS.get(state_part)
        if abbrev:
            return f"{abbrev}-{district_num:02d}"

    return None


def main():
    """Process and save incumbency data for both years."""
    print("=" * 60)
    print("Processing Historical Incumbency Data")
    print("=" * 60)

    # Process 2018
    print("\nProcessing 2018 from candidates_2006-2020.csv...")
    try:
        df_2018 = process_2018_incumbency()
        print(f"  Found {len(df_2018)} districts")
        print(f"  D incumbents: {(df_2018['incumbent_party'] == 'D').sum()}")
        print(f"  R incumbents: {(df_2018['incumbent_party'] == 'R').sum()}")
        print(f"  Open seats: {(df_2018['incumbent_party'] == 'Open').sum()}")

        output_path = DATA_HISTORICAL / "incumbency_2018.csv"
        df_2018.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")
    except Exception as e:
        print(f"  Error: {e}")
        df_2018 = None

    # Process 2022
    print("\nProcessing 2022 from Wikipedia data...")
    try:
        df_2022 = process_2022_incumbency()
        print(f"  Found {len(df_2022)} districts")
        print(f"  D incumbents: {(df_2022['incumbent_party'] == 'D').sum()}")
        print(f"  R incumbents: {(df_2022['incumbent_party'] == 'R').sum()}")
        print(f"  Open seats: {(df_2022['incumbent_party'] == 'Open').sum()}")

        output_path = DATA_HISTORICAL / "incumbency_2022.csv"
        df_2022.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")
    except Exception as e:
        print(f"  Error: {e}")
        df_2022 = None

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    # Load validation files and check alignment
    for year, df_inc in [(2018, df_2018), (2022, df_2022)]:
        if df_inc is None:
            continue

        val_path = DATA_HISTORICAL / f"validation_{year}.csv"
        if val_path.exists():
            df_val = pd.read_csv(val_path)

            # Check district overlap
            inc_districts = set(df_inc["district_id"])
            val_districts = set(df_val["district_id"])

            overlap = inc_districts & val_districts
            missing_in_inc = val_districts - inc_districts
            extra_in_inc = inc_districts - val_districts

            print(f"\n{year}:")
            print(f"  Incumbency districts: {len(inc_districts)}")
            print(f"  Validation districts: {len(val_districts)}")
            print(f"  Overlap: {len(overlap)}")

            if missing_in_inc:
                print(f"  Missing in incumbency: {sorted(missing_in_inc)[:5]}...")
            if extra_in_inc:
                print(f"  Extra in incumbency: {sorted(extra_in_inc)[:5]}...")


if __name__ == "__main__":
    main()
