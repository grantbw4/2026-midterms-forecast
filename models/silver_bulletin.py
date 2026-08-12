"""Silver Bulletin maintained-average ingestion.

Production intentionally consumes the published *averages*, not the underlying
poll rows or Silver's proprietary forecast outputs.  Only the latest average
for a race enters its likelihood.  The daily generic-ballot history is retained
for display, but those highly correlated daily estimates are never multiplied
together as independent evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
import re
from typing import Any, Optional

import pandas as pd
import requests

from .race_polling import CandidateRegistry, SnapshotStore


AVERAGES_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vSyuZYuGnnjFdpjryAiGq6SeRe0ZOoGHKYzPzbxF1X_Ee_cE7411tTGdUbpRerX8_"
    "Xe7uRfw_Rkd1Hj/pub?gid=0&single=true&output=csv"
)
FORECAST_PAGE_URL = (
    "https://www.natesilver.net/p/nate-silver-2026-midterm-election-polls-model"
)
SOURCE_NOTICE = "Publicly downloadable; no redistribution license stated"


@dataclass(frozen=True)
class SilverAverageData:
    generic_history: pd.DataFrame
    race_likelihoods: pd.DataFrame
    status: dict[str, Any]


class SilverBulletinClient:
    """Fetch the public CSV used by Silver Bulletin's live polling embed."""

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def fetch(self) -> pd.DataFrame:
        response = self.session.get(AVERAGES_CSV_URL, timeout=45)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
        required = {
            "id", "group", "place", "date", "cand_D", "party_D", "avg_D",
            "cand_R", "party_R", "avg_R",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Silver Bulletin averages missing columns: {sorted(missing)}")
        return frame


def _forecast_race_id(source_id: str, group: str) -> Optional[str]:
    if group == "Senate":
        match = re.fullmatch(r"2026_([A-Z]{2})-S\d+", source_id)
        return match.group(1) if match else None
    if group == "House":
        match = re.fullmatch(r"2026_([A-Z]{2})-(\d{1,2})", source_id)
        return f"{match.group(1)}-{int(match.group(2)):02d}" if match else None
    return None


def prepare_silver_averages(
    raw: pd.DataFrame,
    registry: Optional[CandidateRegistry] = None,
    aggregate_observation_std: float = 1.5,
) -> SilverAverageData:
    """Build one aggregate likelihood per covered D-v-R race.

    ``aggregate_observation_std`` describes uncertainty in current sentiment
    around the maintained average.  Election-day correlated error is added
    separately by the calibrated race update.
    """

    work = raw.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["avg_D"] = pd.to_numeric(work["avg_D"], errors="coerce")
    work["avg_R"] = pd.to_numeric(work["avg_R"], errors="coerce")
    work = work.dropna(subset=["id", "group", "date"])

    generic = work[
        (work["group"] == "Generic ballot")
        & (work["party_D"] == "D")
        & (work["party_R"] == "R")
    ].dropna(subset=["avg_D", "avg_R"]).copy()
    if generic.empty:
        raise ValueError("Silver Bulletin feed contains no valid generic-ballot average")
    generic = generic.sort_values("date").drop_duplicates("date", keep="last")
    generic_history = pd.DataFrame({
        "date": generic["date"],
        "pollster": "Silver Bulletin maintained average",
        # The aggregate has no defensible single sample size; uncertainty is
        # supplied directly instead of inventing a respondent count.
        "sample_size": None,
        "population": "likely-voter-adjusted average",
        "dem_pct": generic["avg_D"],
        "rep_pct": generic["avg_R"],
        "margin": generic["avg_D"] - generic["avg_R"],
        "observation_std": aggregate_observation_std,
        "partisan": None,
        "internal": False,
        "source_url": FORECAST_PAGE_URL,
        "provider": "Silver Bulletin",
        "aggregation_level": "maintained_average",
    }).reset_index(drop=True)

    latest = (
        work[work["group"].isin(["Senate", "House"])]
        .sort_values("date")
        .groupby("id", as_index=False)
        .tail(1)
    )
    rejected: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    statuses: dict[str, Any] = {}
    for _, row in latest.iterrows():
        race_id = _forecast_race_id(str(row["id"]), str(row["group"]))
        if race_id is None:
            rejected["unsupported_race"] = rejected.get("unsupported_race", 0) + 1
            continue
        if any(
            pd.notna(value) and bool(str(value).strip())
            for value in (row.get("party_I"), row.get("cand_I"))
        ):
            rejected["not_two_party"] = rejected.get("not_two_party", 0) + 1
            statuses[race_id] = {"status": "unresolved_matchup", "polls_used": 0}
            continue
        if row.get("party_D") != "D" or row.get("party_R") != "R" or pd.isna(row["avg_D"]) or pd.isna(row["avg_R"]):
            rejected["not_d_vs_r"] = rejected.get("not_d_vs_r", 0) + 1
            statuses[race_id] = {"status": "unresolved_matchup", "polls_used": 0}
            continue
        dem_name, rep_name = str(row["cand_D"]), str(row["cand_R"])
        if registry is not None:
            dem = registry.resolve(race_id, dem_name)
            rep = registry.resolve(race_id, rep_name)
            if dem is None or rep is None or dem.party != "D" or rep.party != "R":
                rejected["candidate_unmapped"] = rejected.get("candidate_unmapped", 0) + 1
                statuses[race_id] = {"status": "unresolved_matchup", "polls_used": 0}
                continue
            dem_name, rep_name = dem.name, rep.name
        date_value = pd.Timestamp(row["date"])
        records.append({
            "race_id": race_id,
            "date": date_value,
            "pollster": "Silver Bulletin maintained average",
            "grade": "external_aggregate",
            "sample_size": None,
            "population": "likely-voter-adjusted average",
            "margin_of_error": None,
            "observation_std": aggregate_observation_std,
            "partisan": None,
            "dem_candidate": dem_name,
            "rep_candidate": rep_name,
            "dem_pct": float(row["avg_D"]),
            "rep_pct": float(row["avg_R"]),
            "margin": float(row["avg_D"] - row["avg_R"]),
            "matchup": f"{dem_name}|{rep_name}",
            "source_url": FORECAST_PAGE_URL,
            "provider": "Silver Bulletin",
            "aggregation_level": "maintained_average",
        })
        statuses[race_id] = {
            "status": "silver_average",
            "matchup": f"{dem_name}|{rep_name}",
            "polls_used": 1,
            "latest_poll_date": date_value.strftime("%Y-%m-%d"),
        }

    races = pd.DataFrame(records)
    if not races.empty:
        races = races.sort_values(["race_id", "date"]).reset_index(drop=True)
    return SilverAverageData(
        generic_history=generic_history,
        race_likelihoods=races,
        status={
            "summary": {
                "accepted_aggregate_likelihoods": int(len(races)),
                "rejected": rejected,
                "provider": "Silver Bulletin",
                "likelihood_rule": "latest maintained average only",
            },
            "races": statuses,
        },
    )


def save_silver_cache(data_dir: Path, data: SilverAverageData) -> None:
    processed = Path(data_dir) / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    data.generic_history.to_csv(processed / "silver_generic_history.csv", index=False)
    data.race_likelihoods.to_csv(processed / "silver_race_averages.csv", index=False)
    (processed / "silver_average_status.json").write_text(json.dumps(data.status, indent=2))
    generic_payload = data.generic_history.assign(date=data.generic_history["date"].astype(str))
    race_payload = data.race_likelihoods.assign(date=data.race_likelihoods["date"].astype(str))
    payload = {
        "generic_history": generic_payload.astype(object).where(pd.notna(generic_payload), None).to_dict("records"),
        "latest_race_averages": race_payload.astype(object).where(pd.notna(race_payload), None).to_dict("records"),
    }
    SnapshotStore(Path(data_dir) / "raw" / "snapshots").save(
        "silver_bulletin_averages", payload, AVERAGES_CSV_URL, SOURCE_NOTICE
    )


def load_silver_cache(data_dir: Path) -> SilverAverageData:
    processed = Path(data_dir) / "processed"
    generic_path = processed / "silver_generic_history.csv"
    races_path = processed / "silver_race_averages.csv"
    status_path = processed / "silver_average_status.json"
    if not generic_path.exists() or not races_path.exists() or not status_path.exists():
        raise FileNotFoundError("No valid cached Silver Bulletin averages")
    generic = pd.read_csv(generic_path, parse_dates=["date"])
    races = pd.read_csv(races_path, parse_dates=["date"])
    status = json.loads(status_path.read_text())
    return SilverAverageData(generic, races, status)
