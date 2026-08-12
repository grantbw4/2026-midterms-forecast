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

import numpy as np
import pandas as pd
import requests

from .race_polling import CandidateRegistry, SnapshotStore


AVERAGES_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vSyuZYuGnnjFdpjryAiGq6SeRe0ZOoGHKYzPzbxF1X_Ee_cE7411tTGdUbpRerX8_"
    "Xe7uRfw_Rkd1Hj/pub?gid=0&single=true&output=csv"
)
GENERIC_POLLS_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vRsvXNCZ0ubJr8D_yNcU5q6C0_HBa35K7oDK03KpO7Ca43UwdXaIdvVLWoXEmHHph0EREz5430Hm5yZ/"
    "pub?output=csv"
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

    def fetch_generic_polls(self) -> pd.DataFrame:
        """Fetch Silver Bulletin's public, adjusted generic-ballot poll file."""

        response = self.session.get(GENERIC_POLLS_CSV_URL, timeout=45)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
        required = {
            "subgroup", "pollster", "enddate", "samplesize", "population",
            "modeldate", "influence", "adjusted_net", "partisan", "poll_id", "question_id",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Silver Bulletin generic polls missing columns: {sorted(missing)}")
        return frame


def calibrate_generic_average_uncertainty(
    generic_history: pd.DataFrame,
    generic_polls: pd.DataFrame,
    correlated_design_floor: float = 1.25,
) -> dict[str, Any]:
    """Estimate maintained-average observation error from Bulletin's own polls.

    Bulletin's published shaded band predicts a *new poll*, not uncertainty in
    the latent electorate.  We therefore estimate the average's measurement
    error from adjusted-poll dispersion, effective influence, and an explicit
    correlated-design floor.  This uses Bulletin's universe and adjustments
    without re-estimating its pollster house effects.
    """

    polls = generic_polls.copy()
    if "subgroup" in polls:
        polls = polls[polls["subgroup"].astype(str).str.lower().eq("all polls")]
    polls["date"] = pd.to_datetime(polls["enddate"], errors="coerce").dt.normalize()
    polls["adjusted_net"] = pd.to_numeric(polls["adjusted_net"], errors="coerce")
    polls["influence"] = pd.to_numeric(polls["influence"], errors="coerce").fillna(0.0)
    polls = polls.dropna(subset=["date", "adjusted_net"])
    polls = polls.sort_values("influence", ascending=False).drop_duplicates(
        ["poll_id", "question_id"], keep="first"
    )
    polls = polls[polls["influence"] > 0].copy()
    if len(polls) < 5:
        return {
            "observation_std": correlated_design_floor,
            "method": "regularized_prior_insufficient_matched_bulletin_polls",
            "poll_rows": int(len(polls)),
            "effective_polls": 0.0,
        }
    influence = np.maximum(polls["influence"].to_numpy(float), 0.0)
    adjusted_margin = polls["adjusted_net"].to_numpy(float)
    center = float(np.average(adjusted_margin, weights=influence))
    poll_variance = float(np.average((adjusted_margin - center) ** 2, weights=influence))
    effective_polls = float((influence.sum() ** 2) / np.sum(influence**2))
    estimated_variance = poll_variance / max(effective_polls, 1.0)
    # Poll sampling error averages down; correlated design/turnout error does
    # not.  The latter is an explicit Bayesian prior floor, not an invented
    # aggregate respondent count.
    observation_variance = correlated_design_floor**2 + estimated_variance
    return {
        "observation_std": round(float(np.sqrt(observation_variance)), 4),
        "method": "bulletin_adjusted_poll_dispersion_plus_correlated_design_floor",
        "poll_rows": int(len(polls)),
        "effective_polls": round(effective_polls, 3),
        "weighted_average_margin": round(center, 4),
        "weighted_poll_dispersion": round(float(np.sqrt(poll_variance)), 4),
        "correlated_design_floor": correlated_design_floor,
        "provider": "Silver Bulletin",
        "house_effects": "provider_adjusted_not_reestimated",
    }


def prepare_generic_poll_likelihood(
    generic_polls: pd.DataFrame,
    observation_std: float,
) -> pd.DataFrame:
    """Collapse Bulletin-adjusted polls into one current national likelihood.

    ``influence`` is Bulletin's own current model weight, so this neither
    rebuilds nor stacks another pollster model on top of its adjustments.
    Duplicate question rows are removed before the weighted mean is computed.
    """

    polls = generic_polls.copy()
    polls = polls[polls["subgroup"].astype(str).str.lower().eq("all polls")]
    polls["date"] = pd.to_datetime(polls["enddate"], errors="coerce").dt.normalize()
    polls["adjusted_net"] = pd.to_numeric(polls["adjusted_net"], errors="coerce")
    polls["influence"] = pd.to_numeric(polls["influence"], errors="coerce")
    polls = polls.dropna(subset=["date", "adjusted_net", "influence"])
    polls = polls[polls["influence"] > 0].sort_values("influence", ascending=False)
    polls = polls.drop_duplicates(["poll_id", "question_id"], keep="first")
    if polls.empty:
        raise ValueError("No valid Silver Bulletin generic polls remain")
    margin = float(np.average(polls["adjusted_net"], weights=polls["influence"]))
    latest = polls["date"].max()
    return pd.DataFrame([{
        "date": latest,
        "pollster": "Silver Bulletin adjusted poll universe",
        "sample_size": None,
        "population": "Bulletin all-polls model",
        "dem_pct": np.nan,
        "rep_pct": np.nan,
        "margin": margin,
        "observation_std": float(observation_std),
        "partisan": None,
        "internal": False,
        "source_url": "https://www.natesilver.net/p/generic-ballot-average-2026-nate-silver-bulletin-congress-polls",
        "provider": "Silver Bulletin",
        "aggregation_level": "influence_weighted_adjusted_poll_universe",
        "poll_rows": int(len(polls)),
        "total_influence": float(polls["influence"].sum()),
    }])


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
    parsed_dates = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    if parsed_dates.isna().any():
        raise ValueError("Silver Bulletin averages contain malformed dates")
    work["date"] = parsed_dates
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
