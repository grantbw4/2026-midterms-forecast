"""Candidate identity and robust Bayesian race updates.

Production polling averages are prepared in :mod:`models.silver_bulletin`.
This module owns official FEC/Clerk identity, immutable snapshots, and the
historically calibrated update used to combine an external likelihood with a
race fundamentals posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
import gzip
import hashlib
from io import BytesIO, StringIO
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional
from zipfile import ZipFile
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests


FEC_CANDIDATES_URL = "https://api.open.fec.gov/v1/candidates/"
FEC_CANDIDATE_MASTER_URL = "https://www.fec.gov/files/bulk-downloads/2026/cn26.zip"
HOUSE_CLERK_MEMBERS_URL = "https://clerk.house.gov/xml/lists/MemberData.xml"
PARTY_MAP = {"D": "D", "DEM": "D", "DFL": "D", "R": "R", "REP": "R", "GOP": "R"}


def normalize_name(value: str) -> str:
    value = str(value).lower().replace("&amp;", "and")
    if "," in value:
        family, given = value.split(",", 1)
        value = f"{given} {family}"
    value = re.sub(r"\b(jr|sr|ii|iii|iv|dr|hon)\b", " ", value)
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


class SnapshotStore:
    """Write content-addressed immutable upstream snapshots."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def save(self, provider: str, payload: Any, source_url: str, license_name: str) -> Path:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(body).hexdigest()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder = self.root / provider
        folder.mkdir(parents=True, exist_ok=True)
        existing = sorted(folder.glob(f"*-{digest[:12]}.json*"))
        if existing:
            return existing[-1]
        target = folder / f"{timestamp}-{digest[:12]}.json.gz"
        if not target.exists():
            envelope = {
                "metadata": {
                    "provider": provider,
                    "source_url": source_url,
                    "license": license_name,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "sha256": digest,
                },
                "payload": payload,
            }
            with gzip.open(target, "wt", encoding="utf-8") as handle:
                json.dump(envelope, handle, separators=(",", ":"))
        return target


class FECClient:
    def __init__(self, api_key: str = "DEMO_KEY", session: Optional[requests.Session] = None) -> None:
        self.api_key = api_key
        self.session = session or requests.Session()

    def fetch_candidates(self, election_year: int = 2026) -> list[dict[str, Any]]:
        """Fetch the official candidate master in one request.

        The bulk file avoids the very low anonymous pagination limit on the
        public API.  The API key remains supported by callers for future
        endpoint extensions, but candidate identity does not require one.
        """
        bulk_url = FEC_CANDIDATE_MASTER_URL.replace("2026", str(election_year)).replace(
            "cn26.zip", f"cn{str(election_year)[-2:]}.zip"
        )
        response = self.session.get(bulk_url, timeout=60)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as archive:
            filename = archive.namelist()[0]
            text = archive.read(filename).decode("latin-1")
        columns = [
            "candidate_id", "name", "party", "election_year", "state", "office",
            "district", "incumbent_challenge", "status", "committee_id", "street_1",
            "street_2", "city", "mailing_state", "zip_code",
        ]
        frame = pd.read_csv(StringIO(text), sep="|", names=columns, dtype=str)
        frame = frame[frame["office"].isin(["H", "S"])]
        frame["incumbent_challenge_full"] = frame["incumbent_challenge"].map(
            {"I": "Incumbent", "C": "Challenger", "O": "Open seat"}
        ).fillna("")
        return frame.to_dict("records")

    def fetch_candidates_api(self, election_year: int = 2026) -> list[dict[str, Any]]:
        """Paginated API fallback for environments that block bulk files."""
        records: list[dict[str, Any]] = []
        for office in ("H", "S"):
            page = 1
            while True:
                response = self.session.get(
                    FEC_CANDIDATES_URL,
                    params={
                        "api_key": self.api_key,
                        "election_year": election_year,
                        "office": office,
                        "per_page": 100,
                        "page": page,
                    },
                    timeout=45,
                )
                response.raise_for_status()
                payload = response.json()
                page_records = payload.get("results", [])
                if not isinstance(page_records, list):
                    raise ValueError("FEC returned an unexpected candidate schema")
                records.extend(page_records)
                pagination = payload.get("pagination", {})
                pages = int(pagination.get("pages", page))
                if page >= pages:
                    break
                page += 1
        return records


class HouseClerkClient:
    """Current House roster from the official Clerk of the House."""

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()

    def fetch_members(self) -> tuple[list[dict[str, Any]], str]:
        response = self.session.get(HOUSE_CLERK_MEMBERS_URL, timeout=45)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        records: list[dict[str, Any]] = []
        for member in root.findall("./members/member"):
            statedistrict = (member.findtext("statedistrict") or "").strip()
            info = member.find("member-info")
            if info is None or len(statedistrict) < 4:
                continue
            state, district = statedistrict[:2], statedistrict[2:].zfill(2)
            records.append({
                "name": (info.findtext("official-name") or info.findtext("namelist") or "").strip(),
                "party": (info.findtext("party") or "").strip(),
                "office": "H",
                "state": state,
                "district": district,
                "incumbent_challenge": "I",
                "incumbent_challenge_full": "Incumbent",
            })
        if len(records) < 420:
            raise ValueError(f"House Clerk roster unexpectedly contains only {len(records)} members")
        return records, str(root.attrib.get("publish-date", "unknown"))


@dataclass(frozen=True)
class Candidate:
    name: str
    normalized_name: str
    party: str
    race_id: str
    incumbent: bool


class CandidateRegistry:
    def __init__(self, candidates: Iterable[dict[str, Any]]) -> None:
        self.by_race: dict[str, list[Candidate]] = {}
        for row in candidates:
            party = PARTY_MAP.get(str(row.get("party", "")).upper())
            if party is None:
                continue
            office = str(row.get("office", "")).upper()
            state = str(row.get("state", "")).upper()
            district = str(row.get("district", "00")).zfill(2)
            # Forecast files conventionally number at-large House districts 01;
            # the Clerk and FEC encode them as 00.
            if office == "H" and district == "00":
                district = "01"
            race_id = state if office == "S" else f"{state}-{district}"
            name = str(row.get("name") or row.get("candidate_name") or "")
            if not name or not state:
                continue
            incumbent_code = str(row.get("incumbent_challenge", "")).upper()
            incumbent_text = str(row.get("incumbent_challenge_full", "")).lower()
            candidate = Candidate(
                name=name,
                normalized_name=normalize_name(name),
                party=party,
                race_id=race_id,
                incumbent=incumbent_code == "I" or "incumbent" in incumbent_text,
            )
            self.by_race.setdefault(race_id, []).append(candidate)
        # FEC filings and the Clerk roster can contain the same incumbent.
        # Collapse them before deciding whether a race has exactly one incumbent.
        for race_id, race_candidates in self.by_race.items():
            deduped: dict[tuple[str, str], Candidate] = {}
            for candidate in race_candidates:
                key = (candidate.normalized_name, candidate.party)
                previous = deduped.get(key)
                if previous is None or (candidate.incumbent and not previous.incumbent):
                    deduped[key] = candidate
            self.by_race[race_id] = list(deduped.values())

    def resolve(self, race_id: str, poll_name: str) -> Optional[Candidate]:
        query = normalize_name(poll_name.replace("-", " "))
        suffix_matches = [
            candidate for candidate in self.by_race.get(race_id, [])
            if candidate.normalized_name == query
            or candidate.normalized_name.endswith(f" {query}")
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        incumbent_suffix = [candidate for candidate in suffix_matches if candidate.incumbent]
        if suffix_matches and len({candidate.party for candidate in suffix_matches}) == 1 and len(incumbent_suffix) == 1:
            return incumbent_suffix[0]
        if " " not in query:
            token_matches = [
                candidate for candidate in self.by_race.get(race_id, [])
                if query in candidate.normalized_name.split()
            ]
            if len(token_matches) == 1:
                return token_matches[0]
        scored = sorted(
            ((SequenceMatcher(None, query, candidate.normalized_name).ratio(), candidate)
             for candidate in self.by_race.get(race_id, [])),
            key=lambda item: item[0],
            reverse=True,
        )
        if not scored or scored[0][0] < 0.78:
            return None
        if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.06:
            return None
        return scored[0][1]

    def update_fundamentals(self, frame: pd.DataFrame, race_column: str) -> pd.DataFrame:
        """Apply FEC incumbency/open-seat facts without mutating source files."""
        updated = frame.copy()
        for idx, row in updated.iterrows():
            race_id = str(row[race_column])
            incumbents = [candidate for candidate in self.by_race.get(race_id, []) if candidate.incumbent]
            if len(incumbents) == 1:
                updated.at[idx, "incumbent"] = incumbents[0].name.title()
                updated.at[idx, "incumbent_party"] = incumbents[0].party
                updated.at[idx, "open_seat"] = False
            elif not incumbents and self.by_race.get(race_id):
                updated.at[idx, "incumbent"] = "Open seat"
                updated.at[idx, "incumbent_party"] = ""
                updated.at[idx, "open_seat"] = True
        updated["fundamentals_source"] = "FEC candidate registry + 2024 partisan lean"
        updated["fundamentals_effective_date"] = date.today().isoformat()
        return updated


def poll_observation_variance(row: pd.Series, election_date: date) -> float:
    explicit = pd.to_numeric(row.get("observation_std"), errors="coerce")
    if pd.notna(explicit) and float(explicit) > 0:
        return float(explicit) ** 2
    n = max(float(row.get("sample_size", 500)), 100.0)
    published_moe = pd.to_numeric(row.get("margin_of_error"), errors="coerce")
    sampling_std = float(published_moe) if pd.notna(published_moe) else np.sqrt(5000.0 / n)
    poll_date = pd.to_datetime(row["date"]).date()
    days = max((election_date - poll_date).days, 0)
    time_std = min(3.0, 0.012 * days)
    population_std = 0.0 if str(row.get("population", "")).lower() in {"lv", "likely_voters"} else 0.7
    partisan_std = 1.5 if pd.notna(row.get("partisan")) else 0.0
    pollster_std = float(pd.to_numeric(row.get("pollster_std"), errors="coerce"))
    if not np.isfinite(pollster_std):
        pollster_std = 0.8
    design_std = 2.5
    return float(sampling_std**2 + time_std**2 + population_std**2
                 + partisan_std**2 + pollster_std**2 + design_std**2)


def update_race_draws(
    prior_draws: np.ndarray,
    polls: pd.DataFrame,
    rng: np.random.Generator,
    election_date: date = date(2026, 11, 3),
    student_df: float = 5.0,
    correlated_error_floor: float = 5.0,
) -> np.ndarray:
    """Robust Bayesian update with irreducible correlated polling error.

    Polls in one race share turnout and design error, so treating them as
    conditionally independent makes the posterior collapse unrealistically.
    Independent sampling/design components average down; the correlated floor
    does not. The poll aggregate is then combined with the fundamentals prior.
    """
    draws = np.asarray(prior_draws, dtype=float).copy()
    if polls is None or polls.empty:
        return draws
    prior_mean, prior_var = float(np.mean(draws)), float(np.var(draws))
    observations: list[float] = []
    variances: list[float] = []
    for _, row in polls.sort_values("date").iterrows():
        pollster_effect = float(pd.to_numeric(row.get("pollster_effect", 0.0), errors="coerce"))
        if not np.isfinite(pollster_effect):
            pollster_effect = 0.0
        observations.append(float(row["margin"]) - pollster_effect)
        variances.append(poll_observation_variance(row, election_date))

    y = np.asarray(observations)
    variance = np.asarray(variances)
    poll_mean = float(np.average(y, weights=1.0 / variance))
    weights = 1.0 / variance
    for _ in range(6):
        standardized = (y - poll_mean) / np.sqrt(variance + correlated_error_floor**2)
        robust_weight = (student_df + 1.0) / (student_df + standardized**2)
        weights = robust_weight / variance
        poll_mean = float(np.average(y, weights=weights))

    poll_var = correlated_error_floor**2 + 1.0 / max(float(weights.sum()), 1e-9)
    prior_conflict = (poll_mean - prior_mean) / np.sqrt(prior_var + poll_var)
    poll_var *= max(1.0, (prior_conflict**2 + student_df) / (student_df + 1.0))
    gain = prior_var / (prior_var + poll_var)
    posterior_mean = prior_mean + gain * (poll_mean - prior_mean)
    posterior_var = max((1.0 - gain) * prior_var, 0.25)
    centered = draws - prior_mean
    updated = posterior_mean + centered * np.sqrt(posterior_var / max(prior_var, 1e-9))
    updated += rng.normal(0.0, np.sqrt(posterior_var) * 0.005, len(draws))
    return updated


def refresh_candidate_registry(
    data_dir: Path,
    fec_api_key: str = "DEMO_KEY",
    session: Optional[requests.Session] = None,
) -> CandidateRegistry:
    """Refresh official identities without fetching any polling provider."""

    session = session or requests.Session()
    fec_payload = FECClient(fec_api_key, session).fetch_candidates()
    clerk_payload, _ = HouseClerkClient(session).fetch_members()
    snapshots = SnapshotStore(Path(data_dir) / "raw" / "snapshots")
    snapshots.save("fec_candidates", fec_payload, FEC_CANDIDATE_MASTER_URL, "U.S. government public data")
    snapshots.save("house_clerk", clerk_payload, HOUSE_CLERK_MEMBERS_URL, "U.S. government public data")
    registry_rows = [
        ({**row, "incumbent_challenge": "", "incumbent_challenge_full": ""}
         if str(row.get("office", "")).upper() == "H" else row)
        for row in fec_payload
    ]
    return CandidateRegistry([*registry_rows, *clerk_payload])


def load_cached_candidate_registry(data_dir: Path) -> Optional[CandidateRegistry]:
    """Rebuild the registry from the latest immutable FEC and Clerk snapshots."""
    root = Path(data_dir) / "raw" / "snapshots"
    records: list[dict[str, Any]] = []
    for provider in ("fec_candidates", "house_clerk"):
        snapshots = sorted((root / provider).glob("*.json*"))
        if not snapshots:
            continue
        snapshot = snapshots[-1]
        if snapshot.suffix == ".gz":
            with gzip.open(snapshot, "rt", encoding="utf-8") as handle:
                envelope = json.load(handle)
        else:
            envelope = json.loads(snapshot.read_text())
        payload = envelope.get("payload", [])
        if isinstance(payload, list):
            if provider == "fec_candidates":
                payload = [
                    ({**row, "incumbent_challenge": "", "incumbent_challenge_full": ""}
                     if str(row.get("office", "")).upper() == "H" else row)
                    for row in payload
                ]
            records.extend(payload)
    return CandidateRegistry(records) if records else None
