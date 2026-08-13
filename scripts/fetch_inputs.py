#!/usr/bin/env python3
"""Fetch and atomically promote every input required by the daily forecast.

Network access lives here.  ``scripts/generate_forecast.py`` is deliberately
cache-only so a partially successful refresh can never produce a mixed public
forecast bundle.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.race_polling import (  # noqa: E402
    CandidateRegistry,
    FECClient,
    FEC_CANDIDATE_MASTER_URL,
    HOUSE_CLERK_MEMBERS_URL,
    HouseClerkClient,
    SnapshotStore,
    load_cached_candidate_registry,
)
from models.silver_bulletin import (  # noqa: E402
    AVERAGES_CSV_URL,
    GENERIC_POLLS_CSV_URL,
    SOURCE_NOTICE,
    SilverAverageData,
    SilverBulletinClient,
    load_silver_cache,
    prepare_silver_averages,
)
DATA = PROJECT_ROOT / "data"
PROCESSED = DATA / "processed"
ECONOMIC = DATA / "raw" / "economic"
SNAPSHOTS = DATA / "raw" / "snapshots"
MANIFEST_PATH = PROCESSED / "input_manifest.json"
FORECAST_EPOCH = "2026-08-12"

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_SERIES = {
    "DSPIC96": "real_disposable_income",
    "UNRATE": "unemployment_rate",
    "GDP": "gdp",
    "CPIAUCSL": "cpi",
    "UMCSENT": "consumer_sentiment",
}

FRESHNESS = {
    "silver_averages": (2, 7),
    "silver_generic_polls": (2, 7),
    "fred_monthly": (75, 150),
    "fred_gdp": (180, 300),
    "candidate_registry": (2, 7),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, default=str, allow_nan=True).encode()


def _freshness(observed: date, warning_days: int, blocking_days: int) -> dict[str, Any]:
    age = max((datetime.now(timezone.utc).date() - observed).days, 0)
    state = "blocked" if age > blocking_days else ("degraded" if age > warning_days else "healthy")
    return {
        "observation_date": observed.isoformat(),
        "age_days": age,
        "warning_days": warning_days,
        "blocking_days": blocking_days,
        "state": state,
    }


def _source_record(
    *,
    provider: str,
    source_url: str,
    observed: date,
    rows: int,
    payload: bytes,
    freshness_key: str,
    fallback_used: bool,
    provider_model_date: date | None = None,
) -> dict[str, Any]:
    warning, blocking = FRESHNESS[freshness_key]
    record = {
        "provider": provider,
        "source_url": source_url,
        "row_count": int(rows),
        "sha256": _sha256_bytes(payload),
        "fallback_used": fallback_used,
        **_freshness(observed, warning, blocking),
    }
    if provider_model_date is not None:
        record["provider_model_date"] = provider_model_date.isoformat()
    return record


def _validate_generic_poll_feed(frame: pd.DataFrame) -> tuple[date, date]:
    required = {
        "subgroup", "enddate", "modeldate", "influence", "adjusted_net",
        "poll_id", "question_id",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Silver generic poll cache missing columns: {sorted(missing)}")
    model_dates = pd.to_datetime(frame["modeldate"], errors="coerce")
    end_dates = pd.to_datetime(frame["enddate"], errors="coerce")
    if model_dates.isna().any() or end_dates.isna().any():
        raise ValueError("Silver generic poll feed contains malformed dates")
    return model_dates.max().date(), end_dates.max().date()


def _normalize_candidate_rows(fec_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        ({**row, "incumbent_challenge": "", "incumbent_challenge_full": ""}
         if str(row.get("office", "")).upper() == "H" else row)
        for row in fec_rows
    ]


def _latest_snapshot_date(provider: str) -> date:
    snapshots = sorted((SNAPSHOTS / provider).glob("*.json*"))
    if not snapshots:
        raise FileNotFoundError(f"No {provider} snapshot exists")
    path = snapshots[-1]
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text())
    return datetime.fromisoformat(payload["metadata"]["fetched_at"]).date()


def _fetch_candidate_registry(session: requests.Session) -> tuple[CandidateRegistry, list[dict[str, Any]], list[dict[str, Any]], bool]:
    try:
        fec_rows = FECClient(os.getenv("FEC_API_KEY", "DEMO_KEY"), session).fetch_candidates()
        clerk_rows, _ = HouseClerkClient(session).fetch_members()
        registry = CandidateRegistry([*_normalize_candidate_rows(fec_rows), *clerk_rows])
        return registry, fec_rows, clerk_rows, False
    except Exception:
        registry = load_cached_candidate_registry(DATA)
        if registry is None:
            raise
        return registry, [], [], True


def _fetch_silver(
    session: requests.Session,
    registry: CandidateRegistry,
) -> tuple[SilverAverageData, pd.DataFrame, pd.DataFrame | None, bool]:
    try:
        client = SilverBulletinClient(session)
        raw_averages = client.fetch()
        generic_polls = client.fetch_generic_polls()
        return prepare_silver_averages(raw_averages, registry), generic_polls, raw_averages, False
    except Exception:
        generic_path = PROCESSED / "silver_generic_polls.csv"
        if not generic_path.exists():
            raise
        return load_silver_cache(DATA), pd.read_csv(generic_path), None, True


def _fetch_fred_series(
    session: requests.Session,
    api_key: str,
    series_id: str,
    name: str,
    allow_keyless: bool = False,
) -> tuple[pd.DataFrame, bool]:
    try:
        if api_key:
            response = session.get(
                FRED_URL,
                params={
                    "series_id": series_id,
                    "api_key": api_key,
                    "file_type": "json",
                    "observation_start": "2004-01-01",
                },
                timeout=45,
            )
            response.raise_for_status()
            observations = response.json().get("observations", [])
            frame = pd.DataFrame(observations)
            if frame.empty or not {"date", "value"} <= set(frame.columns):
                raise ValueError(f"FRED {series_id} returned no observations")
            frame = frame[["date", "value"]].rename(columns={"value": name})
        elif allow_keyless:
            response = session.get(FRED_GRAPH_URL, params={"id": series_id}, timeout=45)
            response.raise_for_status()
            from io import StringIO
            frame = pd.read_csv(StringIO(response.text)).rename(
                columns={"DATE": "date", "observation_date": "date", series_id: name}
            )
            if not {"date", name} <= set(frame.columns):
                raise ValueError(f"FRED graph download for {series_id} changed schema")
            frame = frame[["date", name]]
        else:
            raise RuntimeError("FRED_API_KEY is required")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
        frame = frame.dropna().sort_values("date").reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"FRED {series_id} contained no numeric observations")
        return frame, False
    except Exception:
        path = ECONOMIC / f"{name}.csv"
        if not path.exists():
            raise
        return pd.read_csv(path), True


def _write_staged_csv(stage: Path, relative: Path, frame: pd.DataFrame) -> Path:
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and validate daily forecast inputs")
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Development/bootstrap only: write a manifest even when a cached source is catastrophically stale",
    )
    parser.add_argument(
        "--allow-keyless-fred",
        action="store_true",
        help="Local bootstrap only: use FRED's public graph CSV instead of the authenticated API",
    )
    args = parser.parse_args()
    fred_key = os.getenv("FRED_API_KEY", "").strip()
    if fred_key == "your_fred_api_key_here":
        fred_key = ""
    if not fred_key and not args.allow_keyless_fred:
        raise RuntimeError("FRED_API_KEY is required for the daily input refresh")

    session = requests.Session()
    session.headers.update({"User-Agent": "Grants-Election-Forecast/1.0"})
    print("Refreshing candidate registry...", flush=True)
    registry, fec_rows, clerk_rows, candidate_fallback = _fetch_candidate_registry(session)
    print("Refreshing Silver Bulletin feeds...", flush=True)
    silver, generic_polls, raw_averages, silver_fallback = _fetch_silver(session, registry)
    print("Refreshing FRED series...", flush=True)
    fred: dict[str, tuple[pd.DataFrame, bool]] = {
        name: _fetch_fred_series(requests.Session(), fred_key, series_id, name, args.allow_keyless_fred)
        for series_id, name in FRED_SERIES.items()
    }
    print("Validating input freshness...", flush=True)

    generic_polls = generic_polls.copy()
    provider_model_date, latest_poll_date = _validate_generic_poll_feed(generic_polls)
    published_date = pd.to_datetime(silver.generic_history["date"], errors="raise").max().date()
    now = datetime.now(timezone.utc)

    candidate_observed = min(
        _latest_snapshot_date("fec_candidates"),
        _latest_snapshot_date("house_clerk"),
    ) if candidate_fallback else now.date()
    sources: dict[str, dict[str, Any]] = {
        "silver_averages": _source_record(
            provider="Silver Bulletin",
            source_url=AVERAGES_CSV_URL,
            observed=published_date,
            rows=len(raw_averages) if raw_averages is not None else len(silver.generic_history) + len(silver.race_likelihoods),
            payload=_json_bytes({"generic": silver.generic_history.to_dict("records"), "races": silver.race_likelihoods.to_dict("records")}),
            freshness_key="silver_averages",
            fallback_used=silver_fallback,
            provider_model_date=published_date,
        ),
        "silver_generic_polls": _source_record(
            provider="Silver Bulletin",
            source_url=GENERIC_POLLS_CSV_URL,
            observed=provider_model_date,
            rows=len(generic_polls),
            payload=generic_polls.to_csv(index=False).encode(),
            freshness_key="silver_generic_polls",
            fallback_used=silver_fallback,
            provider_model_date=provider_model_date,
        ),
        "candidate_registry": _source_record(
            provider="FEC and Clerk of the House",
            source_url=FEC_CANDIDATE_MASTER_URL,
            observed=candidate_observed,
            rows=len(fec_rows) + len(clerk_rows) if not candidate_fallback else sum(len(v) for v in registry.by_race.values()),
            payload=_json_bytes({"fec": fec_rows, "clerk": clerk_rows}) if not candidate_fallback else _json_bytes(sorted(registry.by_race)),
            freshness_key="candidate_registry",
            fallback_used=candidate_fallback,
        ),
    }
    sources["candidate_registry"]["source_urls"] = [FEC_CANDIDATE_MASTER_URL, HOUSE_CLERK_MEMBERS_URL]
    sources["silver_generic_polls"]["latest_poll_end_date"] = latest_poll_date.isoformat()

    for name, (frame, fallback) in fred.items():
        observed = pd.to_datetime(frame["date"], errors="raise").max().date()
        series_id = next(key for key, value in FRED_SERIES.items() if value == name)
        sources[f"fred_{name}"] = _source_record(
            provider="FRED",
            source_url=f"https://fred.stlouisfed.org/series/{series_id}",
            observed=observed,
            rows=len(frame),
            payload=frame.to_csv(index=False).encode(),
            freshness_key="fred_gdp" if name == "gdp" else "fred_monthly",
            fallback_used=fallback,
        )

    blocked = sorted(name for name, record in sources.items() if record["state"] == "blocked")
    degraded = sorted(name for name, record in sources.items() if record["state"] == "degraded")
    if blocked and not args.allow_blocked:
        raise RuntimeError(f"Refusing to promote catastrophically stale inputs: {', '.join(blocked)}")

    manifest = {
        "schema_version": "1.0.0",
        "forecast_epoch": FORECAST_EPOCH,
        "generated_at": now.isoformat(),
        "status": "blocked" if blocked else ("degraded" if degraded else "healthy"),
        "blocked_sources": blocked,
        "degraded_sources": degraded,
        "fallback_used": any(record["fallback_used"] for record in sources.values()),
        "sources": sources,
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="input-stage-", dir=PROCESSED) as stage_name:
        stage = Path(stage_name)
        staged = {
            PROCESSED / "silver_generic_history.csv": _write_staged_csv(stage, Path("processed/silver_generic_history.csv"), silver.generic_history),
            PROCESSED / "silver_race_averages.csv": _write_staged_csv(stage, Path("processed/silver_race_averages.csv"), silver.race_likelihoods),
            PROCESSED / "silver_generic_polls.csv": _write_staged_csv(stage, Path("processed/silver_generic_polls.csv"), generic_polls),
        }
        status_path = stage / "processed/silver_average_status.json"
        status_path.write_text(json.dumps(silver.status, indent=2, allow_nan=False))
        staged[PROCESSED / "silver_average_status.json"] = status_path
        for name, (frame, _) in fred.items():
            staged[ECONOMIC / f"{name}.csv"] = _write_staged_csv(stage, Path(f"economic/{name}.csv"), frame)
        manifest_stage = stage / "processed/input_manifest.json"
        manifest_stage.write_text(json.dumps(manifest, indent=2, allow_nan=False))
        staged[MANIFEST_PATH] = manifest_stage
        for destination, source in staged.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination.with_suffix(destination.suffix + ".tmp"))
        for destination in staged:
            destination.with_suffix(destination.suffix + ".tmp").replace(destination)

    snapshots = SnapshotStore(SNAPSHOTS)
    if not silver_fallback:
        generic_payload = silver.generic_history.assign(date=silver.generic_history["date"].astype(str))
        race_payload = silver.race_likelihoods.assign(date=silver.race_likelihoods["date"].astype(str))
        snapshots.save(
            "silver_bulletin_averages",
            {"generic_history": generic_payload.to_dict("records"), "latest_race_averages": race_payload.to_dict("records")},
            AVERAGES_CSV_URL,
            SOURCE_NOTICE,
        )
        snapshots.save(
            "silver_bulletin_generic_polls",
            generic_polls.astype(object).where(pd.notna(generic_polls), None).to_dict("records"),
            GENERIC_POLLS_CSV_URL,
            SOURCE_NOTICE,
        )
    if not candidate_fallback:
        snapshots.save("fec_candidates", fec_rows, FEC_CANDIDATE_MASTER_URL, "U.S. government public data")
        snapshots.save("house_clerk", clerk_rows, HOUSE_CLERK_MEMBERS_URL, "U.S. government public data")

    print(f"Promoted validated input manifest: {manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
