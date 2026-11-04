"""Protect the published Election Day snapshot while allowing isolated research."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_PATH = PROJECT_ROOT / "website" / "archive.json"
ARTIFACTS = ("forecast.json", "senate_forecast.json", "timeline.csv", "senate_timeline.csv")


def archive_is_frozen() -> bool:
    # A damaged marker must fail closed, before any data acquisition or writes.
    if not ARCHIVE_PATH.exists():
        return False
    if json.loads(ARCHIVE_PATH.read_text())["status"] != "frozen":
        raise ValueError("Invalid Election Day archive status")
    return True


def guard_input_refresh() -> None:
    if archive_is_frozen():
        raise RuntimeError("The Election Day archive is frozen; input acquisition is disabled.")


def guard_forecast_output(output_dir: Path, skip_website_copy: bool) -> None:
    if not archive_is_frozen():
        return
    destination = output_dir.resolve()
    root = PROJECT_ROOT.resolve()
    if not skip_website_copy or destination.is_relative_to(root) or root.is_relative_to(destination):
        raise RuntimeError(
            "The Election Day archive is frozen. Research requires --skip-website-copy "
            "and an --output-dir outside this checkout (for example /tmp/forecast-research)."
        )


def verify_archive(root: Path = PROJECT_ROOT) -> None:
    archive = json.loads((root / "website" / "archive.json").read_text())
    if archive["status"] != "frozen" or set(archive["sha256"]) != set(ARTIFACTS):
        raise ValueError("Invalid Election Day archive manifest")
    for name in ARTIFACTS:
        expected = archive["sha256"][name]
        for directory in ("website", "outputs"):
            digest = hashlib.sha256((root / directory / name).read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(f"Frozen artifact changed: {directory}/{name}")
    for name in ARTIFACTS[:2]:
        metadata = json.loads((root / "website" / name).read_text())["metadata"]
        for key, expected in (
            ("updated_at", archive["forecast_updated_at"]),
            ("run_id", archive["run_id"]),
            ("election_date", archive["election_date"]),
            ("forecast_timezone", archive["forecast_timezone"]),
        ):
            if metadata[key] != expected:
                raise ValueError(f"Archive metadata mismatch: {name}: {key}")


if __name__ == "__main__":
    verify_archive()
    print("Verified: all eight artifact copies match the frozen Election Day snapshot.")
