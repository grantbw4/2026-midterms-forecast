"""Archive integrity and write protection must hold even during research runs."""

import json
from pathlib import Path
import shutil
import sys

import pytest

from models import forecast_archive as archive
from scripts import fetch_inputs, generate_forecast


def test_selected_election_day_archive():
    archive.verify_archive()
    manifest = json.loads(archive.ARCHIVE_PATH.read_text())
    assert manifest["source_commit"] == "6e39b248d4531902b024a337bd43a11e35aafb66"
    assert manifest["forecast_updated_at"] == "2026-11-03T08:16:10.819058+00:00"
    assert manifest["run_id"] == "forecast-20261103T081610Z-22f8084f3cb1"


@pytest.mark.parametrize("directory", ["website", "outputs"])
@pytest.mark.parametrize("filename", archive.ARTIFACTS)
def test_any_modified_artifact_blocks_publication(tmp_path, directory, filename):
    for folder in ("website", "outputs"):
        (tmp_path / folder).mkdir()
        for artifact in archive.ARTIFACTS:
            shutil.copyfile(archive.PROJECT_ROOT / folder / artifact, tmp_path / folder / artifact)
    shutil.copyfile(archive.ARCHIVE_PATH, tmp_path / "website" / "archive.json")
    with (tmp_path / directory / filename).open("ab") as output:
        output.write(b"\n")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        archive.verify_archive(tmp_path)


def test_default_refresh_refuses_before_credentials_or_network(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fetch_inputs.py", "--allow-blocked", "--allow-keyless-fred"])
    monkeypatch.setattr(fetch_inputs, "_environment_value", lambda _: pytest.fail("must stop before credentials"))
    with pytest.raises(RuntimeError, match="archive is frozen"):
        fetch_inputs.main()


@pytest.mark.parametrize("arguments", [
    [], ["--skip-website-copy"], ["--allow-blocked-inputs"],
    ["--output-dir", "/tmp/forecast-research"],
    ["--skip-website-copy", "--output-dir", "website"],
    ["--skip-website-copy", "--output-dir", "outputs/../outputs"],
])
def test_generation_refuses_before_model_or_writes(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["generate_forecast.py", *arguments])
    monkeypatch.setattr(generate_forecast, "_load_manifest", lambda _: pytest.fail("must stop before model inputs"))
    with pytest.raises(RuntimeError, match="archive is frozen"):
        generate_forecast.main()


def test_research_output_must_be_isolated_including_symlinks(tmp_path):
    archive.guard_forecast_output(tmp_path / "research", True)
    with pytest.raises(RuntimeError):
        archive.guard_forecast_output(archive.PROJECT_ROOT.parent, True)
    alias = tmp_path / "alias"
    alias.symlink_to(archive.PROJECT_ROOT / "outputs", target_is_directory=True)
    with pytest.raises(RuntimeError):
        archive.guard_forecast_output(alias, True)


@pytest.mark.parametrize("payload,exception", [
    ("invalid json", json.JSONDecodeError),
    ('{"status":"unknown"}', ValueError),
])
def test_damaged_archive_marker_fails_closed(tmp_path, monkeypatch, payload, exception):
    marker = tmp_path / "archive.json"
    marker.write_text(payload)
    monkeypatch.setattr(archive, "ARCHIVE_PATH", marker)
    with pytest.raises(exception):
        archive.guard_input_refresh()
    with pytest.raises(exception):
        archive.guard_forecast_output(tmp_path / "research", True)


def test_automated_updates_removed_and_publication_verifies_archive():
    workflows = archive.PROJECT_ROOT / ".github" / "workflows"
    assert not (workflows / "daily-forecast.yml").exists()
    assert all("cron:" not in path.read_text() for path in workflows.glob("*.yml"))
    for filename in ("verify.yml", "deploy-pages.yml"):
        workflow = (workflows / filename).read_text()
        assert "python -m models.forecast_archive" in workflow
        assert "workflow_run:" not in workflow
