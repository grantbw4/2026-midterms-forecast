"""Synthetic recovery and behavioral verification for the production forecast."""

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.backtesting import evaluate_predictions
from models.chamber_forecast import _house_parameters, _senate_parameters
from models.dynamic_polling import DynamicNationalModel, build_fundamentals_prior
from models.house_model import fit_house_calibration
from models.race_polling import (
    CandidateRegistry,
    SnapshotStore,
    poll_observation_variance,
    update_race_draws,
)
from models.silver_bulletin import (
    SilverBulletinClient,
    calibrate_generic_average_uncertainty,
    prepare_generic_poll_likelihood,
    prepare_silver_averages,
)
import scripts.fetch_inputs as fetch_inputs
from scripts.fetch_inputs import _environment_value, _freshness, _validate_generic_poll_feed
from scripts.generate_forecast import _publish_staged_bundle


def test_snapshot_store_deduplicates_identical_payloads(tmp_path):
    store = SnapshotStore(tmp_path)
    first = store.save("provider", {"rows": [1, 2]}, "https://example.com", "test")
    second = store.save("provider", {"rows": [1, 2]}, "https://example.com", "test")
    assert first == second
    assert len(list((tmp_path / "provider").glob("*.json*"))) == 1


def _national_polls() -> pd.DataFrame:
    start = date(2026, 5, 1)
    rows = []
    for index in range(24):
        latent = -1.0 + 0.15 * index
        rows.append({
            "date": start + timedelta(days=index * 3),
            "pollster": ["Alpha", "Beta", "Gamma"][index % 3],
            "sample_size": 1000,
            "population": "rv",
            "dem_pct": 45 + latent / 2,
            "rep_pct": 45 - latent / 2,
            "margin": latent + [0.3, -0.2, 0.0][index % 3],
            "partisan": None,
            "internal": False,
        })
    return pd.DataFrame(rows)


def test_dynamic_model_recovers_trend_and_reports_analytic_diagnostics():
    result = DynamicNationalModel(random_seed=7).fit(
        _national_polls(), build_fundamentals_prior(), n_draws=2000
    )
    assert result.current_mean > 0.5
    assert result.diagnostics["inference"] == "robust_dynamic_kalman_filter"
    assert result.diagnostics["r_hat"] is None
    assert result.diagnostics["divergences"] == 0
    assert len(result.trend) > 50


def test_economy_only_prior_retains_economic_and_structural_uncertainty():
    prior = build_fundamentals_prior(economic_index=-1.76, economic_std=1.0)
    expected_mean = -0.34 * -1.76
    expected_variance = 3.5**2 + (-0.34 * 1.0) ** 2 + (-1.76 * 0.33) ** 2
    assert np.isclose(prior.mean, expected_mean)
    assert np.isclose(prior.std**2, expected_variance)
    assert set(prior.components) == {"economy", "structural_uncertainty"}


def test_external_average_enters_national_likelihood_once():
    latest = {
        "date": "2026-08-11", "pollster": "Silver Bulletin maintained average",
        "sample_size": 1000, "population": "likely-voter-adjusted average",
        "dem_pct": 50.4, "rep_pct": 42.3, "margin": 8.1,
        "observation_std": 1.5,
    }
    history = pd.DataFrame([
        {**latest, "date": (date(2026, 8, 1) + timedelta(days=index)).isoformat(), "margin": 7.0 + index / 10}
        for index in range(10)
    ] + [latest])
    prior = build_fundamentals_prior()
    full = DynamicNationalModel(random_seed=10).fit_external_average(history, prior, n_draws=1000)
    one = DynamicNationalModel(random_seed=10).fit_external_average(pd.DataFrame([latest]), prior, n_draws=1000)
    assert full.current_mean == one.current_mean
    assert full.current_std == one.current_std
    assert full.diagnostics["n_polls"] == 1
    assert full.diagnostics["published_average_days"] == 11


def test_bulletin_adjusted_poll_universe_is_one_weighted_likelihood():
    polls = pd.DataFrame([
        {"subgroup": "All polls", "pollster": "A", "enddate": "2026-08-10",
         "adjusted_net": 4.0, "influence": 1.0, "poll_id": 1, "question_id": 1},
        {"subgroup": "All polls", "pollster": "B", "enddate": "2026-08-11",
         "adjusted_net": 8.0, "influence": 3.0, "poll_id": 2, "question_id": 2},
    ])
    likelihood = prepare_generic_poll_likelihood(polls, observation_std=1.3)
    assert len(likelihood) == 1
    assert likelihood.iloc[0]["margin"] == 7.0
    assert likelihood.iloc[0]["aggregation_level"] == "influence_weighted_adjusted_poll_universe"


def test_bulletin_dispersion_calibration_keeps_correlated_floor():
    polls = pd.DataFrame([
        {"subgroup": "All polls", "enddate": f"2026-08-{day:02d}",
         "adjusted_net": float(day % 3), "influence": 1.0,
         "poll_id": day, "question_id": day}
        for day in range(1, 10)
    ])
    calibration = calibrate_generic_average_uncertainty(pd.DataFrame(), polls)
    assert calibration["observation_std"] >= calibration["correlated_design_floor"]


def test_smoothed_bulletin_history_cannot_reduce_process_prior():
    history = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=60),
        "margin": np.linspace(2.0, 2.1, 60),
    })
    calibration = DynamicNationalModel.calibrate_process_std(history)
    assert calibration["process_std_per_day"] >= 0.09


def test_house_cluster_level_posterior_uncertainty_has_floor():
    rows = []
    for year, national in ((2018, 8.0), (2022, -3.0)):
        for index in range(200):
            lean = (index % 21) - 10
            rows.append({
                "year": year, "district_id": f"X-{year}-{index}",
                "margin": lean + national, "pvi_numeric": lean,
                "incumbency_code": 0, "national_margin": national,
                "region": "Southeast",
            })
    calibration = fit_house_calibration([pd.DataFrame(rows)])
    national_index = calibration.coefficient_names.index("national")
    assert calibration.posterior_covariance[national_index, national_index] >= 0.25**2


def test_production_parameter_sign_conventions():
    house = _house_parameters(PROJECT_ROOT / "data")
    senate = _senate_parameters(PROJECT_ROOT / "data")
    assert house.posterior_mean[house.coefficient_names.index("lean")] > 0
    assert house.posterior_mean[house.coefficient_names.index("incumbency")] > 0
    assert house.posterior_mean[house.coefficient_names.index("national")] > 0
    assert senate.beta_lean_mean > 0
    assert senate.beta_inc_mean > 0
    assert senate.beta_national_mean > 0


def test_freshness_thresholds_fail_closed():
    today = datetime.now(timezone.utc).date()
    assert _freshness(today - timedelta(days=2), 2, 7)["state"] == "healthy"
    assert _freshness(today - timedelta(days=3), 2, 7)["state"] == "degraded"
    assert _freshness(today - timedelta(days=8), 2, 7)["state"] == "blocked"


def test_environment_value_prefers_injected_secret(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "actions-secret")
    assert _environment_value("FRED_API_KEY") == "actions-secret"


def test_environment_value_reads_local_dotenv_without_dependency(tmp_path, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(fetch_inputs, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text("FRED_API_KEY='local-secret'\n")
    assert fetch_inputs._environment_value("FRED_API_KEY") == "local-secret"


def test_silver_generic_feed_rejects_changed_columns_and_malformed_dates():
    base = pd.DataFrame([{
        "subgroup": "All polls", "enddate": "2026-08-10", "modeldate": "2026-08-12",
        "influence": 1.0, "adjusted_net": 7.0, "poll_id": 1, "question_id": 2,
    }])
    model_date, latest_poll = _validate_generic_poll_feed(base)
    assert model_date.isoformat() == "2026-08-12"
    assert latest_poll.isoformat() == "2026-08-10"
    try:
        _validate_generic_poll_feed(base.drop(columns="influence"))
    except ValueError as error:
        assert "missing columns" in str(error)
    else:
        raise AssertionError("changed Silver columns must fail validation")
    try:
        _validate_generic_poll_feed(base.assign(modeldate="not-a-date"))
    except ValueError as error:
        assert "malformed dates" in str(error)
    else:
        raise AssertionError("malformed Silver model dates must fail validation")


def test_silver_client_schema_fixture_rejects_missing_modeldate():
    class Response:
        text = "subgroup,pollster,enddate,samplesize,population,influence,adjusted_net,partisan,poll_id,question_id\nAll polls,A,2026-08-10,1000,lv,1,5,,1,1\n"

        @staticmethod
        def raise_for_status():
            return None

    class Session:
        @staticmethod
        def get(*_args, **_kwargs):
            return Response()

    try:
        SilverBulletinClient(Session()).fetch_generic_polls()
    except ValueError as error:
        assert "modeldate" in str(error)
    else:
        raise AssertionError("missing modeldate must fail schema validation")


def test_silver_average_rejects_malformed_dates():
    raw = pd.DataFrame([{
        "id": "2026_US-GB", "group": "Generic ballot", "place": "National",
        "date": "not-a-date", "cand_D": "Democrats", "party_D": "D", "avg_D": 50,
        "cand_R": "Republicans", "party_R": "R", "avg_R": 43,
    }])
    try:
        prepare_silver_averages(raw)
    except ValueError as error:
        assert "malformed dates" in str(error)
    else:
        raise AssertionError("malformed maintained-average dates must fail")


def test_silver_uses_only_latest_d_vs_r_average_per_race():
    raw = pd.DataFrame([
        {"id": "2026_NC-S2", "group": "Senate", "place": "North Carolina", "date": "2026-08-10",
         "cand_D": "Cooper", "party_D": "D", "avg_D": 50.0,
         "cand_R": "Whatley", "party_R": "R", "avg_R": 44.0},
        {"id": "2026_NC-S2", "group": "Senate", "place": "North Carolina", "date": "2026-08-11",
         "cand_D": "Cooper", "party_D": "D", "avg_D": 51.4,
         "cand_R": "Whatley", "party_R": "R", "avg_R": 42.5},
        {"id": "2026_ID-S2", "group": "Senate", "place": "Idaho", "date": "2026-08-11",
         "cand_D": "", "party_D": "", "avg_D": None,
         "cand_R": "Risch", "party_R": "R", "avg_R": 49.0},
        {"id": "2026_MT-S2", "group": "Senate", "place": "Montana", "date": "2026-08-11",
         "cand_D": "Bankhead", "party_D": "D", "avg_D": 23.4,
         "cand_R": "Alme", "party_R": "R", "avg_R": 42.1,
         "cand_I": "Bodnar", "party_I": "I", "avg_I": 23.7},
        {"id": "2026_US-GB", "group": "Generic ballot", "place": "National", "date": "2026-08-11",
         "cand_D": "Democrats", "party_D": "D", "avg_D": 50.4,
         "cand_R": "Republicans", "party_R": "R", "avg_R": 42.3},
    ])
    prepared = prepare_silver_averages(raw)
    assert len(prepared.race_likelihoods) == 1
    assert prepared.race_likelihoods.iloc[0]["race_id"] == "NC"
    assert np.isclose(prepared.race_likelihoods.iloc[0]["margin"], 8.9)
    assert prepared.status["races"]["NC"]["polls_used"] == 1
    assert prepared.status["summary"]["rejected"]["not_d_vs_r"] == 1
    assert prepared.status["summary"]["rejected"]["not_two_party"] == 1


def test_no_race_polls_preserves_fundamentals_draws():
    draws = np.random.default_rng(1).normal(2.0, 4.0, 2000)
    updated = update_race_draws(draws, pd.DataFrame(), np.random.default_rng(2))
    np.testing.assert_array_equal(updated, draws)


def test_more_race_polls_tighten_posterior():
    rng = np.random.default_rng(3)
    prior = rng.normal(0.0, 8.0, 5000)
    one = pd.DataFrame([{
        "date": "2026-09-01", "margin": 3.0, "sample_size": 1000,
        "margin_of_error": 3.1, "population": "likely_voters", "partisan": None,
    }])
    many = pd.concat([one.assign(margin=value) for value in [2.5, 3.0, 3.5, 2.8]], ignore_index=True)
    posterior_one = update_race_draws(prior, one, np.random.default_rng(4))
    posterior_many = update_race_draws(prior, many, np.random.default_rng(4))
    assert np.std(posterior_many) < np.std(posterior_one) < np.std(prior)


def test_old_and_partisan_polls_keep_more_uncertainty():
    base = pd.Series({
        "date": "2026-10-25", "sample_size": 1000, "margin_of_error": 3.0,
        "population": "likely_voters", "partisan": None,
    })
    old_partisan = base.copy()
    old_partisan["date"] = "2026-04-01"
    old_partisan["partisan"] = "R"
    assert poll_observation_variance(old_partisan, date(2026, 11, 3)) > poll_observation_variance(
        base, date(2026, 11, 3)
    )


def test_outlier_is_downweighted_not_discarded():
    prior = np.random.default_rng(5).normal(0.0, 3.0, 5000)
    outlier = pd.DataFrame([{
        "date": "2026-10-20", "margin": 25.0, "sample_size": 1000,
        "margin_of_error": 3.0, "population": "likely_voters", "partisan": None,
    }])
    posterior = update_race_draws(prior, outlier, np.random.default_rng(6))
    assert 0 < np.mean(posterior) < 8


def test_candidate_registry_accepts_official_clerk_party_codes():
    registry = CandidateRegistry([{
        "name": "Example Member", "party": "R", "office": "H",
        "state": "AK", "district": "00", "incumbent_challenge": "I",
    }])
    incumbent = registry.by_race["AK-01"][0]
    assert incumbent.party == "R"
    assert incumbent.incumbent is True


def test_candidate_registry_does_not_erase_sourced_open_seat():
    registry = CandidateRegistry([{
        "name": "Retiring Member", "party": "R", "office": "H",
        "state": "AK", "district": "00", "incumbent_challenge": "I",
    }])
    frame = pd.DataFrame([{
        "district_id": "AK-01", "incumbent": "OPEN", "incumbent_party": "R",
        "open_seat": True, "pvi_source": "Cook PVI",
    }])
    updated = registry.update_fundamentals(frame, "district_id")
    assert updated.iloc[0]["open_seat"]
    assert updated.iloc[0]["incumbent_party"] == ""


def test_backtest_gate_and_interval_coverage_contract():
    rows = []
    for model, offset in (("v3", 0.1), ("v2", 0.2), ("fundamentals", 0.25), ("polls_only", 0.3)):
        for horizon in (120, 90, 60, 30, 14, 7):
            for idx, actual in enumerate((-4, -2, 2, 4)):
                rows.append({
                    "year": 2022, "horizon": horizon, "model": model,
                    "race_id": f"X-{idx}", "actual_margin": actual,
                    "pred_mean": actual * (1 - offset), "pred_std": 2.0,
                    "prob_dem": offset if actual < 0 else 1 - offset,
                })
    report = evaluate_predictions(pd.DataFrame(rows))
    assert report["status"] == "complete"
    assert report["race_polling_gate"]["status"] == "production"


def test_shared_national_shock_increases_seat_tail_variance():
    rng = np.random.default_rng(8)
    n_sims, n_races = 5000, 40
    local = rng.normal(0, 2, (n_sims, n_races))
    shared = rng.normal(0, 3, (n_sims, 1))
    correlated_seats = np.sum(shared + local > 0, axis=1)
    independent_seats = np.sum(rng.normal(0, np.sqrt(13), (n_sims, n_races)) > 0, axis=1)
    assert np.var(correlated_seats) > np.var(independent_seats) * 3


def test_public_schema_and_website_artifacts_match():
    required = {
        "prior_margin", "posterior_margin", "credible_interval_90", "prob_dem",
        "polling_adjustment", "polls_used", "latest_poll_date", "data_quality",
    }
    for filename, race_key, expected in (
        ("forecast.json", "districts", 435),
        ("senate_forecast.json", "races", 35),
    ):
        output = json.loads((PROJECT_ROOT / "outputs" / filename).read_text())
        website = json.loads((PROJECT_ROOT / "website" / filename).read_text())
        assert output == website
        assert output["metadata"]["model_version"] == "5.0.0"
        assert output["metadata"]["schema_version"] == "5.0.0"
        assert output["metadata"]["forecast_epoch"] == "2026-08-12"
        assert output["metadata"]["run_id"].startswith("forecast-")
        change = output["change_decomposition"]
        if change is not None:
            assert set(change) == {
                "probability_change", "median_seat_change", "national_update",
            }
            assert set(change["national_update"]) == {
                "fundamentals_prior", "poll_updated_current", "election_day_mean",
                "polling_contribution", "future_uncertainty_std",
            }
        assert output["backtest"]["race_polling_gate"]["status"] == "production"
        assert len(output[race_key]) == expected
        assert all(required <= set(race) for race in output[race_key])
        assert "generic_ballot_margin" not in output["summary"]
        assert "published_generic_ballot_margin" in output["summary"]
        assert "national_likelihood_margin" in output["summary"]
        assert "national_likelihood_date" in output["summary"]
        assert "poll_updated_current_margin" in output["summary"]
        assert "approval_rating" not in output["summary"]
        assert "net_approval" not in output["summary"]
        assert "data_through" not in output["metadata"]
        assert "data_through_definition" not in output["metadata"]
        assert "approval" not in output["polling"]
        assert "votehub_approval" not in output["metadata"]["source_freshness"]
        environment = output["national_environment"]
        assert set(environment) == {
            "fundamentals_prior", "published_sentiment", "polling_input",
            "poll_updated_current", "election_day", "economy",
        }
        assert len(environment["economy"]["components"]) == 5
        for component in environment["economy"]["components"].values():
            assert {"model_oriented_change", "weight", "contribution", "unit", "observation_date"} <= set(component)

    senate = json.loads((PROJECT_ROOT / "outputs" / "senate_forecast.json").read_text())
    assert senate["metadata"]["model_type"] == "senate_bayesian_external_average"
    assert senate["metadata"]["races_total"] == 35
    assert senate["summary"]["dem_not_up"] == 34
    assert senate["summary"]["rep_not_up"] == 31
    assert senate["summary"]["national_likelihood_margin"] == senate["summary"]["published_generic_ballot_margin"]
    assert senate["national_environment"]["polling_input"]["poll_rows"] == 1
    house = json.loads((PROJECT_ROOT / "outputs" / "forecast.json").read_text())
    assert house["national_environment"]["polling_input"]["poll_rows"] > 1


def test_public_methodology_explains_model_and_probability():
    page = (PROJECT_ROOT / "website" / "index.html").read_text()
    methodology = (PROJECT_ROOT / "METHODOLOGY.md").read_text()
    for phrase in (
        "One forecast, six auditable steps",
        "What the model is tracking nationally",
        "Five measured changes, fully exposed",
        "How to read the probability",
        "34 Democratic and 31 Republican not-up seats",
    ):
        assert phrase in page
    for phrase in (
        "The short version",
        "The national environment, metric by metric",
        "Chamber polling input",
        "Poll-updated current margin",
        "From race margins to chamber probabilities",
        "Glossary",
    ):
        assert phrase in methodology


def test_public_map_uses_openstreetmap_without_carto():
    app = (PROJECT_ROOT / "website" / "js" / "app.js").read_text()
    assert "https://tile.openstreetmap.org/{z}/{x}/{y}.png" in app
    assert "https://www.openstreetmap.org/copyright" in app
    assert "carto" not in app.lower()


def test_timeline_history_matches_latest_public_summaries():
    for filename, output_name, probability in (
        ("timeline.csv", "forecast.json", "prob_dem_majority"),
        ("senate_timeline.csv", "senate_forecast.json", "prob_dem_control"),
    ):
        timeline = pd.read_csv(
            PROJECT_ROOT / "outputs" / filename,
            dtype={"forecast_epoch": str, "schema_version": str, "model_version": str},
        )
        output = json.loads((PROJECT_ROOT / "outputs" / output_name).read_text())
        assert not timeline.empty
        assert timeline.iloc[0]["date"] == "2026-08-12"
        assert timeline["date"].is_unique
        assert timeline["date"].tolist() == sorted(timeline["date"].tolist())
        assert (timeline["forecast_epoch"] == "2026-08-12").all()
        assert (timeline["schema_version"] == "5.0.0").all()
        assert (timeline["model_version"] == "5.0.0").all()
        latest = timeline.iloc[-1]
        assert np.isclose(latest[probability], output["summary"][probability])
        assert latest["median_dem_seats"] == output["summary"]["median_dem_seats"]
        assert np.isclose(
            latest["published_generic_ballot"],
            output["summary"]["published_generic_ballot_margin"],
        )
        assert "approval" not in timeline.columns
        assert "national_env" not in timeline.columns
        assert np.isclose(
            latest["election_day_national_margin"],
            output["summary"]["election_day_national_margin"],
        )


def test_public_branding_and_copy_exclude_removed_concepts():
    public_text = "\n".join(
        path.read_text()
        for path in (
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "METHODOLOGY.md",
            PROJECT_ROOT / "MODEL_CARD.md",
            PROJECT_ROOT / "website" / "index.html",
            PROJECT_ROOT / "website" / "js" / "app.js",
        )
    ).lower()
    assert "grant's election forecast" in public_text
    assert "model data through" not in public_text
    assert "votehub" not in public_text
    assert "approval" not in public_text
    assert "forecast v4" not in public_text


def test_invalid_staged_chamber_leaves_public_bundle_untouched(tmp_path):
    stage = tmp_path / "stage"
    public = tmp_path / "public"
    stage.mkdir()
    public.mkdir()
    filenames = ["forecast.json", "senate_forecast.json", "timeline.csv"]
    (public / "forecast.json").write_text('{"old":"house"}')
    (public / "senate_forecast.json").write_text('{"old":"senate"}')
    (public / "timeline.csv").write_text("date,value\n2026-08-12,1\n")
    (stage / "forecast.json").write_text('{"new":"house"}')
    (stage / "senate_forecast.json").write_text("not valid json")
    (stage / "timeline.csv").write_text("date,value\n2026-08-12,2\n")
    try:
        _publish_staged_bundle(stage, public, filenames)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("invalid staged JSON must block the whole publication")
    assert json.loads((public / "forecast.json").read_text()) == {"old": "house"}
    assert json.loads((public / "senate_forecast.json").read_text()) == {"old": "senate"}
    assert "2026-08-12,1" in (public / "timeline.csv").read_text()


if __name__ == "__main__":
    for name, function in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"PASS {name}")
