#!/usr/bin/env python3
"""
Validation tests for model sign conventions.

These tests verify that the model's fundamental assumptions are correct:
1. Increasing Republican PVI lowers Democratic vote share
2. Democratic incumbency increases Democratic vote share
3. Positive national environment (D advantage) increases Democratic vote share

If any of these tests fail, the model has a critical sign error.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.hierarchical_model import HierarchicalForecastModel, NationalPosterior
from models.senate_forecast import SenateForecastModel
from models.parameter_fitting import LearnedParameters, REGIONS


def create_test_params() -> LearnedParameters:
    """Create test parameters with known values."""
    return LearnedParameters(
        beta_pvi_mean=0.5,  # Each point of D-lean PVI adds 0.5 to Dem vote share
        beta_pvi_std=0.01,  # Small std for deterministic testing
        beta_inc_mean=3.0,  # D incumbent adds 3 points
        beta_inc_std=0.01,
        beta_national_mean=1.0,  # Full national swing
        beta_national_std=0.01,
        regional_effects={r: 0.0 for r in REGIONS},
        regional_effects_std={r: 0.01 for r in REGIONS},
        sigma_national=0.01,  # Minimal noise for testing
        sigma_regional=0.01,
        sigma_district=0.01,
        n_districts_fitted=100,
        years_used=[2018, 2022],
        rmse=4.0,
        r_squared=0.94,
    )


def create_test_districts(pvi_values: list, inc_parties: list) -> pd.DataFrame:
    """Create test district DataFrame."""
    n = len(pvi_values)
    return pd.DataFrame({
        "district_id": [f"TEST-{i:02d}" for i in range(n)],
        "state": ["CA"] * n,
        "district_number": list(range(n)),
        "pvi": pvi_values,
        "incumbent": ["Test Inc"] * n,
        "incumbent_party": inc_parties,
        "region": ["Pacific"] * n,
        "open_seat": [False] * n,
    })


def test_pvi_direction():
    """
    Test: Increasing Republican PVI (more negative) should LOWER Democratic vote share.

    PVI Sign Convention: positive = D-leaning, negative = R-leaning
    """
    print("\n" + "=" * 60)
    print("TEST: PVI Direction")
    print("=" * 60)

    params = create_test_params()

    # Create two districts: one D+10, one R+10
    districts = create_test_districts(
        pvi_values=[10.0, -10.0],  # D+10 and R+10
        inc_parties=["D", "D"],  # Same incumbency to isolate PVI effect
    )

    national = NationalPosterior(mean=0.0, std=0.01)  # Neutral environment

    model = HierarchicalForecastModel(
        districts_df=districts,
        national_posterior=national,
        learned_params=params,
        n_simulations=1000,
        random_seed=42,
    )

    result = model.simulate_elections()

    d_lean_vote = result.mean_vote_share[0]  # D+10 district
    r_lean_vote = result.mean_vote_share[1]  # R+10 district

    print(f"D+10 district mean Dem vote share: {d_lean_vote:.1f}%")
    print(f"R+10 district mean Dem vote share: {r_lean_vote:.1f}%")
    print(f"Difference (D+10 - R+10): {d_lean_vote - r_lean_vote:.1f} points")

    # D+10 should have HIGHER Dem vote share than R+10
    assert d_lean_vote > r_lean_vote, \
        f"CRITICAL ERROR: D+10 district ({d_lean_vote:.1f}%) should have higher Dem " \
        f"vote share than R+10 district ({r_lean_vote:.1f}%)"

    # The difference should be approximately 2 × β_pvi × 10 = 10 points
    expected_diff = 2 * params.beta_pvi_mean * 10  # 10 points
    actual_diff = d_lean_vote - r_lean_vote

    assert abs(actual_diff - expected_diff) < 2.0, \
        f"PVI effect magnitude wrong: expected ~{expected_diff:.1f} points difference, " \
        f"got {actual_diff:.1f}"

    print("✓ PASSED: PVI direction is correct")
    return True


def test_incumbency_direction():
    """
    Test: Democratic incumbent should INCREASE Democratic vote share.

    Incumbency Code: D = +1, R = -1, Open = 0
    """
    print("\n" + "=" * 60)
    print("TEST: Incumbency Direction")
    print("=" * 60)

    params = create_test_params()

    # Create three districts: D incumbent, R incumbent, Open
    districts = create_test_districts(
        pvi_values=[0.0, 0.0, 0.0],  # All neutral PVI
        inc_parties=["D", "R", None],
    )
    districts.loc[2, "open_seat"] = True

    national = NationalPosterior(mean=0.0, std=0.01)

    model = HierarchicalForecastModel(
        districts_df=districts,
        national_posterior=national,
        learned_params=params,
        n_simulations=1000,
        random_seed=42,
    )

    result = model.simulate_elections()

    d_inc_vote = result.mean_vote_share[0]  # D incumbent
    r_inc_vote = result.mean_vote_share[1]  # R incumbent
    open_vote = result.mean_vote_share[2]   # Open seat

    print(f"D incumbent district: {d_inc_vote:.1f}%")
    print(f"R incumbent district: {r_inc_vote:.1f}%")
    print(f"Open seat district: {open_vote:.1f}%")

    # D incumbent should have HIGHEST Dem vote share
    assert d_inc_vote > open_vote, \
        f"CRITICAL ERROR: D incumbent ({d_inc_vote:.1f}%) should have higher Dem " \
        f"vote share than open seat ({open_vote:.1f}%)"

    assert open_vote > r_inc_vote, \
        f"CRITICAL ERROR: Open seat ({open_vote:.1f}%) should have higher Dem " \
        f"vote share than R incumbent ({r_inc_vote:.1f}%)"

    # Check magnitude: D vs R should differ by 2 × β_inc = 6 points
    expected_inc_diff = 2 * params.beta_inc_mean  # 6 points
    actual_inc_diff = d_inc_vote - r_inc_vote

    assert abs(actual_inc_diff - expected_inc_diff) < 2.0, \
        f"Incumbency effect magnitude wrong: expected ~{expected_inc_diff:.1f} points, " \
        f"got {actual_inc_diff:.1f}"

    print("✓ PASSED: Incumbency direction is correct")
    return True


def test_national_environment_direction():
    """
    Test: Positive national environment (D advantage) should INCREASE Democratic vote share.
    """
    print("\n" + "=" * 60)
    print("TEST: National Environment Direction")
    print("=" * 60)

    params = create_test_params()

    # Create identical district
    districts = create_test_districts(
        pvi_values=[0.0],
        inc_parties=[None],
    )
    districts.loc[0, "open_seat"] = True

    # Test with D+10 environment
    national_d = NationalPosterior(mean=10.0, std=0.01)
    model_d = HierarchicalForecastModel(
        districts_df=districts.copy(),
        national_posterior=national_d,
        learned_params=params,
        n_simulations=1000,
        random_seed=42,
    )
    result_d = model_d.simulate_elections()
    vote_d_env = result_d.mean_vote_share[0]

    # Test with R+10 environment
    national_r = NationalPosterior(mean=-10.0, std=0.01)
    model_r = HierarchicalForecastModel(
        districts_df=districts.copy(),
        national_posterior=national_r,
        learned_params=params,
        n_simulations=1000,
        random_seed=42,
    )
    result_r = model_r.simulate_elections()
    vote_r_env = result_r.mean_vote_share[0]

    print(f"D+10 national environment: {vote_d_env:.1f}% Dem vote")
    print(f"R+10 national environment: {vote_r_env:.1f}% Dem vote")
    print(f"Difference: {vote_d_env - vote_r_env:.1f} points")

    # D+10 environment should have HIGHER Dem vote share
    assert vote_d_env > vote_r_env, \
        f"CRITICAL ERROR: D+10 environment ({vote_d_env:.1f}%) should have higher Dem " \
        f"vote share than R+10 environment ({vote_r_env:.1f}%)"

    print("✓ PASSED: National environment direction is correct")
    return True


def test_senate_pvi_direction():
    """
    Test: Same PVI direction test for Senate model.
    """
    print("\n" + "=" * 60)
    print("TEST: Senate PVI Direction")
    print("=" * 60)

    params = create_test_params()

    # Test with D+10 environment
    model = SenateForecastModel(
        national_environment=0.0,  # Neutral
        n_simulations=1000,
        random_seed=42,
        learned_params=params,
        use_cook_data=False,  # Use hardcoded races
    )

    model.simulate_elections()

    # Find a safe D race (high positive PVI) and safe R race (high negative PVI)
    safe_d_race = None
    safe_r_race = None

    for forecast in model.forecasts:
        if forecast.pvi >= 10 and safe_d_race is None:
            safe_d_race = forecast
        if forecast.pvi <= -10 and safe_r_race is None:
            safe_r_race = forecast

    if safe_d_race and safe_r_race:
        print(f"D-lean state ({safe_d_race.state}, PVI={safe_d_race.pvi:+.0f}): "
              f"{safe_d_race.prob_dem:.1%} Dem win prob")
        print(f"R-lean state ({safe_r_race.state}, PVI={safe_r_race.pvi:+.0f}): "
              f"{safe_r_race.prob_dem:.1%} Dem win prob")

        assert safe_d_race.prob_dem > safe_r_race.prob_dem, \
            f"CRITICAL ERROR: D-lean state should have higher Dem win probability"

        print("✓ PASSED: Senate PVI direction is correct")
    else:
        print("⚠ WARNING: Could not find suitable test races")

    return True


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("SIGN CONVENTION VALIDATION TESTS")
    print("=" * 60)
    print("These tests verify the model's fundamental assumptions.")
    print("If any test fails, there is a critical sign error in the model.")

    all_passed = True

    try:
        test_pvi_direction()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        all_passed = False

    try:
        test_incumbency_direction()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        all_passed = False

    try:
        test_national_environment_direction()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        all_passed = False

    try:
        test_senate_pvi_direction()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("Sign conventions are correct.")
    else:
        print("SOME TESTS FAILED ✗")
        print("CRITICAL: Sign errors detected. Do not use model until fixed.")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
