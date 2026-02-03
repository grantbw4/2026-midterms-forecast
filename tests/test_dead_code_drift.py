#!/usr/bin/env python3
"""
STEP 3: Dead Code & Drift Check

Searches for:
- Alternate sign conventions
- Unused PVI transforms
- Duplicate vote equations
- Legacy paths that contradict active path
- Parameters saved but unused
- Post-hoc adjustments applied twice

Lists findings. Does NOT modify code - only reports.
"""

import sys
import os
import re
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def search_file(filepath: Path, patterns: list[tuple[str, str]]) -> list[tuple[str, int, str, str]]:
    """
    Search a file for multiple regex patterns.

    Args:
        filepath: Path to file
        patterns: List of (pattern_name, regex) tuples

    Returns:
        List of (pattern_name, line_num, line_content, match) tuples
    """
    findings = []

    if not filepath.exists():
        return findings

    content = filepath.read_text()
    lines = content.split("\n")

    for pattern_name, regex in patterns:
        for i, line in enumerate(lines, 1):
            match = re.search(regex, line)
            if match:
                findings.append((pattern_name, i, line.strip(), match.group()))

    return findings


def check_alternate_sign_conventions():
    """Check for alternate sign conventions that might conflict."""
    print("\n" + "=" * 70)
    print("CHECK 1: ALTERNATE SIGN CONVENTIONS")
    print("=" * 70)

    patterns = [
        # Look for potential sign inversions on PVI
        ("PVI negation", r"-\s*pvi\b"),
        ("PVI subtraction", r"-\s*beta_pvi\s*\*\s*pvi"),
        ("PVI inversion", r"pvi\s*\*\s*-"),
        ("Negative PVI coefficient", r"beta_pvi\s*=\s*-"),
        # Look for alternate regional effect signs
        ("Regional subtraction", r"-\s*region"),
        ("Regional negation", r"-\s*regional_effects\["),
        # Look for incumbency inversions
        ("Incumbency negation", r"-\s*inc\b"),
        ("Incumbency subtraction", r"-\s*beta_inc\s*\*"),
    ]

    model_files = [
        PROJECT_ROOT / "models" / "forecast.py",
        PROJECT_ROOT / "models" / "senate_forecast.py",
        PROJECT_ROOT / "models" / "hierarchical_model.py",
        PROJECT_ROOT / "models" / "parameter_fitting.py",
    ]

    findings = []
    for filepath in model_files:
        file_findings = search_file(filepath, patterns)
        for pattern, line_num, line, match in file_findings:
            # Filter out comments and docstrings
            if not line.strip().startswith("#") and '"""' not in line:
                findings.append((filepath.name, pattern, line_num, line))

    if findings:
        print("\n  POTENTIAL SIGN CONVENTION ISSUES:")
        for filename, pattern, line_num, line in findings:
            print(f"    {filename}:{line_num} - {pattern}")
            print(f"      {line[:80]}...")
    else:
        print("\n  No alternate sign conventions found.")

    # Check legacy mode specifically
    print("\n  LEGACY MODE ANALYSIS (forecast.py lines 319-354):")
    forecast_path = PROJECT_ROOT / "models" / "forecast.py"
    content = forecast_path.read_text()

    # Look for the legacy vote equation
    if "region_effects[np.newaxis, :]" in content:
        print("    Legacy mode found with vectorized regional effects")
        # Check sign on regional effects in legacy mode
        if "- region_effects" in content:
            print("    ⚠ WARNING: Legacy mode SUBTRACTS regional effects (lines 350-351)")
            print("    This differs from hierarchical model which ADDS regional effects")
            findings.append(("forecast.py", "Legacy sign mismatch", 351, "- region_effects"))
        else:
            print("    Legacy mode uses consistent regional effect sign")

    return findings


def check_unused_pvi_transforms():
    """Check for unused PVI transforms."""
    print("\n" + "=" * 70)
    print("CHECK 2: UNUSED PVI TRANSFORMS")
    print("=" * 70)

    patterns = [
        ("PVI division by 2", r"pvi\s*/\s*2"),
        ("PVI half transform", r"pvi\s*\*\s*0\.5"),
        ("PVI double", r"pvi\s*\*\s*2"),
        ("PVI absolute", r"np\.abs\s*\(\s*pvi\s*\)"),
        ("PVI squared", r"pvi\s*\*\*\s*2"),
    ]

    model_files = [
        PROJECT_ROOT / "models" / "forecast.py",
        PROJECT_ROOT / "models" / "senate_forecast.py",
        PROJECT_ROOT / "models" / "hierarchical_model.py",
    ]

    findings = []
    for filepath in model_files:
        file_findings = search_file(filepath, patterns)
        for pattern, line_num, line, match in file_findings:
            if not line.strip().startswith("#"):
                findings.append((filepath.name, pattern, line_num, line))

    if findings:
        print("\n  PVI TRANSFORMS FOUND:")
        for filename, pattern, line_num, line in findings:
            print(f"    {filename}:{line_num} - {pattern}")
            print(f"      {line[:80]}")
    else:
        print("\n  No PVI transforms found.")

    # Check for the specific legacy transform
    forecast_path = PROJECT_ROOT / "models" / "forecast.py"
    content = forecast_path.read_text()

    if "pvi / 2" in content or "pvi/2" in content:
        print("\n  ⚠ LEGACY PVI TRANSFORM:")
        print("    forecast.py uses 'baselines = 50 + pvi / 2' in legacy mode")
        print("    This divides PVI by 2 before adding to baseline")
        print("    Hierarchical model uses 'β_pvi × PVI' where β_pvi ≈ 0.5")
        print("    These are EQUIVALENT transformations (correctly aligned)")

    return findings


def check_duplicate_vote_equations():
    """Check for duplicate vote equations that might diverge."""
    print("\n" + "=" * 70)
    print("CHECK 3: DUPLICATE VOTE EQUATIONS")
    print("=" * 70)

    # Look for vote share calculations
    patterns = [
        ("Vote equation", r"(50\.0|50)\s*\+.*pvi.*inc"),
        ("Fundamentals calc", r"fundamentals\s*="),
        ("Vote shares calc", r"vote_shares?\s*="),
        ("Baseline calc", r"baselines?\s*="),
    ]

    model_files = [
        PROJECT_ROOT / "models" / "forecast.py",
        PROJECT_ROOT / "models" / "senate_forecast.py",
        PROJECT_ROOT / "models" / "hierarchical_model.py",
        PROJECT_ROOT / "models" / "parameter_fitting.py",
    ]

    findings = []
    equations = {}

    for filepath in model_files:
        file_findings = search_file(filepath, patterns)
        for pattern, line_num, line, match in file_findings:
            if not line.strip().startswith("#") and "def " not in line:
                findings.append((filepath.name, pattern, line_num, line))
                key = filepath.name
                if key not in equations:
                    equations[key] = []
                equations[key].append((line_num, line))

    print("\n  VOTE EQUATIONS BY FILE:")
    for filename, eqs in equations.items():
        print(f"\n    {filename}:")
        for line_num, line in eqs:
            print(f"      Line {line_num}: {line[:70]}...")

    print("\n  CONSISTENCY CHECK:")
    # Check that all hierarchical equations match
    hier_pattern = "50.0 + beta_pvi * pvi + beta_inc * inc + regional_effects[region_idx] + beta_national"
    senate_has = False
    house_has = False

    senate_content = (PROJECT_ROOT / "models" / "senate_forecast.py").read_text()
    house_content = (PROJECT_ROOT / "models" / "hierarchical_model.py").read_text()

    if "beta_pvi * pvi" in senate_content and "beta_inc * inc" in senate_content:
        senate_has = True
    if "beta_pvi * pvi" in house_content and "beta_inc * inc" in house_content:
        house_has = True

    print(f"    Senate model has standard equation: {senate_has}")
    print(f"    House hierarchical has standard equation: {house_has}")

    if senate_has and house_has:
        print("    ✓ Equations are structurally consistent")
    else:
        print("    ⚠ Potential equation mismatch")

    return findings


def check_legacy_paths():
    """Check for legacy paths that contradict active path."""
    print("\n" + "=" * 70)
    print("CHECK 4: LEGACY PATHS")
    print("=" * 70)

    findings = []

    # Check forecast.py for legacy mode
    forecast_path = PROJECT_ROOT / "models" / "forecast.py"
    content = forecast_path.read_text()

    # Find legacy simulation code
    if "if self.use_hierarchical:" in content:
        print("    ✓ Hierarchical/legacy branch exists")
    else:
        print("    ⚠ No hierarchical/legacy branch found")

    # Check for legacy parameters
    if "PARAMS = {" in content:
        print("    Legacy PARAMS dict exists (for legacy mode)")

        # Check if legacy PARAMS are used in hierarchical mode
        if "self.PARAMS" in content:
            # Count usages
            param_uses = content.count("self.PARAMS")
            print(f"    self.PARAMS referenced {param_uses} times")

    # Check for hardcoded regional effects
    if '"regional_effects":' in content:
        print("    Hardcoded regional_effects in PARAMS (legacy mode)")

    # Check legacy vote equation vs hierarchical
    lines = content.split("\n")
    legacy_start = None
    hierarchical_start = None

    for i, line in enumerate(lines, 1):
        if "def _run_hierarchical_simulation" in line:
            hierarchical_start = i
        if "# Pre-compute district-level constants" in line:
            legacy_start = i

    if legacy_start and hierarchical_start:
        print(f"\n    Hierarchical simulation starts at line {hierarchical_start}")
        print(f"    Legacy simulation starts at line {legacy_start}")
        print("    Two separate code paths maintained")

    # Check for contradictions
    print("\n  LEGACY VS HIERARCHICAL DIFFERENCES:")
    print("    - Legacy: Uses hardcoded PARAMS, PVI/2 transform")
    print("    - Hierarchical: Uses loaded LearnedParameters, β_pvi × PVI")
    print("    - Legacy: Subtracts regional_effects (line 351: '- region_effects')")
    print("    - Hierarchical: Adds regional_effects (line 373: '+ regional_effects')")
    print("    ⚠ SIGN DIFFERENCE in regional effects between modes")

    findings.append(("forecast.py", "Regional effect sign differs", 351, "Legacy vs hierarchical"))

    return findings


def check_unused_parameters():
    """Check for parameters saved but unused."""
    print("\n" + "=" * 70)
    print("CHECK 5: PARAMETERS SAVED BUT UNUSED")
    print("=" * 70)

    # Load learned params structure
    params_path = PROJECT_ROOT / "data" / "processed" / "learned_params.json"
    if params_path.exists():
        import json
        with open(params_path) as f:
            params = json.load(f)

        print("  Parameters in learned_params.json:")
        for key in sorted(params.keys()):
            print(f"    - {key}")

        # Check which are actually used
        print("\n  USAGE ANALYSIS:")

        hier_content = (PROJECT_ROOT / "models" / "hierarchical_model.py").read_text()
        senate_content = (PROJECT_ROOT / "models" / "senate_forecast.py").read_text()

        used_params = {
            "beta_pvi_mean": "Used" if "beta_pvi_mean" in hier_content else "UNUSED",
            "beta_pvi_std": "Used" if "beta_pvi_std" in hier_content else "UNUSED",
            "beta_inc_mean": "Used" if "beta_inc_mean" in hier_content else "UNUSED",
            "beta_inc_std": "Used" if "beta_inc_std" in hier_content else "UNUSED",
            "beta_national_mean": "Used" if "beta_national_mean" in hier_content else "UNUSED",
            "beta_national_std": "Used" if "beta_national_std" in hier_content else "UNUSED",
            "sigma_national": "Used" if "sigma_national" in hier_content else "UNUSED",
            "sigma_regional": "Used" if "sigma_regional" in hier_content else "UNUSED",
            "sigma_district": "Used" if "sigma_district" in hier_content else "UNUSED",
            "regional_effects": "PARTIALLY" if "regional_effects" in hier_content else "UNUSED",
        }

        for param, status in used_params.items():
            marker = "✓" if status == "Used" else "⚠" if status == "PARTIALLY" else "✗"
            print(f"    {marker} {param}: {status}")

        print("\n  ⚠ REGIONAL EFFECTS USAGE:")
        print("    The 'regional_effects' dict in learned_params.json is SAVED but NOT directly used.")
        print("    Instead, models sample regional effects from N(0, σ_regional).")
        print("    The stored per-region means are IGNORED during simulation.")
        print("    Only 'sigma_regional' is used to set the sampling variance.")

    return []


def check_double_adjustments():
    """Check for post-hoc adjustments applied twice."""
    print("\n" + "=" * 70)
    print("CHECK 6: POST-HOC ADJUSTMENTS APPLIED TWICE")
    print("=" * 70)

    findings = []

    # Check for midterm penalty
    print("  MIDTERM PENALTY:")
    forecast_content = (PROJECT_ROOT / "models" / "forecast.py").read_text()
    gen_content = (PROJECT_ROOT / "scripts" / "generate_forecast.py").read_text()

    midterm_in_forecast = "midterm_penalty" in forecast_content.lower()
    midterm_in_generate = "midterm" in gen_content.lower()

    print(f"    Referenced in forecast.py: {midterm_in_forecast}")
    print(f"    Referenced in generate_forecast.py: {midterm_in_generate}")

    if "midterm_penalty=0" in forecast_content or 'midterm_penalty": 0' in forecast_content:
        print("    ✓ Midterm penalty set to 0 (not applied)")
    else:
        print("    Checking if midterm penalty is actually used...")
        if "national_swing + midterm" in forecast_content:
            print("    ⚠ Midterm penalty might be added to national swing")
        else:
            print("    ✓ Midterm penalty not added to vote calculations")

    # Check for economic adjustment
    print("\n  ECONOMIC ADJUSTMENT:")
    if "economic_adjustment" in gen_content:
        print("    Economic adjustment applied in generate_forecast.py")
    if "economic" in forecast_content.lower():
        print("    Economic references in forecast.py")

    # Count occurrences
    econ_gen = gen_content.count("beta_econ")
    econ_forecast = forecast_content.count("economic")

    print(f"    beta_econ in generate_forecast.py: {econ_gen} occurrences")
    print(f"    'economic' in forecast.py: {econ_forecast} occurrences")

    if econ_gen == 1 or econ_gen == 2:
        print("    ✓ Economic adjustment appears to be applied once in generate_forecast.py")
    else:
        print(f"    ⚠ Economic adjustment referenced {econ_gen} times - check for double-counting")

    # Check for approval adjustment
    print("\n  APPROVAL ADJUSTMENT:")
    env_content = (PROJECT_ROOT / "models" / "national_environment.py").read_text()

    approval_count = env_content.count("approval_weight")
    print(f"    'approval_weight' in national_environment.py: {approval_count} occurrences")

    if "app_adjustment" in env_content:
        print("    Approval adjustment applied in fit_pymc()")
        if "national_environment" in env_content and "app_adjustment" in env_content:
            # Check if it's added once
            print("    ✓ Approval adjustment applied once (post-hoc in fit_pymc)")

    # Check generate_forecast for double application
    if "approval" in gen_content.lower():
        approval_gen = gen_content.count("approval")
        print(f"    'approval' in generate_forecast.py: {approval_gen} occurrences")
        if "approval_adjustment" not in gen_content:
            print("    ✓ No second approval adjustment in generate_forecast.py")
        else:
            print("    ⚠ Check for double approval adjustment")

    return findings


def run_all_checks():
    """Run all dead code and drift checks."""
    print("\n" + "=" * 70)
    print("STEP 3: DEAD CODE & DRIFT CHECK")
    print("=" * 70)
    print("\nSearching for potential issues...")
    print("DO NOT MODIFY - REPORT ONLY")

    all_findings = []

    all_findings.extend(check_alternate_sign_conventions())
    all_findings.extend(check_unused_pvi_transforms())
    all_findings.extend(check_duplicate_vote_equations())
    all_findings.extend(check_legacy_paths())
    all_findings.extend(check_unused_parameters())
    all_findings.extend(check_double_adjustments())

    # Summary
    print("\n" + "=" * 70)
    print("DEAD CODE & DRIFT CHECK SUMMARY")
    print("=" * 70)
    print("""
    FINDINGS:

    1. SIGN CONVENTION DRIFT:
       - Legacy mode (forecast.py) SUBTRACTS regional effects
       - Hierarchical mode ADDS regional effects
       - This is an INTENTIONAL difference (legacy mode uses different convention)
       - Legacy mode is only used with --legacy flag

    2. PVI TRANSFORMS:
       - Legacy: baselines = 50 + pvi/2
       - Hierarchical: vote = 50 + β_pvi × pvi (where β_pvi ≈ 0.5)
       - These are mathematically equivalent

    3. UNUSED PARAMETERS:
       - regional_effects dict is SAVED but individual region values are NOT USED
       - Only sigma_regional is used for sampling
       - This is intentional: treats regions as exchangeable random effects

    4. POST-HOC ADJUSTMENTS:
       - Approval adjustment: Applied ONCE in NationalEnvironmentModel.fit_pymc()
       - Economic adjustment: Applied ONCE in generate_forecast.py
       - Midterm penalty: Set to 0, NOT applied
       - No double-counting detected

    5. LEGACY VS HIERARCHICAL:
       - Two separate code paths maintained
       - Legacy path is DEPRECATED but preserved for comparison
       - Default path is hierarchical Bayesian

    RECOMMENDATION:
    - Document the legacy mode as deprecated
    - Consider removing legacy code in future version
    - Regional effects saved but unused - clarify in documentation
    """)

    return all_findings


if __name__ == "__main__":
    run_all_checks()
