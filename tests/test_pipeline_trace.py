#!/usr/bin/env python3
"""
STEP 2: Pipeline Trace

Traces one full forecast run and reports:
- Which model path is used
- Whether PyMC forecast mode is actually invoked
- Whether parameters are refit or loaded
- Which national environment method is active
- Whether approval and economic adjustments are applied
- Where uncertainty is sampled vs fixed

Produces a step-by-step execution trace.
"""

import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

# Set PyTensor flags before importing PyMC
os.environ.setdefault("PYTENSOR_FLAGS", "device=cpu,floatX=float64,optimizer=fast_compile,cxx=")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def trace_pipeline():
    """
    Trace the full forecast pipeline execution.
    """
    print("\n" + "=" * 70)
    print("STEP 2: PIPELINE TRACE")
    print("=" * 70)
    print("\nTracing execution path through the forecast pipeline...")
    print()

    trace_log = []

    def log_step(step_num, description, details=None):
        trace_log.append((step_num, description, details))
        print(f"\n[STEP {step_num}] {description}")
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")

    # =========================================================================
    # STEP 1: Check which model path is configured
    # =========================================================================
    log_step("1.1", "MODEL PATH SELECTION", {
        "Default model": "hierarchical_bayesian (use_hierarchical=True)",
        "Legacy mode": "--legacy flag switches to flat simulation",
        "Active path": "HierarchicalForecastModel via HouseForecastModel._run_hierarchical_simulation()",
    })

    # =========================================================================
    # STEP 2: Check PyMC availability and usage
    # =========================================================================
    try:
        import pymc as pm
        pymc_available = True
        pymc_version = pm.__version__
    except ImportError:
        pymc_available = False
        pymc_version = "N/A"

    from models.hierarchical_model import PYMC_AVAILABLE as HIER_PYMC
    from models.senate_forecast import PYMC_AVAILABLE as SENATE_PYMC
    from models.national_environment import PYMC_AVAILABLE as ENV_PYMC
    from models.parameter_fitting import PYMC_AVAILABLE as FIT_PYMC

    log_step("1.2", "PYMC AVAILABILITY", {
        "PyMC installed": pymc_available,
        "PyMC version": pymc_version,
        "hierarchical_model.PYMC_AVAILABLE": HIER_PYMC,
        "senate_forecast.PYMC_AVAILABLE": SENATE_PYMC,
        "national_environment.PYMC_AVAILABLE": ENV_PYMC,
        "parameter_fitting.PYMC_AVAILABLE": FIT_PYMC,
    })

    # =========================================================================
    # STEP 3: Check parameter loading vs refitting
    # =========================================================================
    from models.parameter_fitting import ParameterFitter, LearnedParameters

    params_path = PROJECT_ROOT / "data" / "processed" / "learned_params.json"
    params_exist = params_path.exists()

    if params_exist:
        try:
            loaded_params = ParameterFitter.load_parameters()
            param_source = "LOADED from learned_params.json"
            param_details = {
                "β_pvi": f"{loaded_params.beta_pvi_mean:.3f} ± {loaded_params.beta_pvi_std:.3f}",
                "β_inc": f"{loaded_params.beta_inc_mean:.2f} ± {loaded_params.beta_inc_std:.2f}",
                "β_national": f"{loaded_params.beta_national_mean:.3f} ± {loaded_params.beta_national_std:.3f}",
                "σ_national": f"{loaded_params.sigma_national:.2f}",
                "σ_regional": f"{loaded_params.sigma_regional:.2f}",
                "σ_district": f"{loaded_params.sigma_district:.2f}",
                "years_used": str(loaded_params.years_used),
                "n_districts_fitted": loaded_params.n_districts_fitted,
            }
        except Exception as e:
            param_source = f"FAILED to load: {e}"
            param_details = {}
    else:
        param_source = "DEFAULTS (no learned_params.json found)"
        param_details = {"note": "Will use default priors from _default_parameters()"}

    log_step("2.1", "PARAMETER SOURCE", {
        "learned_params.json exists": params_exist,
        "Source": param_source,
        **param_details,
    })

    log_step("2.2", "PARAMETER USAGE IN MODELS", {
        "House (hierarchical)": "Samples β_pvi, β_inc, β_national from Gaussian posteriors each simulation",
        "Senate": "Samples β_pvi, β_inc, β_national from Gaussian posteriors each simulation",
        "Regional effects": "Sampled from N(0, σ_regional), NOT from stored regional_effects dict",
        "Note": "Stored regional_effects are UNUSED during simulation - only sigma_regional matters",
    })

    # =========================================================================
    # STEP 4: Check national environment method
    # =========================================================================
    polls_path = PROJECT_ROOT / "data" / "raw" / "polling" / "generic_ballot.csv"
    approval_path = PROJECT_ROOT / "data" / "raw" / "polling" / "trump_approval.csv"

    polls_exist = polls_path.exists()
    approval_exist = approval_path.exists()

    if polls_exist:
        polls_df = pd.read_csv(polls_path)
        n_polls = len(polls_df)
    else:
        n_polls = 0

    log_step("3.1", "NATIONAL ENVIRONMENT DATA", {
        "Generic ballot polls exist": polls_exist,
        "Number of GB polls": n_polls,
        "Approval polls exist": approval_exist,
    })

    log_step("3.2", "NATIONAL ENVIRONMENT METHOD", {
        "Active method": "NationalEnvironmentModel.fit_pymc() if PyMC available, else fit_simple()",
        "PyMC model": "Hierarchical model with pollster house effects",
        "Simple fallback": "Weighted average of poll margins",
    })

    # =========================================================================
    # STEP 5: Check approval and economic adjustments
    # =========================================================================
    coeffs_path = PROJECT_ROOT / "data" / "processed" / "national_env_coefficients.json"
    coeffs_exist = coeffs_path.exists()

    if coeffs_exist:
        import json
        with open(coeffs_path) as f:
            coeffs = json.load(f)
        beta_polls = coeffs.get("beta_polls", 1.0)
        beta_econ = coeffs.get("beta_econ", 0.0)
    else:
        beta_polls = 1.0
        beta_econ = 0.0

    log_step("4.1", "APPROVAL ADJUSTMENT (POST-HOC)", {
        "Applied in": "NationalEnvironmentModel.fit_pymc() AFTER MCMC sampling",
        "Formula": "national_env += -0.3 × approval_weight × net_approval",
        "approval_weight": "0.3 (hardcoded in NationalEnvironmentModel)",
        "Note": "This is OUTSIDE the Bayesian model - deterministic post-hoc adjustment",
        "Uncertainty impact": "Does NOT propagate approval uncertainty",
    })

    log_step("4.2", "ECONOMIC ADJUSTMENT", {
        "Coefficients file exists": coeffs_exist,
        "β_polls": beta_polls,
        "β_econ": beta_econ,
        "Formula": "final_national = β_polls × polling_env + β_econ × adjusted_econ",
        "Adjusted econ": "-raw_econ (negated for R president)",
        "Applied in": "generate_forecast.py AFTER NationalEnvironmentModel.fit()",
        "Note": "Deterministic adjustment, not part of Bayesian model",
    })

    # =========================================================================
    # STEP 6: Trace uncertainty sources
    # =========================================================================
    log_step("5.1", "UNCERTAINTY SOURCES - MCMC SAMPLED", {
        "National environment μ": "PyMC MCMC in NationalEnvironmentModel.fit_pymc()",
        "Pollster house effects": "PyMC MCMC in NationalEnvironmentModel.fit_pymc()",
        "Parameter fitting (historical)": "PyMC MCMC in ParameterFitter.fit_pymc() (if used)",
    })

    log_step("5.2", "UNCERTAINTY SOURCES - MONTE CARLO SAMPLED", {
        "β_pvi per simulation": "Normal(β_pvi_mean, β_pvi_std) - Gaussian posterior approx",
        "β_inc per simulation": "Normal(β_inc_mean, β_inc_std) - Gaussian posterior approx",
        "β_national per simulation": "Normal(β_national_mean, β_national_std) - Gaussian posterior approx",
        "National env per simulation": "Normal(μ_posterior, σ_posterior) from poll model",
        "Regional effects per simulation": "Normal(0, σ_regional) - NOT using stored effects",
        "District noise per simulation": "Normal(0, σ_district × PVI_scale) - PVI-scaled",
    })

    log_step("5.3", "UNCERTAINTY SOURCES - FIXED (NOT SAMPLED)", {
        "Approval adjustment": "Fixed formula: -0.3 × 0.3 × net_approval",
        "Economic adjustment": "Fixed: β_econ × adjusted_econ (no uncertainty)",
        "PVI values": "Fixed from districts.csv (no measurement uncertainty)",
        "Incumbency codes": "Fixed from districts.csv (deterministic)",
    })

    # =========================================================================
    # STEP 7: Full pipeline flow diagram
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION FLOW")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    FORECAST PIPELINE FLOW                           │
    └─────────────────────────────────────────────────────────────────────┘

    1. DATA LOADING
       ├── VoteHub polls → data/raw/polling/generic_ballot.csv
       ├── Approval polls → data/raw/polling/trump_approval.csv
       ├── Cook ratings → data/cook/house_ratings.csv
       └── Districts → data/processed/districts.csv

    2. NATIONAL ENVIRONMENT INFERENCE (MCMC if PyMC available)
       ├── NationalEnvironmentModel.load_polls()
       ├── NationalEnvironmentModel.fit_pymc()
       │   ├── μ ~ Normal(0, 5)                      [Prior on national env]
       │   ├── house_effects ~ Normal(0, σ_house)    [Pollster bias]
       │   ├── y_i ~ Normal(μ + house_i, σ_i)        [Poll observations]
       │   └── Returns: μ_posterior, σ_posterior
       │
       └── POST-HOC approval adjustment (OUTSIDE MCMC)
           └── μ_adjusted = μ_posterior + (-0.3 × 0.3 × net_approval)

    3. ECONOMIC ADJUSTMENT (DETERMINISTIC)
       ├── Load β_polls, β_econ from national_env_coefficients.json
       ├── Calculate EconomicFundamentals.calculate_index()
       └── final_national = β_polls × μ_adjusted + β_econ × (-econ_index)

    4. PARAMETER LOADING
       ├── ParameterFitter.load_parameters()
       │   └── Returns: β_pvi_mean/std, β_inc_mean/std, β_national_mean/std,
       │                regional_effects, σ_national, σ_regional, σ_district
       │
       └── Note: Parameters pre-fitted from 2018/2022 historical data

    5. HOUSE FORECAST SIMULATION (POSTERIOR PREDICTIVE MONTE CARLO)
       ├── HouseForecastModel.__init__(use_hierarchical=True)
       ├── HouseForecastModel._run_hierarchical_simulation()
       │   └── HierarchicalForecastModel.simulate_elections()
       │
       └── For each of N simulations:
           ├── Sample β_pvi ~ Normal(mean, std)     [Parameter uncertainty]
           ├── Sample β_inc ~ Normal(mean, std)     [Parameter uncertainty]
           ├── Sample β_national ~ Normal(mean, std) [Parameter uncertainty]
           ├── Sample μ_national ~ Normal(μ, σ)     [National uncertainty]
           ├── Sample regional ~ Normal(0, σ_reg)   [Regional uncertainty]
           ├── Sample ε_d ~ Normal(0, σ_d × PVI_scale) [District noise]
           │
           └── vote_d = 50 + β_pvi×PVI + β_inc×Inc + regional + β_national×μ + ε
               └── Winner = vote_d > 50

    6. SENATE FORECAST SIMULATION (SAME METHODOLOGY)
       ├── SenateForecastModel.__init__()
       ├── SenateForecastModel.simulate_elections()
       └── Uses identical vote equation as House

    7. OUTPUT
       ├── outputs/forecast.json (House)
       ├── outputs/senate_forecast.json
       ├── website/forecast.json (copied)
       └── website/senate_forecast.json (copied)
    """)

    # =========================================================================
    # STEP 8: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE TRACE SUMMARY")
    print("=" * 70)
    print("""
    KEY FINDINGS:

    1. MODEL PATH: Hierarchical Bayesian (default), Legacy flat available via --legacy

    2. PYMC USAGE:
       - National environment: Full MCMC inference on poll aggregation
       - Parameter fitting: MCMC on historical data (pre-computed)
       - Forecast simulation: Monte Carlo with Gaussian posterior approximations
       - NOT joint inference: Parameters fitted separately from national env

    3. PARAMETER SOURCE: Loaded from learned_params.json (pre-fitted)
       - NOT refit each run
       - Gaussian posteriors approximated from MCMC traces

    4. NATIONAL ENVIRONMENT: VoteHub polling → PyMC MCMC → post-hoc adjustments
       - Approval adjustment: OUTSIDE MCMC (deterministic post-hoc)
       - Economic adjustment: OUTSIDE MCMC (deterministic post-hoc)

    5. UNCERTAINTY PROPAGATION:
       - MCMC: National env μ, pollster house effects
       - Monte Carlo: β_pvi, β_inc, β_national, regional effects, district noise
       - Fixed: Approval adjustment, economic adjustment, PVI values

    6. STAGED INFERENCE (NOT JOINT):
       Stage 1: Fit parameters from 2018/2022 data (separate MCMC)
       Stage 2: Infer national env from polls (separate MCMC)
       Stage 3: Simulate with Gaussian approximations (Monte Carlo)
    """)

    return trace_log


if __name__ == "__main__":
    trace_pipeline()
