# 2026 Midterms Forecast Methodology

**Version 2.0.0** | Updated: 2026-02-02

This document describes the methodology of the 2026 House and Senate forecast model as implemented in the codebase. All claims are verified against executable code.

---

## Model Overview

This is a **staged Bayesian inference model** that produces probabilistic forecasts for the 2026 U.S. House and Senate elections. The model uses three sequential inference stages, not joint Bayesian inference.

### Key Characteristics

- **Poll-anchored**: The generic ballot is the single driving variable
- **Staged inference**: Parameters, national environment, and forecasts are estimated separately
- **Posterior predictive simulation**: Monte Carlo sampling with Gaussian posterior approximations
- **Uncertainty propagation**: Some uncertainty is MCMC-derived, some is Monte Carlo-sampled, some is fixed

---

## Model Structure

### Stage 1: Historical Parameter Fitting (Pre-computed)

Parameters are fitted from 2018 and 2022 midterm election results using PyMC MCMC:

**Model specification** (parameter_fitting.py lines 381-418):
```
y_i ~ Normal(μ_i, σ_district)

μ_i = 50 + β_pvi × PVI_i + β_inc × Inc_i + β_national × National_i + Regional_i + Year_i
```

**Priors**:
- β_pvi ~ Normal(0.5, 0.2)
- β_inc ~ Normal(3.0, 2.0)
- β_national ~ Normal(1.0, 0.3)
- Regional_i ~ Normal(0, σ_region)
- Year_i ~ Normal(0, σ_year)
- σ_district ~ HalfNormal(10)
- σ_region ~ HalfNormal(3)
- σ_year ~ HalfNormal(3)

**Output**: Posterior means and standard deviations stored in `learned_params.json`.

**Fitted values** (from learned_params.json):
- β_pvi = 0.478 ± 0.007
- β_inc = 2.16 ± 0.22
- β_national = 0.66 ± 0.23
- σ_district = 3.69
- σ_regional = 0.54
- n_districts = 797 (2018 + 2022 contested races)

### Stage 2: National Environment Inference (Per Run)

The national environment is inferred from VoteHub generic ballot polling using PyMC MCMC:

**Model specification** (national_environment.py lines 212-235):
```
μ ~ Normal(0, 5)                              # Latent national environment
house_effects_j ~ Normal(0, σ_house)          # Pollster-specific bias
y_i ~ Normal(μ + house_effects[pollster_i], σ_i)  # Poll observations
```

**Prior justification**: The prior `μ ~ Normal(0, 5)` is a weakly informative prior centered at a neutral national environment (neither party favored). The standard deviation of 5 points on the generic ballot scale assigns roughly 95% probability to environments between D+10 and R+10, covering essentially all plausible national environments in modern U.S. elections. This allows the polling data to drive the posterior while preventing extreme inferences from small samples.

**Output**: Posterior mean μ and uncertainty σ.

**Post-hoc adjustments** (applied OUTSIDE the MCMC model):
1. **Approval adjustment**: `μ_adjusted = μ + (-0.3 × 0.3 × net_approval)`
   - This is a deterministic heuristic, not a fitted relationship
   - Applied in `fit_pymc()` after MCMC sampling completes
   - Uncertainty does NOT account for approval-to-GB relationship uncertainty

2. **Economic adjustment**: `final_national = β_polls × μ_adjusted + β_econ × adjusted_econ`
   - Applied in `generate_forecast.py`
   - β_econ ≈ 0.34 (fitted from historical data)
   - `adjusted_econ = -raw_econ` (negated for Republican president)

### Stage 3: Election Simulation (Posterior Predictive Monte Carlo)

For each of N=10,000 simulations:

1. **Sample parameters from Gaussian approximations**:
   ```
   β_pvi[s] ~ Normal(β_pvi_mean, β_pvi_std)
   β_inc[s] ~ Normal(β_inc_mean, β_inc_std)
   β_national[s] ~ Normal(β_national_mean, β_national_std)
   ```

2. **Sample national environment**:
   ```
   μ_national[s] ~ Normal(μ_posterior, σ_posterior)
   ```

3. **Sample regional effects**:
   ```
   Regional[s, r] ~ Normal(0, σ_regional)
   ```
   Note: Stored regional_effects dict is NOT used; only σ_regional matters.

4. **Sample district noise with PVI-scaled uncertainty**:
   ```
   σ_d = σ_floor + (σ_base - σ_floor) × logistic(|PVI_d|)
   ε_d[s] ~ Normal(0, σ_d)
   ```
   Safe districts (high |PVI|) get less noise than competitive districts.

5. **Calculate vote share**:
   ```
   vote_d[s] = 50 + β_pvi[s] × PVI_d + β_inc[s] × Inc_d
             + Regional[s, region_d] + β_national[s] × μ_national[s] + ε_d[s]
   ```

6. **Determine winner**: Democrat wins if `vote_d[s] > 50`

7. **Count seats**: Sum winners across all districts for each simulation

---

## Regional Structure

The model uses **FiveThirtyEight's 10 political regions** to capture geographic correlation in election outcomes. This regional taxonomy groups states by political behavior rather than Census divisions.

**Attribution**: Regional definitions adapted from [FiveThirtyEight's 2024 presidential election forecast methodology](https://abcnews.go.com/538/538s-2024-presidential-election-forecast-works/story?id=113068753).

| Region | States |
|--------|--------|
| New England | ME, NH, VT, MA |
| Mid-Atlantic/Northeast | NY, NJ, DE, MD, RI, CT |
| Rust Belt | IL, IN, OH, MI, WI, PA, MN, IA |
| Southeast | FL, GA, NC, VA |
| Deep South | SC, AL, MS, AR, TN, KY, WV, MO |
| Texas Region | TX, OK, LA |
| Plains | ND, SD, NE, KS |
| Mountain | ID, MT, WY, UT, AK |
| Southwest | AZ, NV, NM, CO |
| Pacific | CA, OR, WA, HI |

**Usage in the model**: During each simulation, a regional effect is sampled from Normal(0, σ_regional) for each region. All districts/races within a region share the same regional effect for that simulation, inducing correlation in outcomes among geographically proximate races.

---

## Sign Conventions

### PVI (Partisan Voter Index)
- **Positive = Democratic-leaning** (e.g., D+10 = +10)
- **Negative = Republican-leaning** (e.g., R+10 = -10)
- Adding β_pvi × PVI to 50% baseline increases Dem vote share for D-leaning districts

### Incumbency
- **D incumbent = +1**: Adds β_inc to Dem vote share
- **R incumbent = -1**: Subtracts β_inc from Dem vote share
- **Open seat = 0**: No incumbency effect

### National Environment
- **Positive = Democratic advantage** (e.g., D+5 generic ballot)
- **Negative = Republican advantage** (e.g., R+5 generic ballot)

### Regional Effects
- Sampled from Normal(0, σ_regional) each simulation
- Positive regional effect increases Dem vote share
- Stored regional_effects values from historical fitting are NOT used directly

---

## Vote Share Equation

The core vote share equation (hierarchical_model.py lines 369-375):

```
vote_share_d = 50.0
             + β_pvi × PVI_d           # Partisan lean effect
             + β_inc × Inc_d           # Incumbency advantage
             + Regional[region_d]      # Regional random effect
             + β_national × μ_national # National environment
             + ε_d                     # District-specific noise
```

This equation is structurally identical in:
- `hierarchical_model.py` (House)
- `senate_forecast.py` (Senate)
- `parameter_fitting.py` (historical fitting)

---

## Uncertainty Sources

### MCMC-Sampled (Full Posterior)
- National environment μ from poll aggregation
- Pollster house effects σ_house
- Historical parameters (β_pvi, β_inc, etc.) during fitting stage

### Monte Carlo-Sampled (Gaussian Approximation)
- β_pvi, β_inc, β_national per simulation (from stored means/stds)
- National environment per simulation (from posterior mean/std)
- Regional effects per simulation (from N(0, σ_regional))
- District noise per simulation (from N(0, σ_d × PVI_scale))

### Fixed (No Uncertainty Propagated)
- Approval adjustment formula (-0.3 × 0.3 × net_approval)
- Economic adjustment (β_econ × adjusted_econ)
- PVI values from district fundamentals
- Incumbency assignments

---

## Simulation Structure

### House Forecast
1. Load districts.csv (435 districts)
2. Load learned parameters from learned_params.json
3. Get national environment from NationalEnvironmentModel
4. Run 10,000 posterior predictive simulations
5. For each district: calculate P(Dem win) = fraction of simulations where vote > 50%
6. For aggregate: calculate seat distribution from simulation totals

### Senate Forecast
1. Load Senate races (35 seats up in 2026)
2. Use same learned parameters as House
3. Get same national environment as House
4. Run 10,000 posterior predictive simulations
5. Calculate P(Dem control) = fraction where total Dem seats ≥ 51

---

## Model Limitations

### Data Limitations

1. **No race-level polling**: The model does not incorporate district-level or state-level polling for individual races. All race-level predictions are driven by fundamentals (PVI, incumbency) and the national environment. This is a significant limitation for competitive races where local polling would provide valuable information.

2. **No candidate quality adjustment**: The model treats all candidates as generic party representatives. It does not account for candidate quality, fundraising, scandals, or other race-specific factors that can significantly affect outcomes.

3. **No special election data**: Recent special election results, which can be leading indicators of the national environment, are not systematically incorporated.

4. **Limited historical data**: Parameters are fitted on only two election cycles (2018, 2022). This small sample may not capture the full range of possible electoral dynamics.

### Methodological Limitations

5. **Staged inference, not joint**: Parameters are fitted separately from national environment. This may underestimate total uncertainty by not accounting for covariance between stages.

6. **Gaussian posterior approximations**: MCMC posteriors are summarized as (mean, std) and sampled as Normal. True posteriors may be skewed or multimodal.

7. **Post-hoc adjustments**: Approval and economic adjustments are applied deterministically after Bayesian inference. Their uncertainty is not propagated through the model.

8. **Regional effects as random, not fixed**: Stored regional effect estimates from historical fitting are NOT used during simulation. Only σ_regional is used, treating regions as exchangeable rather than having learned fixed effects per region.

### Structural Limitations

9. **PVI measurement uncertainty**: PVI values are treated as fixed, but Cook PVI has its own measurement error that is not propagated.

10. **Cross-year extrapolation**: Parameters fitted on 2018/2022 midterms may not generalize to 2026 if political dynamics have shifted.

11. **Correlation structure**: District outcomes are correlated through shared national environment and regional effects, but within-region correlation may be understated. The model may underestimate the probability of wave elections.

12. **No redistricting uncertainty**: District boundaries and PVI values are treated as fixed, but some states may have ongoing redistricting litigation.

---

## Data Sources

- **Generic ballot polls**: VoteHub API
- **Approval polls**: VoteHub API
- **Cook Political ratings**: Scraped from Cook Political Report
- **Historical results**: 2018, 2022 House election results
- **District fundamentals**: PVI, incumbency, regional assignments

---

## Code References

| Component | Primary File | Key Lines |
|-----------|-------------|-----------|
| Vote equation | hierarchical_model.py | 369-375 |
| Parameter fitting | parameter_fitting.py | 381-418 |
| National environment | national_environment.py | 212-235 |
| PVI-scaled sigma | hierarchical_model.py | 50-84 |
| Senate simulation | senate_forecast.py | 480-505 |
| Pipeline execution | generate_forecast.py | 287-358 |

---

## Verification

All sign conventions and behavioral properties have been verified programmatically:

1. ✓ Increasing D PVI increases D vote share
2. ✓ Increasing R PVI decreases D vote share
3. ✓ D incumbency increases D vote share vs open seat
4. ✓ R incumbency decreases D vote share vs open seat
5. ✓ Increasing national D margin increases vote share everywhere
6. ✓ Regional effects are correctly indexed by region
7. ✓ PVI-scaled sigma decreases as |PVI| increases
8. ✓ Senate and House use identical sign conventions

See `tests/test_behavioral_verification.py` for executable verification tests.
