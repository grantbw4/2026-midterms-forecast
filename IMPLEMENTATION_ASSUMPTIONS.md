# Implementation Assumptions

**Version 2.0.0** | Updated: 2026-02-02

This document lists all approximations, heuristics, fixed weights, and implementation choices that affect model behavior.

---

## 1. Approximations

### 1.1 Gaussian Posterior Approximation
**Location**: `hierarchical_model.py:333-348`, `senate_forecast.py:460-473`

MCMC posteriors from historical parameter fitting are approximated as Gaussian:
```python
beta_pvi_samples = rng.normal(self.params.beta_pvi_mean, self.params.beta_pvi_std, size=n_sims)
```

**Assumption**: True posterior is approximately Normal. In practice, most posteriors from well-specified models are approximately Gaussian by CLT, but this may underestimate tail probabilities.

### 1.2 National Environment Gaussian Sampling
**Location**: `hierarchical_model.py:97-101`

```python
def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
    if self.samples is not None and len(self.samples) >= n:
        return rng.choice(self.samples, size=n, replace=True)
    return rng.normal(self.mean, self.std, size=n)
```

**Assumption**: If full MCMC samples are not available, the national environment is sampled as Normal(mean, std).

### 1.3 Exchangeable Regional Effects
**Location**: `hierarchical_model.py:347-348`, `senate_forecast.py:471-473`

```python
regional_effects_samples = self.rng.normal(0, self.params.sigma_regional, size=(n_sims, n_regions))
```

**Assumption**: Regional effects are sampled from N(0, σ_regional) each simulation, treating regions as exchangeable. The stored `regional_effects` dictionary containing fitted region-specific means is NOT used during simulation.

---

## 2. Heuristics

### 2.1 Approval-to-Generic-Ballot Conversion
**Location**: `national_environment.py:280-297`

```python
app_adjustment = -0.3 * app_mean * self.approval_weight
result["national_environment"] = round(mu_mean + app_adjustment, 2)
```

**Heuristic**: Each point of net approval shifts the generic ballot by -0.3 × 0.3 = -0.09 points. This is a rule-of-thumb conversion, not a fitted relationship. Applied OUTSIDE the MCMC model as a deterministic post-hoc adjustment.

### 2.2 Economic Index Party Adjustment
**Location**: `generate_forecast.py:343`

```python
adjusted_econ = -raw_econ  # R president: bad economy helps Dems
```

**Heuristic**: The economic index is negated when the president is Republican, assuming bad economic conditions help the opposing party.

### 2.3 PVI-Scaled Uncertainty
**Location**: `hierarchical_model.py:50-84`, `senate_forecast.py:86-120`

```python
def compute_pvi_scaled_sigma(
    pvi: np.ndarray,
    sigma_base: float = 4.5,
    sigma_floor: float = 2.0,
    midpoint: float = 15.0,
    steepness: float = 0.15,
) -> np.ndarray:
    abs_pvi = np.abs(pvi)
    logistic = 1 / (1 + np.exp(steepness * (abs_pvi - midpoint)))
    return sigma_floor + (sigma_base - sigma_floor) * logistic
```

**Heuristic**: District-level uncertainty decreases with |PVI|. Safe seats (|PVI| > 15) have uncertainty approaching σ_floor = 2.0, while competitive seats (|PVI| ≈ 0) have uncertainty approaching σ_base = 4.5. The logistic function provides smooth transition.

---

## 3. Fixed Weights

### 3.1 Approval Weight
**Location**: `national_environment.py:77`

```python
self.approval_weight = 0.3
```

**Fixed value**: Weight given to approval adjustment is hardcoded at 0.3. This is not fitted from data.

### 3.2 Approval-to-GB Conversion Factor
**Location**: `national_environment.py:294-295`

```python
app_adjustment = -0.3 * app_mean * self.approval_weight
```

**Fixed value**: The -0.3 multiplier is hardcoded, not fitted.

### 3.3 Economic Coefficient (when fitted)
**Location**: `generate_forecast.py:325-334`

```python
beta_polls = coeffs.get("beta_polls", 1.0)
beta_econ = coeffs.get("beta_econ", 0.0)
```

**From file**: β_econ ≈ 0.34 when fitted, but defaults to 0.0 if no coefficients file exists.

### 3.4 PVI-Scaled Sigma Parameters
**Location**: `hierarchical_model.py:52-55`

| Parameter | House | Senate | Description |
|-----------|-------|--------|-------------|
| sigma_base | 4.5 | 5.0 | Max uncertainty for competitive races |
| sigma_floor | 2.0 | 2.5 | Min uncertainty for safe races |
| midpoint | 15.0 | 12.0 | |PVI| at which sigma is halfway |
| steepness | 0.15 | 0.2 | Transition sharpness |

**Fixed values**: These parameters are hardcoded, not fitted from data.

---

## 4. Staged Inference

### 4.1 Three-Stage Pipeline

**Stage 1**: Parameter fitting (MCMC on 2018/2022 data)
- Produces: β_pvi, β_inc, β_national, σ_district, σ_regional posterior summaries
- Stored in: `learned_params.json`
- Run: Pre-computed, not per forecast run

**Stage 2**: National environment inference (MCMC on polls)
- Produces: μ_national, σ_national
- Post-hoc: Approval and economic adjustments applied deterministically
- Run: Each forecast run

**Stage 3**: Forecast simulation (Monte Carlo)
- Uses: Gaussian approximations from Stages 1 and 2
- Produces: District probabilities, seat distribution
- Run: Each forecast run

**Implication**: Total uncertainty may be underestimated because stages are not jointly inferred. Correlations between parameters and national environment are not captured.

---

## 5. Hardcoded Race Data

### 5.1 Senate Races
**Location**: `senate_forecast.py:177-216`

```python
RACES_2026 = [
    SenateRace("GA", 2, "Jon Ossoff", "D", True, 0, "toss_up"),
    SenateRace("MI", 2, "Gary Peters", "D", True, 1, "lean_d"),
    # ... 35 races total
]
```

**Hardcoded**: PVI values, incumbents, and initial ratings for all 35 Senate races are hardcoded in the source file. Cook Political ratings are loaded separately and can override some values.

### 5.2 Cook Rating to PVI Mapping
**Location**: `senate_forecast.py:73-83`

```python
COOK_TO_PVI = {
    "Solid D": 12.0,
    "Likely D": 7.0,
    "Lean D": 3.0,
    "Toss-up": 0.0,
    "Lean R": -3.0,
    "Likely R": -7.0,
    "Solid R": -12.0,
}
```

**Fixed mapping**: Cook ratings are converted to PVI using this fixed mapping when Cook data is used as fallback.

### 5.3 Regional Assignments
**Location**: `parameter_fitting.py:52-77`

```python
STATE_TO_REGION = {
    "ME": "New_England", "NH": "New_England", ...
    "TX": "Texas_Region", "OK": "Texas_Region", ...
}
```

**Fixed mapping**: States are assigned to 10 FiveThirtyEight regions via hardcoded dictionary.

---

## 6. Parameter Reuse Across Chambers

### 6.1 Shared Parameters
**Location**: `senate_forecast.py:370-378`

```python
if learned_params is not None:
    self.params = learned_params
else:
    try:
        self.params = ParameterFitter.load_parameters()
```

**Assumption**: Senate model uses the same learned parameters (β_pvi, β_inc, β_national, σ values) as House model. These parameters are fitted only from House data (2018/2022 House elections). Senate-specific fitting is NOT performed.

**Justification**: Limited historical Senate data makes separate fitting unreliable.

---

## 7. Unused but Stored Data

### 7.1 Regional Effect Point Estimates
**Location**: `learned_params.json`

```json
"regional_effects": {
    "New_England": -1.23,
    "Southeast": 0.45,
    ...
}
```

**Unused**: These fitted regional effect means are stored but NOT used during simulation. Only `sigma_regional` is used for sampling from N(0, σ_regional).

### 7.2 Regional Effect Standard Deviations
**Location**: `learned_params.json`

```json
"regional_effects_std": {
    "New_England": 0.8,
    ...
}
```

**Unused**: Per-region uncertainty estimates are stored but not used. The shared `sigma_regional` is used instead.

---

## 8. Default Fallbacks

### 8.1 Default Parameters
**Location**: `hierarchical_model.py:212-230`

```python
def _default_parameters(self) -> LearnedParameters:
    return LearnedParameters(
        beta_pvi_mean=0.5,
        beta_pvi_std=0.1,
        beta_inc_mean=3.0,
        beta_inc_std=1.0,
        ...
    )
```

**Fallback**: If `learned_params.json` cannot be loaded, hardcoded default values are used.

### 8.2 Default National Environment
**Location**: `hierarchical_model.py:204-207`

```python
if national_posterior is None:
    self.national = NationalPosterior(mean=0.0, std=self.DEFAULT_SIGMA_NATIONAL)
```

**Fallback**: If no national environment is provided, defaults to D+0 with σ=3.0.

---

## 9. Manual PVI Adjustments

### 9.1 Cook Rating PVI Adjustments
**Location**: `scripts/fetch_cook_ratings.py`

Redistricted districts have their PVI values adjusted based on Cook Political ratings when Cook data is more recent than the stored PVI values.

**Process**:
1. Load existing districts.csv
2. Identify redistricted states from Cook's redistricting status
3. For districts in redistricted states, potentially update PVI based on Cook rating
4. Save adjusted districts.csv

---

## Summary Table

| Category | Item | Value/Method | Location |
|----------|------|--------------|----------|
| Approximation | Posterior shape | Gaussian | hierarchical_model.py:333 |
| Approximation | Regional effects | Exchangeable | hierarchical_model.py:347 |
| Heuristic | Approval→GB | -0.3 × 0.3 | national_environment.py:294 |
| Heuristic | PVI→σ | Logistic decay | hierarchical_model.py:50 |
| Fixed Weight | approval_weight | 0.3 | national_environment.py:77 |
| Fixed Weight | sigma_base | 4.5 / 5.0 | hierarchical_model.py:52 |
| Staged | Inference | 3 separate stages | generate_forecast.py |
| Hardcoded | Senate races | 35 races | senate_forecast.py:177 |
| Reuse | House→Senate params | Same params | senate_forecast.py:370 |
| Unused | regional_effects dict | Stored but ignored | learned_params.json |
