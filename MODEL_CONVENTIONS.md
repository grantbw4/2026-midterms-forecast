# Model Conventions

**Version 2.0.0** | Frozen: 2026-02-02

This file documents the sign conventions, encodings, and structural decisions that MUST remain consistent across all model files. Any changes to these conventions require updating ALL affected files and re-running verification tests.

---

## 1. PVI Sign Convention

**Convention**: Positive = Democratic-leaning, Negative = Republican-leaning

| PVI Value | Interpretation |
|-----------|----------------|
| +10 | D+10 (Democratic +10 points) |
| 0 | Even/Neutral |
| -10 | R+10 (Republican +10 points) |

**Effect in vote equation**: β_pvi × PVI is ADDED to 50% baseline
- Positive PVI → increases Dem vote share
- Negative PVI → decreases Dem vote share

**Files using this convention**:
- `models/forecast.py` (line 19-20, 319-321)
- `models/senate_forecast.py` (line 16-17, 73-83, 289-296)
- `models/hierarchical_model.py` (line 16-17, 140-156)
- `models/parameter_fitting.py`
- `data/processed/districts.csv`

---

## 2. Incumbency Encoding

**Convention**: D = +1, R = -1, Open = 0

| incumbent_party | incumbency_code |
|-----------------|-----------------|
| "D" | +1 |
| "R" | -1 |
| "" or None | 0 |
| open_seat=True | 0 (overrides party) |

**Effect in vote equation**: β_inc × Inc is ADDED to 50% baseline
- D incumbent (+1) → increases Dem vote share by β_inc
- R incumbent (-1) → decreases Dem vote share by β_inc
- Open seat (0) → no incumbency effect

**Encoding location**: `hierarchical_model.py:244-252`
```python
inc_map = {"D": 1, "R": -1}
self.districts["incumbency_code"] = self.districts["incumbent_party"].map(inc_map).fillna(0)
if "open_seat" in self.districts.columns:
    self.districts.loc[self.districts["open_seat"] == True, "incumbency_code"] = 0
```

---

## 3. National Environment Sign

**Convention**: Positive = Democratic advantage, Negative = Republican advantage

| National Env | Interpretation |
|--------------|----------------|
| +5 | D+5 on generic ballot |
| 0 | Even |
| -5 | R+5 on generic ballot |

**Effect in vote equation**: β_national × μ_national is ADDED to fundamentals
- Positive national env → increases Dem vote share
- Negative national env → decreases Dem vote share

**Source**: Generic ballot margin = Dem% - Rep%

---

## 4. Regional Effects Sign

**Convention**: Positive = increases Democratic vote share

Regional effects are sampled from N(0, σ_regional) each simulation.
The stored regional_effects dictionary is NOT used during simulation.

**Effect in vote equation**: Regional effect is ADDED to fundamentals
- Positive regional → increases Dem vote share
- Negative regional → decreases Dem vote share

**Important**: The hierarchical model ADDS regional effects.
The legacy mode SUBTRACTS regional effects (different convention, deprecated).

---

## 5. Vote Equation Form

**Canonical form** (from hierarchical_model.py:369-375):

```
vote_share_d = 50.0
             + β_pvi × PVI_d
             + β_inc × Inc_d
             + Regional[region_d]
             + β_national × μ_national
             + ε_d
```

**Sign summary**:
| Term | Sign | Effect of positive value |
|------|------|--------------------------|
| PVI | + | Increases D vote |
| Inc | + | D incumbent increases D vote |
| Regional | + | Increases D vote |
| National | + | Increases D vote |
| ε_d | + | Random noise |

**All terms are ADDED**. No subtraction in the hierarchical model vote equation.

---

## 6. Simulation Structure

### Monte Carlo Simulation Loop

```
For s = 1 to N_simulations:
    1. Sample β_pvi[s] ~ Normal(mean, std)
    2. Sample β_inc[s] ~ Normal(mean, std)
    3. Sample β_national[s] ~ Normal(mean, std)
    4. Sample μ_national[s] ~ Normal(μ_posterior, σ_posterior)
    5. Sample Regional[s, r] ~ Normal(0, σ_regional) for each region r
    6. For each district d:
       a. Compute σ_d from PVI-scaled function
       b. Sample ε_d[s] ~ Normal(0, σ_d)
       c. Compute vote_d[s] using vote equation
       d. D wins if vote_d[s] > 50
    7. Count D seats: seats[s] = sum(vote_d[s] > 50)

Output:
    - P(D win district d) = mean(vote_d > 50 over s)
    - Seat distribution from seats[s] histogram
```

---

## 7. Uncertainty Sources

### MCMC-Derived (Stage 1 & 2)
- β_pvi, β_inc, β_national posterior distributions (from historical fitting)
- μ_national, σ_national (from poll aggregation)
- σ_district, σ_regional (from historical fitting)

### Monte Carlo-Sampled (Stage 3)
- β_pvi, β_inc, β_national per simulation (Gaussian approximation)
- μ_national per simulation (Gaussian approximation)
- Regional effects per simulation (N(0, σ_regional))
- District noise per simulation (N(0, σ_d × PVI_scale))

### Fixed (No Uncertainty)
- Approval adjustment multiplier (-0.3 × 0.3)
- Economic adjustment coefficient (β_econ)
- PVI values
- Incumbency assignments
- Regional assignments

---

## 8. File Consistency Requirements

When modifying the model, ensure these files remain consistent:

| File | Must Match |
|------|------------|
| `forecast.py` | PVI sign, inc encoding, vote equation signs |
| `senate_forecast.py` | Same as above |
| `hierarchical_model.py` | Same as above |
| `parameter_fitting.py` | Same as above (for training data) |
| `districts.csv` | PVI sign convention |
| `learned_params.json` | Coefficient signs assume this convention |

---

## 9. Verification Requirements

Before any release, run:
```bash
python tests/test_behavioral_verification.py
```

All 8 tests must pass:
1. ✓ Increasing D PVI increases D vote share
2. ✓ Increasing R PVI decreases D vote share
3. ✓ D incumbency increases D vote share vs open seat
4. ✓ R incumbency decreases D vote share vs open seat
5. ✓ Increasing national D margin increases vote share everywhere
6. ✓ Regional effects are correctly indexed by region
7. ✓ PVI-scaled sigma decreases as |PVI| increases
8. ✓ Senate and House use identical sign conventions

---

## 10. Legacy Mode Warning

**DEPRECATED**: The legacy mode (`--legacy` flag) uses different conventions:

| Aspect | Hierarchical | Legacy |
|--------|--------------|--------|
| PVI transform | β_pvi × PVI | PVI / 2 |
| Regional sign | ADDED | SUBTRACTED |
| Parameters | Loaded | Hardcoded |

Legacy mode is preserved for comparison only. Do not use for production forecasts.

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-02 | 2.0.0 | Initial freeze of conventions |
