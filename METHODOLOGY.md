# Grant's Election Forecast methodology

## The short version

Grant's Election Forecast estimates the probability that each party controls the U.S. House and Senate after the 2026 election. It does not select a single deterministic outcome. It builds a distribution of plausible national conditions and race results, then simulates the election 10,000 times.

The model has six steps:

1. Validate polling, economic, map, and candidate inputs.
2. Build a broad national prior from economics only.
3. Update that prior once with the relevant chamber's national polling input.
4. Build a fundamentals distribution for every race.
5. Update covered races with the latest valid Silver Bulletin candidate average.
6. Simulate correlated outcomes and count seats.

All margins use Democratic minus Republican two-party margin. `D+5` means Democrats lead by five points. `R+5` is stored internally as `-5`.

## The national environment, metric by metric

The dashboard exposes every national quantity the model tracks. These numbers have different jobs and should not be substituted for one another.

### 1. Economy-only fundamentals prior

This is the model's economy-based expectation before national polling is used.

Five FRED series enter the economic composite:

| Component | Weight | Change used | Direction |
|---|---:|---|---|
| Consumer sentiment | 35% | Change in the latest three-observation average from the same period one year earlier | Higher is better for the incumbent party |
| Unemployment rate | 25% | Prior-year average minus current average | Lower unemployment is better for the incumbent party |
| Real disposable income | 20% | Year-over-year percent change in three-observation averages | Higher growth is better for the incumbent party |
| Gross domestic product | 15% | Year-over-year percent change in three-observation averages | Higher growth is better for the incumbent party |
| Consumer prices | 5% | Prior-year CPI average minus current CPI average | Lower inflation is better for the incumbent party |

The weighted raw composite is standardized against the 2006, 2010, 2014, 2018, and 2022 midterm environments. Positive standardized values favor the incumbent president's party; negative values are unfavorable to it.

With a Republican president, the prior mean on the Democratic-margin scale is:

```text
prior_mean = -0.34 * standardized_economic_index
```

The prior variance is:

```text
prior_variance =
    3.5^2
    + (economic_coefficient * economic_input_std)^2
    + (standardized_economic_index * coefficient_std)^2
```

The economic coefficient standard deviation is `0.33`, and the standardized input uncertainty is `1.0`. The 3.5-point structural term is intentionally large because five historical midterms cannot precisely identify a stable economic effect. Economics enters here once and is never added again after polling.

### 2. Silver published sentiment

This is Silver Bulletin's displayed likely-voter maintained generic-ballot average, with Silver's publication date. It describes current national sentiment.

For the House it is context, not the direct likelihood. For the Senate it is also the national polling input.

### 3. Chamber polling input

The two chambers intentionally use different inputs:

- **House:** the adjusted generic-ballot poll file is collapsed into one observation using Silver's published influence weights. Its displayed uncertainty includes a correlated design-error floor, because many rows share pollsters, methods, and voters.
- **Senate:** the latest Silver published likely-voter maintained average enters once.

This one-observation rule prevents pseudo-replication. Adjacent daily averages reuse most of the same polls, so multiplying the full history as if each day were new independent evidence would make the model falsely certain.

### 4. Poll-updated current margin

The current margin is the posterior after combining the economy-only prior with the chamber polling input. The update is an analytic robust Bayesian calculation. A surprising observation receives a larger effective observation variance rather than being discarded.

Conceptually:

```text
prior:       theta ~ Normal(prior_mean, prior_variance)
observation: polling_input ~ Student-t(theta, input_uncertainty)
posterior:   poll_updated_current = prior updated once by observation
```

The dashboard shows the posterior mean, standard deviation, 90% credible interval, and polling-input date.

### 5. Election Day national margin

Current sentiment is not assumed to remain fixed. The model adds uncertainty for the number of days until November 3 and a 1.5-point Election Day error floor:

```text
election_variance =
    current_variance
    + days_to_election * daily_process_std^2
    + 1.5^2
```

The House daily process standard deviation is calibrated from first differences in Silver's maintained-average history but cannot fall below the regularized `0.09`-point prior. The Senate retains the `0.09` setting. The dashboard shows the Election Day mean, standard deviation, 90% credible interval, and election date.

## House race model

Each of the 435 current-map districts starts with a structural Democratic-margin distribution:

```text
district_margin =
    intercept
    + beta_lean * Cook_PVI
    + beta_incumbency * incumbency_code
    + beta_national * election_day_national_margin
    + regional_effect
    + local_error
```

The House calibration uses 2018, 2022, and map-comparable 2024 results. Coefficients remain on the Democratic-margin scale. Cluster-level uncertainty floors prevent hundreds of district rows from being mistaken for hundreds of independent national elections.

National and regional draws are shared across districts within a simulation. A national wave therefore moves many seats together, which creates more realistic chamber tails than independent district simulations.

If Silver supplies a current, unambiguous Democratic-versus-Republican maintained average for a district, that latest average robustly updates the structural distribution. If a matchup is missing, third-party, stale, or cannot be matched to official candidates, the race remains fundamentals-only.

## Senate race model

Thirty-five 2026 races are simulated. The structural form also uses partisan lean, incumbency, the Senate national environment, and broad race error, but its coefficients use wider regularized priors rather than the House calibration.

The latest valid Silver candidate average updates a covered race once. Sixty-five seats are not up: 34 Democratic and 31 Republican. Those values come directly from the forecast output. Democrats are treated as controlling the Senate only at 51 or more seats under the current tie-breaking assumption.

## From race margins to chamber probabilities

Each simulation draws a coherent national environment, regional movement, and race-level error. The model converts simulated Democratic margins to winners and counts seats.

- House Democratic control: at least 218 Democratic seats.
- Senate Democratic control: at least 51 Democratic seats.

If Democrats control the House in 8,700 of 10,000 simulations, the displayed probability is 87%. It is not a vote-share forecast, not the expected share of seats, and not a guarantee.

The 90% credible interval contains the central 90% of simulated seat outcomes under the model and its inputs.

## Data acquisition and publication safety

Network access is isolated in `scripts/fetch_inputs.py`. It fetches:

- Silver Bulletin's public maintained-average and adjusted generic-poll feeds.
- Consumer sentiment, unemployment, real disposable income, GDP, and CPI from FRED.
- Candidate and incumbent records from the FEC and Clerk of the House.

Each source record contains its URL, fetch or provider date, latest observation date, checksum, row count, fallback status, and freshness state. Warning and blocking thresholds are source-specific. Forecast generation performs no network requests.

Inputs are staged and promoted only after validation. House JSON, Senate JSON, and both timelines are also staged and validated as one bundle. A partial failure cannot replace only part of the public forecast.

The website recalculates source age in the browser. If a daily workflow fails and an older forecast remains online, its stale sources still become visibly degraded or blocked.

## Forecast history and compatibility

The comparable forecast epoch begins August 12, 2026. Internal schema and model version `5.0.0` identify the current machine-readable contract but are not public branding.

Timeline rows from before the epoch or a different internal contract are discarded. Same-day runs upsert one row rather than creating duplicates. The baseline has no change decomposition because there is no prior comparable forecast. Later changes compare only with the preceding compatible row.

## Validation

The production suite checks sign conventions, polling aggregation, uncertainty floors, candidate matching, stale-source behavior, atomic publication, 435 House districts, 35 Senate races, the complete national-environment contract, the five economic components, and the one-row August 12 baseline.

The House structural layer passes whole-cycle holdouts for 2022 and 2024 conditional on the realized national House margin: Brier 0.0290, log loss 0.1002, signed error -0.73 points, and 95.4% coverage for nominal 90% district intervals. This validates the House margin-to-seat layer, not Silver's proprietary aggregation or 2026 calibration.

## Known limitations

- Only five historical midterms anchor the economic standardization, so the prior is deliberately weak.
- Silver's aggregation is external and is not independently reproduced.
- Candidate averages are sparse early in the cycle.
- The Senate structural layer is less empirically calibrated than the House layer.
- Cook PVI is not an individual-level voter model and can miss demographic or legal changes.
- The model does not include fundraising, endorsements, scandals, or subjective candidate quality.
- All probabilities are conditional on the model, source data, and current electoral rules.

## Glossary

- **Prior:** the probability distribution before the current polling input is used.
- **Likelihood:** the model for how the observed polling input relates to the unknown margin.
- **Posterior:** the updated distribution after combining prior and polling input.
- **Credible interval:** a range containing a stated share of posterior draws.
- **National environment:** the complete set of prior, polling, posterior, Election Day, and economic metrics—not one ambiguous margin.
- **Correlated simulation:** a draw in which shared national and regional shocks move multiple races together.
- **Fundamentals-only race:** a race without a valid candidate-average update.
