# Forecast v4 methodology

This forecast answers one question: **if the election were run many times under the uncertainty we can measure today, how often would each party control the chamber?** It does not simply project the latest poll margin onto every race. It builds a broad starting expectation, updates it with polling, translates the national environment into race-level margins, and simulates correlated election outcomes 10,000 times.

All margins use Democratic two-party margin points throughout: `D+5` is `+5`, `R+5` is `-5`, and an even race is `0`.

## The short version

1. **Refresh and validate the inputs.** Silver Bulletin polling, VoteHub approval, FRED economics, FEC candidates, and the House Clerk roster are fetched separately from forecast generation. Stale or malformed required data cannot replace the public forecast.
2. **Build a national fundamentals prior.** Presidential approval and the economy create a deliberately uncertain pre-poll expectation for the national Democratic margin.
3. **Update that prior with national polling.** The House uses one weighted observation reconstructed from Silver's adjusted generic-ballot poll file. The Senate retains its existing method and uses Silver's latest published likely-voter maintained average once.
4. **Build a prior for every race.** Partisan lean, incumbency, the simulated national environment, regional movement, and local error create a distribution of possible margins for all 435 House districts and 35 Senate races.
5. **Use a race average when it is valid.** The latest Silver maintained average updates a race only when it is a verified Democratic-versus-Republican matchup. Otherwise the race remains fundamentals-only.
6. **Count seats in 10,000 correlated simulations.** The probability of control is the share of simulations in which Democrats reach 218 House seats or 51 Senate seats.

## Three national numbers that must not be confused

The dashboard deliberately shows three related but different quantities:

| Quantity | Meaning | Role in the House forecast |
|---|---|---|
| Silver published likely-voter average | Silver's displayed maintained average of current voter sentiment | Context only; it is not stacked on top of the House polling likelihood |
| House national polling likelihood | One influence-weighted average of Silver's adjusted underlying poll universe | The single polling observation used to update the House fundamentals prior |
| Election-day national environment | The poll-updated posterior after adding uncertainty for movement and election-day error | The national draw passed into every House district simulation |

For the Senate, the latest published likely-voter average is the single national polling observation. The resulting poll-updated posterior is reported separately. This preserves the existing Senate method rather than silently applying the House redesign to it.

## 1. Data acquisition and freshness

The daily acquisition job runs before the forecast model. It validates schemas and dates, writes normalized caches, records checksums and row counts, and stores immutable compressed source snapshots. Forecast generation then runs without network access.

The required inputs are:

- **Silver Bulletin maintained averages:** the published generic-ballot history and candidate-race averages.
- **Silver Bulletin adjusted generic polls:** adjusted margins and Silver's current `influence` weights, used only for the House national likelihood.
- **VoteHub presidential approval polls:** used to estimate current net approval.
- **FRED economics:** consumer sentiment, unemployment, real disposable income, GDP, and CPI.
- **FEC and House Clerk records:** used to verify candidate identity and incumbency.
- **Current House district fundamentals:** 435 current-map partisan-lean records with source and effective-date provenance.

Each source is classified as `healthy`, `degraded`, or `blocked`. A degraded source may publish with a visible warning. A source beyond its blocking threshold prevents a new bundle from replacing the last valid forecast. The website recomputes ages from the source dates in the browser, so a retained forecast becomes visibly stale even if a failed update cannot publish new metadata.

## 2. National fundamentals prior

Before national polling is used, the model creates a broad election-day prior:

```text
national_margin ~ Normal(
    approval_coefficient × net_approval
  + economy_coefficient × economic_index,
    prior_uncertainty
)
```

Because the incumbent president is Republican, the Democratic-margin coefficients are negative: `-0.08` for net approval and `-0.34` for the economic index. A negative Trump net approval therefore increases the expected Democratic margin; an economy favorable to the incumbent party decreases it.

Approval is the median net-approval value in the latest 90-day window available in the validated VoteHub cache. Its uncertainty uses a robust median-absolute-deviation estimate. The economic index combines year-over-year changes in three-month averages using these weights:

| Economic series | Weight | Direction before weighting |
|---|---:|---|
| Consumer sentiment | 35% | Higher is better for the incumbent party |
| Unemployment | 25% | Lower is better |
| Real disposable income | 20% | Higher growth is better |
| GDP | 15% | Higher growth is better |
| CPI | 5% | Lower inflation is better |

The composite is standardized against the 2006, 2010, 2014, 2018, and 2022 midterm environments. This is a weak prior, not a claim that five cycles precisely identify economic effects. The prior includes a 3.5-point structural standard deviation plus uncertainty in the approval and economic coefficients and inputs. Fundamentals are used once here; they are not added again after polling.

## 3. House national polling update

The House national likelihood starts with Silver's adjusted generic-ballot poll file. Only the `All polls` subgroup is used. Duplicate poll-question rows are removed, zero-influence rows are excluded, and the remaining adjusted Democratic margins are averaged using Silver's published influence weights:

```text
house_polling_input = sum(adjusted_margin[i] × influence[i]) / sum(influence[i])
```

Silver has already applied its pollster adjustments and weighting system. This model does not estimate another set of house effects and does not treat the rows as independent posterior updates. The weighted universe is collapsed into **one** observation.

The observation standard deviation combines:

- the weighted dispersion of the adjusted poll rows;
- their effective number of independent weighted observations; and
- a 1.25-point correlated design-error floor that cannot average away.

The one observation updates the Normal fundamentals prior analytically. A Student-t-style conflict adjustment increases the observation variance when polling and fundamentals disagree sharply. In simplified form:

```text
gain = prior_variance / (prior_variance + adjusted_poll_variance)
current_mean = prior_mean + gain × (polling_input - prior_mean)
```

The model then adds daily process variance through Election Day and a 1.5-point election-error floor. The daily process standard deviation cannot fall below 0.09 points merely because a maintained average is smooth.

This calculation is analytic rather than MCMC. `r_hat` is therefore not applicable, and the output reports it as null instead of inventing a convergence statistic.

## 4. Senate national polling update

The Senate intentionally preserves its existing national method. Silver's latest published likely-voter maintained average is used once as the external observation. Older days are displayed as history but are not sequentially multiplied into the posterior because adjacent maintained averages reuse most of the same polls.

The same broad fundamentals prior and robust analytic update are used. The Senate output distinguishes:

- `national_likelihood_margin`: the latest published average that enters the update;
- `poll_updated_current_margin`: the posterior current sentiment after combining the average with fundamentals; and
- `national_environment`: the Election Day mean after future uncertainty is added.

## 5. House race priors

The House model is calibrated on Democratic two-party district margins from 2018, 2022, and comparable 2024 districts:

```text
district_margin[r] = intercept
                   + lean_coefficient × district_lean[r]
                   + incumbency_coefficient × incumbency[r]
                   + national_coefficient × national_margin
                   + region_effect[region[r]]
                   + national_cycle_error
                   + regional_cycle_error[region[r]]
                   + local_race_error[r]
```

`incumbency` is `+1` for a Democratic incumbent, `-1` for a Republican incumbent, and `0` for an open seat. Current district lean uses the stored current-map Cook PVI table. Positive lean favors Democrats.

The coefficients are fit with regularized Bayesian linear regression and Student-t reweighting, which reduces the influence of extreme residuals. Production draws retain the full coefficient covariance matrix. Explicit uncertainty floors prevent hundreds of districts from pretending to be hundreds of independent national elections: only three recent cycles inform national-cycle behavior.

Within each simulation, all districts share one national draw, districts in the same region share a regional shock, and each district receives a local Student-t shock. This dependence is essential: a wave election should move many seats together.

## 6. Senate race priors

The Senate simulates the 35 seats scheduled for election. Each race starts from state partisan lean, incumbency, the Senate national draw, a shared regional shock, and a local Student-t race shock.

Unlike the House, the Senate structural coefficients are currently a deliberately wide regularized prior rather than a newly refit chamber model:

- state-lean coefficient: `0.50 ± 0.08`;
- incumbency effect: `2.5 ± 0.8` points;
- national coefficient: `0.50 ± 0.10`;
- regional standard deviation: `1.2` points; and
- local race standard deviation: `5.5` points.

This is a major limitation and is stated explicitly in the model card. The current Senate backtest supports the robust race-poll update and error floor; it does not validate every Senate structural coefficient.

The chamber total begins with the 65 seats not up in 2026: 34 Democratic and 31 Republican. Each simulation adds the winners of the 35 races. Democratic control requires at least 51 seats under the model's current tie-breaking assumption.

## 7. Candidate-race polling update

Silver's feed may contain repeated dates, incomplete matchups, or third-party candidates. The scraper keeps only the latest maintained row for each race and accepts it only when:

- both a Democratic and Republican candidate and average are present;
- no third-party candidate makes it a multi-candidate average;
- the race ID maps to a forecast race; and
- the candidate identities can be resolved against the FEC/Clerk registry.

Rejected matchups are counted and recorded. An uncovered or rejected race remains `fundamentals_only` or `unresolved_matchup`; the model does not invent a poll estimate.

For an accepted race, the maintained average updates the entire vector of simulated prior margins. The update has a historically calibrated correlated polling-error floor—currently about 5.9 points—that cannot average away. Student-t weights limit the effect of outliers or sharp prior conflict. The transformation preserves the rank order of the existing draws, so simulations that are nationally or regionally favorable remain favorable after the race update.

## 8. From race margins to chamber probabilities

The model runs 10,000 posterior-predictive simulations. In each simulation:

1. draw national conditions and model coefficients;
2. draw shared regional and local race errors;
3. apply any valid race-average update;
4. mark a race Democratic when its simulated Democratic two-party share exceeds 50%; and
5. count chamber seats.

The displayed control probability is a frequency, not a vote-share forecast or a confidence level. For example, an 87% House probability means Democrats reached 218 seats in about 8,700 of 10,000 model simulations. It does **not** mean Democrats are expected to win 87% of seats, and it does not guarantee the result.

The median seat count is the middle simulated outcome. A 90% credible interval contains the middle 90% of simulated seat totals conditional on the model and its inputs; it is not a claim that every possible source of uncertainty has been captured.

Race ratings are probability bins:

| Rating | Democratic win probability |
|---|---:|
| Safe D | 85% or higher |
| Likely D | 70% to below 85% |
| Lean D | 55% to below 70% |
| Toss-up | 45% to below 55% |
| Lean R | 30% to below 45% |
| Likely R | 15% to below 30% |
| Safe R | Below 15% |

## 9. Validation and what it does not prove

Tests enforce the sign convention, input schemas, latest-average-only rule, candidate matching, rejection of third-party matchups, no-poll identity, outlier robustness, deterministic seeds, correlated chamber tails, output contract, baseline history, and fail-closed publication.

The House structural layer is evaluated with whole-cycle holdouts for 2022 and 2024, conditional on each election's realized national House margin. It records a 0.0290 Brier score, 0.1002 log loss, -0.73-point signed error, and 95.4% coverage for nominal 90% district intervals. Both official seat totals fall inside their 90% simulated intervals.

Those results validate the House margin-to-seat structure, not Silver's proprietary 2026 weighting model. A comparable public historical archive of Silver candidate-race maintained averages does not exist, so provider-specific polling calibration remains externally unvalidated. The Senate evidence validates the robust update layer, not the full Senate structure.

## 10. Publication and comparison rules

Schema `4.0.0` begins a new comparable forecast epoch on August 12, 2026. Rows from earlier methods are excluded from public timelines and change calculations. Same-day runs replace that day's row instead of adding duplicates. Later forecasts compare only with the preceding forecast from the same epoch, schema, and model version.

House JSON, Senate JSON, and both timelines are staged and validated as one bundle. Invalid JSON, a chamber failure, or a catastrophically stale required source leaves the prior public bundle untouched.

## Glossary

- **Prior:** the distribution before the current polling observation is applied.
- **Likelihood:** the polling evidence used to update the prior.
- **Posterior:** the updated distribution after combining prior and likelihood.
- **Margin:** Democratic percentage minus Republican percentage among the two parties.
- **PVI / partisan lean:** a race's underlying partisan tendency; positive is Democratic here.
- **Posterior-predictive simulation:** one plausible election generated from all modeled uncertainties.
- **Credible interval:** an interval containing a stated share of posterior simulations, conditional on the model.
- **Correlated error:** uncertainty shared across polls or races that does not disappear by averaging more observations.
- **Fundamentals-only:** no valid candidate-race polling average was used for that race.
