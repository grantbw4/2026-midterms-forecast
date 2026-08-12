# Forecast v4 methodology

All margins use a single sign convention: positive values favor Democrats and negative values favor Republicans.

## 1. Election-day fundamentals prior

Approval and economic data inform a broad national prior:

```text
theta_election ~ Normal(beta_approval * approval + beta_economy * economy, sigma_structural)
```

Coefficient uncertainty and input uncertainty are propagated into the prior variance. The structural standard deviation is deliberately broad because the number of modern midterm cycles is small. These variables are not added to the posterior after polling; doing so would double-count the national environment.

## 2. National polling likelihood

The national likelihood is constructed only from Silver Bulletin's public generic-ballot poll universe. Bulletin's adjusted margins and current influence weights are collapsed into one observation:

```text
bulletin_adjusted_aggregate ~ Student-t(theta_current, sigma_aggregate, nu)
theta_election ~ Normal(theta_current, days_to_election * sigma_process^2 + sigma_election^2)
```

Silver already adjusts pollster house effects and supplies influence weights. The model neither re-estimates those effects nor updates once per poll. Aggregate observation variance combines Bulletin poll dispersion/effective sample size with a correlated design-error floor. A Student-t robustification inflates variance when polling sharply conflicts with fundamentals. The daily movement prior cannot be reduced by the first differences of a smoothed published average.

The production update is conjugate/analytic, not MCMC. It therefore reports schema checks, the one-likelihood invariant, finite posterior checks, and posterior draw count, while `r_hat` is explicitly null. It would be statistically misleading to invent an R-hat for an analytic update.

## 3. House fundamentals

The race prior is:

```text
race_margin[r] = intercept + beta_lean * partisan_lean[r]
               + beta_inc * incumbency[r]
               + beta_nat * theta_election
               + region_effect[region[r]]
               + cycle/race_error[r]
```

Every term is measured in Democratic two-party margin points. House coefficients use robust regularized Bayesian regression over 2018, 2022, and map-comparable 2024 districts. National-coefficient, intercept, and regional posterior floors prevent district rows from masquerading as independent election cycles. Cycle, region, and local Student-t errors are propagated separately.

Current House lean comes from the Cook Political Report's 435-row current-map PVI table, which incorporates mid-cycle redistricting. Cook race ratings are not converted into PVI. The model fails closed for missing provenance, missing districts, implausible values, or invalid source URLs. A sourced open-seat flag cannot be overwritten merely because a retiring member remains in the Clerk roster.

## 4. Candidate-race polling averages

For every House or Senate race covered by Silver Bulletin, exactly the latest maintained average is mapped to the forecast race and checked against FEC candidate identities. Missing D-vs-R pairs, third-party matchups, ambiguous surnames, and unmapped candidates are excluded. A race absent from the feed or failing validation remains at its fundamentals posterior with `unresolved_matchup` or `fundamentals_only` status.

The aggregate receives a regularized current-sentiment measurement error plus an irreducible election-day polling-error component estimated from prior Senate cycles. Only one average enters per race, so neither the daily history nor the underlying polls can create false precision. Robust updates preserve the rank order of existing race draws so national and regional correlation is not destroyed.

## 5. Posterior-predictive chamber simulation

Each simulation shares the same national draw across every race and the same regional draw within a region. Race shocks remain local. This creates the correlated tails expected in wave elections; summing independent win probabilities would be too narrow. House control requires 218 seats. Senate control requires 51 Democratic seats in the current tie-breaking configuration.

## 6. Publication contract

Output is schema version `3.0.0`. Public files are written atomically only after both chambers succeed. Network failure may use the latest valid cache and sets `fallback_used`; stale feeds set `degraded`. A catastrophically stale national feed prevents publication unless explicitly overridden for development. Raw provider snapshots are content-addressed and retain fetch time, URL, license, and SHA-256 checksum.

Every race reports its prior and posterior margins, 90% credible interval, probability, polling adjustment, polls used, latest poll date, source URLs, and data quality. The dashboard distinguishes current voter sentiment from the election-day forecast.

## 7. Validation

Behavioral tests verify the one-average likelihood invariant, no-poll identity, candidate matching, third-party rejection, outlier robustness, sign conventions, deterministic seeds, and wider tails under correlated shocks. The lower-level poll updater retains synthetic tests for posterior contraction and correlated error because it supplies the historical calibration evidence.

House whole-cycle holdouts train on earlier cycles and score 2022 and 2024 district margins, probabilities, intervals, seat totals, and control. The gate requires competitive Brier/log loss, absolute signed error no greater than 1.5 points, and 90% seat-interval coverage. The candidate passes, improving Brier from 0.0335 to 0.0290 and signed error from −2.51 to −0.73 points versus legacy v3. Because a comparable historical Bulletin maintained-average archive is unavailable, polling-layer validation remains explicitly separate.
