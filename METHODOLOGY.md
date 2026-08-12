# Forecast v3 methodology

All margins use a single sign convention: positive values favor Democrats and negative values favor Republicans.

## 1. Election-day fundamentals prior

Approval and economic data inform a broad national prior:

```text
theta_election ~ Normal(beta_approval * approval + beta_economy * economy, sigma_structural)
```

Coefficient uncertainty and input uncertainty are propagated into the prior variance. The structural standard deviation is deliberately broad because the number of modern midterm cycles is small. These variables are not added to the posterior after polling; doing so would double-count the national environment.

## 2. National polling likelihood

Silver Bulletin's likely-voter-adjusted generic-ballot average is the sole national polling likelihood:

```text
silver_average_latest ~ Student-t(theta_current, sigma_aggregate, nu)
theta_election ~ Normal(theta_current, days_to_election * sigma_process^2 + sigma_election^2)
```

Silver already adjusts and weights the underlying polls. Re-estimating house effects from the same inputs would double-count those adjustments, so the model treats the published average as one external measurement with a regularized observation error. The entire daily average history is shown on the site, but only its latest value enters the posterior. This pseudo-replication guard matters because adjacent daily averages reuse nearly all the same polls. A Student-t robustification inflates the measurement variance when the average sharply conflicts with fundamentals. Future uncertainty grows toward Election Day through the latent random-walk variance and an election-error floor.

The production update is conjugate/analytic, not MCMC. It therefore reports schema checks, the one-likelihood invariant, finite posterior checks, and posterior draw count, while `r_hat` is explicitly null. It would be statistically misleading to invent an R-hat for an analytic update.

## 3. House and Senate fundamentals

The race prior is:

```text
race_margin[r] = beta_lean * partisan_lean[r]
               + beta_inc * incumbency[r]
               + beta_nat * theta_election
               + region_effect[region[r]]
               + cycle/race_error[r]
```

House coefficients are posterior summaries from historical House fits. Senate uses a separate, wider regularized prior until a sufficiently complete Senate training panel passes rolling-origin validation. Residuals are Student-t. Current House incumbents come from the Clerk of the House; FEC filings establish candidate identity and open-seat status. Inputs carry source and effective-date fields and are validated for 435 unique districts and plausible lean values.

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

Rolling-origin backtests cover 2018–2024 at 120, 90, 60, 30, 14, and 7 days. Metrics are margin RMSE, Brier score, log loss, calibration, and 50%/80%/95% interval coverage, compared with v2, polls-only, and fundamentals-only. The underlying Senate race-update design passes the final-60-day matched-sample gate: Brier improves from 0.0910 to 0.0584 and log loss from 0.2933 to 0.1909 over 275 forecasts. At every holdout, model parameters and the common-error floor use earlier cycles only. Because a public historical archive of Silver's maintained race averages is unavailable, this evidence validates the update/error structure—not the external average provider itself or the full House chamber model.
