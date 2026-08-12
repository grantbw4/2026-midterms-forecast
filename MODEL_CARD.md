# Model card: Forecast v3.0.0

## Intended use

Estimate probabilistic 2026 U.S. House and Senate outcomes and demonstrate reproducible Bayesian forecasting practice. It is not a voting recommendation or a substitute for campaign-specific intelligence.

## Current validation status

Synthetic and behavioral tests are enforced in CI. The robust Senate race-update design passes a leakage-resistant rolling-origin evaluation for 2018, 2020, 2022, and 2024 at 120, 90, 60, 30, 14, and 7 days. Over 275 matched forecasts in the final 60 days, it records a 0.0584 Brier score and 0.1909 log loss versus 0.0910 and 0.2933 for v2. Silver Bulletin's maintained 2026 averages lack a comparable public historical archive, so they are transparently marked as an external likelihood whose provider-specific calibration is not independently backtested here.

## Promotion criteria

Across the final 60 days of 2018–2024, the race-update design must not materially worsen Brier score or log loss relative to v2 and must improve at least one. It passes both metrics. The current Silver-average input remains `external_unvalidated` until a suitable historical average archive can be evaluated; the older backtest is retained as supporting evidence, not relabeled as a direct validation.

## Diagnostics contract

- Historical MCMC fits: R-hat < 1.01, bulk ESS > 400, zero divergences.
- Analytic average update: schema, latest-only likelihood invariant, residual, and finite-posterior checks; MCMC diagnostics marked not applicable.
- Any failed fit retains the prior successful public artifacts.

## Main limitations

- The Senate chamber parameter layer is regularized; the backtest validates its candidate-race update rather than every structural component.
- Silver Bulletin's aggregation is externally maintained and its full weighting model is not reproduced or audited here.
- Early-cycle candidate averages are sparse; absent, ambiguous, or third-party races remain fundamentals-only.
- District partisan lean is not MRP and may lag redistricting or demographic change.
- The model excludes fundraising, endorsements, scandals, and subjective candidate quality.
- Probability estimates are conditional on model structure and source data.

## Reproducibility

Dependencies are pinned, random seeds are deterministic, upstream data are snapshotted with checksums, and public artifacts include a content-derived run ID, sources, freshness, fallbacks, and inference method.
