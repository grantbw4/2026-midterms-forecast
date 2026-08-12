# Model card: Forecast v4.0.0

## Intended use

Estimate probabilistic 2026 U.S. House and Senate outcomes and demonstrate reproducible Bayesian forecasting practice. It is not a voting recommendation or a substitute for campaign-specific intelligence.

## Current validation status

Synthetic and behavioral tests are enforced in CI. The House structural layer passes whole-cycle 2022 and 2024 holdouts conditional on the realized national House margin: Brier 0.0290, log loss 0.1002, signed error −0.73 points, and 95.4% coverage for nominal 90% district intervals. Both official House seat outcomes are inside their 90% posterior intervals. The separate robust Senate race-update gate remains in production. Silver Bulletin's 2026 poll feed lacks a comparable public historical archive, so its provider-specific calibration remains externally unvalidated.

## Promotion criteria

House v4 must match the best structural baseline within 0.005 Brier and 0.01 log loss, keep absolute signed error at or below 1.5 margin points, and cover each held-out official seat outcome at 90%. It passes. Bulletin polling remains `external_unvalidated` until a suitable historical archive can be evaluated.

## Diagnostics contract

- House robust Bayesian regression: finite covariance, cluster-level uncertainty floors, structural holdout gate, and posterior-predictive seat checks.
- Analytic national update: one-aggregate likelihood invariant, finite posterior checks, and MCMC diagnostics marked not applicable.
- Any failed fit retains the prior successful public artifacts.

## Main limitations

- The Senate chamber parameter layer is regularized; the backtest validates its candidate-race update rather than every structural component.
- Silver Bulletin's aggregation is externally maintained and its full weighting model is not reproduced or audited here.
- Early-cycle candidate averages are sparse; absent, ambiguous, or third-party races remain fundamentals-only.
- Cook PVI is not MRP and can miss demographic change or late litigation affecting maps.
- Only three recent House cycles inform cluster-level uncertainty, so the model deliberately retains broad priors and control tails.
- The model excludes fundraising, endorsements, scandals, and subjective candidate quality.
- Probability estimates are conditional on model structure and source data.

## Reproducibility

Dependencies are pinned, random seeds are deterministic, upstream data are snapshotted with checksums, and public artifacts include a content-derived run ID, sources, freshness, fallbacks, and inference method.
