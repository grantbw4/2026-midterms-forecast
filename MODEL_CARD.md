# Model card: Grant's Election Forecast

## Intended use

Estimate probabilistic 2026 U.S. House and Senate outcomes and demonstrate reproducible Bayesian forecasting practice. It is not a voting recommendation or a substitute for campaign-specific intelligence.

## Architecture

The model validates its inputs, builds a broad economy-only national prior, updates national conditions once with polling, converts those conditions into a probability distribution for every race, applies the latest valid candidate-race average where available, and counts seats in 10,000 correlated simulations.

The chambers use different national polling inputs. The House uses one influence-weighted aggregate of Silver's adjusted generic-ballot poll file. The Senate uses the latest published likely-voter average once. Silver's published sentiment remains separately visible so readers can tell what Silver publishes from what the House model reconstructs and uses.

## Current validation status

Synthetic and behavioral tests are enforced in CI. The House structural layer passes whole-cycle 2022 and 2024 holdouts conditional on the realized national House margin: Brier 0.0290, log loss 0.1002, signed error -0.73 points, and 95.4% coverage for nominal 90% district intervals. Both official House seat outcomes are inside their 90% posterior intervals. The separate robust Senate race-update gate remains in production. Silver Bulletin's 2026 poll feed lacks a comparable public historical archive, so its provider-specific calibration remains externally unvalidated.

## Diagnostics contract

- House robust Bayesian regression: finite covariance, cluster-level uncertainty floors, structural holdout gate, and posterior-predictive seat checks.
- Analytic national update: one-aggregate likelihood invariant, finite posterior checks, and MCMC diagnostics marked not applicable.
- Economy-only prior: the economic coefficient, input uncertainty, and 3.5-point structural uncertainty remain explicit.
- Publication: any failed input validation or forecast bundle validation retains the prior successful public artifacts.

## Main limitations

- The economy coefficient is deliberately weak and estimated from few comparable midterms.
- The Senate parameter layer is regularized; its backtest validates the candidate-race update rather than every structural component.
- Silver Bulletin's aggregation is externally maintained and its full weighting model is not reproduced or audited here.
- Early-cycle candidate averages are sparse; absent, ambiguous, or third-party races remain fundamentals-only.
- Cook PVI is not MRP and can miss demographic change or map litigation.
- The model excludes fundraising, endorsements, scandals, and subjective candidate quality.
- Probabilities are conditional on model structure and source data.

## Reproducibility

Random seeds are deterministic, upstream data are snapshotted with checksums, and public artifacts include a content-derived run ID, explicit source dates, freshness, fallbacks, and inference method. See [METHODOLOGY.md](METHODOLOGY.md) for the full calculation.
