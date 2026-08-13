# Production implementation assumptions

The public forecast epoch begins on August 12, 2026. Earlier forecast rows used
different methods and are intentionally excluded from the published timelines
and from change comparisons.

All partisan leans and predicted margins use Democratic two-party margin:
positive values favor Democrats and negative values favor Republicans. House
parameters are calibrated on 2018, 2022, and map-comparable 2024 districts.
Cluster-level posterior standard-deviation floors are 1.5 points for the
intercept, 0.25 for the national coefficient, 0.5 for incumbency and regional
coefficients, and 0.08 for district lean. The national cycle-shock scale uses a
3-point regularizing prior because only three recent cycles are available.

The House national likelihood collapses Silver Bulletin's adjusted generic
poll file and published influence weights into one observation. A 1.25-point
correlated design-error floor prevents false precision. Silver's separately
published likely-voter maintained average is displayed as current sentiment,
but is not stacked as another House likelihood.

The national fundamentals prior uses economics only. The existing economic
coefficient, coefficient uncertainty, standardized-input uncertainty, and
3.5-point structural standard deviation are retained; economics enters once
before polling.

The Senate retains its regularized production parameter prior pending a
Senate-specific recalibration. It uses Silver's latest likely-voter maintained
average once at the national level and the latest validated maintained average
once for each covered candidate race.

Daily acquisition is fail-closed. Source freshness and fallback use are
recorded in `data/processed/input_manifest.json`; catastrophically stale inputs
cannot replace the public bundle. Forecast generation performs no network
requests and publishes House, Senate, and timeline artifacts atomically.
