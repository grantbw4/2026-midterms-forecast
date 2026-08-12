# 2026 Midterms Forecast v4

A dynamic Bayesian forecast for the 2026 U.S. House and Senate elections. House v4 keeps all structural quantities on the Democratic two-party-margin scale and explicitly preserves cycle-level uncertainty.

**Live dashboard:** [grantbw4.github.io/2026-midterms-forecast](https://grantbw4.github.io/2026-midterms-forecast/)

## What v4 demonstrates

- A single national Bayesian likelihood built from Silver Bulletin's adjusted generic-ballot poll universe and its published influence weights; Bulletin's pollster house effects are not estimated twice.
- A fundamentals prior from approval and economic uncertainty, used once rather than added after polling.
- One externally aggregated Silver Bulletin likelihood per covered race, robustly combined with the fundamentals posterior.
- Official candidate validation from FEC records; ambiguous, third-party, and unmapped matchups remain fundamentals-only.
- A House-specific robust hierarchical calibration on 2018, 2022, and comparable 2024 districts, with shared national and regional posterior draws.
- Current-map Cook PVI values for all 435 districts, including mid-cycle redistricting, with row-level provenance and fail-closed validation.
- Fundamentals-only behavior for unpolled or unresolved races—no invented poll estimate.
- Atomic, fail-closed publication with immutable source snapshots and explicit degraded status.
- Synthetic recovery, behavioral verification, output schemas, and a rolling-origin backtest promotion gate.

## Reproduce a forecast

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python scripts/generate_forecast.py
```

The live run uses only free public sources. `FEC_API_KEY` is optional because candidate identity uses the FEC bulk candidate master.

For an offline deterministic run using the latest valid cache:

```bash
python scripts/generate_forecast.py --skip-fetch --skip-race-fetch --skip-timeline
```

If inputs are invalid, catastrophically stale, or diagnostics fail, public JSON is not replaced. A valid cache may be used, but that fact is exposed in `metadata.fallbacks` and `metadata.model_status`.

## Model sketch

```text
fundamentals prior:   election_margin ~ Normal(approval + economy, structural_error)
national likelihood: bulletin_adjusted_poll_aggregate ~ Student-t(theta_current, sigma_aggregate)
House race prior:     margin[r] ~ Student-t(intercept + lean + incumbency + national + region, sigma_race)
race likelihood:      silver_average_latest[r] ~ Student-t(margin[r], sigma_aggregate + common_error)
```

The national update is analytic, so MCMC convergence statistics do not apply. Bulletin-adjusted polls are collapsed to one influence-weighted likelihood rather than treated as independent posterior updates. House parameter covariance and national-cycle variance retain explicit floors because hundreds of district rows represent only three election cycles.

## Outputs

Every race exposes `prior_margin`, `posterior_margin`, a 90% credible interval, `prob_dem`, `polling_adjustment`, `polls_used`, `latest_poll_date`, source URLs, and `data_quality`. Forecast metadata includes stable schema and model versions, `run_id`, `data_through`, freshness, inference method, diagnostics, and fallbacks.

See [METHODOLOGY.md](METHODOLOGY.md) for assumptions and [MODEL_CARD.md](MODEL_CARD.md) for validation and limitations.

## Data sources

| Purpose | Source |
|---|---|
| Generic-ballot polls | [Silver Bulletin generic ballot](https://www.natesilver.net/p/generic-ballot-average-2026-nate-silver-bulletin-congress-polls), adjusted public poll file |
| Candidate-race likelihoods | [Silver Bulletin 2026 forecast](https://www.natesilver.net/p/nate-silver-2026-midterm-election-polls-model), public maintained-average feed |
| Current House district lean | [Cook Political Report race table](https://www.cookpolitical.com/races), current-map Cook PVI |
| Approval prior input | [VoteHub API](https://votehub.com/polls/api/) |
| Candidate identity | [FEC candidate master](https://www.fec.gov/campaign-finance-data/candidate-master-file-description/) |
| Current House roster | [Clerk of the House](https://clerk.house.gov/xml/lists/MemberData.xml) |
| District lean / historical results | Processed public election results with source and effective-date fields |

## Validation

Run behavioral tests with `python -m pytest -q`. Rolling-origin evaluation expects frozen historical prediction snapshots:

```bash
python scripts/backtest_v3.py --input data/backtests/predictions.csv
```

The House structural layer passes whole-cycle holdouts for 2022 and 2024 conditional on the realized national House margin. Across 740 contested, map-comparable districts, v4 records a 0.0290 Brier score, 0.1002 log loss, −0.73-point signed error, and 95.4% coverage for nominal 90% intervals, improving on legacy v3. Both official seat outcomes fall inside the 90% posterior interval. This validates the House margin-to-seat layer, not Bulletin's 2026 polling calibration.

## License

Code is MIT. Upstream data remain subject to their providers' terms; the repository stores only attributed maintained-average snapshots needed for reproducibility, not Silver Bulletin's underlying poll database or proprietary forecast outputs.
