# 2026 Midterms Forecast v3

A dynamic Bayesian forecast for the 2026 U.S. House and Senate elections. The project is designed as an auditable data-science portfolio piece: assumptions, likelihoods, uncertainty propagation, data provenance, diagnostics, and validation gates are visible in the code and public output.

**Live dashboard:** [grantbw4.github.io/2026-midterms-forecast](https://grantbw4.github.io/2026-midterms-forecast/)

## What v3 demonstrates

- A Bayesian update from Silver Bulletin's maintained generic-ballot average, with an explicit guard against treating correlated daily averages as independent evidence.
- A fundamentals prior from approval and economic uncertainty, used once rather than added after polling.
- One externally aggregated Silver Bulletin likelihood per covered race, robustly combined with the fundamentals posterior.
- Official candidate validation from FEC records; ambiguous, third-party, and unmapped matchups remain fundamentals-only.
- Shared national and regional posterior draws, preserving correlated chamber outcomes.
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
national likelihood: silver_average_latest ~ Student-t(theta_current, sigma_aggregate)
race prior:           margin[r] ~ Student-t(partisan_lean + incumbency + national + region, sigma_race)
race likelihood:      silver_average_latest[r] ~ Student-t(margin[r], sigma_aggregate + common_error)
```

The national update is analytic, so MCMC convergence statistics do not apply to that step. Only the latest Silver value enters the likelihood; its history is displayed but never multiplied as repeated evidence. Historical parameter fits are offline and may be published only with R-hat < 1.01, bulk ESS > 400, and zero divergences.

## Outputs

Every race exposes `prior_margin`, `posterior_margin`, a 90% credible interval, `prob_dem`, `polling_adjustment`, `polls_used`, `latest_poll_date`, source URLs, and `data_quality`. Forecast metadata includes v3 schema and model versions, `run_id`, `data_through`, freshness, inference method, diagnostics, and fallbacks.

See [METHODOLOGY.md](METHODOLOGY.md) for assumptions and [MODEL_CARD.md](MODEL_CARD.md) for validation and limitations.

## Data sources

| Purpose | Source |
|---|---|
| Generic ballot and race likelihoods | [Silver Bulletin 2026 forecast](https://www.natesilver.net/p/nate-silver-2026-midterm-election-polls-model), public maintained-average feed |
| Approval prior input | [VoteHub API](https://votehub.com/polls/api/) |
| Candidate identity | [FEC candidate master](https://www.fec.gov/campaign-finance-data/candidate-master-file-description/) |
| Current House roster | [Clerk of the House](https://clerk.house.gov/xml/lists/MemberData.xml) |
| District lean / historical results | Processed public election results with source and effective-date fields |

## Validation

Run behavioral tests with `python -m pytest -q`. Rolling-origin evaluation expects frozen historical prediction snapshots:

```bash
python scripts/backtest_v3.py --input data/backtests/predictions.csv
```

The underlying robust race-update design passes its Senate promotion gate on 275 matched final-60-day forecasts from 2018–2024: Brier score improves from 0.0910 to 0.0584 and log loss from 0.2933 to 0.1909. Each holdout fits fundamentals, pollster effects, and the correlated-error floor using earlier cycles only. Silver's 2026 maintained averages do not have a comparable public historical archive, so this result supports the update/error model but is not presented as an out-of-sample validation of Silver's averages themselves.

## License

Code is MIT. Upstream data remain subject to their providers' terms; the repository stores only attributed maintained-average snapshots needed for reproducibility, not Silver Bulletin's underlying poll database or proprietary forecast outputs.
