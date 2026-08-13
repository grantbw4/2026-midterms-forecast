# Grant's Election Forecast

A transparent Bayesian forecast for the 2026 U.S. House and Senate elections.

**Live dashboard:** [grantbw4.github.io/2026-midterms-forecast](https://grantbw4.github.io/2026-midterms-forecast/)

**Start with the methodology:** [METHODOLOGY.md](METHODOLOGY.md) explains every national metric, the House/Senate differences, the race model, the simulation, and how to interpret the probabilities.

## How the model works

The forecast follows six auditable steps:

1. Fetch and validate Silver Bulletin polling, five FRED economic series, and FEC/Clerk candidate records.
2. Turn the five economic changes into a broad, economy-only national prior.
3. Update that prior once with a chamber-specific national polling input.
4. Build a fundamentals distribution for every race from partisan lean, incumbency, national conditions, regional movement, and local error.
5. Update covered races with the latest valid Silver Democratic-versus-Republican maintained average.
6. Count seats in 10,000 correlated simulations.

The House uses one influence-weighted aggregate of Silver's adjusted generic-ballot poll file. The Senate uses Silver's latest published likely-voter average once. The published Silver average, the model polling input, the poll-updated current margin, and the Election Day forecast are separate quantities and are labeled separately everywhere.

In this project, “fundamentals prior” means economics only.

## Model sketch

```text
economic composite:   z_econ = standardize(weighted five-series changes)
fundamentals prior:   theta_election ~ Normal(-0.34 * z_econ, sigma_fundamentals)
national update:      chamber_polling_input ~ Student-t(theta_current, sigma_input)
House race prior:     margin[r] ~ Student-t(intercept + lean + incumbency + national + region, sigma_race)
race update:          silver_average_latest[r] ~ Student-t(margin[r], sigma_average + common_error)
control probability:  share of correlated simulations reaching 218 House or 51 Senate seats
```

The prior standard deviation includes a 3.5-point structural term, uncertainty in the economic coefficient, and uncertainty in the standardized economic input. Economics enters only in the prior; it is not added again after polling.

## Reproduce the forecast

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m pytest -q
python3 scripts/generate_forecast.py
```

Forecast generation is deterministic and network-free. Daily acquisition runs separately with `python3 scripts/fetch_inputs.py`, requires `FRED_API_KEY`, and writes a validated manifest. Invalid or catastrophically stale inputs cannot replace the existing public bundle.

## Public outputs

The House and Senate JSON artifacts contain:

- An explicit `national_environment` object with the economy-only prior, published Silver sentiment, chamber polling input, posterior current margin, Election Day forecast, economic composite, and all five economic components.
- Per-source observation dates and live freshness thresholds.
- Race-level prior and posterior margins, 90% credible intervals, probabilities, polling use, source links, and data quality.
- Internal schema and model versions for compatibility; these are not reader-facing branding.

The public timeline begins on August 12, 2026. Same-day runs replace that date's baseline instead of adding a duplicate.

## Data sources

| Purpose | Source |
|---|---|
| Published generic ballot and adjusted poll input | [Silver Bulletin generic ballot](https://www.natesilver.net/p/generic-ballot-average-2026-nate-silver-bulletin-congress-polls) |
| Candidate-race maintained averages | [Silver Bulletin 2026 forecast](https://www.natesilver.net/p/nate-silver-2026-midterm-election-polls-model) |
| Economic fundamentals | [FRED](https://fred.stlouisfed.org/) |
| Current House district lean | [Cook Political Report](https://www.cookpolitical.com/races) |
| Candidate identity | [FEC candidate master](https://www.fec.gov/campaign-finance-data/candidate-master-file-description/) |
| Current House roster | [Clerk of the House](https://clerk.house.gov/xml/lists/MemberData.xml) |

## Validation and limits

Run `python3 -m pytest -q` for the production contract and behavioral checks. The House structural layer passes whole-cycle 2022 and 2024 holdouts conditional on the realized national House margin: 0.0290 Brier score, 0.1002 log loss, -0.73-point signed error, and 95.4% coverage for nominal 90% district intervals. This validates the House margin-to-seat layer, not Silver's 2026 polling calibration.

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, validation, and limitations.

## License

Code is MIT. Upstream data remain subject to provider terms.
