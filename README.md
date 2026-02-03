# 2026 Midterms Forecast

A Bayesian hierarchical forecasting model for the 2026 U.S. House and Senate elections.

## Live Forecast

**[grantbw4.github.io/2026-midterms-forecast](https://grantbw4.github.io/2026-midterms-forecast/)**

Updated daily at 9am ET via GitHub Actions.

## Features

- **House & Senate Forecasts**: Full probability distributions for all 435 House districts and 33 Senate races
- **Bayesian Inference**: National environment estimated via PyMC with pollster house effects
- **Monte Carlo Simulation**: 10,000 election scenarios with correlated regional effects
- **Hierarchical Model**:
  - National environment (generic ballot polls + presidential approval)
  - Regional effects (10 FiveThirtyEight-style political regions)
  - District/state-level predictions (PVI + incumbency)
- **Interactive Website**: Live probability dashboard, interactive maps, race tables, and historical timeline

## Model Overview

The model combines polling data with district fundamentals to generate probabilistic forecasts:

1. **National Environment**: Bayesian inference on generic ballot polls with pollster house effects, adjusted for presidential approval
2. **District Vote Share**: `vote_share = 50 + β_pvi × PVI + β_inc × Inc + μ_region + β_nat × μ_national + ε`
3. **Monte Carlo**: 10,000 simulations sampling from posterior distributions at each level

Parameters were fitted on 2018 and 2022 midterm results (R² = 0.94, RMSE = 3.9 points).

## Quick Start

```bash
# 1. Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env with your API keys
cp .env.example .env
# Edit .env and add your keys

# 4. Fetch data
python scripts/fetch_data.py
python scripts/fetch_votehub.py
python scripts/fetch_cook_ratings.py

# 5. Generate forecast
python scripts/generate_forecast.py

# 6. View website
open website/index.html
```

## Project Structure

```
2026-midterms-forecast/
├── data/
│   ├── raw/                    # Fetched data
│   │   ├── generic_ballot.csv
│   │   ├── approval.csv
│   │   └── cook_ratings.json
│   └── processed/
│       ├── districts.csv       # 435 House district fundamentals
│       ├── senate_races.csv    # 33 Senate races
│       └── learned_params.json # Fitted model parameters
├── models/
│   ├── national_environment.py # PyMC Bayesian inference
│   ├── hierarchical_model.py   # House Monte Carlo simulation
│   ├── senate_forecast.py      # Senate Monte Carlo simulation
│   ├── parameter_fitting.py    # Historical parameter training
│   └── forecast.py             # Main forecast orchestration
├── scripts/
│   ├── fetch_data.py           # FRED economic data
│   ├── fetch_votehub.py        # Polling data from VoteHub API
│   ├── fetch_cook_ratings.py   # Cook Political ratings scraper
│   └── generate_forecast.py    # Run model, output JSON
├── outputs/
│   ├── forecast.json           # House predictions
│   ├── senate_forecast.json    # Senate predictions
│   └── timeline.json           # Historical tracking
├── website/
│   ├── index.html
│   ├── css/style.css
│   ├── js/app.js
│   ├── forecast.json
│   └── senate_forecast.json
├── .github/workflows/
│   ├── daily-forecast.yml      # Daily update automation
│   └── deploy-pages.yml        # GitHub Pages deployment
└── requirements.txt
```

## Data Sources

| Data | Source |
|------|--------|
| Generic Ballot Polls | [VoteHub API](https://votehub.com) |
| Presidential Approval | [VoteHub API](https://votehub.com) |
| District PVI | Wikipedia (2024 presidential results) |
| Race Ratings | [Cook Political Report](https://cookpolitical.com) |
| Congressional Maps | U.S. Census TIGER/Line |

## Fitted Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| β_pvi | 0.48 | PVI coefficient |
| β_inc | 2.2 | Incumbency advantage |
| β_nat | 0.66 | National environment coefficient |
| σ_regional | 0.54 | Regional effect standard deviation |
| σ_district | 3.7 | Base district uncertainty |

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for full technical details, or the methodology section on the [live website](https://grantbw4.github.io/2026-midterms-forecast/#methodology).

## License

MIT
