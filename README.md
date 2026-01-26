# 2026 House Forecast

A Bayesian hierarchical forecasting model for the 2026 U.S. House of Representatives elections.

## Live Demo

View the forecast at: `website/index.html` (open in browser)

## Features

- **Probabilistic Predictions**: Full probability distributions for all 435 districts
- **Monte Carlo Simulation**: 10,000 election scenarios with proper uncertainty
- **Three-Layer Model**:
  - National environment (polls + fundamentals)
  - Regional effects (Northeast, South, Midwest, West)
  - District-level predictions (PVI + incumbency)
- **Interactive Website**: Hero dashboard, district table, seat distribution chart

## Quick Start

```bash
# 1. Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env with your FRED API key
cp .env.example .env
# Edit .env and add your key

# 4. Fetch data
python scripts/fetch_data.py

# 5. Generate forecast
python scripts/generate_forecast.py

# 6. View website
open website/index.html
```

## Project Structure

```
2026-midterms-forecast/
├── data/
│   ├── raw/                    # Fetched data (polls, economic)
│   │   ├── generic_ballot.csv
│   │   ├── approval.csv
│   │   └── economic/
│   └── processed/
│       └── districts.csv       # 435 district fundamentals
├── models/
│   └── forecast.py             # Bayesian hierarchical model
├── scripts/
│   ├── fetch_data.py           # Data pipeline
│   └── generate_forecast.py    # Run model, output JSON
├── outputs/
│   ├── forecast.json           # Model predictions
│   └── timeline.csv            # Historical tracking
├── website/
│   ├── index.html              # Main page
│   ├── css/style.css
│   ├── js/app.js
│   └── forecast.json           # Copy for website
├── requirements.txt
└── README.md
```

## Data Sources

- **FRED API**: Economic indicators (unemployment, GDP, income)
- **Polling**: Generic ballot and presidential approval
- **Fundamentals**: District PVI from 2024 presidential results

## Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Incumbency Advantage | 3.0 pts | Boost for incumbent party |
| Midterm Penalty | 3.5 pts | Out-party bonus in midterms |
| National Uncertainty | 3.0 SD | Uncertainty in national environment |
| District Uncertainty | 4.5 SD | District-level variation |

## Output Format

`forecast.json` contains:

```json
{
  "metadata": {
    "updated_at": "2026-01-25T...",
    "days_until_election": 281
  },
  "summary": {
    "prob_dem_majority": 0.82,
    "median_dem_seats": 273,
    "ci_90_low": 174,
    "ci_90_high": 364
  },
  "categories": {
    "dem": { "safe": 151, "likely": 60, "lean": 41 },
    "toss_up": 35,
    "rep": { "safe": 38, "likely": 45, "lean": 65 }
  },
  "districts": [
    {
      "id": "CA-13",
      "prob_dem": 0.63,
      "category": "lean_d",
      ...
    }
  ]
}
```

## Automation

Run daily with cron or GitHub Actions:

```bash
# Fetch fresh data and regenerate forecast
python scripts/fetch_data.py && python scripts/generate_forecast.py
```

## License

MIT
