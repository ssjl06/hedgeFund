# CVaR Portfolio Optimization Web Server

A web application wrapping the Mean-CVaR portfolio optimization pipeline
(based on [NVIDIA's cvar_basic.ipynb](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/blob/main/notebooks/cvar_basic.ipynb))
into an interactive web interface.

## Architecture

```
User Input (tickers, dates, risk params)
        │
        ▼
  Flask Web Server (app.py)
        │
        ▼
  Optimization Engine (optimizer.py)
   ├─ yfinance: download latest price data
   ├─ Scenario generation: historical returns
   └─ scipy.optimize.linprog: solve Mean-CVaR LP
        │
        ▼
  Chart Generator (charts.py)
   └─ Plotly: 6 interactive charts
        │
        ▼
  Frontend (index.html)
   └─ Bootstrap 5 + Plotly.js
```

## Quick Start

```bash
cd webapp
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

Or use the launch script:
```bash
./run.sh          # dev mode  (port 5000)
./run.sh --prod   # gunicorn  (port 8000)
```

## User Inputs

| Parameter | Description | Default |
|-----------|-------------|---------|
| Tickers | Stock symbols (min 2) | — |
| Start / End Date | Historical data range | 2 years |
| Confidence Level (β) | CVaR tail percentile | 0.95 |
| Risk Aversion (λ) | 0 = min risk, 1 = max return | 0.50 |
| Min / Max Weight | Per-asset weight bounds | 0.0 / 1.0 |

## Output Charts

1. **Portfolio Weights** — Pie + Bar chart of optimal allocation
2. **Cumulative Returns** — Portfolio vs equal-weight & individual assets
3. **Efficient Frontier** — Mean-CVaR tradeoff curve
4. **Return Distribution** — Histogram with VaR & CVaR markers
5. **Correlation Matrix** — Asset return correlations heatmap
6. **Drawdown** — Portfolio drawdown over time

## API Endpoints

- `GET /` — Main UI
- `POST /api/optimize` — Run optimization (JSON body)
- `GET /api/presets` — Preset ticker groups

## Theory

The optimizer solves the Rockafellar-Uryasev (2000) LP formulation:

```
min  -λ·μ'w  +  (1-λ)·[α + 1/((1-β)S) · Σ z_s]

s.t.  z_s ≥ -r_s'w - α    ∀s
      z_s ≥ 0              ∀s
      Σ w_i = 1
      lo ≤ w_i ≤ hi
```

Where `w` = portfolio weights, `α` = VaR auxiliary, `z_s` = CVaR auxiliaries,
`r_s` = return scenarios, `β` = confidence level, `λ` = risk aversion.
