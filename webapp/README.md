# CVaR Portfolio Optimization Web Server

A web application wrapping the Mean-CVaR portfolio optimization pipeline
(based on [NVIDIA's cvar_basic.ipynb](https://github.com/NVIDIA-AI-Blueprints/quantitative-portfolio-optimization/blob/main/notebooks/cvar_basic.ipynb))
into an interactive web interface.

## Architecture

```
Browser (any device on LAN / internet)
    │
    ▼
Windows Host (:80)  ←  port forwarding (netsh portproxy)
    │
    ▼
WSL2 Nginx (:80)    ←  reverse proxy, gzip, static files
    │
    ▼
Gunicorn (:8000)    ←  multi-worker WSGI server
    │
    ▼
Flask App (app.py)
    ├─ optimizer.py  →  yfinance + scipy Mean-CVaR LP
    └─ charts.py     →  Plotly interactive charts
```

## Quick Start (Development)

```bash
cd webapp
pip install -r requirements.txt
python app.py
# → http://localhost:5000 (only you can access)
```

## Production Deployment (LAN accessible)

### Step 1: WSL2 side (one-time setup)

```bash
cd /home/user/hedgeFund/webapp
sudo bash deploy/setup.sh
```

This installs and starts:
- Nginx (reverse proxy on port 80)
- Gunicorn (app server on port 8000)
- Log files in `/var/log/cvar-optimizer/`

### Step 2: Windows side (re-run after each WSL restart)

Open **PowerShell as Administrator**:

```powershell
cd \\wsl$\Ubuntu\home\user\hedgeFund\webapp\deploy
.\windows-port-forward.ps1
```

This creates:
- Port forwarding: Windows:80 → WSL2:80
- Firewall rule allowing inbound TCP 80

### Step 3: Access

```
Your machine:     http://localhost
LAN colleagues:   http://<your-windows-ip>
```

Find your Windows IP with `ipconfig` in CMD.

### Management commands

```bash
# Check status
ps aux | grep gunicorn
sudo service nginx status

# Restart
sudo service nginx restart
pkill -HUP -f gunicorn

# Logs
tail -f /var/log/cvar-optimizer/access.log
tail -f /var/log/cvar-optimizer/error.log
tail -f /var/log/nginx/cvar-optimizer.error.log

# Stop everything
pkill -f gunicorn
sudo service nginx stop
```

### Teardown (Windows side)

```powershell
.\windows-port-forward.ps1 -Remove
```

## File Structure

```
webapp/
├── app.py                      # Flask application
├── optimizer.py                # Mean-CVaR optimization engine
├── charts.py                   # Plotly chart generators
├── templates/
│   └── index.html              # Frontend (Bootstrap 5 + Plotly.js)
├── static/                     # Static assets (served by Nginx)
├── requirements.txt            # Python dependencies
├── run.sh                      # Dev/prod launcher
├── deploy/
│   ├── setup.sh                # One-command WSL deployment
│   ├── nginx.conf              # Nginx reverse proxy config
│   ├── gunicorn.conf.py        # Gunicorn production config
│   ├── cvar-optimizer.service  # Systemd service unit
│   └── windows-port-forward.ps1  # Windows port forwarding
└── README.md
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
