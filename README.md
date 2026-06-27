# Options Pricing Engine

[![CI](https://github.com/KAANSSAR/options-pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/KAANSSAR/options-pricer/actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, well-tested option pricing engine implementing three canonical models with live market data integration and an interactive Streamlit dashboard.

## Models

| Model | Type | Supports |
|---|---|---|
| Black-Scholes | Closed-form analytical | European calls & puts |
| Binomial Tree (CRR) | Lattice / backward induction | European & American |
| Monte Carlo (GBM) | Simulation | European, with variance reduction |

All five Greeks computed analytically via Black-Scholes: **Δ Delta, Γ Gamma, ν Vega, Θ Theta, ρ Rho**.

## Project structure

```
options-pricer/
├── src/
│   └── options_pricer/
│       ├── core/
│       │   ├── black_scholes.py     # Analytical pricing + Greeks
│       │   ├── binomial_tree.py     # CRR tree, European + American
│       │   └── monte_carlo.py       # GBM simulation, antithetic variates
│       └── data/
│           └── market_data.py       # yfinance wrapper
├── app/
│   └── dashboard.py                 # Streamlit UI (5 tabs)
├── notebooks/
│   └── options_pricing.ipynb        # Step-by-step walkthrough
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── test_black_scholes.py        # 20 tests
│   ├── test_binomial_tree.py        # 12 tests
│   └── test_monte_carlo.py          # 14 tests
├── .github/workflows/ci.yml         # GitHub Actions CI (Python 3.10–3.12)
├── pyproject.toml
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/KAANSSAR/options-pricer.git
cd options-pricer
pip install -e ".[dev]"

# Run tests
pytest

# Launch dashboard
streamlit run app/dashboard.py
```

## Usage

```python
from options_pricer.core import black_scholes as bs
from options_pricer.core import binomial_tree as bt
from options_pricer.core import monte_carlo as mc

S, K, T, r, sigma = 150.0, 155.0, 0.25, 0.05, 0.25

# Price a European call
bs.price(S, K, T, r, sigma, "call")       # → 6.1138
bt.price(S, K, T, r, sigma, "call")       # → 6.1133  (200 steps, CRR)
mc.price(S, K, T, r, sigma, "call")       # → {"price": 6.127, "std_error": ..., "confidence_interval": ...}

# All Greeks (analytical)
bs.greeks(S, K, T, r, sigma, "call")
# → {"delta": 0.460, "gamma": 0.0212, "vega": 0.298, "theta": -0.049, "rho": 0.157}

# American put (early exercise premium)
bt.price(S, K, T, r, sigma, "put", style="american")

# Live market data
from options_pricer.data.market_data import get_stock_data, get_risk_free_rate
data = get_stock_data("AAPL")    # current price, 30d/252d historical vol
r    = get_risk_free_rate()      # US 13-week T-bill yield
```

## Dashboard

Five tabs: **Pricing** (model comparison) · **Greeks** (surface charts) · **Binomial Tree** (convergence + heatmap) · **Monte Carlo** (path simulation + distribution) · **Sensitivity** (vol surface + P&L diagram + scenario table).

Toggle live data in the sidebar to auto-populate spot price and historical volatility for any ticker.

## Implementation notes

- **Black-Scholes**: Greeks computed analytically — no finite differences. Put-call parity holds to machine precision (~1e-14).
- **Binomial tree**: CRR parameterisation (`u = e^(σ√Δt)`, `d = 1/u`). Vectorised terminal payoffs, backward induction loop. American early exercise checked at every node.
- **Monte Carlo**: log-normal increments under risk-neutral measure. Antithetic variates halve variance without doubling computation. Returns 95% CI and standard error alongside price.
- **Market data**: live spot price and historical volatility via `yfinance`; risk-free rate approximated from US 13-week T-bill yield (`^IRX`) with fallback to 5%.

## Running tests

```bash
pytest                          # all tests
pytest -v                       # verbose
pytest --cov=src/options_pricer # with coverage
pytest tests/test_black_scholes.py  # single module
```
