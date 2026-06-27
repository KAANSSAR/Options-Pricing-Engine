# Options Pricing Engine

[![CI](https://github.com/KAANSSAR/Options-Pricing-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/KAANSSAR/Options-Pricing-Engine/actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-220%20passing-brightgreen)](https://github.com/KAANSSAR/Options-Pricing-Engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-quality quantitative finance library implementing three canonical options pricing models from scratch, with a live Streamlit dashboard, implied volatility solver, real-time market data, and 220 pytest tests across Python 3.10–3.12.

## Models

| Model | Type | Supports |
|---|---|---|
| Black-Scholes | Closed-form analytical | European calls & puts |
| Binomial Tree (CRR) | Lattice / backward induction | European & American |
| Monte Carlo (GBM) | Simulation | European, with antithetic variance reduction |

All five Greeks computed analytically: **Δ Delta, Γ Gamma, ν Vega, Θ Theta, ρ Rho** — verified against finite differences.

## Project Structure

```
Options-Pricing-Engine/
├── src/
│   └── options_pricer/
│       ├── core/
│       │   ├── black_scholes.py     # Analytical pricing + Greeks
│       │   ├── binomial_tree.py     # CRR tree, European + American
│       │   └── monte_carlo.py       # GBM simulation, antithetic variates
│       └── data/
│           └── market_data.py       # yfinance wrapper
├── app/
│   └── dashboard.py                 # Streamlit dashboard (7 tabs)
├── ticker_search.py                 # Local fuzzy ticker search (~220 symbols)
├── notebooks/
│   └── options_pricing.ipynb        # Step-by-step walkthrough
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── test_black_scholes.py        # 80 tests
│   ├── test_binomial_tree.py        # 48 tests
│   ├── test_monte_carlo.py          # 48 tests
│   ├── test_iv_solver.py            # 30 tests
│   └── test_market_data.py          # 20 tests (6 total: 220 passing)
├── .github/workflows/ci.yml         # GitHub Actions CI (Python 3.10–3.12)
├── pyproject.toml
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/KAANSSAR/Options-Pricing-Engine.git
cd Options-Pricing-Engine
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

# Implied volatility (Brent's method)
from scipy.optimize import brentq
def implied_vol(market_price, S, K, T, r, option_type):
    return brentq(lambda s: bs.price(S, K, T, r, s, option_type) - market_price, 1e-6, 10.0)

# Live market data
from options_pricer.data.market_data import get_stock_data, get_risk_free_rate
data = get_stock_data("AAPL")    # current price, 30d/252d historical vol
r    = get_risk_free_rate()      # US 13-week T-bill yield
```

## Dashboard

Seven tabs built with Streamlit and Plotly:

- **Pricing** — model comparison (BS / Binomial / MC), intrinsic vs time value decomposition, put-call parity verification
- **Greeks** — animated curve charts for all 5 Greeks with live interpretation text
- **Binomial Tree** — European vs American prices, early exercise premium, convergence chart, node heatmap
- **Monte Carlo** — GBM path simulation, terminal price distribution, probability of profit, convergence vs simulations
- **Sensitivity** — rotating 3D vol surface (Vol × Time), P&L diagram, scenario analysis table
- **IV Solver** — back-solve implied volatility from a market price, vol smile chart, live options chain with real IV
- **Help** — parameter guide, Greeks reference, model descriptions, FAQ

Live data features: search by ticker symbol or company name with instant suggestions, auto-populates spot price and historical volatility, locks stock price to live feed, interactive stock chart (1D candlestick through 5Y line) with timeframe switching.

## Implementation Notes

- **Black-Scholes**: Greeks computed analytically — no finite differences. Put-call parity holds to machine precision (~1e-14). All 5 Greeks verified against central finite differences in the test suite.
- **Binomial Tree**: CRR parameterisation (`u = e^(σ√Δt)`, `d = 1/u`). Vectorised terminal payoffs, backward induction loop. American early exercise checked at every node. Convergence to BS verified across 8 step counts.
- **Monte Carlo**: log-normal increments under the risk-neutral measure. Antithetic variates halve variance without doubling computation. Returns 95% CI and standard error alongside price. E[S_T] verified to equal S·e^(rT) within 1%.
- **IV Solver**: Brent's method with bracket [1e-6, 10.0]. Round-trip accuracy < 1e-4 across 7 volatility levels and 162 parameter combinations.
- **Market data**: live spot price and historical volatility via `yfinance`; risk-free rate from US 13-week T-bill yield (`^IRX`) with 5% fallback.

## Test Coverage

```
220 tests across 5 modules — all passing on Python 3.10, 3.11, 3.12

test_black_scholes.py  (80)  — known values, put-call parity (162 combos), bounds,
                               monotonicity, edge cases, all 5 Greeks vs finite diff
test_binomial_tree.py  (48)  — BS convergence, American ≥ European (grid sweep),
                               early exercise premium, price_grid node verification
test_monte_carlo.py    (48)  — CI coverage, SE scaling (√N), antithetic variance
                               reduction, GBM E[S_T] = S·e^rT, path properties
test_iv_solver.py      (30)  — round-trip accuracy, monotonicity, no-arbitrage bounds,
                               deep ITM/OTM, high vol, short/long expiry
test_market_data.py    (20)  — time_to_expiry correctness, floor enforcement,
                               risk-free rate plausibility
```

## Running Tests

```bash
pytest                               # all 220 tests
pytest -v                            # verbose
pytest --cov=src/options_pricer      # with coverage
pytest tests/test_black_scholes.py   # single module
```