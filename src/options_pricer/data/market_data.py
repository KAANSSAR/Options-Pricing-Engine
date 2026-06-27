"""
Market data fetcher using yfinance.
Fetches current price, historical volatility, and options chain data.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime


def get_stock_data(ticker: str, period: str = "1y"):
    """
    Fetch stock data and compute historical volatility.

    Parameters
    ----------
    ticker : str — e.g. 'AAPL', 'SPY', 'TSLA'
    period : str — yfinance period string, e.g. '1y', '6mo'

    Returns
    -------
    dict with:
        ticker : str
        current_price : float
        hist_volatility_30d : float — 30-day annualised vol
        hist_volatility_252d : float — 252-day annualised vol
        history : pd.DataFrame
    """
    tkr = yf.Ticker(ticker)
    hist = tkr.history(period=period)

    if hist.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

    current_price = float(hist["Close"].iloc[-1])

    # Log returns
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

    vol_30 = float(log_returns.tail(30).std() * np.sqrt(252))
    vol_252 = float(log_returns.std() * np.sqrt(252))

    return {
        "ticker": ticker.upper(),
        "current_price": round(current_price, 4),
        "hist_volatility_30d": round(vol_30, 4),
        "hist_volatility_252d": round(vol_252, 4),
        "history": hist,
    }


def get_risk_free_rate():
    """
    Approximate risk-free rate using the 13-week US T-bill yield (^IRX).
    Falls back to 5% if unavailable.
    """
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if not hist.empty:
            rate = float(hist["Close"].iloc[-1]) / 100
            return round(rate, 4)
    except Exception:
        pass
    return 0.05  # fallback


def get_options_chain(ticker: str):
    """
    Fetch the real options chain for a ticker.

    Returns
    -------
    dict with:
        expiry_dates : list[str]
        calls : pd.DataFrame (for nearest expiry)
        puts : pd.DataFrame (for nearest expiry)
        nearest_expiry : str
    """
    tkr = yf.Ticker(ticker)
    expirations = tkr.options
    if not expirations:
        raise ValueError(f"No options data found for '{ticker}'.")

    nearest = expirations[0]
    chain = tkr.option_chain(nearest)

    return {
        "expiry_dates": list(expirations),
        "calls": chain.calls,
        "puts": chain.puts,
        "nearest_expiry": nearest,
    }


def time_to_expiry(expiry_str: str) -> float:
    """Convert an expiry date string (YYYY-MM-DD) to years from today."""
    expiry = datetime.strptime(expiry_str, "%Y-%m-%d")
    today = datetime.today()
    days = (expiry - today).days
    return max(days / 365, 1 / 365)  # floor at 1 day