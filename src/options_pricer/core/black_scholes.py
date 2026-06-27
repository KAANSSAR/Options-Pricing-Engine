"""
Black-Scholes analytical option pricing model.
Supports European calls and puts.
"""

import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma):
    """Compute d1 term in Black-Scholes formula."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma):
    """Compute d2 term in Black-Scholes formula."""
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes option price.

    Parameters
    ----------
    S : float — current stock price
    K : float — strike price
    T : float — time to expiry in years
    r : float — risk-free rate (annualised, e.g. 0.05 for 5%)
    sigma : float — implied volatility (annualised, e.g. 0.20 for 20%)
    option_type : str — 'call' or 'put'

    Returns
    -------
    float — option price
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        return max(K - S, 0)

    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)

    if option_type == "call":
        return S * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def greeks(S, K, T, r, sigma, option_type="call"):
    """
    Compute all Black-Scholes Greeks analytically.

    Returns
    -------
    dict with keys: delta, gamma, vega, theta, rho
    """
    if T <= 0:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}

    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)

    # Gamma and vega are same for call/put
    gamma = norm.pdf(_d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(_d1) * np.sqrt(T) / 100  # per 1% vol move

    if option_type == "call":
        delta = norm.cdf(_d1)
        theta = (
            -S * norm.pdf(_d1) * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(_d2)
        ) / 365  # per day
        rho = K * T * np.exp(-r * T) * norm.cdf(_d2) / 100  # per 1% rate move
    elif option_type == "put":
        delta = norm.cdf(_d1) - 1
        theta = (
            -S * norm.pdf(_d1) * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-_d2)
        ) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-_d2) / 100
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {
        "delta": round(delta, 6),
        "gamma": round(gamma, 6),
        "vega": round(vega, 6),
        "theta": round(theta, 6),
        "rho": round(rho, 6),
    }