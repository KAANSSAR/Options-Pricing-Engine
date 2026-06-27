"""
Binomial tree option pricing model.
Supports European and American options (calls and puts).
Uses Cox-Ross-Rubinstein (CRR) parameterisation.
"""

import numpy as np


def price(S, K, T, r, sigma, option_type="call", style="european", steps=200):
    """
    Binomial tree option price (CRR).

    Parameters
    ----------
    S : float — current stock price
    K : float — strike price
    T : float — time to expiry in years
    r : float — risk-free rate (annualised)
    sigma : float — volatility (annualised)
    option_type : str — 'call' or 'put'
    style : str — 'european' or 'american'
    steps : int — number of time steps (higher = more accurate, slower)

    Returns
    -------
    float — option price
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))       # up factor
    d = 1 / u                             # down factor (CRR)
    p = (np.exp(r * dt) - d) / (u - d)   # risk-neutral probability

    # Terminal stock prices (vectorised)
    j = np.arange(steps + 1)
    ST = S * (u ** (steps - j)) * (d ** j)

    # Terminal payoffs
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Backward induction
    discount = np.exp(-r * dt)
    for i in range(steps - 1, -1, -1):
        payoffs = discount * (p * payoffs[:-1] + (1 - p) * payoffs[1:])
        if style == "american":
            # Early exercise check
            j_i = np.arange(i + 1)
            ST_i = S * (u ** (i - j_i)) * (d ** j_i)
            if option_type == "call":
                intrinsic = np.maximum(ST_i - K, 0)
            else:
                intrinsic = np.maximum(K - ST_i, 0)
            payoffs = np.maximum(payoffs, intrinsic)

    return float(payoffs[0])


def price_grid(S, K, T, r, sigma, option_type="call", style="european", steps=50):
    """
    Return full binomial price tree for visualisation.
    Uses fewer steps to keep the output manageable.

    Returns
    -------
    stock_tree : np.ndarray (steps+1, steps+1) — stock prices at each node
    price_tree : np.ndarray (steps+1, steps+1) — option prices at each node
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Build stock tree
    stock_tree = np.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)

    # Terminal payoffs
    price_tree = np.zeros((steps + 1, steps + 1))
    if option_type == "call":
        price_tree[:, steps] = np.maximum(stock_tree[:, steps] - K, 0)
    else:
        price_tree[:, steps] = np.maximum(K - stock_tree[:, steps], 0)

    # Backward induction
    discount = np.exp(-r * dt)
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            hold = discount * (p * price_tree[j, i + 1] + (1 - p) * price_tree[j + 1, i + 1])
            if style == "american":
                if option_type == "call":
                    intrinsic = max(stock_tree[j, i] - K, 0)
                else:
                    intrinsic = max(K - stock_tree[j, i], 0)
                price_tree[j, i] = max(hold, intrinsic)
            else:
                price_tree[j, i] = hold

    return stock_tree, price_tree