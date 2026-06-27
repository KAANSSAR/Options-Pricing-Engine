"""
Monte Carlo simulation option pricing.
Supports European options (calls and puts) with variance reduction.
Uses antithetic variates for variance reduction.
"""

import numpy as np


def price(
    S, K, T, r, sigma,
    option_type="call",
    n_simulations=100_000,
    n_steps=252,
    seed=42,
    antithetic=True,
):
    """
    Monte Carlo option price using Geometric Brownian Motion.

    Parameters
    ----------
    S : float — current stock price
    K : float — strike price
    T : float — time to expiry in years
    r : float — risk-free rate (annualised)
    sigma : float — volatility (annualised)
    option_type : str — 'call' or 'put'
    n_simulations : int — number of simulated paths
    n_steps : int — number of time steps per path
    seed : int — random seed for reproducibility
    antithetic : bool — use antithetic variates for variance reduction

    Returns
    -------
    dict with:
        price : float — estimated option price
        std_error : float — standard error of the estimate
        confidence_interval : tuple — 95% CI (lower, upper)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    discount = np.exp(-r * T)

    if antithetic:
        # Generate half the paths, reflect for the other half
        n_half = n_simulations // 2
        Z = rng.standard_normal((n_steps, n_half))
        Z = np.concatenate([Z, -Z], axis=1)  # antithetic pairs
        actual_sims = Z.shape[1]
    else:
        Z = rng.standard_normal((n_steps, n_simulations))
        actual_sims = n_simulations

    # Simulate paths using log-normal increments
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_returns = drift + diffusion * Z
    log_price_paths = np.log(S) + np.cumsum(log_returns, axis=0)
    ST = np.exp(log_price_paths[-1])  # terminal stock prices

    # Compute payoffs
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_payoffs = discount * payoffs
    price_est = float(np.mean(discounted_payoffs))
    std_err = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(actual_sims))
    ci_lower = price_est - 1.96 * std_err
    ci_upper = price_est + 1.96 * std_err

    return {
        "price": round(price_est, 6),
        "std_error": round(std_err, 6),
        "confidence_interval": (round(ci_lower, 6), round(ci_upper, 6)),
        "n_simulations": actual_sims,
    }


def simulate_paths(
    S, T, r, sigma,
    n_paths=50,
    n_steps=252,
    seed=42,
):
    """
    Simulate and return price paths for visualisation.

    Returns
    -------
    times : np.ndarray shape (n_steps+1,) — time axis
    paths : np.ndarray shape (n_steps+1, n_paths) — simulated price paths
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_steps, n_paths))

    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_returns = drift + diffusion * Z

    log_paths = np.zeros((n_steps + 1, n_paths))
    log_paths[0] = np.log(S)
    log_paths[1:] = np.log(S) + np.cumsum(log_returns, axis=0)

    paths = np.exp(log_paths)
    times = np.linspace(0, T, n_steps + 1)

    return times, paths