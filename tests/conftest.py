"""Shared pytest fixtures for the options pricer test suite."""

import pytest


@pytest.fixture
def base_params():
    """Standard option parameters used across tests."""
    return {
        "S": 100.0,   # stock price
        "K": 100.0,   # strike (ATM)
        "T": 1.0,     # 1 year
        "r": 0.05,    # 5% risk-free rate
        "sigma": 0.20, # 20% volatility
    }


@pytest.fixture
def itm_call_params():
    return {"S": 110.0, "K": 100.0, "T": 0.5, "r": 0.05, "sigma": 0.20}


@pytest.fixture
def otm_call_params():
    return {"S": 90.0, "K": 100.0, "T": 0.5, "r": 0.05, "sigma": 0.20}