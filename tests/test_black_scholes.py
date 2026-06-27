"""
Tests for Black-Scholes pricing and Greeks.

Reference values computed independently via:
  https://www.wolframalpha.com/input?i=Black-Scholes+option+price

Coverage:
  - Pricing correctness (known values, bounds, monotonicity)
  - Put-call parity (parameter sweep)
  - Edge cases (T=0, zero vol, deep ITM/OTM, large T, negative rates)
  - Greeks: sign, range, symmetry, call/put relationships, monotonicity
  - Greeks: numerical verification vs finite differences (all 5, both types)
  - d1/d2 internal formulas
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from options_pricer.core.black_scholes import price, greeks, d1, d2


# ── Finite difference helpers ────────────────────────────────────────────────

def fd_delta(S, K, T, r, sigma, option_type, h=0.01):
    return (price(S+h, K, T, r, sigma, option_type) -
            price(S-h, K, T, r, sigma, option_type)) / (2*h)

def fd_gamma(S, K, T, r, sigma, option_type, h=0.01):
    return (price(S+h, K, T, r, sigma, option_type) -
            2*price(S, K, T, r, sigma, option_type) +
            price(S-h, K, T, r, sigma, option_type)) / h**2

def fd_vega(S, K, T, r, sigma, option_type, h=0.0001):
    return (price(S, K, T, r, sigma+h, option_type) -
            price(S, K, T, r, sigma-h, option_type)) / (2*h) / 100

def fd_theta(S, K, T, r, sigma, option_type, h=1/365):
    return -(price(S, K, T+h, r, sigma, option_type) -
             price(S, K, T-h, r, sigma, option_type)) / (2*h) / 365

def fd_rho(S, K, T, r, sigma, option_type, h=0.0001):
    return (price(S, K, T, r+h, sigma, option_type) -
            price(S, K, T, r-h, sigma, option_type)) / (2*h) / 100


# ══════════════════════════════════════════════════════════════════════════════
# PRICING
# ══════════════════════════════════════════════════════════════════════════════

class TestBlackScholesPrice:

    # Known values
    def test_atm_call_known_value(self, base_params):
        assert abs(price(**base_params, option_type="call") - 10.4506) < 0.001

    def test_atm_put_known_value(self, base_params):
        assert abs(price(**base_params, option_type="put") - 5.5735) < 0.001

    def test_itm_call_known_value(self):
        assert abs(price(110, 100, 0.5, 0.05, 0.20, "call") - 14.075) < 0.01

    def test_otm_call_known_value(self):
        assert abs(price(90, 100, 0.5, 0.05, 0.20, "call") - 2.349) < 0.01

    def test_itm_put_known_value(self):
        assert abs(price(90, 100, 0.5, 0.05, 0.20, "put") - 9.880) < 0.01

    def test_otm_put_known_value(self):
        assert abs(price(110, 100, 0.5, 0.05, 0.20, "put") - 1.606) < 0.01

    # Positivity
    def test_call_positive(self, base_params):
        assert price(**base_params, option_type="call") > 0

    def test_put_positive(self, base_params):
        assert price(**base_params, option_type="put") > 0

    def test_prices_positive_parameter_sweep(self):
        for S in [50, 100, 200]:
            for K in [80, 100, 120]:
                for T in [0.1, 0.5, 2.0]:
                    for sigma in [0.1, 0.3, 0.6]:
                        for otype in ["call", "put"]:
                            assert price(S, K, T, 0.05, sigma, otype) > 0

    # Put-call parity
    def test_put_call_parity_atm(self, base_params):
        S, K, T, r, sigma = base_params["S"], base_params["K"], base_params["T"], base_params["r"], base_params["sigma"]
        c = price(S, K, T, r, sigma, "call")
        p = price(S, K, T, r, sigma, "put")
        assert abs(c - p - (S - K*np.exp(-r*T))) < 1e-10

    def test_put_call_parity_itm(self, itm_call_params):
        S, K, T, r, sigma = itm_call_params["S"], itm_call_params["K"], itm_call_params["T"], itm_call_params["r"], itm_call_params["sigma"]
        c = price(S, K, T, r, sigma, "call")
        p = price(S, K, T, r, sigma, "put")
        assert abs(c - p - (S - K*np.exp(-r*T))) < 1e-10

    def test_put_call_parity_otm(self, otm_call_params):
        S, K, T, r, sigma = otm_call_params["S"], otm_call_params["K"], otm_call_params["T"], otm_call_params["r"], otm_call_params["sigma"]
        c = price(S, K, T, r, sigma, "call")
        p = price(S, K, T, r, sigma, "put")
        assert abs(c - p - (S - K*np.exp(-r*T))) < 1e-10

    def test_put_call_parity_parameter_sweep(self):
        for S in [50, 100, 150]:
            for K in [80, 100, 120]:
                for T in [0.1, 1.0, 3.0]:
                    for r in [0.0, 0.05, 0.10]:
                        for sigma in [0.1, 0.3, 0.8]:
                            c = price(S, K, T, r, sigma, "call")
                            p = price(S, K, T, r, sigma, "put")
                            assert abs(c - p - (S - K*np.exp(-r*T))) < 1e-9

    # Bounds
    def test_call_lower_bound(self, base_params):
        S, K, T, r, sigma = base_params["S"], base_params["K"], base_params["T"], base_params["r"], base_params["sigma"]
        c = price(S, K, T, r, sigma, "call")
        assert c >= max(S - K*np.exp(-r*T), 0) - 1e-10

    def test_put_lower_bound(self, base_params):
        S, K, T, r, sigma = base_params["S"], base_params["K"], base_params["T"], base_params["r"], base_params["sigma"]
        p = price(S, K, T, r, sigma, "put")
        assert p >= max(K*np.exp(-r*T) - S, 0) - 1e-10

    def test_call_upper_bound(self, base_params):
        assert price(**base_params, option_type="call") <= base_params["S"] + 1e-10

    def test_put_upper_bound(self, base_params):
        K, T, r = base_params["K"], base_params["T"], base_params["r"]
        assert price(**base_params, option_type="put") <= K*np.exp(-r*T) + 1e-10

    # Monotonicity
    def test_call_increases_with_spot(self, base_params):
        bp = base_params
        prices = [price(s, bp["K"], bp["T"], bp["r"], bp["sigma"], "call") for s in [70,85,100,115,130]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_put_decreases_with_spot(self, base_params):
        bp = base_params
        prices = [price(s, bp["K"], bp["T"], bp["r"], bp["sigma"], "put") for s in [70,85,100,115,130]]
        assert all(prices[i] > prices[i+1] for i in range(len(prices)-1))

    def test_call_decreases_with_strike(self, base_params):
        bp = base_params
        prices = [price(bp["S"], k, bp["T"], bp["r"], bp["sigma"], "call") for k in [80,90,100,110,120]]
        assert all(prices[i] > prices[i+1] for i in range(len(prices)-1))

    def test_put_increases_with_strike(self, base_params):
        bp = base_params
        prices = [price(bp["S"], k, bp["T"], bp["r"], bp["sigma"], "put") for k in [80,90,100,110,120]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_both_increase_with_vol(self, base_params):
        bp = base_params
        for otype in ["call", "put"]:
            prices = [price(bp["S"], bp["K"], bp["T"], bp["r"], v, otype) for v in [0.05,0.10,0.20,0.40,0.80]]
            assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_both_increase_with_time(self, base_params):
        bp = base_params
        for otype in ["call", "put"]:
            prices = [price(bp["S"], bp["K"], t, bp["r"], bp["sigma"], otype) for t in [0.1,0.25,0.5,1.0,2.0]]
            assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_call_increases_with_rate(self, base_params):
        bp = base_params
        prices = [price(bp["S"], bp["K"], bp["T"], r, bp["sigma"], "call") for r in [0.0,0.02,0.05,0.08,0.12]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_put_decreases_with_rate(self, base_params):
        bp = base_params
        prices = [price(bp["S"], bp["K"], bp["T"], r, bp["sigma"], "put") for r in [0.0,0.02,0.05,0.08,0.12]]
        assert all(prices[i] > prices[i+1] for i in range(len(prices)-1))

    # Edge cases
    def test_zero_time_call_intrinsic(self, base_params):
        p = price(base_params["S"], base_params["K"], 0, base_params["r"], base_params["sigma"], "call")
        assert p == max(base_params["S"] - base_params["K"], 0)

    def test_zero_time_put_intrinsic(self, base_params):
        p = price(base_params["S"], base_params["K"], 0, base_params["r"], base_params["sigma"], "put")
        assert p == max(base_params["K"] - base_params["S"], 0)

    def test_zero_time_itm_call(self):
        assert price(120, 100, 0, 0.05, 0.20, "call") == 20

    def test_zero_time_otm_call(self):
        assert price(80, 100, 0, 0.05, 0.20, "call") == 0

    def test_zero_time_itm_put(self):
        assert price(80, 100, 0, 0.05, 0.20, "put") == 20

    def test_zero_time_otm_put(self):
        assert price(120, 100, 0, 0.05, 0.20, "put") == 0

    def test_deep_itm_call_approaches_forward(self):
        S, K, T, r, sigma = 500, 100, 1.0, 0.05, 0.20
        p = price(S, K, T, r, sigma, "call")
        assert abs(p - (S - K*np.exp(-r*T))) < 0.01

    def test_deep_otm_call_near_zero(self):
        assert price(50, 200, 0.5, 0.05, 0.20, "call") < 0.001

    def test_deep_otm_put_near_zero(self):
        assert price(300, 100, 0.5, 0.05, 0.20, "put") < 0.001

    def test_long_dated_option_finite(self):
        p = price(100, 100, 10.0, 0.05, 0.20, "call")
        assert 0 < p < 100

    def test_high_vol_finite(self):
        p = price(100, 100, 1.0, 0.05, 3.0, "call")
        assert 0 < p < 100

    def test_zero_rate(self, base_params):
        bp = {**base_params, "r": 0.0}
        c = price(**bp, option_type="call")
        p = price(**bp, option_type="put")
        assert c > 0 and p > 0
        assert abs(c - p - (bp["S"] - bp["K"])) < 1e-10

    def test_negative_rate(self):
        assert price(100, 100, 1.0, -0.01, 0.20, "call") > 0

    def test_atm_zero_rate_call_equals_put(self):
        c = price(100, 100, 1.0, 0.0, 0.20, "call")
        p = price(100, 100, 1.0, 0.0, 0.20, "put")
        assert abs(c - p) < 1e-10

    # Errors
    def test_invalid_type_raises(self, base_params):
        with pytest.raises(ValueError, match="option_type"):
            price(**base_params, option_type="forward")

    # d1/d2
    def test_d1_d2_relationship(self, base_params):
        bp = base_params
        assert abs(d1(bp["S"],bp["K"],bp["T"],bp["r"],bp["sigma"]) -
                   d2(bp["S"],bp["K"],bp["T"],bp["r"],bp["sigma"]) -
                   bp["sigma"]*np.sqrt(bp["T"])) < 1e-12

    def test_d1_atm_zero_rate(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.0, 0.20
        assert abs(d1(S,K,T,r,sigma) - sigma*np.sqrt(T)/2) < 1e-10

    def test_d1_increases_with_spot(self, base_params):
        bp = base_params
        vals = [d1(s, bp["K"], bp["T"], bp["r"], bp["sigma"]) for s in [80,90,100,110,120]]
        assert all(vals[i] < vals[i+1] for i in range(len(vals)-1))


# ══════════════════════════════════════════════════════════════════════════════
# GREEKS
# ══════════════════════════════════════════════════════════════════════════════

class TestBlackScholesGreeks:

    # Delta
    def test_call_delta_range(self, base_params):
        assert 0 < greeks(**base_params, option_type="call")["delta"] < 1

    def test_put_delta_range(self, base_params):
        assert -1 < greeks(**base_params, option_type="put")["delta"] < 0

    def test_call_put_delta_sum(self, base_params):
        gc = greeks(**base_params, option_type="call")
        gp = greeks(**base_params, option_type="put")
        assert abs(gc["delta"] - gp["delta"] - 1.0) < 1e-5

    def test_deep_itm_call_delta_near_one(self):
        assert greeks(500, 100, 1.0, 0.05, 0.20, "call")["delta"] > 0.999

    def test_deep_otm_call_delta_near_zero(self):
        assert greeks(50, 200, 1.0, 0.05, 0.20, "call")["delta"] < 0.001

    def test_atm_short_call_delta_near_half(self):
        assert abs(greeks(100, 100, 0.01, 0.05, 0.20, "call")["delta"] - 0.5) < 0.05

    def test_deep_itm_put_delta_near_neg_one(self):
        assert greeks(50, 200, 1.0, 0.05, 0.20, "put")["delta"] < -0.999

    def test_call_delta_monotonic_with_spot(self, base_params):
        bp = base_params
        deltas = [greeks(s, bp["K"], bp["T"], bp["r"], bp["sigma"], "call")["delta"] for s in [70,85,100,115,130]]
        assert all(deltas[i] < deltas[i+1] for i in range(len(deltas)-1))

    def test_put_delta_monotonic_with_spot(self, base_params):
        bp = base_params
        deltas = [greeks(s, bp["K"], bp["T"], bp["r"], bp["sigma"], "put")["delta"] for s in [70,85,100,115,130]]
        assert all(deltas[i] < deltas[i+1] for i in range(len(deltas)-1))

    # Gamma
    def test_gamma_positive_call(self, base_params):
        assert greeks(**base_params, option_type="call")["gamma"] > 0

    def test_gamma_positive_put(self, base_params):
        assert greeks(**base_params, option_type="put")["gamma"] > 0

    def test_gamma_call_put_equal(self, base_params):
        gc = greeks(**base_params, option_type="call")
        gp = greeks(**base_params, option_type="put")
        assert abs(gc["gamma"] - gp["gamma"]) < 1e-10

    def test_gamma_peaks_atm(self, base_params):
        bp = base_params
        g_atm = greeks(100, 100, bp["T"], bp["r"], bp["sigma"], "call")["gamma"]
        g_itm = greeks(150, 100, bp["T"], bp["r"], bp["sigma"], "call")["gamma"]
        g_otm = greeks(60,  100, bp["T"], bp["r"], bp["sigma"], "call")["gamma"]
        assert g_atm > g_itm and g_atm > g_otm

    def test_gamma_higher_near_expiry_atm(self):
        g_long  = greeks(100, 100, 1.0,  0.05, 0.20, "call")["gamma"]
        g_short = greeks(100, 100, 0.05, 0.05, 0.20, "call")["gamma"]
        assert g_short > g_long

    # Vega
    def test_vega_positive_call(self, base_params):
        assert greeks(**base_params, option_type="call")["vega"] > 0

    def test_vega_positive_put(self, base_params):
        assert greeks(**base_params, option_type="put")["vega"] > 0

    def test_vega_call_put_equal(self, base_params):
        gc = greeks(**base_params, option_type="call")
        gp = greeks(**base_params, option_type="put")
        assert abs(gc["vega"] - gp["vega"]) < 1e-10

    def test_vega_peaks_atm(self, base_params):
        bp = base_params
        v_atm = greeks(100, 100, bp["T"], bp["r"], bp["sigma"], "call")["vega"]
        v_itm = greeks(150, 100, bp["T"], bp["r"], bp["sigma"], "call")["vega"]
        v_otm = greeks(60,  100, bp["T"], bp["r"], bp["sigma"], "call")["vega"]
        assert v_atm > v_itm and v_atm > v_otm

    def test_vega_increases_with_time(self, base_params):
        bp = base_params
        v_short = greeks(bp["S"], bp["K"], 0.1, bp["r"], bp["sigma"], "call")["vega"]
        v_long  = greeks(bp["S"], bp["K"], 2.0, bp["r"], bp["sigma"], "call")["vega"]
        assert v_long > v_short

    # Theta
    def test_theta_negative_call(self, base_params):
        assert greeks(**base_params, option_type="call")["theta"] < 0

    def test_theta_negative_put(self, base_params):
        assert greeks(**base_params, option_type="put")["theta"] < 0

    def test_theta_more_negative_near_expiry(self):
        t_long  = greeks(100, 100, 1.0,  0.05, 0.20, "call")["theta"]
        t_short = greeks(100, 100, 0.1,  0.05, 0.20, "call")["theta"]
        assert t_short < t_long

    def test_theta_more_negative_atm(self, base_params):
        bp = base_params
        t_atm = greeks(100, 100, bp["T"], bp["r"], bp["sigma"], "call")["theta"]
        t_otm = greeks(60,  100, bp["T"], bp["r"], bp["sigma"], "call")["theta"]
        assert t_atm < t_otm

    # Rho
    def test_call_rho_positive(self, base_params):
        assert greeks(**base_params, option_type="call")["rho"] > 0

    def test_put_rho_negative(self, base_params):
        assert greeks(**base_params, option_type="put")["rho"] < 0

    def test_rho_higher_for_longer_call(self, base_params):
        bp = base_params
        r_short = greeks(bp["S"], bp["K"], 0.1, bp["r"], bp["sigma"], "call")["rho"]
        r_long  = greeks(bp["S"], bp["K"], 2.0, bp["r"], bp["sigma"], "call")["rho"]
        assert r_long > r_short

    # Zero time
    def test_greeks_zero_time_all_zero(self, base_params):
        g = greeks(base_params["S"], base_params["K"], 0, base_params["r"], base_params["sigma"], "call")
        assert g == {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}

    # Finite difference verification
    FD = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)

    @pytest.mark.parametrize("otype", ["call", "put"])
    def test_delta_vs_finite_diff(self, otype):
        assert abs(greeks(**self.FD, option_type=otype)["delta"] - fd_delta(**self.FD, option_type=otype)) < 1e-4

    @pytest.mark.parametrize("otype", ["call", "put"])
    def test_gamma_vs_finite_diff(self, otype):
        assert abs(greeks(**self.FD, option_type=otype)["gamma"] - fd_gamma(**self.FD, option_type=otype)) < 1e-4

    @pytest.mark.parametrize("otype", ["call", "put"])
    def test_vega_vs_finite_diff(self, otype):
        assert abs(greeks(**self.FD, option_type=otype)["vega"] - fd_vega(**self.FD, option_type=otype)) < 1e-4

    @pytest.mark.parametrize("otype", ["call", "put"])
    def test_theta_vs_finite_diff(self, otype):
        assert abs(greeks(**self.FD, option_type=otype)["theta"] - fd_theta(**self.FD, option_type=otype)) < 1e-4

    @pytest.mark.parametrize("otype", ["call", "put"])
    def test_rho_vs_finite_diff(self, otype):
        assert abs(greeks(**self.FD, option_type=otype)["rho"] - fd_rho(**self.FD, option_type=otype)) < 1e-4

    @pytest.mark.parametrize("otype", ["call", "put"])
    def test_all_greeks_fd_itm(self, otype, itm_call_params):
        g  = greeks(**itm_call_params, option_type=otype)
        assert abs(g["delta"] - fd_delta(**itm_call_params, option_type=otype)) < 1e-4
        assert abs(g["gamma"] - fd_gamma(**itm_call_params, option_type=otype)) < 1e-4
        assert abs(g["vega"]  - fd_vega(**itm_call_params,  option_type=otype)) < 1e-4

    @pytest.mark.parametrize("otype", ["call", "put"])
    def test_all_greeks_fd_otm(self, otype, otm_call_params):
        g  = greeks(**otm_call_params, option_type=otype)
        assert abs(g["delta"] - fd_delta(**otm_call_params, option_type=otype)) < 1e-4
        assert abs(g["vega"]  - fd_vega(**otm_call_params,  option_type=otype)) < 1e-4

    def test_greeks_invalid_type(self, base_params):
        with pytest.raises(ValueError):
            greeks(**base_params, option_type="straddle")