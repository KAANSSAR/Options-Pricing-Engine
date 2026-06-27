"""
Tests for the Binomial Tree (CRR) pricing model.

Coverage:
  - Convergence to Black-Scholes (European, both types)
  - American >= European (all cases)
  - American call = European call (no dividends)
  - Put-call parity for European binomial
  - Monotonicity (spot, strike, vol, time, rate)
  - Accuracy improvement with more steps
  - Edge cases (T~0, very few steps, deep ITM/OTM)
  - American early exercise: deep ITM puts, high rate
  - price_grid shape and node values
  - Error handling
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from options_pricer.core.binomial_tree import price, price_grid
from options_pricer.core.black_scholes import price as bs_price


class TestBinomialTreePrice:

    # ── Convergence to BS ─────────────────────────────────────────────────────

    def test_convergence_call_european(self, base_params):
        bs = bs_price(**base_params, option_type="call")
        bt = price(**base_params, option_type="call", style="european", steps=500)
        assert abs(bt - bs) < 0.01

    def test_convergence_put_european(self, base_params):
        bs = bs_price(**base_params, option_type="put")
        bt = price(**base_params, option_type="put", style="european", steps=500)
        assert abs(bt - bs) < 0.01

    def test_convergence_itm_call(self, itm_call_params):
        bs = bs_price(**itm_call_params, option_type="call")
        bt = price(**itm_call_params, option_type="call", style="european", steps=500)
        assert abs(bt - bs) < 0.01

    def test_convergence_otm_put(self, otm_call_params):
        bs = bs_price(**otm_call_params, option_type="put")
        bt = price(**otm_call_params, option_type="put", style="european", steps=500)
        assert abs(bt - bs) < 0.01

    def test_convergence_monotonic(self, base_params):
        """Larger step counts should produce prices closer to BS."""
        bs = bs_price(**base_params, option_type="call")
        err_50  = abs(price(**base_params, option_type="call", steps=50)  - bs)
        err_500 = abs(price(**base_params, option_type="call", steps=500) - bs)
        assert err_500 < err_50

    def test_convergence_across_moneyness(self):
        """Convergence holds for ITM, ATM, OTM."""
        for S in [80, 100, 120]:
            bs = bs_price(S, 100, 1.0, 0.05, 0.20, "call")
            bt = price(S, 100, 1.0, 0.05, 0.20, "call", steps=400)
            assert abs(bt - bs) < 0.015

    # ── American >= European ──────────────────────────────────────────────────

    def test_american_put_geq_european(self, base_params):
        euro = price(**base_params, option_type="put", style="european", steps=200)
        amer = price(**base_params, option_type="put", style="american", steps=200)
        assert amer >= euro - 1e-10

    def test_american_call_equals_european_no_dividends(self, base_params):
        """Without dividends, never optimal to exercise call early."""
        euro = price(**base_params, option_type="call", style="european", steps=200)
        amer = price(**base_params, option_type="call", style="american", steps=200)
        assert abs(amer - euro) < 0.01

    def test_american_put_premium_positive_deep_itm(self):
        """Deep ITM put with high rate → significant early exercise premium."""
        euro = price(50, 100, 1.0, 0.10, 0.20, "put", style="european", steps=200)
        amer = price(50, 100, 1.0, 0.10, 0.20, "put", style="american", steps=200)
        assert amer > euro + 0.01  # meaningful premium

    def test_american_put_premium_increases_with_rate(self):
        """Higher rates → larger early exercise premium for puts."""
        p_low  = price(80, 100, 1.0, 0.01, 0.20, "put", style="american", steps=200) - \
                 price(80, 100, 1.0, 0.01, 0.20, "put", style="european", steps=200)
        p_high = price(80, 100, 1.0, 0.10, 0.20, "put", style="american", steps=200) - \
                 price(80, 100, 1.0, 0.10, 0.20, "put", style="european", steps=200)
        assert p_high > p_low

    # ── Positivity ────────────────────────────────────────────────────────────

    def test_call_positive(self, base_params):
        assert price(**base_params, option_type="call") > 0

    def test_put_positive(self, base_params):
        assert price(**base_params, option_type="put") > 0

    def test_american_put_positive(self, base_params):
        assert price(**base_params, option_type="put", style="american") > 0

    # ── Put-call parity (European) ────────────────────────────────────────────

    def test_put_call_parity_european(self, base_params):
        S, K, T, r, sigma = base_params["S"], base_params["K"], base_params["T"], base_params["r"], base_params["sigma"]
        c = price(S, K, T, r, sigma, "call", "european", steps=500)
        p = price(S, K, T, r, sigma, "put",  "european", steps=500)
        assert abs(c - p - (S - K*np.exp(-r*T))) < 0.02

    def test_put_call_parity_itm(self, itm_call_params):
        bp = itm_call_params
        c = price(bp["S"], bp["K"], bp["T"], bp["r"], bp["sigma"], "call", "european", steps=400)
        p = price(bp["S"], bp["K"], bp["T"], bp["r"], bp["sigma"], "put",  "european", steps=400)
        assert abs(c - p - (bp["S"] - bp["K"]*np.exp(-bp["r"]*bp["T"]))) < 0.02

    # ── Monotonicity ──────────────────────────────────────────────────────────

    def test_call_increases_with_spot(self, base_params):
        bp = base_params
        prices = [price(s, bp["K"], bp["T"], bp["r"], bp["sigma"], "call", steps=100) for s in [80,90,100,110,120]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_put_decreases_with_spot(self, base_params):
        bp = base_params
        prices = [price(s, bp["K"], bp["T"], bp["r"], bp["sigma"], "put", steps=100) for s in [80,90,100,110,120]]
        assert all(prices[i] > prices[i+1] for i in range(len(prices)-1))

    def test_call_decreases_with_strike(self, base_params):
        bp = base_params
        prices = [price(bp["S"], k, bp["T"], bp["r"], bp["sigma"], "call", steps=100) for k in [80,90,100,110,120]]
        assert all(prices[i] > prices[i+1] for i in range(len(prices)-1))

    def test_both_increase_with_vol(self, base_params):
        bp = base_params
        for otype in ["call", "put"]:
            prices = [price(bp["S"], bp["K"], bp["T"], bp["r"], v, otype, steps=100) for v in [0.1,0.2,0.3,0.5]]
            assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_both_increase_with_time(self, base_params):
        bp = base_params
        for otype in ["call", "put"]:
            prices = [price(bp["S"], bp["K"], t, bp["r"], bp["sigma"], otype, steps=100) for t in [0.1,0.5,1.0,2.0]]
            assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    # ── Specific moneyness ────────────────────────────────────────────────────

    def test_itm_call_exceeds_intrinsic(self, itm_call_params):
        p = price(**itm_call_params, option_type="call", steps=200)
        intrinsic = max(itm_call_params["S"] - itm_call_params["K"], 0)
        assert p >= intrinsic - 1e-10

    def test_itm_call_exceeds_otm_call(self, itm_call_params, otm_call_params):
        itm = price(**itm_call_params, option_type="call", steps=100)
        otm = price(**otm_call_params, option_type="call", steps=100)
        assert itm > otm

    def test_deep_otm_call_near_zero(self):
        assert price(50, 200, 0.5, 0.05, 0.20, "call", steps=100) < 0.01

    def test_deep_otm_put_near_zero(self):
        assert price(300, 100, 0.5, 0.05, 0.20, "put", steps=100) < 0.01

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_very_few_steps(self, base_params):
        """1-step binomial still returns a positive price."""
        p = price(**base_params, option_type="call", steps=1)
        assert p > 0

    def test_single_step_put_call_parity(self, base_params):
        bp = base_params
        c = price(bp["S"], bp["K"], bp["T"], bp["r"], bp["sigma"], "call", steps=1)
        p = price(bp["S"], bp["K"], bp["T"], bp["r"], bp["sigma"], "put",  steps=1)
        assert abs(c - p - (bp["S"] - bp["K"]*np.exp(-bp["r"]*bp["T"]))) < 0.5

    def test_high_vol_finite(self):
        p = price(100, 100, 1.0, 0.05, 2.0, "call", steps=100)
        assert 0 < p < 100

    def test_zero_rate(self, base_params):
        bp = {**base_params, "r": 0.0}
        c = price(**bp, option_type="call", steps=200)
        p = price(**bp, option_type="put",  steps=200)
        assert c > 0 and p > 0

    # ── American vs European comparison parameter sweep ────────────────────────

    def test_american_geq_european_sweep(self):
        """American always >= European across a grid."""
        for S in [80, 100, 120]:
            for otype in ["call", "put"]:
                euro = price(S, 100, 1.0, 0.05, 0.20, otype, "european", steps=100)
                amer = price(S, 100, 1.0, 0.05, 0.20, otype, "american", steps=100)
                assert amer >= euro - 1e-8

    # ── Error handling ────────────────────────────────────────────────────────

    def test_invalid_option_type_raises(self, base_params):
        with pytest.raises(Exception):
            price(**base_params, option_type="straddle")



class TestPriceGrid:

    def test_output_shapes(self, base_params):
        bp = base_params
        st, pt = price_grid(bp["S"], bp["K"], bp["T"], bp["r"], bp["sigma"], "call", "european", 10)
        assert st.shape == (11, 11)
        assert pt.shape == (11, 11)

    def test_root_node_matches_price(self, base_params):
        bp = base_params
        _, pt = price_grid(bp["S"], bp["K"], bp["T"], bp["r"], bp["sigma"], "call", "european", 50)
        scalar = price(**bp, option_type="call", style="european", steps=50)
        assert abs(pt[0, 0] - scalar) < 0.001

    def test_stock_tree_starts_at_S(self, base_params):
        bp = base_params
        st, _ = price_grid(bp["S"], bp["K"], bp["T"], bp["r"], bp["sigma"], "call", "european", 10)
        assert abs(st[0, 0] - bp["S"]) < 1e-10

    def test_terminal_payoffs_non_negative(self, base_params):
        bp = base_params
        _, pt = price_grid(bp["S"], bp["K"], bp["T"], bp["r"], bp["sigma"], "call", "european", 10)
        terminal = pt[:, -1]
        assert (terminal >= -1e-10).all()

    def test_option_values_non_negative(self, base_params):
        bp = base_params
        _, pt = price_grid(bp["S"], bp["K"], bp["T"], bp["r"], bp["sigma"], "call", "european", 10)
        non_nan = pt[~np.isnan(pt)]
        assert (non_nan >= -1e-10).all()

    def test_stock_tree_upper_path_monotone(self, base_params):
        """Top row of stock tree (all up-moves) is strictly increasing."""
        bp = base_params
        st, _ = price_grid(bp["S"], bp["K"], bp["T"], bp["r"], bp["sigma"], "call", "european", 10)
        top_row = [st[0, i] for i in range(11)]
        assert all(top_row[i] < top_row[i+1] for i in range(len(top_row)-1))

    def test_american_geq_european_grid(self, base_params):
        """Every node in American grid >= corresponding European node."""
        bp = base_params
        _, pt_euro = price_grid(bp["S"], bp["K"], bp["T"], bp["r"], bp["sigma"], "put", "european", 10)
        _, pt_amer = price_grid(bp["S"], bp["K"], bp["T"], bp["r"], bp["sigma"], "put", "american", 10)
        mask = ~np.isnan(pt_euro)
        assert (pt_amer[mask] >= pt_euro[mask] - 1e-10).all()