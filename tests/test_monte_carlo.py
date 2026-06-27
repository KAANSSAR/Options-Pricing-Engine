"""
Tests for Monte Carlo option pricing.

Coverage:
  - Convergence to BS within confidence intervals
  - Statistical properties (CI ordered, SE positive, keys present)
  - Variance reduction (antithetic variates)
  - Reproducibility and seed isolation
  - Monotonicity (price responds correctly to parameters)
  - Simulate paths shape, positivity, boundary conditions
  - Edge cases (deep ITM/OTM, short/long T, high vol)
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from options_pricer.core.monte_carlo import price, simulate_paths
from options_pricer.core.black_scholes import price as bs_price


class TestMonteCarloPricing:

    # ── Convergence to BS ─────────────────────────────────────────────────────

    def test_call_converges_to_bs(self, base_params):
        result = price(**base_params, option_type="call", n_simulations=200_000, seed=0)
        bs = bs_price(**base_params, option_type="call")
        assert abs(result["price"] - bs) < 3 * result["std_error"]

    def test_put_converges_to_bs(self, base_params):
        result = price(**base_params, option_type="put", n_simulations=200_000, seed=0)
        bs = bs_price(**base_params, option_type="put")
        assert abs(result["price"] - bs) < 3 * result["std_error"]

    def test_bs_inside_confidence_interval(self, base_params):
        result = price(**base_params, option_type="call", n_simulations=100_000, seed=42)
        bs = bs_price(**base_params, option_type="call")
        lo, hi = result["confidence_interval"]
        assert lo <= bs <= hi

    def test_bs_inside_ci_put(self, base_params):
        result = price(**base_params, option_type="put", n_simulations=100_000, seed=7)
        bs = bs_price(**base_params, option_type="put")
        lo, hi = result["confidence_interval"]
        assert lo <= bs <= hi

    def test_convergence_itm_call(self, itm_call_params):
        result = price(**itm_call_params, option_type="call", n_simulations=100_000, seed=0)
        bs = bs_price(**itm_call_params, option_type="call")
        assert abs(result["price"] - bs) < 3 * result["std_error"]

    def test_convergence_otm_put(self, otm_call_params):
        result = price(**otm_call_params, option_type="put", n_simulations=100_000, seed=0)
        bs = bs_price(**otm_call_params, option_type="put")
        assert abs(result["price"] - bs) < 3 * result["std_error"]

    # ── Result structure ──────────────────────────────────────────────────────

    def test_result_keys_present(self, base_params):
        result = price(**base_params, option_type="call", n_simulations=10_000)
        assert "price" in result
        assert "std_error" in result
        assert "confidence_interval" in result
        assert "n_simulations" in result

    def test_price_positive(self, base_params):
        result = price(**base_params, option_type="call", n_simulations=10_000)
        assert result["price"] > 0

    def test_std_error_positive(self, base_params):
        result = price(**base_params, option_type="call", n_simulations=10_000)
        assert result["std_error"] > 0

    def test_confidence_interval_ordered(self, base_params):
        result = price(**base_params, option_type="call", n_simulations=10_000)
        lo, hi = result["confidence_interval"]
        assert lo < result["price"] < hi

    def test_ci_width_reasonable(self, base_params):
        """95% CI should be fairly narrow with 100k sims."""
        result = price(**base_params, option_type="call", n_simulations=100_000, seed=0)
        lo, hi = result["confidence_interval"]
        assert (hi - lo) < 1.0   # CI width < $1 for a ~$10 option

    def test_n_simulations_reported(self, base_params):
        result = price(**base_params, option_type="call", n_simulations=50_000)
        assert result["n_simulations"] == 50_000

    # ── Variance reduction ────────────────────────────────────────────────────

    def test_antithetic_reduces_se_on_average(self, base_params):
        seeds = range(6)
        plain = [price(**base_params, option_type="call", n_simulations=20_000,
                       seed=s, antithetic=False)["std_error"] for s in seeds]
        anti  = [price(**base_params, option_type="call", n_simulations=20_000,
                       seed=s, antithetic=True)["std_error"] for s in seeds]
        assert np.mean(anti) < np.mean(plain)

    def test_antithetic_convergence_still_correct(self, base_params):
        result = price(**base_params, option_type="call", n_simulations=100_000, seed=0, antithetic=True)
        bs = bs_price(**base_params, option_type="call")
        assert abs(result["price"] - bs) < 3 * result["std_error"]

    # ── Reproducibility ───────────────────────────────────────────────────────

    def test_same_seed_same_result(self, base_params):
        r1 = price(**base_params, option_type="call", n_simulations=10_000, seed=99)
        r2 = price(**base_params, option_type="call", n_simulations=10_000, seed=99)
        assert r1["price"] == r2["price"]
        assert r1["std_error"] == r2["std_error"]

    def test_different_seeds_differ(self, base_params):
        r1 = price(**base_params, option_type="call", n_simulations=10_000, seed=1)
        r2 = price(**base_params, option_type="call", n_simulations=10_000, seed=2)
        assert r1["price"] != r2["price"]

    def test_different_n_sims_differ(self, base_params):
        r1 = price(**base_params, option_type="call", n_simulations=10_000, seed=0)
        r2 = price(**base_params, option_type="call", n_simulations=20_000, seed=0)
        assert r1["price"] != r2["price"]

    # ── SE decreases with more sims ───────────────────────────────────────────

    def test_se_decreases_with_more_sims(self, base_params):
        r1 = price(**base_params, option_type="call", n_simulations=1_000,  seed=0)
        r2 = price(**base_params, option_type="call", n_simulations=50_000, seed=0)
        assert r2["std_error"] < r1["std_error"]

    def test_se_roughly_sqrt_n_scaling(self, base_params):
        """Doubling sims should reduce SE by ~√2."""
        r1 = price(**base_params, option_type="call", n_simulations=10_000, seed=42)
        r2 = price(**base_params, option_type="call", n_simulations=40_000, seed=42)
        ratio = r1["std_error"] / r2["std_error"]
        assert 1.5 < ratio < 2.5   # expect ~2.0

    # ── Monotonicity ──────────────────────────────────────────────────────────

    def test_call_price_increases_with_spot(self, base_params):
        bp = base_params
        prices = [price(s, bp["K"], bp["T"], bp["r"], bp["sigma"], "call", n_simulations=50_000, seed=0)["price"]
                  for s in [80, 100, 120]]
        assert prices[0] < prices[1] < prices[2]

    def test_put_price_decreases_with_spot(self, base_params):
        bp = base_params
        prices = [price(s, bp["K"], bp["T"], bp["r"], bp["sigma"], "put", n_simulations=50_000, seed=0)["price"]
                  for s in [80, 100, 120]]
        assert prices[0] > prices[1] > prices[2]

    def test_both_increase_with_vol(self, base_params):
        bp = base_params
        for otype in ["call", "put"]:
            prices = [price(bp["S"], bp["K"], bp["T"], bp["r"], v, otype, n_simulations=50_000, seed=0)["price"]
                      for v in [0.10, 0.25, 0.50]]
            assert prices[0] < prices[1] < prices[2]

    def test_itm_call_gt_otm_put(self, itm_call_params):
        """ITM call price > deeply OTM put price."""
        c = price(**itm_call_params, option_type="call", n_simulations=50_000)
        p = price(**itm_call_params, option_type="put",  n_simulations=50_000)
        assert c["price"] > p["price"]

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_deep_itm_call_converges(self):
        result = price(200, 100, 1.0, 0.05, 0.20, "call", n_simulations=100_000, seed=0)
        bs = bs_price(200, 100, 1.0, 0.05, 0.20, "call")
        assert abs(result["price"] - bs) < 3 * result["std_error"]

    def test_deep_otm_call_near_zero(self):
        result = price(50, 200, 0.5, 0.05, 0.20, "call", n_simulations=100_000, seed=0)
        assert result["price"] < 0.05

    def test_high_vol_finite(self):
        result = price(100, 100, 1.0, 0.05, 2.0, "call", n_simulations=50_000, seed=0)
        assert 0 < result["price"] < 100

    def test_short_expiry(self):
        result = price(100, 100, 0.05, 0.05, 0.20, "call", n_simulations=100_000, seed=0)
        bs = bs_price(100, 100, 0.05, 0.05, 0.20, "call")
        assert abs(result["price"] - bs) < 3 * result["std_error"]

    def test_long_expiry(self):
        result = price(100, 100, 5.0, 0.05, 0.20, "call", n_simulations=100_000, seed=0)
        bs = bs_price(100, 100, 5.0, 0.05, 0.20, "call")
        assert abs(result["price"] - bs) < 3 * result["std_error"]

    # ── Error handling ────────────────────────────────────────────────────────

    def test_invalid_type_raises(self, base_params):
        with pytest.raises(ValueError):
            price(**base_params, option_type="butterfly")


class TestSimulatePaths:

    # ── Shape and structure ───────────────────────────────────────────────────

    def test_output_shape_default(self, base_params):
        bp = base_params
        times, paths = simulate_paths(bp["S"], bp["T"], bp["r"], bp["sigma"],
                                      n_paths=50, n_steps=252)
        assert paths.shape == (253, 50)
        assert times.shape == (253,)

    def test_output_shape_custom(self, base_params):
        bp = base_params
        times, paths = simulate_paths(bp["S"], bp["T"], bp["r"], bp["sigma"],
                                      n_paths=100, n_steps=100)
        assert paths.shape == (101, 100)
        assert times.shape == (101,)

    def test_paths_start_at_S(self, base_params):
        bp = base_params
        _, paths = simulate_paths(bp["S"], bp["T"], bp["r"], bp["sigma"],
                                  n_paths=200, n_steps=252)
        assert np.allclose(paths[0, :], bp["S"])

    def test_time_axis_starts_at_zero(self, base_params):
        bp = base_params
        times, _ = simulate_paths(bp["S"], bp["T"], bp["r"], bp["sigma"])
        assert times[0] == 0.0

    def test_time_axis_ends_at_T(self, base_params):
        bp = base_params
        times, _ = simulate_paths(bp["S"], bp["T"], bp["r"], bp["sigma"])
        assert abs(times[-1] - bp["T"]) < 1e-10

    def test_time_axis_monotone(self, base_params):
        bp = base_params
        times, _ = simulate_paths(bp["S"], bp["T"], bp["r"], bp["sigma"])
        assert (np.diff(times) > 0).all()

    # ── GBM properties ───────────────────────────────────────────────────────

    def test_paths_always_positive(self, base_params):
        """GBM paths are log-normal — always positive."""
        bp = base_params
        _, paths = simulate_paths(bp["S"], bp["T"], bp["r"], bp["sigma"],
                                  n_paths=500, n_steps=252, seed=0)
        assert (paths > 0).all()

    def test_paths_positive_high_vol(self):
        """Even with very high vol, GBM paths stay positive."""
        _, paths = simulate_paths(100, 1.0, 0.05, 2.0, n_paths=200, n_steps=252, seed=0)
        assert (paths > 0).all()

    def test_mean_terminal_price_near_forward(self, base_params):
        """E[S_T] under risk-neutral measure = S*e^(rT)."""
        bp = base_params
        _, paths = simulate_paths(bp["S"], bp["T"], bp["r"], bp["sigma"],
                                  n_paths=100_000, n_steps=1, seed=42)
        expected_forward = bp["S"] * np.exp(bp["r"] * bp["T"])
        sample_mean = paths[-1].mean()
        # Should be within 1% of true forward
        assert abs(sample_mean / expected_forward - 1) < 0.01

    def test_reproducible_with_seed(self, base_params):
        bp = base_params
        _, p1 = simulate_paths(bp["S"], bp["T"], bp["r"], bp["sigma"], n_paths=50, seed=7)
        _, p2 = simulate_paths(bp["S"], bp["T"], bp["r"], bp["sigma"], n_paths=50, seed=7)
        assert np.allclose(p1, p2)

    def test_different_seeds_differ(self, base_params):
        bp = base_params
        _, p1 = simulate_paths(bp["S"], bp["T"], bp["r"], bp["sigma"], n_paths=50, seed=1)
        _, p2 = simulate_paths(bp["S"], bp["T"], bp["r"], bp["sigma"], n_paths=50, seed=2)
        assert not np.allclose(p1, p2)

    def test_higher_vol_wider_spread(self, base_params):
        """Higher vol → wider terminal distribution."""
        bp = base_params
        _, p_low  = simulate_paths(bp["S"], bp["T"], bp["r"], 0.10, n_paths=5000, seed=0)
        _, p_high = simulate_paths(bp["S"], bp["T"], bp["r"], 0.50, n_paths=5000, seed=0)
        assert p_high[-1].std() > p_low[-1].std()

    def test_longer_time_wider_spread(self, base_params):
        bp = base_params
        _, p_short = simulate_paths(bp["S"], 0.1, bp["r"], bp["sigma"], n_paths=5000, seed=0)
        _, p_long  = simulate_paths(bp["S"], 2.0, bp["r"], bp["sigma"], n_paths=5000, seed=0)
        assert p_long[-1].std() > p_short[-1].std()

    def test_single_step_paths(self, base_params):
        bp = base_params
        times, paths = simulate_paths(bp["S"], bp["T"], bp["r"], bp["sigma"],
                                      n_paths=100, n_steps=1)
        assert paths.shape == (2, 100)
        assert np.allclose(paths[0, :], bp["S"])
        assert (paths[1, :] > 0).all()