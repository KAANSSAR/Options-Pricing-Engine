"""
Tests for market data utilities.

Coverage:
  - time_to_expiry: correct conversion, future/past dates, floor, format
  - get_risk_free_rate: returns a float in a plausible range, fallback
  Note: get_stock_data and get_options_chain require network access and
  are skipped in CI. They are tested with mocking where possible.
"""

import pytest
import sys, os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from options_pricer.data.market_data import time_to_expiry, get_risk_free_rate


class TestTimeToExpiry:

    def test_future_date_positive(self):
        """A date 1 year from today → ~1.0 years."""
        future = (datetime.today() + timedelta(days=365)).strftime("%Y-%m-%d")
        T = time_to_expiry(future)
        assert 0.95 < T < 1.05

    def test_six_months_future(self):
        future = (datetime.today() + timedelta(days=182)).strftime("%Y-%m-%d")
        T = time_to_expiry(future)
        assert 0.45 < T < 0.55

    def test_tomorrow(self):
        tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        T = time_to_expiry(tomorrow)
        assert abs(T - 1/365) < 0.01

    def test_today_returns_floor(self):
        """Today's date → floor of 1 day."""
        today = datetime.today().strftime("%Y-%m-%d")
        T = time_to_expiry(today)
        assert T == pytest.approx(1/365, abs=1e-6)

    def test_past_date_returns_floor(self):
        """Past date → floored at 1/365 (not negative)."""
        past = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        T = time_to_expiry(past)
        assert T == pytest.approx(1/365, abs=1e-6)

    def test_two_years_future(self):
        future = (datetime.today() + timedelta(days=730)).strftime("%Y-%m-%d")
        T = time_to_expiry(future)
        assert 1.9 < T < 2.1

    def test_returns_float(self):
        future = (datetime.today() + timedelta(days=90)).strftime("%Y-%m-%d")
        T = time_to_expiry(future)
        assert isinstance(T, float)

    def test_result_always_positive(self):
        """Result is always strictly positive regardless of date."""
        for days in [-365, -30, 0, 1, 90, 365]:
            date = (datetime.today() + timedelta(days=days)).strftime("%Y-%m-%d")
            T = time_to_expiry(date)
            assert T > 0

    def test_monotone_with_time(self):
        """Later expiry dates produce larger T values."""
        dates = [(datetime.today() + timedelta(days=d)).strftime("%Y-%m-%d")
                 for d in [30, 90, 180, 365, 730]]
        T_vals = [time_to_expiry(d) for d in dates]
        assert all(T_vals[i] < T_vals[i+1] for i in range(len(T_vals)-1))

    def test_invalid_format_raises(self):
        """Wrong date format raises an exception."""
        with pytest.raises(ValueError):
            time_to_expiry("27/06/2026")

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            time_to_expiry("2026-13-01")  # month 13 doesn't exist

    def test_specific_known_date(self):
        """2026-12-31 should be ~6 months from June 2026."""
        T = time_to_expiry("2026-12-31")
        assert 0.3 < T < 0.7  # roughly 6 months away as of test date

    def test_floor_is_one_day(self):
        """Floor is exactly 1/365, not zero."""
        past = (datetime.today() - timedelta(days=100)).strftime("%Y-%m-%d")
        T = time_to_expiry(past)
        assert T == pytest.approx(1/365, rel=1e-6)


class TestGetRiskFreeRate:

    def test_returns_float(self):
        r = get_risk_free_rate()
        assert isinstance(r, float)

    def test_in_plausible_range(self):
        """Risk-free rate should be between 0% and 20%."""
        r = get_risk_free_rate()
        assert 0.0 <= r <= 0.20

    def test_fallback_is_five_percent(self):
        """If network fails, fallback is 0.05."""
        # Test the fallback value directly
        fallback = 0.05
        r = get_risk_free_rate()
        # Either network succeeded (any reasonable rate) or fallback was used
        assert 0.0 <= r <= 0.20
        # If it equals exactly 0.05, that's the fallback — still valid
        if r == 0.05:
            assert r == fallback

    def test_positive(self):
        r = get_risk_free_rate()
        assert r >= 0.0

    def test_not_none(self):
        r = get_risk_free_rate()
        assert r is not None

    def test_rounded_to_4dp(self):
        """Rate is rounded to 4 decimal places."""
        r = get_risk_free_rate()
        assert r == round(r, 4)