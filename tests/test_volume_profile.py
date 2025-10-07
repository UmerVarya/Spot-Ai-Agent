import pandas as pd
import pytest

from volume_profile import compute_volume_profile


def test_compute_volume_profile_identifies_poc_and_lvns():
    prices = [100.0] * 5 + [101.0] * 5 + [102.0] * 2 + [103.0] * 5
    volumes = [40, 42, 38, 41, 39, 65, 66, 64, 63, 67, 10, 9, 55, 54, 53, 56, 55]
    df = pd.DataFrame({"close": prices, "volume": volumes})
    result = compute_volume_profile(df, bin_pct=0.001)
    assert result.poc is not None
    assert pytest.approx(result.poc, rel=0.01) == 101.0
    assert result.lvns, "Expected at least one LVN"
    assert any(abs(lvn - 102.0) < 0.2 for lvn in result.lvns)
    assert not result.histogram.empty
    assert pytest.approx(result.histogram["volume"].sum()) == sum(volumes)


def test_compute_volume_profile_handles_missing_columns():
    df = pd.DataFrame({"close": [1, 2, 3]})
    result = compute_volume_profile(df)
    assert result.poc is None
    assert result.lvns == []
    assert result.histogram.empty
