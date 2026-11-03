import pandas as pd
import pytest

from vol_trading.grid_search_experiment import (
    BandConfig,
    BinaryConfig,
    band_weights,
    binary_weights,
    first_oos_year,
    generate_band_grid,
    generate_binary_grid,
)


def test_binary_weights_respect_threshold_and_weights():
    ratio = pd.Series([0.95, 1.00, 1.01, float("nan")])
    config = BinaryConfig(threshold=1.0, low_weight=0.25, high_weight=1.0)

    weights = binary_weights(ratio, config)

    assert weights.iloc[0] == pytest.approx(0.25)
    assert weights.iloc[1] == pytest.approx(0.25)
    assert weights.iloc[2] == pytest.approx(1.0)
    assert pd.isna(weights.iloc[3])


def test_band_weights_respect_bands_and_weights():
    ratio = pd.Series([0.90, 0.95, 1.00, 1.05, 1.10, float("nan")])
    config = BandConfig(
        low_threshold=0.95,
        high_threshold=1.05,
        low_weight=0.0,
        mid_weight=0.5,
        high_weight=1.0,
    )

    weights = band_weights(ratio, config)

    expected = [0.0, 0.5, 0.5, 0.5, 1.0]
    for actual, target in zip(weights.iloc[:5], expected):
        assert actual == pytest.approx(target)
    assert pd.isna(weights.iloc[5])


def test_invalid_binary_config_is_rejected():
    with pytest.raises(ValueError):
        BinaryConfig(threshold=1.0, low_weight=0.75, high_weight=0.50)


def test_invalid_band_config_is_rejected():
    with pytest.raises(ValueError):
        BandConfig(
            low_threshold=1.05,
            high_threshold=0.95,
            low_weight=0.0,
            mid_weight=0.5,
        )
    with pytest.raises(ValueError):
        BandConfig(
            low_threshold=0.95,
            high_threshold=1.05,
            low_weight=0.75,
            mid_weight=0.50,
        )


def test_default_grids_have_expected_sizes():
    assert len(generate_binary_grid()) == 54
    assert len(generate_band_grid()) == 192


def test_first_oos_year_requires_full_minimum_training_window():
    # First signal is Sep 2010, so three full years of history means first OOS year is 2014.
    assert first_oos_year(pd.Timestamp("2010-09-30"), 3) == 2014
    assert first_oos_year(pd.Timestamp("2010-01-04"), 3) == 2013
