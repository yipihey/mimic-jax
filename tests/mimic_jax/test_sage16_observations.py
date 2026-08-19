from pathlib import Path

import numpy as np

from mimic_jax.sage16.observations import (
    BALDRY2008_CHABRIER_SHIFT_DEX,
    load_baldry2008_stellar_mass_function,
)

DATA = Path(__file__).parents[2] / "data/observations/baldry2008_stellar_mass_function.csv"


def test_baldry_table_reproduces_sage_h_and_imf_conversions():
    observation = load_baldry2008_stellar_mass_function(DATA, hubble_h=0.73)
    expected_mass = np.log10(10.0**7.05 / 0.73**2) + BALDRY2008_CHABRIER_SHIFT_DEX
    assert observation.coordinate.shape == (50,)
    assert np.isclose(observation.coordinate[0], expected_mass)
    assert np.isclose(observation.values[0], 1.3531e-1 * 0.73**3)
    assert np.isclose(observation.lower_errors[0], 6.0741e-2 * 0.73**3)
    np.testing.assert_array_equal(observation.lower_errors, observation.upper_errors)


def test_baldry_loader_keeps_salpeter_mass_when_requested():
    observation = load_baldry2008_stellar_mass_function(DATA, hubble_h=0.73, chabrier_imf=False)
    assert np.isclose(observation.coordinate[0], np.log10(10.0**7.05 / 0.73**2))
