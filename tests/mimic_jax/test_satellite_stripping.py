"""Shared-central satellite-stripping physics and numerical behavior."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16 import (
    apply_satellite_stripping,
    baryonic_mass,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    metal_mass,
    step_context,
)


def _stripping_case(*, num_substeps=10, satellite_type=1, **satellite_overrides):
    values = {
        "HaloBaryonFraction": 0.17,
        "HotGas": 5.0,
        "MetalsHotGas": 0.1,
    }
    values.update(satellite_overrides)
    return (
        initial_galaxy_state(**values),
        initial_galaxy_state(HotGas=100.0, MetalsHotGas=2.0),
        initial_halo_forcing(Type=satellite_type, Mvir=10.0),
        step_context(num_substeps=num_substeps),
        fiducial_parameters(),
    )


def test_stripping_matches_the_recomputed_excess_transfer():
    satellite, central, halo, context, parameters = _stripping_case()
    result = apply_satellite_stripping(
        satellite,
        central,
        halo,
        context,
        parameters,
    )
    np.testing.assert_allclose(result.transfer.gas, 0.33, rtol=1.0e-14)
    np.testing.assert_allclose(result.transfer.metals, 0.0066, rtol=1.0e-7)
    np.testing.assert_allclose(result.satellite.HotGas, 4.67, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(result.central.HotGas, 100.33, rtol=0.0, atol=1.0e-5)


def test_stripping_conserves_combined_baryons_and_metals_with_float_storage():
    satellite, central, halo, context, parameters = _stripping_case(num_substeps=1)
    result = apply_satellite_stripping(
        satellite,
        central,
        halo,
        context,
        parameters,
    )
    before_baryons = baryonic_mass(satellite) + baryonic_mass(central)
    after_baryons = baryonic_mass(result.satellite) + baryonic_mass(result.central)
    before_metals = metal_mass(satellite) + metal_mass(central)
    after_metals = metal_mass(result.satellite) + metal_mass(result.central)
    np.testing.assert_allclose(after_baryons, before_baryons, rtol=0.0, atol=8.0e-6)
    np.testing.assert_allclose(after_metals, before_metals, rtol=0.0, atol=2.0e-7)


def test_gas_and_metal_caps_use_the_same_symmetric_transfer():
    satellite, central, halo, context, parameters = _stripping_case(
        num_substeps=1,
        HotGas=2.0,
        MetalsHotGas=3.0,
        StellarMass=10.0,
        HaloBaryonFraction=0.2,
    )
    halo = halo._replace(Mvir=jnp.asarray(10.0, dtype=jnp.float64))
    result = apply_satellite_stripping(
        satellite,
        central,
        halo,
        context,
        parameters,
    )
    np.testing.assert_allclose(result.transfer.gas, 2.0)
    np.testing.assert_allclose(result.transfer.metals, 3.0)
    np.testing.assert_array_equal(result.satellite.HotGas, 0.0)
    np.testing.assert_array_equal(result.satellite.MetalsHotGas, 0.0)
    np.testing.assert_allclose(
        result.central.MetalsHotGas - central.MetalsHotGas,
        3.0,
        rtol=0.0,
        atol=0.0,
    )


def test_only_type1_satellites_with_hot_gas_are_active():
    for satellite_type in (0, 2, 3):
        satellite, central, halo, context, parameters = _stripping_case(
            satellite_type=satellite_type
        )
        result = apply_satellite_stripping(
            satellite,
            central,
            halo,
            context,
            parameters,
        )
        np.testing.assert_array_equal(result.satellite.HotGas, satellite.HotGas)
        np.testing.assert_array_equal(result.central.HotGas, central.HotGas)


def test_repeated_stripping_preserves_upstream_geometric_substep_dependence():
    satellite, central, halo, _, parameters = _stripping_case(num_substeps=8)
    context = step_context(num_substeps=8)
    for _ in range(8):
        result = apply_satellite_stripping(
            satellite,
            central,
            halo,
            context,
            parameters,
        )
        satellite, central = result.satellite, result.central
    expected = 1.7 + 3.3 * (1.0 - 1.0 / 8.0) ** 8
    np.testing.assert_allclose(satellite.HotGas, expected, rtol=0.0, atol=6.0e-7)


def test_stripping_jit_fractional_derivative_and_derivative_conservation():
    satellite, central, halo, context, parameters = _stripping_case()
    eager = apply_satellite_stripping(
        satellite,
        central,
        halo,
        context,
        parameters,
    )
    compiled = jax.jit(apply_satellite_stripping)(
        satellite,
        central,
        halo,
        context,
        parameters,
    )
    np.testing.assert_array_equal(compiled.satellite.HotGas, eager.satellite.HotGas)

    def stripped(epsilon):
        return apply_satellite_stripping(
            satellite,
            central,
            halo,
            context,
            parameters,
            epsilon,
        ).transfer.gas

    automatic = jax.grad(stripped)(jnp.asarray(0.0, dtype=jnp.float64))
    step = jnp.asarray(1.0e-4, dtype=jnp.float64)
    finite_difference = (stripped(step) - stripped(-step)) / (2.0 * step)
    np.testing.assert_allclose(automatic, finite_difference, rtol=1.0e-8)

    def combined_hot_change(epsilon):
        result = apply_satellite_stripping(
            satellite,
            central,
            halo,
            context,
            parameters,
            epsilon,
        )
        return result.satellite.HotGas + result.central.HotGas

    np.testing.assert_allclose(
        jax.grad(combined_hot_change)(jnp.asarray(0.0, dtype=jnp.float64)),
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )
