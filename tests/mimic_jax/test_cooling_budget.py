"""Sutherland-Dopita interpolation and SAGE16 cooling-budget tests."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16 import (
    apply_cooling,
    baryonic_mass,
    calculate_cooling_budget,
    initial_galaxy_state,
    initial_halo_forcing,
    load_cooling_tables,
    metal_dependent_cooling_rate,
    metal_mass,
    sage16_units,
    step_context,
)


def test_cooling_tables_preserve_upstream_float_input_values():
    tables = load_cooling_tables()

    assert tables.log_cooling_rates.shape == (8, 91)
    assert tables.log_metallicities.shape == (8,)
    assert tables.log_cooling_rates.dtype == jnp.float64
    assert float(tables.log_cooling_rates[4, 0]) == float(np.float32(-23.31))
    assert float(tables.log_metallicities[6]) == np.log10(0.02)


def test_temperature_and_metallicity_interpolation_matches_piecewise_formula():
    tables = load_cooling_tables()
    log_temperature = jnp.asarray(5.025, dtype=jnp.float64)
    log_metallicity = 0.5 * (tables.log_metallicities[4] + tables.log_metallicities[5])
    actual = metal_dependent_cooling_rate(log_temperature, log_metallicity, tables)

    lower_temperature_index = 20
    rate_at_lower_metallicity = 0.5 * (
        tables.log_cooling_rates[4, lower_temperature_index]
        + tables.log_cooling_rates[4, lower_temperature_index + 1]
    )
    rate_at_upper_metallicity = 0.5 * (
        tables.log_cooling_rates[5, lower_temperature_index]
        + tables.log_cooling_rates[5, lower_temperature_index + 1]
    )
    expected = 10.0 ** (0.5 * (rate_at_lower_metallicity + rate_at_upper_metallicity))

    np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=0.0)


def test_high_temperature_behavior_matches_upstream_last_interval_extrapolation():
    tables = load_cooling_tables()
    log_temperature = jnp.asarray(9.0, dtype=jnp.float64)
    log_metallicity = tables.log_metallicities[6]
    actual = metal_dependent_cooling_rate(log_temperature, log_metallicity, tables)
    rate1 = tables.log_cooling_rates[6, 89]
    rate2 = tables.log_cooling_rates[6, 90]
    expected_log_rate = rate1 + (rate2 - rate1) * 20.0 * (log_temperature - 8.45)

    np.testing.assert_allclose(actual, 10.0**expected_log_rate, rtol=2.0e-15, atol=0.0)


def test_cooling_budget_calculates_transport_without_mutating_reservoirs():
    state = initial_galaxy_state(HotGas=8.0, MetalsHotGas=0.16)
    halo = initial_halo_forcing(Rvir=0.2, Vvir=200.0, dT=0.01)
    result = calculate_cooling_budget(
        state,
        halo,
        step_context(num_substeps=10, time_interval=0.01),
        sage16_units(),
        load_cooling_tables(),
    )

    assert float(result.state.HotGas) == float(state.HotGas)
    assert float(result.state.ColdGas) == float(state.ColdGas)
    assert float(result.state.CoolingGas) == float(result.budget.gas)
    assert float(result.state.Rcool) == float(result.budget.radius)
    assert float(result.state.CoolingLambda) == float(result.budget.cooling_lambda)
    assert 0.0 < float(result.budget.gas) <= float(state.HotGas)
    assert float(result.budget.radius) > 0.0
    assert float(result.budget.cooling_lambda) > 0.0

    applied = apply_cooling(result.state, halo)
    np.testing.assert_allclose(
        baryonic_mass(applied.state),
        baryonic_mass(state),
        rtol=0.0,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        metal_mass(applied.state),
        metal_mass(state),
        rtol=0.0,
        atol=2.0e-7,
    )


def test_cooling_budget_zeroes_transport_for_empty_hot_reservoir():
    state = initial_galaxy_state(
        HotGas=0.0,
        CoolingGas=2.0,
        Rcool=3.0,
        CoolingLambda=4.0,
    )
    result = calculate_cooling_budget(
        state,
        initial_halo_forcing(),
        step_context(),
        sage16_units(),
        load_cooling_tables(),
    )

    np.testing.assert_array_equal(jnp.asarray(result.budget), jnp.zeros(3))
    assert float(result.state.CoolingGas) == 0.0
    assert float(result.state.Rcool) == 0.0
    assert float(result.state.CoolingLambda) == 0.0


def test_cooling_interpolation_and_budget_are_jittable_and_differentiable():
    tables = load_cooling_tables()
    state = initial_galaxy_state(HotGas=8.0, MetalsHotGas=0.16)
    halo = initial_halo_forcing(Rvir=0.2, Vvir=200.0, dT=0.01)
    context = step_context(num_substeps=10, time_interval=0.01)
    units = sage16_units()

    compiled = jax.jit(calculate_cooling_budget)(state, halo, context, units, tables)
    eager = calculate_cooling_budget(state, halo, context, units, tables)
    np.testing.assert_allclose(compiled.budget, eager.budget, rtol=0.0, atol=0.0)

    def cooling_for_hot_metal_mass(metals_hot_gas):
        varied = state._replace(MetalsHotGas=metals_hot_gas)
        return calculate_cooling_budget(varied, halo, context, units, tables).budget.gas

    derivative = jax.grad(cooling_for_hot_metal_mass)(state.MetalsHotGas)
    assert bool(jnp.isfinite(derivative))
    assert float(derivative) != 0.0
