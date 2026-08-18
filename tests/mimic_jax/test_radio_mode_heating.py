"""Radio-mode AGN equivalence behavior, ledgers, and derivatives."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16 import (
    PROCESS_NAMES,
    apply_radio_mode_heating,
    baryonic_mass,
    evolve_upstream_sequential_central_history,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    load_cooling_tables,
    metal_mass,
    perturbations_from_matrix,
    sage16_units,
    step_context,
)
from mimic_jax.sage16.precision import as_float64


def _radio_case(**state_overrides):
    state_values = {
        "CoolingGas": 1.0,
        "Rcool": 0.05,
        "Rheat": 0.01,
        "HotGas": 10.0,
        "MetalsHotGas": 0.2,
        "BlackHoleMass": 0.01,
        "CoolingLambda": 1.0e-20,
    }
    state_values.update(state_overrides)
    return (
        initial_galaxy_state(**state_values),
        initial_halo_forcing(Mvir=100.0, Vvir=200.0, Rvir=0.2, dT=0.01),
        step_context(time_interval=0.01),
        fiducial_parameters(),
        sage16_units(),
    )


def test_prior_heating_suppresses_current_cooling_and_new_heating_only_grows_radius():
    state, halo, context, parameters, units = _radio_case()
    result = apply_radio_mode_heating(state, halo, context, parameters, units)

    expected = (1.0 - as_float64(state.Rheat) / state.Rcool) * state.CoolingGas
    np.testing.assert_allclose(result.state.CoolingGas, expected, rtol=0.0, atol=0.0)
    assert float(result.transfer.heating_mass) > 0.0
    assert float(result.state.CoolingGas) == float(result.transfer.cooling_after_prior_heating)
    assert float(result.state.Rheat) > float(state.Rheat)


def test_radio_mode_has_explicit_baryon_transfer_and_metal_sink_ledgers():
    state, halo, context, parameters, units = _radio_case()
    result = apply_radio_mode_heating(state, halo, context, parameters, units)

    np.testing.assert_allclose(
        baryonic_mass(result.state),
        baryonic_mass(state),
        rtol=0.0,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        metal_mass(result.state) - metal_mass(state),
        -result.transfer.hot_metals_accreted,
        rtol=0.0,
        atol=2.0e-8,
    )


def test_radio_mode_off_is_a_complete_no_op_even_with_stored_heating_radius():
    state, halo, context, parameters, units = _radio_case()
    off = parameters._replace(AGNrecipe=jnp.asarray(0, dtype=jnp.int32))
    result = apply_radio_mode_heating(state, halo, context, off, units)

    for before, after in zip(
        jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(result.state)
    ):
        np.testing.assert_array_equal(after, before)


def test_radio_mode_jit_and_fractional_derivative_match_finite_difference():
    state, halo, context, parameters, units = _radio_case()
    eager = apply_radio_mode_heating(state, halo, context, parameters, units)
    compiled = jax.jit(apply_radio_mode_heating)(state, halo, context, parameters, units)
    np.testing.assert_array_equal(compiled.state.BlackHoleMass, eager.state.BlackHoleMass)

    def accreted(epsilon):
        return apply_radio_mode_heating(
            state,
            halo,
            context,
            parameters,
            units,
            epsilon,
        ).transfer.black_hole_accreted

    automatic = jax.grad(accreted)(jnp.asarray(0.0, dtype=jnp.float64))
    step = jnp.asarray(1.0e-4, dtype=jnp.float64)
    finite_difference = (accreted(step) - accreted(-step)) / (2.0 * step)
    assert float(automatic) > 0.0
    np.testing.assert_allclose(automatic, finite_difference, rtol=1.0e-8)


def test_baryon_conservation_derivative_is_zero():
    state, halo, context, parameters, units = _radio_case()

    def baryon_change(efficiency):
        varied = parameters._replace(RadioModeEfficiency=efficiency)
        result = apply_radio_mode_heating(state, halo, context, varied, units)
        return (
            as_float64(result.state.HotGas)
            - as_float64(state.HotGas)
            + as_float64(result.state.BlackHoleMass)
            - as_float64(state.BlackHoleMass)
        )

    derivative = jax.grad(baryon_change)(parameters.RadioModeEfficiency)
    np.testing.assert_allclose(derivative, 0.0, rtol=0.0, atol=1.0e-15)


def test_upstream_ordered_history_exposes_agn_epoch_response():
    state, halo, context, parameters, units = _radio_case(
        Rheat=0.0,
        ColdGas=2.0,
        StellarMass=1.0,
        DiskScaleRadius=0.01,
    )
    count = 3
    stack = lambda record: jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (count,) + value.shape), record
    )
    halos = stack(halo)
    contexts = stack(context._replace(num_substeps=jnp.asarray(10, dtype=jnp.int32)))
    tables = load_cooling_tables()
    baseline = evolve_upstream_sequential_central_history(
        state,
        halos,
        contexts,
        parameters,
        units,
        tables,
    )
    assert baseline.diagnostics.radio_mode.heating_mass.shape == (count,)
    assert bool(jnp.any(baseline.diagnostics.radio_mode.heating_mass > 0.0))

    def final_stellar_mass(agn_epsilon):
        values = jnp.zeros((len(PROCESS_NAMES), count), dtype=jnp.float64)
        agn_index = PROCESS_NAMES.index("agn_heating")
        values = values.at[agn_index, 0].set(agn_epsilon)
        history = evolve_upstream_sequential_central_history(
            state,
            halos,
            contexts,
            parameters,
            units,
            tables,
            perturbations_from_matrix(values),
        )
        return history.final_state.StellarMass

    derivative = jax.grad(final_stellar_mass)(jnp.asarray(0.0, dtype=jnp.float64))
    assert bool(jnp.isfinite(derivative))
    assert float(derivative) < 0.0
