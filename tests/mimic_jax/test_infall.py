"""Reionization, group infall-budget, and finite infall-application tests."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16 import (
    apply_infall,
    apply_reionization,
    baryonic_mass,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    prepare_infall_budget,
    sage16_units,
    step_context,
)


def _stack_records(records):
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *records)


def test_reionization_is_mass_dependent_jittable_and_differentiable():
    state = initial_galaxy_state()
    context = step_context(redshift=2.0)
    parameters = fiducial_parameters()
    units = sage16_units()
    low = initial_halo_forcing(Mvir=0.5)
    high = initial_halo_forcing(Mvir=50.0)

    low_result = apply_reionization(state, low, context, parameters, units)
    high_result = jax.jit(apply_reionization)(state, high, context, parameters, units)
    assert (
        0.0
        < float(low_result.state.HaloBaryonFraction)
        < float(high_result.state.HaloBaryonFraction)
    )
    assert float(high_result.state.HaloBaryonFraction) <= float(parameters.GlobalBaryonFraction)

    def baryon_fraction(global_fraction):
        varied = parameters._replace(GlobalBaryonFraction=global_fraction)
        return apply_reionization(state, high, context, varied, units).state.HaloBaryonFraction

    derivative = jax.grad(baryon_fraction)(parameters.GlobalBaryonFraction)
    np.testing.assert_allclose(derivative, high_result.modifier, rtol=1.0e-13)


def test_reionization_skips_type3_and_zeroes_nonpositive_halo_mass():
    state = initial_galaxy_state(HaloBaryonFraction=0.123)
    parameters = fiducial_parameters()
    units = sage16_units()
    context = step_context(redshift=0.0)
    type3 = apply_reionization(
        state,
        initial_halo_forcing(Type=3, Mvir=100.0),
        context,
        parameters,
        units,
    )
    zero = apply_reionization(
        state,
        initial_halo_forcing(Mvir=0.0),
        context,
        parameters,
        units,
    )
    np.testing.assert_array_equal(type3.state.HaloBaryonFraction, state.HaloBaryonFraction)
    np.testing.assert_array_equal(zero.state.HaloBaryonFraction, 0.0)


def test_prepare_infall_budget_consolidates_only_ejected_gas_and_ics():
    central = initial_galaxy_state(
        HaloBaryonFraction=0.17,
        StellarMass=5.0,
        ColdGas=3.0,
        HotGas=8.0,
        EjectedGas=1.0,
        ICS=0.5,
        BlackHoleMass=0.1,
        MetalsEjectedGas=0.02,
        MetalsICS=0.01,
    )
    satellite = initial_galaxy_state(
        HaloBaryonFraction=0.17,
        HotGas=3.0,
        EjectedGas=2.0,
        ICS=1.5,
        MetalsHotGas=0.06,
        MetalsEjectedGas=0.04,
        MetalsICS=0.03,
    )
    states = _stack_records((central, satellite))
    halos = _stack_records(
        (
            initial_halo_forcing(Type=0, Mvir=100.0),
            initial_halo_forcing(Type=2, Mvir=0.0),
        )
    )
    result = jax.jit(
        lambda current: prepare_infall_budget(current, halos, 0, fiducial_parameters())
    )(states)

    np.testing.assert_array_equal(result.states.EjectedGas, [3.0, 0.0])
    np.testing.assert_array_equal(result.states.ICS, [2.0, 0.0])
    np.testing.assert_array_equal(result.states.HotGas, [8.0, 3.0])
    np.testing.assert_array_equal(
        result.states.MetalsHotGas,
        jnp.asarray([0.0, 0.06], dtype=jnp.float32),
    )
    np.testing.assert_allclose(result.transfer.satellite_ejected_to_central, 2.0)
    np.testing.assert_allclose(result.transfer.satellite_ics_to_central, 1.5)
    np.testing.assert_allclose(result.transfer.target_baryons, 17.0)
    np.testing.assert_allclose(result.transfer.group_baryons, 24.1, rtol=1.0e-8)
    np.testing.assert_allclose(result.states.InfallingGas[0], -7.1, rtol=1.0e-7)

    before = sum(
        float(baryonic_mass(jax.tree_util.tree_map(lambda x: x[i], states))) for i in range(2)
    )
    after = sum(
        float(baryonic_mass(jax.tree_util.tree_map(lambda x: x[i], result.states)))
        for i in range(2)
    )
    np.testing.assert_allclose(after, before, rtol=0.0, atol=3.0e-7)


def test_positive_infall_partitions_a_fixed_external_source_over_substeps():
    initial = initial_galaxy_state(InfallingGas=12.0, HotGas=5.0)
    context = step_context(num_substeps=4)
    state = initial
    for _ in range(4):
        result = apply_infall(state, context)
        state = result.state
        np.testing.assert_allclose(result.transfer.external_to_hot, 3.0)
    np.testing.assert_allclose(state.HotGas, 17.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        baryonic_mass(state) - baryonic_mass(initial),
        12.0,
        rtol=0.0,
        atol=0.0,
    )


def test_negative_infall_removes_ejected_then_hot_and_tracks_metal_sinks():
    initial = initial_galaxy_state(
        InfallingGas=-8.0,
        EjectedGas=3.0,
        MetalsEjectedGas=0.06,
        HotGas=10.0,
        MetalsHotGas=0.2,
    )
    result = apply_infall(initial, step_context())
    np.testing.assert_array_equal(result.state.EjectedGas, 0.0)
    np.testing.assert_array_equal(result.state.MetalsEjectedGas, 0.0)
    np.testing.assert_array_equal(result.state.HotGas, 5.0)
    np.testing.assert_allclose(result.state.MetalsHotGas, 0.1, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(result.transfer.ejected_to_external, 3.0)
    np.testing.assert_allclose(result.transfer.hot_to_external, 5.0)
    np.testing.assert_allclose(result.transfer.unfulfilled_removal, 0.0)
    np.testing.assert_allclose(
        baryonic_mass(result.state) - baryonic_mass(initial),
        -8.0,
        rtol=0.0,
        atol=0.0,
    )


def test_infall_fractional_derivative_matches_finite_difference_and_ledger_derivative():
    initial = initial_galaxy_state(InfallingGas=2.0, HotGas=5.0)
    context = step_context(num_substeps=2)

    def hot_mass(epsilon):
        return apply_infall(initial, context, epsilon).state.HotGas

    automatic = jax.grad(hot_mass)(jnp.asarray(0.0, dtype=jnp.float64))
    step = jnp.asarray(1.0e-4, dtype=jnp.float64)
    finite_difference = (hot_mass(step) - hot_mass(-step)) / (2.0 * step)
    np.testing.assert_allclose(automatic, finite_difference, rtol=2.0e-3)

    def ledger_residual(epsilon):
        result = apply_infall(initial, context, epsilon)
        return (
            baryonic_mass(result.state) - baryonic_mass(initial) - result.transfer.external_to_hot
        )

    np.testing.assert_allclose(
        jax.grad(ledger_residual)(jnp.asarray(0.0, dtype=jnp.float64)),
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )
