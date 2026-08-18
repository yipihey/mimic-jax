"""Full FoF-group orchestration for the faithful fiducial SAGE16 schedule."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16 import (
    active_group_baryonic_mass,
    evolve_upstream_sequential_group_final,
    evolve_upstream_sequential_group_interval,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    load_cooling_tables,
    prepare_upstream_sequential_group,
    process_perturbations,
    sage16_units,
    step_context,
    upstream_sequential_group_substep,
)


def _stack(*records):
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *records)


def _record_at(records, index):
    return jax.tree_util.tree_map(lambda values: values[index], records)


def _controlled_group(hot_gas=8.0):
    states = _stack(
        initial_galaxy_state(
            HotGas=hot_gas,
            ColdGas=3.0,
            StellarMass=5.0,
            EjectedGas=1.0,
            ICS=0.5,
            BlackHoleMass=0.01,
            MetalsHotGas=0.16,
            MetalsColdGas=0.06,
            MetalsStellarMass=0.1,
            MetalsEjectedGas=0.02,
            MetalsICS=0.01,
            MergTime=999.9,
        ),
        initial_galaxy_state(
            HotGas=3.0,
            ColdGas=1.5,
            StellarMass=1.0,
            EjectedGas=0.2,
            ICS=0.1,
            BlackHoleMass=0.005,
            MetalsHotGas=0.06,
            MetalsColdGas=0.03,
            MetalsStellarMass=0.02,
            MetalsEjectedGas=0.004,
            MetalsICS=0.002,
            DiskScaleRadius=0.005,
            MergTime=10.0,
        ),
        initial_galaxy_state(ColdGas=7.0, HotGas=11.0, MergTime=4.0),
    )
    halos = _stack(
        initial_halo_forcing(
            Type=0,
            CentralHalo=0,
            Len=1000,
            Mvir=100.0,
            Rvir=0.2,
            Vvir=200.0,
            Vmax=200.0,
            Spin=(1.0, 2.0, 3.0),
            dT=0.001,
        ),
        initial_halo_forcing(
            Type=1,
            CentralHalo=0,
            Len=200,
            Mvir=20.0,
            Rvir=0.1,
            Vvir=100.0,
            Vmax=100.0,
            dT=0.002,
        ),
        initial_halo_forcing(
            Type=3,
            CentralHalo=0,
            Len=0,
            Mvir=0.0,
            Rvir=0.0,
            Vvir=0.0,
            Vmax=0.0,
            dT=0.001,
        ),
    )
    return states, halos


def test_pre_timestep_runs_once_in_upstream_order_and_skips_type3():
    states, halos = _controlled_group()
    result = prepare_upstream_sequential_group(
        states,
        halos,
        step_context(redshift=1.0, num_substeps=2, time_interval=0.002),
        0,
        fiducial_parameters(),
        sage16_units(),
    )

    assert result.states.HaloBaryonFraction[0] >= 0.0
    assert result.states.InfallingGas[0] == result.diagnostics.infall_budget.infalling_gas
    assert result.states.DiskScaleRadius[0] > 0.0
    np.testing.assert_array_equal(result.states.DiskScaleRadius[1], np.float32(0.005))
    np.testing.assert_array_equal(result.states.ColdGas[2], states.ColdGas[2])
    np.testing.assert_array_equal(result.diagnostics.reionization_modifiers[2], 1.0)


def test_group_substep_is_galaxy_major_and_conserves_closed_baryon_transfers():
    states, halos = _controlled_group()
    states = states._replace(InfallingGas=states.InfallingGas.at[0].set(0.0))
    context = step_context(num_substeps=2, time_interval=0.002)
    before = active_group_baryonic_mass(states, halos.Type)
    result = upstream_sequential_group_substep(
        states,
        halos,
        context,
        0,
        fiducial_parameters(),
        sage16_units(),
        load_cooling_tables(),
    )

    assert bool(result.success)
    np.testing.assert_array_equal(result.diagnostics.galaxies.active, (True, True, False))
    assert result.diagnostics.galaxies.satellite_stripping.gas[1] > 0.0
    np.testing.assert_allclose(
        active_group_baryonic_mass(result.states, result.halos.Type),
        before,
        rtol=0.0,
        atol=3.0e-6,
    )
    np.testing.assert_array_equal(result.states.ColdGas[2], states.ColdGas[2])
    np.testing.assert_array_equal(result.states.HotGas[2], states.HotGas[2])


def test_full_group_interval_is_jittable_vmappable_and_uses_object_local_dt():
    states, halos = _controlled_group()
    context = step_context(redshift=0.5, time_interval=0.002)
    parameters = fiducial_parameters()
    units = sage16_units()
    tables = load_cooling_tables()

    def run(group_states, group_halos):
        return evolve_upstream_sequential_group_interval(
            group_states,
            group_halos,
            context,
            0,
            parameters,
            units,
            tables,
            num_substeps=2,
        )

    eager = run(states, halos)
    compiled = jax.jit(run)(states, halos)
    np.testing.assert_array_equal(compiled.final_halos.Type, eager.final_halos.Type)
    np.testing.assert_array_equal(compiled.final_states.ColdGas, eager.final_states.ColdGas)
    assert bool(compiled.success)
    expected_clock = np.float32(np.float32(10.0) - np.float32(0.001) - np.float32(0.001))
    np.testing.assert_array_equal(compiled.states.MergTime[-1, 1], expected_clock)

    batched_states = jax.tree_util.tree_map(lambda value: jnp.stack((value, value)), states)
    batched_halos = jax.tree_util.tree_map(lambda value: jnp.stack((value, value)), halos)
    batched = jax.vmap(run)(batched_states, batched_halos)
    np.testing.assert_array_equal(
        batched.final_states.StellarMass[0], eager.final_states.StellarMass
    )


def test_final_only_group_path_is_bitwise_equal_and_vmappable():
    states, halos = _controlled_group()
    context = step_context(redshift=0.5, time_interval=0.002)
    parameters = fiducial_parameters()
    units = sage16_units()
    tables = load_cooling_tables()

    history = evolve_upstream_sequential_group_interval(
        states,
        halos,
        context,
        0,
        parameters,
        units,
        tables,
        num_substeps=2,
    )

    def run(group_states, group_halos):
        return evolve_upstream_sequential_group_final(
            group_states,
            group_halos,
            context,
            0,
            parameters,
            units,
            tables,
            num_substeps=2,
        )

    final = jax.jit(run)(states, halos)
    for observed, expected in zip(
        jax.tree_util.tree_leaves(final.final_states),
        jax.tree_util.tree_leaves(history.final_states),
    ):
        np.testing.assert_array_equal(observed, expected)
    for observed, expected in zip(
        jax.tree_util.tree_leaves(final.final_halos),
        jax.tree_util.tree_leaves(history.final_halos),
    ):
        np.testing.assert_array_equal(observed, expected)
    assert bool(final.success)

    batched_states = jax.tree_util.tree_map(lambda value: jnp.stack((value, value)), states)
    batched_halos = jax.tree_util.tree_map(lambda value: jnp.stack((value, value)), halos)
    batched = jax.jit(jax.vmap(run))(batched_states, batched_halos)
    np.testing.assert_array_equal(
        batched.final_states.StellarMass[0], final.final_states.StellarMass
    )

    def run_dynamic(group_states, group_halos, central_index):
        return evolve_upstream_sequential_group_final(
            group_states,
            group_halos,
            context,
            central_index,
            parameters,
            units,
            tables,
            num_substeps=2,
        )

    dynamic = jax.jit(jax.vmap(run_dynamic))(
        batched_states,
        batched_halos,
        jnp.asarray((0, 0), dtype=jnp.int32),
    )
    np.testing.assert_array_equal(dynamic.final_states.ColdGas[0], final.final_states.ColdGas)


def test_trailing_inactive_member_padding_is_bitwise_neutral():
    states, halos = _controlled_group()
    padded_states = _stack(
        *[_record_at(states, index) for index in range(3)],
        initial_galaxy_state(),
    )
    padded_halos = _stack(
        *[_record_at(halos, index) for index in range(3)],
        initial_halo_forcing(
            Type=3,
            CentralHalo=0,
            Len=0,
            Mvir=0.0,
            Rvir=0.0,
            Vvir=0.0,
            Vmax=0.0,
            dT=0.0,
        ),
    )
    context = step_context(redshift=0.5, time_interval=0.002)
    parameters = fiducial_parameters()
    units = sage16_units()
    tables = load_cooling_tables()

    def run(group_states, group_halos):
        return evolve_upstream_sequential_group_final(
            group_states,
            group_halos,
            context,
            0,
            parameters,
            units,
            tables,
            num_substeps=2,
        )

    exact = jax.jit(run)(states, halos)
    padded = jax.jit(run)(padded_states, padded_halos)
    for observed, expected in zip(
        jax.tree_util.tree_leaves(padded.final_states),
        jax.tree_util.tree_leaves(exact.final_states),
    ):
        np.testing.assert_array_equal(observed[:3], expected)
    for observed, expected in zip(
        jax.tree_util.tree_leaves(padded.final_halos),
        jax.tree_util.tree_leaves(exact.final_halos),
    ):
        np.testing.assert_array_equal(observed[:3], expected)


def test_group_baryon_conservation_derivative_is_zero():
    _, halos = _controlled_group()
    context = step_context(num_substeps=2, time_interval=0.002)
    parameters = fiducial_parameters()
    units = sage16_units()
    tables = load_cooling_tables()
    perturbations = process_perturbations()

    def residual(hot_gas):
        states, _ = _controlled_group(hot_gas)
        states = states._replace(InfallingGas=states.InfallingGas.at[0].set(0.0))
        before = active_group_baryonic_mass(states, halos.Type)
        result = upstream_sequential_group_substep(
            states,
            halos,
            context,
            0,
            parameters,
            units,
            tables,
            perturbations,
        )
        return active_group_baryonic_mass(result.states, result.halos.Type) - before

    derivative = jax.grad(residual)(jnp.asarray(8.0, dtype=jnp.float64))
    np.testing.assert_allclose(derivative, 0.0, rtol=0.0, atol=2.0e-7)
