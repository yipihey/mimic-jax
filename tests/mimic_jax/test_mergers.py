"""Ordered SAGE16 disruption, merger, and immediate-consumer contracts."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16 import (
    active_group_baryonic_mass,
    active_group_metal_mass,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    resolve_mergers_and_disruption,
    sage16_units,
    step_context,
)
from mimic_jax.sage16.processes.mergers import (
    MERGER_ACTION_DISRUPTION,
    MERGER_ACTION_MERGER,
    MERGER_ERROR_INVALID_DT,
)


def _stack(*records):
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *records)


def _group(states, halos, *, substeps=1, substep=0, parameters=None):
    return resolve_mergers_and_disruption(
        states,
        halos,
        step_context(
            time=13.8,
            time_interval=0.1,
            num_substeps=substeps,
            substep_number=substep,
        ),
        fiducial_parameters() if parameters is None else parameters,
        sage16_units(),
    )


def test_disruption_has_explicit_destinations_and_black_hole_sink():
    central = initial_galaxy_state(
        HotGas=5.0,
        EjectedGas=1.0,
        ICS=0.5,
        MetalsHotGas=0.1,
        MetalsEjectedGas=0.02,
        MetalsICS=0.01,
    )
    satellite = initial_galaxy_state(
        MergTime=0.4,
        ColdGas=1.0,
        HotGas=2.0,
        EjectedGas=0.3,
        StellarMass=1.0,
        ICS=0.2,
        BlackHoleMass=0.05,
        MetalsColdGas=0.02,
        MetalsHotGas=0.04,
        MetalsEjectedGas=0.006,
        MetalsStellarMass=0.02,
        MetalsICS=0.004,
    )
    states = _stack(central, satellite)
    halos = _stack(
        initial_halo_forcing(Type=0, Mvir=100.0, dT=0.1),
        initial_halo_forcing(Type=1, Mvir=2.0, dT=0.1, CentralHalo=0),
    )
    before_baryons = active_group_baryonic_mass(states, halos.Type)
    before_metals = active_group_metal_mass(states, halos.Type)
    result = _group(states, halos)

    np.testing.assert_array_equal(result.diagnostics.action, (0, MERGER_ACTION_DISRUPTION))
    np.testing.assert_array_equal(result.halos.Type, (0, 3))
    np.testing.assert_array_equal(result.states.HotGas[0], np.float32(8.0))
    np.testing.assert_array_equal(result.states.ICS[0], np.float32(1.7))
    np.testing.assert_array_equal(
        result.diagnostics.ownership.black_hole_sink[1],
        np.float64(np.float32(0.05)),
    )
    np.testing.assert_allclose(
        active_group_baryonic_mass(result.states, result.halos.Type),
        before_baryons - satellite.BlackHoleMass,
        rtol=0.0,
        atol=6.0e-7,
    )
    np.testing.assert_allclose(
        active_group_metal_mass(result.states, result.halos.Type),
        before_metals,
        rtol=0.0,
        atol=1.0e-8,
    )


def test_major_merger_transfers_ownership_and_sets_event_times_after_consumers():
    states = _stack(
        initial_galaxy_state(
            ColdGas=0.0,
            StellarMass=10.0,
            BulgeMass=2.0,
            BlackHoleMass=0.1,
            MetalsStellarMass=0.2,
            MetalsBulgeMass=0.04,
            TimeOfLastMajorMerger=-1.0,
            TimeOfLastMinorMerger=-1.0,
        ),
        initial_galaxy_state(
            MergTime=-0.1,
            StellarMass=5.0,
            BulgeMass=1.0,
            BlackHoleMass=0.02,
            MetalsStellarMass=0.1,
            MetalsBulgeMass=0.02,
        ),
    )
    halos = _stack(
        initial_halo_forcing(Type=0, Mvir=100.0, dT=0.1),
        initial_halo_forcing(Type=1, Mvir=2.0, dT=0.1, CentralHalo=0),
    )
    result = _group(states, halos)
    expected_time = np.float32(13.8 + 0.1 - 0.5 * 0.1)

    np.testing.assert_array_equal(result.diagnostics.action, (0, MERGER_ACTION_MERGER))
    np.testing.assert_array_equal(result.diagnostics.mass_ratio[1], 0.5)
    np.testing.assert_array_equal(result.states.StellarMass[0], np.float32(15.0))
    np.testing.assert_array_equal(result.states.BulgeMass[0], np.float32(15.0))
    expected_black_hole = np.float32(np.float32(0.1) + np.float32(0.02))
    np.testing.assert_array_equal(result.states.BlackHoleMass[0], expected_black_hole)
    np.testing.assert_array_equal(result.states.TimeOfLastMinorMerger[0], expected_time)
    np.testing.assert_array_equal(result.states.TimeOfLastMajorMerger[0], expected_time)


def test_live_scan_redirects_a_type2_source_through_a_consumed_target_once():
    states = _stack(
        initial_galaxy_state(StellarMass=20.0, BulgeMass=3.0),
        initial_galaxy_state(MergTime=-0.1, StellarMass=3.0, BulgeMass=0.3),
        initial_galaxy_state(MergTime=5.0, StellarMass=1.5, BulgeMass=0.2),
        initial_galaxy_state(MergTime=-0.1, StellarMass=1.8, BulgeMass=0.1),
    )
    halos = _stack(
        initial_halo_forcing(Type=0, Mvir=80.0, dT=0.1),
        initial_halo_forcing(Type=1, CentralHalo=2, Mvir=1.0, dT=0.1),
        initial_halo_forcing(Type=1, CentralHalo=0, Mvir=20.0, dT=0.1),
        initial_halo_forcing(Type=2, CentralHalo=1, Mvir=0.5, dT=0.1),
    )
    result = _group(states, halos)

    np.testing.assert_array_equal(result.diagnostics.action, (0, 2, 0, 2))
    np.testing.assert_array_equal(result.diagnostics.target_index, (-1, 0, -1, 2))
    np.testing.assert_allclose(result.diagnostics.mass_ratio[1], 3.0 / 20.0)
    np.testing.assert_allclose(result.diagnostics.mass_ratio[3], 1.5 / 1.8)
    np.testing.assert_array_equal(result.halos.Type, (0, 3, 1, 3))
    np.testing.assert_array_equal(result.states.StellarMass[2], np.float32(3.3))
    np.testing.assert_array_equal(result.states.BulgeMass[2], np.float32(3.3))


def test_clock_decrement_uses_object_dt_and_live_substep_mvir():
    states = _stack(
        initial_galaxy_state(),
        initial_galaxy_state(MergTime=2.0, StellarMass=1.0),
    )
    halos = _stack(
        initial_halo_forcing(Type=0, dT=0.1),
        initial_halo_forcing(Type=1, Mvir=8.0, deltaMvir=4.0, dT=0.4),
    )
    result = _group(states, halos, substeps=4, substep=1)
    np.testing.assert_array_equal(result.states.MergTime[1], np.float32(1.9))
    np.testing.assert_array_equal(result.diagnostics.current_mvir[1], 6.0)
    np.testing.assert_array_equal(result.diagnostics.virial_to_baryons[1], 6.0)
    np.testing.assert_array_equal(result.diagnostics.action[1], 0)


def test_invalid_dt_halts_the_live_scan_and_reports_failure():
    states = _stack(
        initial_galaxy_state(),
        initial_galaxy_state(MergTime=-1.0, StellarMass=1.0),
        initial_galaxy_state(MergTime=-1.0, StellarMass=1.0),
    )
    halos = _stack(
        initial_halo_forcing(Type=0),
        initial_halo_forcing(Type=1, Mvir=0.5, dT=0.0),
        initial_halo_forcing(Type=1, Mvir=0.5, dT=0.1),
    )
    result = _group(states, halos)
    assert not bool(result.success)
    np.testing.assert_array_equal(result.diagnostics.error[1], MERGER_ERROR_INVALID_DT)
    np.testing.assert_array_equal(result.halos.Type, halos.Type)


def test_event_map_jit_vmap_and_derivative_level_ownership_conservation():
    def build(source_stars):
        return _stack(
            initial_galaxy_state(StellarMass=10.0, BulgeMass=10.0),
            initial_galaxy_state(MergTime=-0.1, StellarMass=source_stars, BulgeMass=source_stars),
        )

    halos = _stack(
        initial_halo_forcing(Type=0, Mvir=100.0, dT=0.1),
        initial_halo_forcing(Type=1, Mvir=1.0, dT=0.1),
    )
    states = build(jnp.asarray(2.0, dtype=jnp.float32))
    eager = _group(states, halos)
    compiled = jax.jit(_group)(states, halos)
    np.testing.assert_array_equal(compiled.states.StellarMass, eager.states.StellarMass)

    batched_states = jax.tree_util.tree_map(lambda value: jnp.stack((value, value)), states)
    batched_halos = jax.tree_util.tree_map(lambda value: jnp.stack((value, value)), halos)
    batched = jax.vmap(_group)(batched_states, batched_halos)
    np.testing.assert_array_equal(batched.halos.Type[0], eager.halos.Type)

    def conservation_residual(source_stars):
        current = build(source_stars)
        before = active_group_baryonic_mass(current, halos.Type)
        result = _group(current, halos)
        after = active_group_baryonic_mass(result.states, result.halos.Type)
        return after - before

    derivative = jax.grad(conservation_residual)(jnp.asarray(2.0, dtype=jnp.float32))
    np.testing.assert_allclose(derivative, 0.0, rtol=0.0, atol=0.0)
