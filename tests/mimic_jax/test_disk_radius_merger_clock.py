"""Pre-timestep disk-radius and merger-clock parity contracts."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16 import (
    initial_galaxy_state,
    initial_halo_forcing,
    initialise_merger_clocks,
    sage16_units,
    set_disk_scale_radius,
)


def _stack(*records):
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *records)


def test_disk_radius_preserves_upstream_float_argument_boundary():
    state = initial_galaxy_state(DiskScaleRadius=0.123)
    halo = initial_halo_forcing(
        Type=0,
        Spin=(100.0, 150.0, 200.0),
        Vvir=200.00000001,
        Rvir=0.200000001,
    )
    result = set_disk_scale_radius(state, halo)

    spin = np.asarray((100.0, 150.0, 200.0), dtype=np.float32)
    vvir = np.float32(halo.Vvir)
    rvir = np.float32(halo.Rvir)
    magnitude = np.sqrt(sum(float(component) * float(component) for component in spin))
    expected = np.float32((magnitude / (1.414 * float(vvir) * float(rvir)) / 1.414) * rvir)
    np.testing.assert_array_equal(result.radius, expected)
    assert bool(result.updated)


def test_disk_radius_is_frozen_for_satellites_and_has_float_fallback():
    satellite = initial_galaxy_state(DiskScaleRadius=0.123)
    preserved = set_disk_scale_radius(
        satellite,
        initial_halo_forcing(Type=1, Spin=(1.0, 2.0, 3.0)),
    )
    np.testing.assert_array_equal(preserved.state.DiskScaleRadius, satellite.DiskScaleRadius)
    assert not bool(preserved.updated)

    fallback = set_disk_scale_radius(
        initial_galaxy_state(),
        initial_halo_forcing(Type=0, Vvir=1.0e-12, Rvir=0.2),
    )
    np.testing.assert_array_equal(
        fallback.radius,
        np.float32(0.1) * np.float32(0.2),
    )


def test_disk_radius_supports_jit_vmap_and_halo_input_gradients():
    states = _stack(initial_galaxy_state(), initial_galaxy_state(DiskScaleRadius=0.2))
    halos = _stack(
        initial_halo_forcing(Type=0, Spin=(100.0, 150.0, 200.0)),
        initial_halo_forcing(Type=1, Spin=(50.0, 25.0, 10.0)),
    )
    batched = jax.jit(jax.vmap(set_disk_scale_radius))(states, halos)
    assert bool(batched.updated[0])
    assert not bool(batched.updated[1])
    np.testing.assert_array_equal(batched.state.DiskScaleRadius[1], np.float32(0.2))

    def radius_from_spin_x(spin_x):
        halo = initial_halo_forcing(Spin=(spin_x, 150.0, 200.0))
        return set_disk_scale_radius(initial_galaxy_state(), halo).radius.astype(jnp.float64)

    derivative = jax.grad(radius_from_spin_x)(jnp.asarray(100.0, dtype=jnp.float32))
    assert np.isfinite(derivative)
    assert derivative > 0.0


def test_merger_clock_initialization_reproduces_sentinel_protocol_and_formula():
    states = _stack(
        initial_galaxy_state(MergTime=5.0),
        initial_galaxy_state(MergTime=999.9, StellarMass=5.0, ColdGas=2.0),
        initial_galaxy_state(MergTime=999.9, StellarMass=3.0, ColdGas=1.0),
        initial_galaxy_state(MergTime=999.9),
        initial_galaxy_state(MergTime=5.5),
    )
    halos = _stack(
        initial_halo_forcing(Type=0, Len=1000, Mvir=100.0, Rvir=0.5, Vvir=200.0),
        initial_halo_forcing(Type=1, Len=200, Mvir=20.0),
        initial_halo_forcing(Type=2, Len=0, Mvir=5.0, CentralHalo=1),
        initial_halo_forcing(Type=3, Len=200, Mvir=20.0),
        initial_halo_forcing(Type=1, Len=200, Mvir=20.0),
    )
    units = sage16_units()
    result = initialise_merger_clocks(states, halos, units)

    coulomb = np.log1p(1000.0 / 200.0)
    expected = np.float32(2.0 * 1.17 * 0.5 * 0.5 * 200.0 / (coulomb * units.G * 27.0))
    np.testing.assert_array_equal(result.states.MergTime[0], np.float32(999.9))
    np.testing.assert_array_equal(result.states.MergTime[1], expected)
    np.testing.assert_array_equal(result.states.MergTime[2], np.float32(0.0))
    np.testing.assert_array_equal(result.states.MergTime[3], np.float32(999.9))
    np.testing.assert_array_equal(result.states.MergTime[4], np.float32(5.5))
    np.testing.assert_array_equal(
        result.diagnostics.initialized,
        np.asarray((False, True, False, False, False)),
    )


def test_merger_clock_handles_immediate_ceiling_target_and_no_central_cases():
    states = _stack(
        initial_galaxy_state(),
        initial_galaxy_state(StellarMass=0.0, ColdGas=0.0),
        initial_galaxy_state(StellarMass=1.0, ColdGas=0.0),
    )
    halos = _stack(
        initial_halo_forcing(Type=0, Len=1000, Rvir=1.0e6, Vvir=1.0e6),
        initial_halo_forcing(Type=1, Len=0, Mvir=0.0),
        initial_halo_forcing(Type=1, Len=10, Mvir=0.0),
    )
    result = jax.jit(initialise_merger_clocks)(states, halos, sage16_units())
    np.testing.assert_array_equal(result.states.MergTime[1], np.float32(-1.0))
    np.testing.assert_array_equal(result.states.MergTime[2], np.float32(998.0))
    np.testing.assert_array_equal(result.diagnostics.target_indices, (0, 0, 0))

    no_central_halos = halos._replace(Type=jnp.asarray((1, 2, 3), dtype=jnp.int32))
    unchanged = initialise_merger_clocks(states, no_central_halos, sage16_units())
    np.testing.assert_array_equal(unchanged.states.MergTime, states.MergTime)
    np.testing.assert_array_equal(unchanged.diagnostics.target_indices, (-1, -1, -1))
