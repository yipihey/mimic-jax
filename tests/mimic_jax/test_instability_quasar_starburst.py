"""Disk-instability, quasar-mode, and collisional-starburst physics."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16 import (
    apply_collisional_starburst,
    apply_disk_instability,
    apply_quasar_mode,
    baryonic_mass,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    metal_mass,
    sage16_units,
)

MASS_ATOL = 4.0e-6
METAL_ATOL = 4.0e-7


def test_disk_instability_matches_the_upstream_structural_map():
    state = initial_galaxy_state(
        ColdGas=5.0,
        StellarMass=10.0,
        BulgeMass=2.0,
        MetalsStellarMass=0.2,
        MetalsBulgeMass=0.04,
        DiskScaleRadius=0.003,
    )
    halo = initial_halo_forcing(Vmax=200.0)
    result = apply_disk_instability(
        state,
        halo,
        fiducial_parameters(),
        sage16_units(),
    )

    disk_mass = 13.0
    critical = (
        float(np.float32(200.0 * 200.0))
        * (3.0 * float(np.float32(0.003)))
        / float(sage16_units().G)
    )
    unstable = disk_mass - critical
    expected_stars = (8.0 / disk_mass) * unstable
    expected_gas = (5.0 / disk_mass) * unstable
    np.testing.assert_allclose(result.transfer.disk_mass, disk_mass, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.transfer.critical_mass, critical, rtol=1.0e-14)
    np.testing.assert_allclose(result.transfer.disk_stars_to_bulge, expected_stars, rtol=1.0e-14)
    np.testing.assert_allclose(result.transfer.unstable_gas, expected_gas, rtol=1.0e-14)
    np.testing.assert_allclose(result.state.BulgeMass, 2.0 + expected_stars, atol=6.0e-7)
    np.testing.assert_allclose(
        result.state.UnstableDiskGasFraction,
        expected_gas / 5.0,
        rtol=1.0e-14,
    )


def test_disk_instability_is_a_component_transfer_and_preserves_stellar_metals():
    state = initial_galaxy_state(
        ColdGas=2.0,
        StellarMass=4.0,
        BulgeMass=1.0,
        MetalsStellarMass=0.08,
        MetalsBulgeMass=0.02,
        DiskScaleRadius=0.001,
    )
    result = apply_disk_instability(
        state,
        initial_halo_forcing(Vmax=120.0),
        fiducial_parameters(),
        sage16_units(),
    )
    np.testing.assert_array_equal(result.state.StellarMass, state.StellarMass)
    np.testing.assert_array_equal(result.state.MetalsStellarMass, state.MetalsStellarMass)
    assert float(result.state.BulgeMass) > float(state.BulgeMass)
    assert float(result.state.MetalsBulgeMass) > float(state.MetalsBulgeMass)


def test_quasar_growth_and_wind_conserve_baryons_and_expose_the_metal_sink():
    state = initial_galaxy_state(
        ColdGas=10.0,
        HotGas=5.0,
        EjectedGas=1.0,
        MetalsColdGas=0.2,
        MetalsHotGas=0.1,
        MetalsEjectedGas=0.02,
        BlackHoleMass=0.01,
        UnstableDiskGasFraction=0.5,
    )
    result = apply_quasar_mode(
        state,
        initial_halo_forcing(Vvir=300.0),
        fiducial_parameters(),
        sage16_units(),
    )
    np.testing.assert_allclose(
        baryonic_mass(result.state),
        baryonic_mass(state),
        rtol=0.0,
        atol=MASS_ATOL,
    )
    np.testing.assert_allclose(
        metal_mass(result.state),
        metal_mass(state) - result.transfer.cold_metals_accreted,
        rtol=0.0,
        atol=METAL_ATOL,
    )
    assert float(result.transfer.black_hole_accreted) > 0.0
    np.testing.assert_array_equal(result.state.ColdGas, 0.0)
    np.testing.assert_array_equal(result.state.HotGas, 0.0)
    np.testing.assert_array_equal(
        result.state.UnstableDiskGasFraction,
        state.UnstableDiskGasFraction,
    )


def test_starburst_conserves_baryons_and_adds_exactly_its_yield():
    state = initial_galaxy_state(
        ColdGas=10.0,
        HotGas=5.0,
        EjectedGas=1.0,
        StellarMass=5.0,
        BulgeMass=1.0,
        MetalsColdGas=0.2,
        MetalsHotGas=0.1,
        MetalsEjectedGas=0.02,
        MetalsStellarMass=0.1,
        MetalsBulgeMass=0.02,
        UnstableDiskGasFraction=0.2,
    )
    halo = initial_halo_forcing(Type=0, Mvir=100.0, Vvir=300.0, dT=0.1)
    result = apply_collisional_starburst(
        state,
        state,
        halo,
        halo,
        state.UnstableDiskGasFraction,
        1,
        halo.dT,
        fiducial_parameters(),
        sage16_units(),
    )
    np.testing.assert_allclose(
        baryonic_mass(result.galaxy),
        baryonic_mass(state),
        rtol=0.0,
        atol=MASS_ATOL,
    )
    np.testing.assert_allclose(
        metal_mass(result.galaxy),
        metal_mass(state) + result.transfer.produced_metals,
        rtol=0.0,
        atol=METAL_ATOL,
    )
    np.testing.assert_allclose(
        result.galaxy.BulgeMass - state.BulgeMass,
        result.transfer.locked_stars,
        rtol=0.0,
        atol=4.0e-7,
    )
    np.testing.assert_allclose(
        result.galaxy.StarFormationRate,
        result.transfer.formed_stars / halo.dT,
        rtol=2.0e-7,
    )


def test_satellite_starburst_uses_the_fof_central_feedback_destination():
    satellite = initial_galaxy_state(
        ColdGas=5.0,
        StellarMass=1.0,
        BulgeMass=0.2,
        MetalsColdGas=0.1,
        MetalsStellarMass=0.02,
        MetalsBulgeMass=0.004,
    )
    central = initial_galaxy_state(
        HotGas=8.0,
        EjectedGas=1.0,
        MetalsHotGas=0.16,
        MetalsEjectedGas=0.02,
    )
    satellite_halo = initial_halo_forcing(Type=1, Vvir=100.0, dT=0.1)
    central_halo = initial_halo_forcing(Type=0, Mvir=100.0, Vvir=200.0, dT=0.1)
    result = apply_collisional_starburst(
        satellite,
        central,
        satellite_halo,
        central_halo,
        0.2,
        1,
        satellite_halo.dT,
        fiducial_parameters(),
        sage16_units(),
    )
    np.testing.assert_allclose(
        baryonic_mass(result.galaxy) + baryonic_mass(result.central),
        baryonic_mass(satellite) + baryonic_mass(central),
        rtol=0.0,
        atol=2.0 * MASS_ATOL,
    )
    assert float(result.central.HotGas) != float(central.HotGas)
    np.testing.assert_array_equal(result.galaxy.HotGas, satellite.HotGas)


def test_coupled_kernels_jit_and_fractional_derivatives_match_finite_differences():
    state = initial_galaxy_state(
        ColdGas=5.0,
        StellarMass=10.0,
        BulgeMass=2.0,
        MetalsColdGas=0.1,
        MetalsStellarMass=0.2,
        MetalsBulgeMass=0.04,
        DiskScaleRadius=0.003,
    )
    halo = initial_halo_forcing(Vmax=200.0, Vvir=300.0)
    parameters = fiducial_parameters()._replace(
        QuasarModeEfficiency=jnp.asarray(0.0, dtype=jnp.float64)
    )
    units = sage16_units()
    eager = apply_disk_instability(state, halo, parameters, units)
    compiled = jax.jit(apply_disk_instability)(state, halo, parameters, units)
    np.testing.assert_array_equal(compiled.state.BulgeMass, eager.state.BulgeMass)
    states = jax.tree_util.tree_map(lambda value: jnp.stack((value, value)), state)
    halos = jax.tree_util.tree_map(lambda value: jnp.stack((value, value)), halo)
    batched = jax.vmap(
        lambda current_state, current_halo: apply_disk_instability(
            current_state,
            current_halo,
            parameters,
            units,
        )
    )(states, halos)
    np.testing.assert_array_equal(
        batched.state.BulgeMass,
        jnp.stack((eager.state.BulgeMass, eager.state.BulgeMass)),
    )

    def unstable_gas(epsilon):
        return apply_disk_instability(
            state,
            halo,
            parameters,
            units,
            epsilon,
        ).transfer.unstable_gas

    step = jnp.asarray(1.0e-3, dtype=jnp.float64)
    disk_automatic = jax.grad(unstable_gas)(jnp.asarray(0.0, dtype=jnp.float64))
    disk_finite = (unstable_gas(step) - unstable_gas(-step)) / (2.0 * step)
    np.testing.assert_allclose(disk_automatic, disk_finite, rtol=2.0e-7)

    unstable = eager.state

    def black_hole_mass(epsilon):
        return apply_quasar_mode(
            unstable,
            halo,
            parameters,
            units,
            None,
            epsilon,
        ).state.BlackHoleMass

    automatic = jax.grad(black_hole_mass)(jnp.asarray(0.0, dtype=jnp.float64))
    finite = (black_hole_mass(step) - black_hole_mass(-step)) / (2.0 * step)
    np.testing.assert_allclose(automatic, finite, rtol=2.0e-3, atol=1.0e-7)
    np.testing.assert_allclose(
        jax.jacfwd(black_hole_mass)(jnp.asarray(0.0, dtype=jnp.float64)),
        automatic,
    )
    np.testing.assert_allclose(
        jax.jacrev(black_hole_mass)(jnp.asarray(0.0, dtype=jnp.float64)),
        automatic,
    )

    def quasar_baryons(epsilon):
        return baryonic_mass(
            apply_quasar_mode(
                unstable,
                halo,
                parameters,
                units,
                None,
                epsilon,
            ).state
        )

    np.testing.assert_allclose(
        jax.grad(quasar_baryons)(jnp.asarray(0.0, dtype=jnp.float64)),
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )

    def burst_baryons(epsilon):
        result = apply_collisional_starburst(
            unstable,
            unstable,
            halo,
            halo,
            unstable.UnstableDiskGasFraction,
            1,
            halo.dT,
            parameters,
            units,
            epsilon,
        )
        return baryonic_mass(result.galaxy)

    np.testing.assert_allclose(
        jax.grad(burst_baryons)(jnp.asarray(0.0, dtype=jnp.float64)),
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )
