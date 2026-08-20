"""Ordered SHARK hybrid snapshot orchestration tests."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from mimic_jax.shark.components import (
    black_hole_component,
    galaxy_baryonic_mass,
    initial_shark_galaxy_state,
    initial_shark_subhalo_state,
    initial_shark_system_state,
    rotating_component,
    sized_component,
    system_baryonic_mass,
)
from mimic_jax.shark.evolution import (
    SharkHybridEventSchedule,
    evolve_shark_hybrid_interval,
)
from mimic_jax.shark.interval import lagos23_model_parameters, shark_interval_forcing


def _forcing():
    return shark_interval_forcing(
        redshift=1.0,
        duration_gyr=0.2,
        halo_mass=8.0e11,
        subhalo_mass=8.0e11,
        virial_velocity=180.0,
        subhalo_velocity=180.0,
        virial_radius=0.15,
        halo_dynamical_time=0.9,
        hot_specific_angular_momentum=5.0,
        cooling_specific_angular_momentum=2.0,
        accreted_mass=1.0e9,
        maximum_allowed_baryon_accretion=2.0e9,
        baryon_fraction_excess_after_infall=0.0,
        stripped_hot_halo_mass_for_density=0.0,
        galaxy_velocity=180.0,
        gas_half_mass_radius=0.008,
        stellar_half_mass_radius=0.006,
        is_central_subhalo=True,
        ignore_galaxy_formation=False,
        galaxy_id=42,
        execution_seed=123456,
    )


def _system():
    return initial_shark_system_state(
        galaxy=initial_shark_galaxy_state(
            disk_stars=sized_component(2.0e9, 2.0e7, 4.0e9, 0.006),
            disk_gas=sized_component(3.0e9, 3.0e7, 9.0e9, 0.008),
            black_hole=black_hole_component(2.0e6, spin=0.3),
            maximum_circular_velocity=jnp.asarray(180.0),
        ),
        subhalo=initial_shark_subhalo_state(
            hot_halo_gas=rotating_component(8.0e10, 8.0e8, 4.0e11),
            ejected_gas=rotating_component(1.0e10, 1.0e8, 5.0e10),
        ),
    )


def _satellite():
    return initial_shark_galaxy_state(
        disk_stars=sized_component(8.0e8, 8.0e6, 1.5e9, 0.004),
        disk_gas=sized_component(6.0e8, 6.0e6, 1.2e9, 0.005),
        black_hole=black_hole_component(1.0e5, spin=0.1),
        maximum_circular_velocity=jnp.asarray(100.0),
    )


def test_hybrid_scheduler_routes_merger_burst_instability_and_flow_in_order():
    state = _system()
    satellite = _satellite()
    result = evolve_shark_hybrid_interval(
        state,
        _forcing(),
        lagos23_model_parameters(),
        events=SharkHybridEventSchedule(merging_satellites=(satellite,)),
        formulation="reference",
        num_steps=8,
    )
    assert result.merger_count == 1
    assert result.merger_burst.diagnostics.active
    # The major merger has already consumed the disk before the subsequent
    # disk-instability test, making that event inactive on this branch.
    assert not result.disk_instability.triggered
    expected = (
        system_baryonic_mass(state)
        + galaxy_baryonic_mass(satellite)
        + result.interval.diagnostics.infall_mass
    )
    np.testing.assert_allclose(system_baryonic_mass(result.state), expected, rtol=2.0e-12, atol=0.1)


def test_reference_and_continuous_scheduler_modes_are_explicit_and_finite():
    state = _system()
    reference = evolve_shark_hybrid_interval(
        state,
        _forcing(),
        lagos23_model_parameters(),
        formulation="reference",
        num_steps=8,
    )
    continuous = evolve_shark_hybrid_interval(
        state,
        _forcing(),
        lagos23_model_parameters(),
        formulation="continuous",
        num_steps=8,
    )
    assert np.isfinite(reference.state.galaxy.disk_stars.mass)
    assert np.isfinite(continuous.state.galaxy.disk_stars.mass)
    # They answer different numerical questions and are not silently aliases.
    assert reference.state.galaxy.disk_gas.mass != continuous.state.galaxy.disk_gas.mass


def test_existing_bulge_gas_does_not_trigger_a_new_burst_without_an_event():
    state = _system()
    galaxy = state.galaxy._replace(
        bulge_stars=sized_component(8.0e9, 8.0e7, 5.0e9, 0.03),
        bulge_gas=sized_component(5.0e8, 5.0e6, 2.0e8, 0.02),
    )
    state = state._replace(galaxy=galaxy)
    result = evolve_shark_hybrid_interval(
        state,
        _forcing(),
        lagos23_model_parameters(),
        formulation="reference",
        num_steps=8,
    )
    assert not result.merger_burst.diagnostics.active
    assert not result.disk_instability.triggered
    assert not result.instability_burst.diagnostics.active
    np.testing.assert_allclose(
        result.state.galaxy.bulge_gas.mass,
        state.galaxy.bulge_gas.mass,
        rtol=0.0,
        atol=0.0,
    )
