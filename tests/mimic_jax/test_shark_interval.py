"""Complete Lagos23 interval orchestration and continuous-mode tests."""

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from mimic_jax.shark.components import (
    black_hole_component,
    initial_shark_galaxy_state,
    initial_shark_subhalo_state,
    initial_shark_system_state,
    rotating_component,
    sized_component,
    system_baryonic_mass,
)
from mimic_jax.shark.interval import (
    evolve_shark_continuous_interval,
    evolve_shark_reference_interval,
    lagos23_model_parameters,
    shark_interval_forcing,
)
from mimic_jax.shark.prescriptions.structure import (
    cooling_gas_specific_angular_momentum,
    lagos23_cosmology,
)

_ORACLE = Path(__file__).parent / "fixtures/shark/lagos23_rate_oracle.json"


def _case():
    galaxy = initial_shark_galaxy_state(
        disk_stars=sized_component(2.0e9, 2.0e7, 4.0e9, 0.006),
        disk_gas=sized_component(3.0e9, 3.0e7, 9.0e9, 0.008),
        black_hole=black_hole_component(2.0e6, spin=0.3),
        maximum_circular_velocity=jnp.asarray(180.0),
    )
    subhalo = initial_shark_subhalo_state(
        hot_halo_gas=rotating_component(8.0e10, 8.0e8, 4.0e11),
        ejected_gas=rotating_component(1.0e10, 1.0e8, 5.0e10),
    )
    state = initial_shark_system_state(galaxy=galaxy, subhalo=subhalo)
    forcing = shark_interval_forcing(
        redshift=1.0,
        duration_gyr=0.2,
        halo_mass=8.0e11,
        subhalo_mass=8.0e11,
        virial_velocity=180.0,
        subhalo_velocity=180.0,
        virial_radius=0.15,
        halo_dynamical_time=0.9,
        hot_specific_angular_momentum=5.0,
        cooling_specific_angular_momentum=5.0,
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
    return state, forcing, lagos23_model_parameters()


def _physical_leaves(state):
    leaves = jax.tree_util.tree_leaves(state)
    return np.asarray([float(value) for value in leaves if np.issubdtype(value.dtype, np.floating)])


def test_reference_interval_composes_finite_preparation_and_flow_conservatively():
    state, forcing, parameters = _case()
    result = evolve_shark_reference_interval(state, forcing, parameters, num_steps=32)
    assert np.all(np.isfinite(_physical_leaves(result.state)))
    np.testing.assert_allclose(result.diagnostics.infall_mass, 1.0e9)
    # Yield creates metals but no baryonic mass; all six flow reservoirs and
    # the black hole are included in this complete-system ledger.
    np.testing.assert_allclose(
        system_baryonic_mass(result.state) - system_baryonic_mass(state),
        result.diagnostics.infall_mass,
        rtol=2.0e-13,
        atol=2.0e-3,
    )
    assert result.diagnostics.cooling_rate > 0.0
    assert result.diagnostics.accepted_steps == 32


def test_reference_interval_matches_pinned_upstream_basic_physical_model():
    oracle = json.loads(_ORACLE.read_text(encoding="utf-8"))["reference_interval"]
    galaxy = initial_shark_galaxy_state(
        disk_stars=sized_component(
            2.0e9,
            2.0e7,
            2.0e9 * (180.0 * 0.006 / 0.835),
            0.006,
        ),
        disk_gas=sized_component(
            3.0e9,
            3.0e7,
            3.0e9 * (180.0 * 0.008 / 0.835),
            0.008,
        ),
        black_hole=black_hole_component(2.0e6, spin=0.3),
        maximum_circular_velocity=jnp.asarray(180.0),
    )
    state = initial_shark_system_state(
        galaxy=galaxy,
        subhalo=initial_shark_subhalo_state(
            hot_halo_gas=rotating_component(8.0e10, 8.0e8, 4.0e11),
            ejected_gas=rotating_component(1.0e10, 1.0e8, 5.0e10),
        ),
    )
    cosmology = lagos23_cosmology()
    forcing = shark_interval_forcing(
        redshift=oracle["redshift"],
        duration_gyr=oracle["duration_gyr"],
        halo_mass=oracle["halo_mass"],
        subhalo_mass=oracle["halo_mass"],
        virial_velocity=oracle["virial_velocity"],
        subhalo_velocity=oracle["virial_velocity"],
        virial_radius=oracle["virial_radius"],
        halo_dynamical_time=oracle["halo_dynamical_time"],
        hot_specific_angular_momentum=5.0,
        cooling_specific_angular_momentum=cooling_gas_specific_angular_momentum(
            oracle["halo_mass"], 0.03, oracle["redshift"], cosmology
        ),
        accreted_mass=0.0,
        maximum_allowed_baryon_accretion=0.0,
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
    result = evolve_shark_reference_interval(
        state, forcing, lagos23_model_parameters(), num_steps=1
    )
    comparisons = {
        "stellar_mass": result.state.galaxy.disk_stars.mass,
        "cold_gas": result.state.galaxy.disk_gas.mass,
        "hot_halo_gas": result.state.subhalo.hot_halo_gas.mass,
        "ejected_gas": result.state.subhalo.ejected_gas.mass,
        "stellar_metals": result.state.galaxy.disk_stars.metals,
        "cold_gas_metals": result.state.galaxy.disk_gas.metals,
        "hot_halo_gas_metals": result.state.subhalo.hot_halo_gas.metals,
        "ejected_gas_metals": result.state.subhalo.ejected_gas.metals,
        "black_hole_mass": result.state.galaxy.black_hole.mass,
        "black_hole_spin": result.state.galaxy.black_hole.spin,
        "cooling_rate": result.diagnostics.cooling_rate,
        "star_formation_rate": result.diagnostics.mean_star_formation_rate,
        "heating_radius": result.diagnostics.heating_radius,
    }
    for name, value in comparisons.items():
        # Upstream stores persistent reservoirs in float32 and accepts this
        # interval in one 5%-tolerance Cash--Karp step. One JAX RK4 step tracks
        # every reported field to better than 3e-5 relative.
        np.testing.assert_allclose(value, oracle[name], rtol=3.0e-5, atol=1.0e-12)


def test_continuous_interval_converges_and_retains_nonnegative_reservoirs():
    state, forcing, parameters = _case()
    results = [
        evolve_shark_continuous_interval(state, forcing, parameters, num_substeps=steps)
        for steps in (8, 16, 32, 64)
    ]
    stellar = np.asarray([float(value.state.galaxy.disk_stars.mass) for value in results])
    errors = np.abs(stellar[:-1] - stellar[-1])
    assert errors[-1] < errors[0]
    for result in results:
        assert np.all(np.isfinite(_physical_leaves(result.state)))
        assert result.state.galaxy.disk_gas.mass >= 0.0
        assert result.state.subhalo.hot_halo_gas.mass >= 0.0
        np.testing.assert_allclose(
            system_baryonic_mass(result.state) - system_baryonic_mass(state),
            result.diagnostics.infall_mass,
            rtol=3.0e-11,
            atol=3.0,
        )


def test_continuous_interval_is_jittable_and_has_exact_parameter_derivatives():
    state, forcing, parameters = _case()

    def final_stellar_mass(efficiency):
        star_formation = parameters.star_formation._replace(efficiency_per_gyr=efficiency)
        varied = parameters._replace(star_formation=star_formation)
        return evolve_shark_continuous_interval(
            state, forcing, varied, num_substeps=4
        ).state.galaxy.disk_stars.mass

    eager = final_stellar_mass(parameters.star_formation.efficiency_per_gyr)
    compiled = jax.jit(final_stellar_mass)(parameters.star_formation.efficiency_per_gyr)
    np.testing.assert_allclose(compiled, eager, rtol=2.0e-14)
    derivative = jax.grad(final_stellar_mass)(parameters.star_formation.efficiency_per_gyr)
    assert jnp.isfinite(derivative)
    assert derivative > 0.0
    epsilon = 1.0e-4
    center = float(parameters.star_formation.efficiency_per_gyr)
    finite_difference = (
        float(final_stellar_mass(center * (1.0 + epsilon)))
        - float(final_stellar_mass(center * (1.0 - epsilon)))
    ) / (2.0 * epsilon * center)
    np.testing.assert_allclose(derivative, finite_difference, rtol=5.0e-5)
