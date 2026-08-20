"""Event-triggered Lagos23 starburst and BH-growth validation."""

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from mimic_jax.shark.burst import evolve_shark_starburst
from mimic_jax.shark.components import (
    black_hole_component,
    initial_shark_galaxy_state,
    initial_shark_subhalo_state,
    initial_shark_system_state,
    rotating_component,
    sized_component,
    system_baryonic_mass,
)
from mimic_jax.shark.interval import lagos23_model_parameters

_ORACLE = Path(__file__).parent / "fixtures/shark/lagos23_rate_oracle.json"


def _case():
    return initial_shark_system_state(
        galaxy=initial_shark_galaxy_state(
            bulge_stars=sized_component(2.0e9, 2.0e7, 1.0e9, 0.006),
            bulge_gas=sized_component(3.0e9, 3.0e7, 2.0e9, 0.008),
            black_hole=black_hole_component(2.0e6, spin=0.3),
            maximum_circular_velocity=jnp.asarray(180.0),
        ),
        subhalo=initial_shark_subhalo_state(
            hot_halo_gas=rotating_component(8.0e10, 8.0e8),
            ejected_gas=rotating_component(1.0e10, 1.0e8),
        ),
    )


def _run(state, **options):
    return evolve_shark_starburst(
        state,
        redshift=1.0,
        duration_gyr=0.2,
        virial_velocity=180.0,
        subhalo_velocity=180.0,
        galaxy_id=84,
        execution_seed=123456,
        model_parameters=lagos23_model_parameters(),
        **options,
    )


def test_complete_starburst_sequence_matches_pinned_upstream_oracle():
    oracle = json.loads(_ORACLE.read_text(encoding="utf-8"))["reference_starburst"]
    result = _run(_case(), num_steps=1)
    comparisons = {
        "black_hole_accretion_time": result.diagnostics.black_hole_accretion_time,
        "black_hole_transfer": result.diagnostics.black_hole_transfer,
        "black_hole_metal_transfer": result.diagnostics.black_hole_metal_transfer,
        "black_hole_mass": result.state.galaxy.black_hole.mass,
        "black_hole_metals": result.state.galaxy.black_hole.metals,
        "black_hole_spin": result.state.galaxy.black_hole.spin,
        "bulge_stellar_mass": result.state.galaxy.bulge_stars.mass,
        "bulge_gas_mass": result.state.galaxy.bulge_gas.mass,
        "bulge_stellar_metals": result.state.galaxy.bulge_stars.metals,
        "bulge_gas_metals": result.state.galaxy.bulge_gas.metals,
        "hot_halo_gas": result.state.subhalo.hot_halo_gas.mass,
        "hot_halo_gas_metals": result.state.subhalo.hot_halo_gas.metals,
        "ejected_gas": result.state.subhalo.ejected_gas.mass,
        "ejected_gas_metals": result.state.subhalo.ejected_gas.metals,
        "lost_gas": result.state.subhalo.lost_gas.mass,
        "lost_gas_metals": result.state.subhalo.lost_gas.metals,
        "star_formation_rate": result.diagnostics.mean_star_formation_rate,
    }
    for name, value in comparisons.items():
        # Persistent upstream reservoirs are float32 and its 5%-tolerance
        # Cash--Karp solve accepts this interval in one step. One JAX RK4 step
        # agrees to 1.1e-4; tighter timestep convergence is tested separately.
        np.testing.assert_allclose(value, oracle[name], rtol=1.1e-4, atol=1.0e-9)


def test_starburst_is_conservative_convergent_jittable_and_differentiable():
    state = _case()
    runs = [_run(state, num_steps=steps) for steps in (4, 8, 16, 32)]
    np.testing.assert_allclose(
        [system_baryonic_mass(run.state) for run in runs],
        system_baryonic_mass(state),
        rtol=2.0e-13,
        atol=2.0e-3,
    )
    values = np.asarray([run.state.galaxy.bulge_stars.mass for run in runs])
    errors = np.abs(values[:-1] - values[-1])
    assert errors[-1] < errors[0]

    parameters = lagos23_model_parameters()

    def final_stars(efficiency):
        varied = parameters._replace(
            star_formation=parameters.star_formation._replace(efficiency_per_gyr=efficiency)
        )
        return evolve_shark_starburst(
            state,
            redshift=1.0,
            duration_gyr=0.2,
            virial_velocity=180.0,
            subhalo_velocity=180.0,
            galaxy_id=84,
            execution_seed=123456,
            model_parameters=varied,
            num_steps=4,
        ).state.galaxy.bulge_stars.mass

    center = parameters.star_formation.efficiency_per_gyr
    eager = final_stars(center)
    np.testing.assert_allclose(jax.jit(final_stars)(center), eager, rtol=2.0e-14)
    derivative = jax.grad(final_stars)(center)
    epsilon = 1.0e-4
    finite_difference = (
        final_stars(center * (1.0 + epsilon)) - final_stars(center * (1.0 - epsilon))
    ) / (2.0 * epsilon * center)
    assert derivative > 0.0
    np.testing.assert_allclose(derivative, finite_difference, rtol=5.0e-5)


def test_subthreshold_bulge_is_an_explicit_no_event_branch():
    state = _case()._replace(
        galaxy=_case().galaxy._replace(bulge_gas=sized_component(9.0e4, 900.0, 1.0e4, 0.008))
    )
    result = jax.jit(lambda value: _run(value, num_steps=4))(state)
    assert not result.diagnostics.active
    np.testing.assert_allclose(system_baryonic_mass(result.state), system_baryonic_mass(state))
    np.testing.assert_allclose(result.state.galaxy.bulge_gas.mass, state.galaxy.bulge_gas.mass)
