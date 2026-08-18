"""Numerical-method separation and refinement diagnostics for the faithful slice."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax import (
    ADAPTIVE_SUCCESS,
    conservation_residual,
    integrate_adaptive,
    rhs_jacobian,
    scaled_jacobian_infinity_norm,
    step_to_timescale_ratio,
    timestep_refinement_study,
)
from mimic_jax.sage16 import (
    baryonic_mass,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    load_cooling_tables,
    sage16_units,
    step_context,
    subcycle_upstream_sequential_central,
)


def _fixed_forcing_case():
    return (
        initial_galaxy_state(
            ColdGas=2.0,
            HotGas=10.0,
            EjectedGas=1.0,
            StellarMass=1.0,
            MetalsColdGas=0.04,
            MetalsHotGas=0.2,
            MetalsEjectedGas=0.02,
            MetalsStellarMass=0.02,
            BlackHoleMass=0.01,
            DiskScaleRadius=0.01,
        ),
        initial_halo_forcing(Mvir=100.0, Rvir=0.2, Vvir=200.0, dT=0.01),
        step_context(time_interval=0.01),
        fiducial_parameters(),
        sage16_units(),
        load_cooling_tables(),
    )


def test_upstream_subcycling_is_jittable_conservative_and_positive():
    state, halo, context, parameters, units, tables = _fixed_forcing_case()

    def run(current):
        return subcycle_upstream_sequential_central(
            current,
            halo,
            context,
            parameters,
            units,
            tables,
            num_substeps=4,
        )

    eager = run(state)
    compiled = jax.jit(run)(state)
    np.testing.assert_array_equal(compiled.final_state.StellarMass, eager.final_state.StellarMass)
    residual = conservation_residual(
        baryonic_mass(state),
        baryonic_mass(eager.final_state),
    )
    assert abs(float(residual)) < 3.0e-6
    for reservoir in ("ColdGas", "HotGas", "EjectedGas", "StellarMass", "BlackHoleMass"):
        assert bool(jnp.all(getattr(eager.states, reservoir) >= 0.0))


def test_refinement_study_records_method_forcing_errors_and_empirical_orders(tmp_path):
    state, halo, context, parameters, units, tables = _fixed_forcing_case()

    def run(num_substeps):
        return subcycle_upstream_sequential_central(
            state,
            halo,
            context,
            parameters,
            units,
            tables,
            num_substeps=num_substeps,
        )

    def observables(result):
        final = result.final_state
        return jnp.asarray([final.StellarMass, final.ColdGas, final.HotGas, final.BlackHoleMass])

    study = timestep_refinement_study(
        run,
        observables,
        substeps=(1, 2, 4, 8),
        observable_names=("stellar_mass", "cold_gas", "hot_gas", "black_hole_mass"),
        observable_units=("1e10 Msun/h",) * 4,
    )
    assert study.method == "upstream_sequential"
    assert study.forcing_interpolation == "piecewise_constant"
    assert study.observable_values.shape == (4, 4)
    assert study.observed_orders.shape == (2, 4)
    np.testing.assert_array_equal(study.absolute_errors[-1], jnp.zeros(4))
    assert bool(jnp.any(study.absolute_errors[0] > 0.0))

    archive = tmp_path / "refinement.npz"
    study.save(archive)
    with np.load(archive) as saved:
        assert saved["method"] == "upstream_sequential"
        assert saved["forcing_interpolation"] == "piecewise_constant"


def test_step_to_timescale_ratio_is_dimensionless_and_handles_empty_sources():
    np.testing.assert_allclose(step_to_timescale_ratio(10.0, 2.0), 0.2)
    np.testing.assert_allclose(step_to_timescale_ratio(0.0, 0.0), 0.0)
    assert bool(jnp.isinf(step_to_timescale_ratio(0.0, 1.0)))


def test_radio_mode_parameter_gradient_survives_upstream_subcycling():
    state, halo, context, parameters, units, tables = _fixed_forcing_case()

    def final_black_hole_mass(efficiency):
        varied = parameters._replace(RadioModeEfficiency=efficiency)
        return subcycle_upstream_sequential_central(
            state,
            halo,
            context,
            varied,
            units,
            tables,
            num_substeps=4,
        ).final_state.BlackHoleMass

    derivative = jax.grad(final_black_hole_mass)(parameters.RadioModeEfficiency)
    assert bool(jnp.isfinite(derivative))
    assert float(derivative) > 0.0


def test_adaptive_error_and_scaled_jacobian_control_are_jittable_and_differentiable():
    rate = jnp.asarray(2.0, dtype=jnp.float64)

    def solve(current_rate):
        return integrate_adaptive(
            lambda _time, value: -current_rate * value,
            jnp.asarray(1.0, dtype=jnp.float64),
            duration=1.0,
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-11,
            initial_step=0.5,
            jacobian_stability_factor=1.0,
            require_nonnegative=True,
            max_steps=128,
            max_attempts=512,
        )

    solution = solve(rate)
    assert int(solution.status) == ADAPTIVE_SUCCESS
    accepted = int(solution.accepted_steps)
    assert accepted > 1
    np.testing.assert_allclose(solution.final_state, np.exp(-2.0), rtol=0.0, atol=7.0e-10)
    assert bool(
        jnp.all(
            solution.accepted_step_sizes[:accepted] * solution.accepted_jacobian_norms[:accepted]
            <= 1.0 + 1.0e-12
        )
    )
    np.testing.assert_allclose(jax.jit(lambda value: solve(value).final_state)(rate), np.exp(-2.0))
    derivative = jax.grad(lambda value: solve(value).final_state)(rate)
    np.testing.assert_allclose(derivative, -np.exp(-2.0), rtol=3.0e-7)

    rhs = lambda _time, value: -rate * value
    np.testing.assert_allclose(rhs_jacobian(rhs, 0.0, jnp.asarray(1.0)), [[-2.0]])
    np.testing.assert_allclose(
        scaled_jacobian_infinity_norm(
            rhs,
            0.0,
            jnp.asarray(1.0),
            jnp.asarray(0.1),
        ),
        2.0,
    )
