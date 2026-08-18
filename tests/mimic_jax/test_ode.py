"""Continuous-limit and convergence tests for the quiescent SAGE16 ODE subset."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax import (
    ADAPTIVE_SUCCESS,
    FORWARD_EULER,
    HEUN_RK2,
    RK4,
    method_convergence_study,
)
from mimic_jax.sage16 import (
    ODE_STATE_NAMES,
    UPSTREAM_RATE_SUBSET,
    apply_reincorporation,
    calculate_cooling_budget,
    calculate_star_formation_budget,
    calculate_supernova_feedback_budget,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    integrate_sage16_ode,
    integrate_sage16_ode_adaptive,
    load_cooling_tables,
    ode_state_from_galaxy,
    sage16_ode_rhs_and_rates,
    sage16_units,
    step_context,
    subcycle_upstream_rate_subset,
)


def _ode_case():
    galaxy = initial_galaxy_state(
        ColdGas=2.0,
        HotGas=10.0,
        EjectedGas=1.0,
        StellarMass=1.0,
        MetalsColdGas=0.04,
        MetalsHotGas=0.2,
        MetalsEjectedGas=0.02,
        MetalsStellarMass=0.02,
        DiskScaleRadius=0.01,
    )
    halo = initial_halo_forcing(Mvir=100.0, Rvir=0.2, Vvir=150.0, dT=5.0e-4)
    return (
        galaxy,
        halo,
        step_context(time_interval=5.0e-4),
        fiducial_parameters(),
        sage16_units(),
        load_cooling_tables(),
    )


def _observables(state):
    return jnp.stack([getattr(state, name) for name in ODE_STATE_NAMES])


def test_ode_rates_match_isolated_upstream_rate_times_dt_budgets():
    galaxy, halo, context, parameters, units, tables = _ode_case()
    num_substeps = 128
    substep_context = context._replace(num_substeps=jnp.asarray(num_substeps, dtype=jnp.int32))
    dt = halo.dT / num_substeps
    rates = sage16_ode_rhs_and_rates(
        0.0,
        ode_state_from_galaxy(galaxy),
        halo,
        galaxy.DiskScaleRadius,
        parameters,
        units,
        tables,
    ).rates

    reincorporation = apply_reincorporation(
        galaxy,
        halo,
        substep_context,
        parameters,
    ).transfer
    cooling = calculate_cooling_budget(
        galaxy,
        halo,
        substep_context,
        units,
        tables,
    ).budget
    star_formation = calculate_star_formation_budget(
        galaxy,
        halo,
        substep_context,
        parameters,
    )
    supernova = calculate_supernova_feedback_budget(
        galaxy,
        halo,
        parameters,
        units,
        star_formation,
    )

    np.testing.assert_allclose(rates.reincorporation, reincorporation.gas / dt, rtol=2.0e-14)
    np.testing.assert_allclose(rates.cooling, cooling.gas / dt, rtol=2.0e-14)
    np.testing.assert_allclose(
        rates.star_formation,
        star_formation.NewStellarMass / dt,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        rates.sn_reheating,
        supernova.SupernovaReheatedMass / dt,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        rates.sn_ejection,
        supernova.SupernovaEjectedMass / dt,
        rtol=2.0e-14,
    )


def test_ode_rhs_conserves_baryons_and_exposes_the_metal_source_with_derivatives():
    galaxy, halo, _, parameters, units, tables = _ode_case()
    state = ode_state_from_galaxy(galaxy)

    def ledgers(sfr_efficiency):
        varied = parameters._replace(SfrEfficiency=sfr_efficiency)
        result = sage16_ode_rhs_and_rates(
            0.0,
            state,
            halo,
            galaxy.DiskScaleRadius,
            varied,
            units,
            tables,
        )
        derivative = result.derivative
        baryon_residual = (
            derivative.ColdGas + derivative.HotGas + derivative.EjectedGas + derivative.StellarMass
        )
        metal_residual = (
            derivative.MetalsColdGas
            + derivative.MetalsHotGas
            + derivative.MetalsEjectedGas
            + derivative.MetalsStellarMass
            - result.rates.produced_metals
        )
        return jnp.asarray([baryon_residual, metal_residual])

    residuals = ledgers(parameters.SfrEfficiency)
    np.testing.assert_allclose(residuals, jnp.zeros(2), atol=2.0e-12)
    residual_derivatives = jax.jacfwd(ledgers)(parameters.SfrEfficiency)
    np.testing.assert_allclose(residual_derivatives, jnp.zeros(2), atol=5.0e-11)


def test_fixed_step_methods_and_upstream_split_show_their_expected_orders():
    galaxy, halo, context, parameters, units, tables = _ode_case()
    initial = ode_state_from_galaxy(galaxy)
    reference_steps = 2048
    reference = integrate_sage16_ode(
        initial,
        halo,
        galaxy.DiskScaleRadius,
        parameters,
        units,
        tables,
        num_steps=reference_steps,
        method=RK4,
    ).final_state

    def run(method, num_steps):
        if method == UPSTREAM_RATE_SUBSET:
            result = subcycle_upstream_rate_subset(
                galaxy,
                halo,
                context,
                parameters,
                units,
                tables,
                num_substeps=num_steps,
            )
            return ode_state_from_galaxy(result.final_state)
        return integrate_sage16_ode(
            initial,
            halo,
            galaxy.DiskScaleRadius,
            parameters,
            units,
            tables,
            num_steps=num_steps,
            method=method,
        ).final_state

    study = method_convergence_study(
        run,
        _observables,
        reference,
        methods=(UPSTREAM_RATE_SUBSET, FORWARD_EULER, HEUN_RK2, RK4),
        step_counts=(2, 4, 8, 16, 32, 64),
        observable_names=ODE_STATE_NAMES,
        observable_units=("1e10 Msun/h",) * len(ODE_STATE_NAMES),
        rhs_evaluations_per_step={
            UPSTREAM_RATE_SUBSET: 1,
            FORWARD_EULER: 1,
            HEUN_RK2: 2,
            RK4: 4,
        },
        reference_method=RK4,
        reference_steps=reference_steps,
        duration=float(halo.dT),
    )
    maximum_relative_errors = np.nanmax(np.asarray(study.relative_errors), axis=2)
    asymptotic_orders = np.log2(maximum_relative_errors[:, -3:-1] / maximum_relative_errors[:, -2:])
    median_orders = np.median(asymptotic_orders, axis=1)
    np.testing.assert_allclose(median_orders[0], 1.0, atol=0.06)
    np.testing.assert_allclose(median_orders[1], 1.0, atol=0.06)
    np.testing.assert_allclose(median_orders[2], 2.0, atol=0.08)
    np.testing.assert_allclose(median_orders[3], 4.0, atol=0.12)
    assert maximum_relative_errors[3, -1] < maximum_relative_errors[2, -1]
    assert maximum_relative_errors[2, -1] < maximum_relative_errors[1, -1]


def test_ode_integration_is_jittable_differentiable_positive_and_conservative():
    galaxy, halo, _, parameters, units, tables = _ode_case()
    initial = ode_state_from_galaxy(galaxy)

    def final_state(sfr_efficiency):
        varied = parameters._replace(SfrEfficiency=sfr_efficiency)
        return integrate_sage16_ode(
            initial,
            halo,
            galaxy.DiskScaleRadius,
            varied,
            units,
            tables,
            num_steps=32,
            method=HEUN_RK2,
        ).final_state

    eager = final_state(parameters.SfrEfficiency)
    compiled = jax.jit(final_state)(parameters.SfrEfficiency)
    np.testing.assert_allclose(_observables(compiled), _observables(eager), rtol=2.0e-14)
    derivative = jax.grad(lambda efficiency: final_state(efficiency).StellarMass)(
        parameters.SfrEfficiency
    )
    assert bool(jnp.isfinite(derivative))
    assert float(derivative) > 0.0
    assert min(float(value) for value in _observables(eager)) > 0.0

    initial_baryons = sum(float(getattr(initial, name)) for name in ODE_STATE_NAMES[:4])
    final_baryons = sum(float(getattr(eager, name)) for name in ODE_STATE_NAMES[:4])
    assert abs(final_baryons - initial_baryons) < 5.0e-13


def test_adaptive_ode_converges_preserves_baryons_and_validates_its_gradient():
    galaxy, halo, _, parameters, units, tables = _ode_case()
    initial = ode_state_from_galaxy(galaxy)
    reference = integrate_sage16_ode(
        initial,
        halo,
        galaxy.DiskScaleRadius,
        parameters,
        units,
        tables,
        num_steps=2048,
        method=RK4,
    ).final_state
    reference_values = np.asarray(_observables(reference))
    solutions = []
    errors = []
    for tolerance in (1.0e-3, 1.0e-5, 1.0e-7):
        solution = integrate_sage16_ode_adaptive(
            initial,
            halo,
            galaxy.DiskScaleRadius,
            parameters,
            units,
            tables,
            relative_tolerance=tolerance,
            absolute_tolerance=tolerance * 1.0e-3,
            initial_step=halo.dT,
            jacobian_stability_factor=1.0,
            max_steps=128,
            max_attempts=512,
        )
        assert int(solution.status) == ADAPTIVE_SUCCESS
        solutions.append(solution)
        values = np.asarray(_observables(solution.final_state))
        errors.append(np.max(np.abs((values - reference_values) / reference_values)))
        assert np.min(values) > 0.0
        accepted = int(solution.accepted_steps)
        assert np.all(
            np.asarray(solution.accepted_step_sizes[:accepted])
            * np.asarray(solution.accepted_jacobian_norms[:accepted])
            <= 1.0 + 1.0e-12
        )
        initial_baryons = sum(float(getattr(initial, name)) for name in ODE_STATE_NAMES[:4])
        final_baryons = sum(
            float(getattr(solution.final_state, name)) for name in ODE_STATE_NAMES[:4]
        )
        assert abs(final_baryons - initial_baryons) < 2.0e-12
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 2.0e-8

    def final_stellar_mass(efficiency):
        varied = parameters._replace(SfrEfficiency=efficiency)
        return integrate_sage16_ode_adaptive(
            initial,
            halo,
            galaxy.DiskScaleRadius,
            varied,
            units,
            tables,
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-9,
            initial_step=halo.dT,
            jacobian_stability_factor=0.5,
            max_steps=64,
            max_attempts=128,
        ).final_state.StellarMass

    automatic = jax.grad(final_stellar_mass)(parameters.SfrEfficiency)
    epsilon = 1.0e-4
    finite_difference = (
        final_stellar_mass(parameters.SfrEfficiency * (1.0 + epsilon))
        - final_stellar_mass(parameters.SfrEfficiency * (1.0 - epsilon))
    ) / (2.0 * epsilon * parameters.SfrEfficiency)
    np.testing.assert_allclose(automatic, finite_difference, rtol=1.0e-6)
    np.testing.assert_allclose(
        jax.jit(final_stellar_mass)(parameters.SfrEfficiency),
        final_stellar_mass(parameters.SfrEfficiency),
        rtol=0.0,
        atol=0.0,
    )
