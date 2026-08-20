"""Reference equations, conservation, transforms, and convergence for SHARK flows."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax import ADAPTIVE_SUCCESS, FORWARD_EULER, HEUN_RK2, RK4
from mimic_jax.shark import (
    SHARK_ODE_STATE_NAMES,
    SharkAugmentedFlowRates,
    SharkFlowRates,
    augmented_baryonic_mass,
    augmented_metal_mass,
    baryonic_mass,
    croton06_unheated_cooling,
    direct_cooling_flow_derivative,
    flow_conservation_residuals,
    initial_shark_continuous_state,
    initial_shark_state,
    integrate_shark_augmented_flow,
    integrate_shark_augmented_flow_adaptive,
    integrate_shark_flow,
    integrate_shark_flow_adaptive,
    lagos23_croton06_cooling_parameters,
    shark_augmented_continuous_rhs_from_rates,
    shark_continuous_rhs_from_rates,
    shark_flow_parameters,
    shark_rhs_from_rates,
    stack_shark_states,
    zero_flow_rates,
)


def _reference_case():
    state = initial_shark_state(
        stellar_mass=4.0,
        cold_gas=10.0,
        cold_halo_gas=8.0,
        hot_halo_gas=30.0,
        ejected_gas=5.0,
        lost_gas=1.0,
        stellar_metals=0.08,
        cold_gas_metals=0.2,
        cold_halo_gas_metals=0.12,
        hot_halo_gas_metals=0.3,
        ejected_gas_metals=0.05,
        lost_gas_metals=0.01,
        stellar_angular_momentum=8.0,
        cold_gas_angular_momentum=50.0,
        cold_halo_angular_momentum=40.0,
        hot_halo_angular_momentum=90.0,
        ejected_angular_momentum=10.0,
    )
    rates = SharkFlowRates(
        cooling=jnp.asarray(3.0),
        star_formation=jnp.asarray(2.0),
        star_formation_angular_momentum=jnp.asarray(7.0),
        stellar_reheating_loading=jnp.asarray(2.0),
        stellar_ejection_loading=jnp.asarray(0.75),
        angular_momentum_reheating_loading=jnp.asarray(1.5),
        angular_momentum_ejection_loading=jnp.asarray(0.5),
        qso_reheating_loading=jnp.asarray(0.25),
        qso_ejection_loading=jnp.asarray(0.1),
        cooling_metallicity=jnp.asarray(0.015),
        cooling_specific_angular_momentum=jnp.asarray(5.0),
    )
    parameters = shark_flow_parameters(
        recycle_fraction=0.4,
        yield_mass_fraction=0.03,
        evolving_yield=True,
        pre_enrichment_metallicity=1.0e-4,
    )
    return state, rates, parameters


def _state_array(state):
    return jnp.stack([getattr(state, name) for name in SHARK_ODE_STATE_NAMES])


def test_rhs_matches_upstream_basic_physical_model_equations():
    state, rates, parameters = _reference_case()
    result = shark_rhs_from_rates(0.0, state, rates, parameters)

    expected = np.asarray(
        [
            1.2,
            -2.7,
            -3.0,
            2.8,
            1.5,
            0.2,
            0.024,
            -0.019,
            -0.045,
            0.056,
            0.03,
            0.004,
            2.0,
            0.04,
            4.2,
            0.3,
            -15.0,
            7.0,
            3.5,
        ]
    )
    np.testing.assert_allclose(_state_array(result.derivative), expected, rtol=5.0e-15)
    np.testing.assert_allclose(result.cold_gas_metallicity, 0.02, rtol=1.0e-15)
    np.testing.assert_allclose(result.effective_yield, 0.025, rtol=1.0e-15)


def test_flow_conservation_and_its_rate_derivatives_are_structural():
    state, rates, parameters = _reference_case()

    def residuals(star_formation):
        varied = rates._replace(star_formation=star_formation)
        result = shark_rhs_from_rates(0.0, state, varied, parameters)
        return jnp.stack(flow_conservation_residuals(result))

    np.testing.assert_allclose(residuals(rates.star_formation), np.zeros(3), atol=2.0e-15)
    np.testing.assert_allclose(
        jax.jacfwd(residuals)(rates.star_formation), np.zeros(3), atol=2.0e-15
    )


def test_continuous_cooling_bypasses_reference_staging_reservoir_conservatively():
    state, rates, parameters = _reference_case()
    reference = shark_rhs_from_rates(0.0, state, rates, parameters).derivative
    continuous = shark_continuous_rhs_from_rates(0.0, state, rates, parameters)

    assert reference.cold_halo_gas == -rates.cooling
    assert reference.hot_halo_gas != continuous.hot_halo_gas
    assert continuous.cold_halo_gas == 0.0
    np.testing.assert_allclose(continuous.hot_halo_gas - reference.hot_halo_gas, -rates.cooling)
    np.testing.assert_allclose(continuous.cold_gas - reference.cold_gas, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(baryonic_mass(continuous), 0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        direct_cooling_flow_derivative(state, rates.cooling, 5.0).hot_halo_gas,
        -rates.cooling,
    )


def test_state_dependent_croton06_continuous_cooling_converges_with_order():
    initial = initial_shark_state(
        cold_gas=1.0e9,
        hot_halo_gas=1.0e11,
        cold_gas_metals=1.0e7,
        hot_halo_gas_metals=1.0e9,
        hot_halo_angular_momentum=5.0e11,
    )
    parameters = shark_flow_parameters(evolving_yield=False)
    cooling_parameters = lagos23_croton06_cooling_parameters()

    def rate_law(time, state):
        del time
        solution = croton06_unheated_cooling(
            state.hot_halo_gas,
            state.hot_halo_gas,
            0.25,
            220.0,
            1.1,
            -24.0,
            cooling_parameters,
        )
        return SharkFlowRates(
            cooling=solution.cooling_rate,
            star_formation=jnp.asarray(0.0),
            star_formation_angular_momentum=jnp.asarray(0.0),
            stellar_reheating_loading=jnp.asarray(0.0),
            stellar_ejection_loading=jnp.asarray(0.0),
            angular_momentum_reheating_loading=jnp.asarray(0.0),
            angular_momentum_ejection_loading=jnp.asarray(0.0),
            qso_reheating_loading=jnp.asarray(0.0),
            qso_ejection_loading=jnp.asarray(0.0),
            cooling_metallicity=jnp.asarray(0.0),
            cooling_specific_angular_momentum=jnp.asarray(5.0),
        )

    # In these fixed halo conditions the slow-cooling branch is
    # dM_hot/dt = -k M_hot^(3/2), which has an analytic solution.  This avoids
    # measuring fourth-order convergence against floating-point noise in a
    # numerically integrated reference.
    initial_rate = float(rate_law(0.0, initial).cooling)
    coefficient = initial_rate / float(initial.hot_halo_gas) ** 1.5
    exact_hot_mass = 1.0 / (float(initial.hot_halo_gas) ** -0.5 + 0.5 * coefficient * 0.5) ** 2
    for method, expected_order in (
        (FORWARD_EULER, 1.0),
        (HEUN_RK2, 2.0),
        (RK4, 4.0),
    ):
        errors = []
        step_counts = (1, 2, 4, 8) if method == RK4 else (8, 16, 32, 64)
        for steps in step_counts:
            final = integrate_shark_flow(
                initial,
                rate_law,
                parameters,
                duration=0.5,
                num_steps=steps,
                method=method,
                formulation="continuous",
            ).final_state
            errors.append(abs(float(final.hot_halo_gas) - exact_hot_mass))
            np.testing.assert_allclose(baryonic_mass(final), baryonic_mass(initial), atol=4.0e-5)
        orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
        np.testing.assert_allclose(np.median(orders[-2:]), expected_order, atol=0.15)


def test_augmented_hot_mode_bh_flow_conserves_mass_and_metals_in_values_and_derivatives():
    state = initial_shark_continuous_state(
        reservoirs=initial_shark_state(
            hot_halo_gas=100.0,
            hot_halo_gas_metals=2.0,
            hot_halo_angular_momentum=500.0,
        ),
        black_hole_mass=2.0,
        black_hole_metals=0.01,
        black_hole_spin=0.4,
        heating_radius=0.03,
    )
    parameters = shark_flow_parameters()

    def ledgers(rate):
        rates = SharkAugmentedFlowRates(
            reservoirs=zero_flow_rates(),
            hot_halo_black_hole_accretion=rate,
            reincorporation=jnp.asarray(0.0),
        )
        result = shark_augmented_continuous_rhs_from_rates(0.0, state, rates, parameters)
        return jnp.asarray(
            [
                augmented_baryonic_mass(result.derivative),
                augmented_metal_mass(result.derivative),
                result.derivative.reservoirs.hot_halo_angular_momentum
                + result.black_hole_angular_momentum_sink,
            ]
        )

    np.testing.assert_allclose(ledgers(3.0), np.zeros(3), atol=2.0e-15)
    np.testing.assert_allclose(jax.jacfwd(ledgers)(3.0), np.zeros(3), atol=2.0e-15)


def test_augmented_hot_mode_bh_growth_supports_fixed_and_adaptive_integration():
    initial = initial_shark_continuous_state(
        reservoirs=initial_shark_state(
            hot_halo_gas=100.0,
            hot_halo_gas_metals=2.0,
            hot_halo_angular_momentum=500.0,
        ),
        black_hole_mass=2.0,
        black_hole_metals=0.01,
        black_hole_spin=0.4,
        heating_radius=0.03,
    )
    parameters = shark_flow_parameters()

    def rate_law(time, state):
        del time
        return SharkAugmentedFlowRates(
            reservoirs=zero_flow_rates(),
            hot_halo_black_hole_accretion=0.1 * state.reservoirs.hot_halo_gas,
            reincorporation=jnp.asarray(0.0),
        )

    fixed = integrate_shark_augmented_flow(
        initial,
        rate_law,
        parameters,
        duration=1.0,
        num_steps=128,
        method=RK4,
    ).final_state
    adaptive = integrate_shark_augmented_flow_adaptive(
        initial,
        rate_law,
        parameters,
        duration=1.0,
        relative_tolerance=1.0e-10,
        absolute_tolerance=1.0e-12,
        initial_step=0.1,
        maximum_step=0.25,
        require_nonnegative=True,
    )
    exact_hot = 100.0 * np.exp(-0.1)
    np.testing.assert_allclose(fixed.reservoirs.hot_halo_gas, exact_hot, rtol=2.0e-14)
    np.testing.assert_allclose(
        adaptive.final_state.reservoirs.hot_halo_gas, exact_hot, rtol=2.0e-11
    )
    np.testing.assert_allclose(
        augmented_baryonic_mass(fixed), augmented_baryonic_mass(initial), atol=2.0e-13
    )
    np.testing.assert_allclose(
        augmented_metal_mass(fixed), augmented_metal_mass(initial), atol=2.0e-15
    )
    np.testing.assert_allclose(fixed.heating_radius, initial.heating_radius)


def test_rhs_supports_jit_vmap_grad_jacfwd_and_jacrev():
    state, rates, parameters = _reference_case()

    def final_stellar_rate(yield_value, one_state):
        varied = parameters._replace(yield_mass_fraction=yield_value)
        result = shark_rhs_from_rates(0.0, one_state, rates, varied)
        return result.derivative.cold_gas_metals

    compiled = jax.jit(shark_rhs_from_rates)(0.0, state, rates, parameters)
    np.testing.assert_allclose(
        _state_array(compiled.derivative),
        _state_array(shark_rhs_from_rates(0.0, state, rates, parameters).derivative),
    )

    batch = stack_shark_states((state, state._replace(cold_gas=20.0)))
    vmapped = jax.vmap(lambda one: shark_rhs_from_rates(0.0, one, rates, parameters))(batch)
    assert vmapped.derivative.cold_gas.shape == (2,)
    np.testing.assert_allclose(jax.grad(final_stellar_rate)(0.03, state), rates.star_formation)

    flat_state = _state_array(state)

    def flat_rhs(values):
        one_state = type(state)(*values)
        return _state_array(shark_rhs_from_rates(0.0, one_state, rates, parameters).derivative)

    forward = jax.jacfwd(flat_rhs)(flat_state)
    reverse = jax.jacrev(flat_rhs)(flat_state)
    np.testing.assert_allclose(forward, reverse, atol=2.0e-15)


def test_fixed_and_adaptive_integrators_converge_for_coupled_shark_flow():
    initial = initial_shark_state(
        cold_gas=8.0,
        cold_halo_gas=20.0,
        hot_halo_gas=5.0,
        ejected_gas=1.0,
        cold_gas_metals=0.08,
        cold_halo_gas_metals=0.2,
        hot_halo_gas_metals=0.05,
        ejected_gas_metals=0.01,
        cold_gas_angular_momentum=24.0,
        cold_halo_angular_momentum=80.0,
    )
    parameters = shark_flow_parameters(evolving_yield=False)

    def rate_law(time, state):
        del time
        return SharkFlowRates(
            cooling=0.7 * state.cold_halo_gas,
            star_formation=0.35 * state.cold_gas,
            star_formation_angular_momentum=0.35 * state.cold_gas_angular_momentum,
            stellar_reheating_loading=jnp.asarray(1.4),
            stellar_ejection_loading=jnp.asarray(0.4),
            angular_momentum_reheating_loading=jnp.asarray(1.4),
            angular_momentum_ejection_loading=jnp.asarray(0.4),
            qso_reheating_loading=jnp.asarray(0.0),
            qso_ejection_loading=jnp.asarray(0.0),
            cooling_metallicity=jnp.where(
                state.cold_halo_gas > 0.0,
                state.cold_halo_gas_metals / state.cold_halo_gas,
                0.0,
            ),
            cooling_specific_angular_momentum=jnp.where(
                state.cold_halo_gas > 0.0,
                state.cold_halo_angular_momentum / state.cold_halo_gas,
                0.0,
            ),
        )

    reference = integrate_shark_flow(
        initial,
        rate_law,
        parameters,
        duration=1.0,
        num_steps=4096,
        method=RK4,
    ).final_state
    methods = (FORWARD_EULER, HEUN_RK2, RK4)
    expected_orders = (1.0, 2.0, 4.0)
    for method, expected_order in zip(methods, expected_orders):
        errors = []
        for steps in (8, 16, 32, 64):
            final = integrate_shark_flow(
                initial,
                rate_law,
                parameters,
                duration=1.0,
                num_steps=steps,
                method=method,
            ).final_state
            errors.append(abs(float(final.stellar_mass - reference.stellar_mass)))
            np.testing.assert_allclose(baryonic_mass(final), baryonic_mass(initial), atol=2.0e-12)
        orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
        np.testing.assert_allclose(np.median(orders[-2:]), expected_order, atol=0.12)

    adaptive = integrate_shark_flow_adaptive(
        initial,
        rate_law,
        parameters,
        duration=1.0,
        relative_tolerance=1.0e-9,
        absolute_tolerance=1.0e-11,
        initial_step=0.1,
        maximum_step=0.25,
        require_nonnegative=True,
    )
    assert int(adaptive.status) == ADAPTIVE_SUCCESS
    np.testing.assert_allclose(
        _state_array(adaptive.final_state), _state_array(reference), rtol=3.0e-9, atol=3.0e-10
    )
