"""Local state-space and Laplace-response tests."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax import (
    frequency_response,
    impulse_response,
    linearize_state_space,
    local_poles,
    step_response,
    transfer_matrix,
)
from mimic_jax.sage16 import (
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    load_cooling_tables,
    ode_state_from_galaxy,
    process_perturbations,
    sage16_ode_rhs_and_rates,
    sage16_units,
)

jax.config.update("jax_enable_x64", True)


def test_one_reservoir_transfer_impulse_step_frequency_and_pole_are_analytic():
    decay_rate = 2.5
    input_gain = 1.7
    output_gain = 0.4

    def rhs(state, control):
        return -decay_rate * state + input_gain * control

    def output(state, control):
        del control
        return output_gain * state

    model = linearize_state_space(rhs, output, jnp.asarray(3.0), jnp.asarray(0.0))
    np.testing.assert_allclose(model.state_jacobian, [[-decay_rate]])
    np.testing.assert_allclose(model.input_jacobian, [[input_gain]])
    np.testing.assert_allclose(model.output_jacobian, [[output_gain]])
    np.testing.assert_allclose(model.direct_input_jacobian, [[0.0]])

    expected_gain = output_gain * input_gain
    frequency = 0.8
    expected_transfer = expected_gain / (1j * frequency + decay_rate)
    np.testing.assert_allclose(transfer_matrix(model, 1j * frequency), [[expected_transfer]])
    np.testing.assert_allclose(frequency_response(model, [frequency]), [[[expected_transfer]]])

    times = np.asarray([0.0, 0.2, 1.0])
    expected_impulse = expected_gain * np.exp(-decay_rate * times)
    expected_step = expected_gain / decay_rate * (1.0 - np.exp(-decay_rate * times))
    np.testing.assert_allclose(impulse_response(model, times)[:, 0, 0], expected_impulse)
    np.testing.assert_allclose(step_response(model, times)[:, 0, 0], expected_step)
    np.testing.assert_allclose(local_poles(model), [-decay_rate])


def test_linearization_accepts_pytree_state_control_and_observable():
    state = {"cold": jnp.asarray(2.0), "hot": jnp.asarray(3.0)}
    control = (jnp.asarray(0.0), jnp.asarray(0.0))

    def rhs(current, inputs):
        cooling, feedback = inputs
        return {
            "cold": -current["cold"] + cooling,
            "hot": current["cold"] - 2.0 * current["hot"] - feedback,
        }

    def output(current, inputs):
        return (current["cold"] + current["hot"], inputs[0])

    model = linearize_state_space(rhs, output, state, control)
    assert model.state_jacobian.shape == (2, 2)
    assert model.input_jacobian.shape == (2, 2)
    assert model.output_jacobian.shape == (2, 2)
    assert model.direct_input_jacobian.shape == (2, 2)
    np.testing.assert_allclose(jnp.sort(jnp.real(local_poles(model))), [-2.0, -1.0])


def test_transfer_and_time_responses_are_jittable_and_differentiable():
    def response(decay_rate):
        model = linearize_state_space(
            lambda state, control: -decay_rate * state + control,
            lambda state, control: state,
            jnp.asarray(1.0),
            jnp.asarray(0.0),
        )
        frequency_gain = jnp.abs(transfer_matrix(model, 1j)[0, 0])
        final_step = step_response(model, jnp.asarray([2.0]))[-1, 0, 0]
        return frequency_gain + final_step

    value = jax.jit(response)(jnp.asarray(2.0))
    derivative = jax.grad(response)(jnp.asarray(2.0))
    assert np.isfinite(value)
    assert np.isfinite(derivative)


def test_sage16_cooling_response_uses_actual_rhs_and_preserves_baryon_tangents():
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
    state = ode_state_from_galaxy(galaxy)
    halo = initial_halo_forcing(Mvir=100.0, Rvir=0.2, Vvir=150.0, dT=5.0e-4)
    parameters = fiducial_parameters()
    units = sage16_units()
    tables = load_cooling_tables()
    control = jnp.zeros((1,), dtype=jnp.float64)

    def evaluate(current, epsilon):
        return sage16_ode_rhs_and_rates(
            0.0,
            current,
            halo,
            galaxy.DiskScaleRadius,
            parameters,
            units,
            tables,
            process_perturbations(cooling=epsilon[0]),
        )

    model = linearize_state_space(
        lambda current, epsilon: evaluate(current, epsilon).derivative,
        lambda current, epsilon: jnp.stack(
            (current.ColdGas, evaluate(current, epsilon).rates.star_formation)
        ),
        state,
        control,
    )
    baryon_state_derivative = np.sum(np.asarray(model.state_jacobian)[:4], axis=0)
    baryon_input_derivative = np.sum(np.asarray(model.input_jacobian)[:4], axis=0)
    np.testing.assert_allclose(baryon_state_derivative, 0.0, atol=3.0e-13)
    np.testing.assert_allclose(baryon_input_derivative, 0.0, atol=3.0e-13)
    response = transfer_matrix(model, 1j)
    assert response.shape == (2, 1)
    assert np.all(np.isfinite(response))
