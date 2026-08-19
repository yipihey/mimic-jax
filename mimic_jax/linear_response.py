"""Local linear-response tools for differentiable reservoir dynamics.

The functions in this module do not assume that the underlying model is
globally linear or time invariant.  They construct and interrogate the
frozen-coefficient linearization at one explicitly chosen state and forcing.
"""

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jax.flatten_util import ravel_pytree

Array = Any


class LocalStateSpace(NamedTuple):
    """Frozen local state-space matrices around one nonlinear operating point."""

    state_jacobian: Array
    input_jacobian: Array
    output_jacobian: Array
    direct_input_jacobian: Array


def linearize_state_space(
    rhs: Callable,
    output: Callable,
    state,
    control,
) -> LocalStateSpace:
    """Linearize ``dx/dt = rhs(x, u)`` and ``y = output(x, u)``.

    Arbitrary JAX PyTrees are accepted for ``state``, ``control``, and the
    observable returned by ``output``.  The resulting dense matrices use the
    leaf order selected by :func:`jax.flatten_util.ravel_pytree`.
    """

    flat_state, unravel_state = ravel_pytree(state)
    flat_control, unravel_control = ravel_pytree(control)
    flat_output, _ = ravel_pytree(output(state, control))

    def flat_rhs(current_state, current_control):
        value = rhs(unravel_state(current_state), unravel_control(current_control))
        return ravel_pytree(value)[0]

    def flat_observable(current_state, current_control):
        value = output(unravel_state(current_state), unravel_control(current_control))
        return ravel_pytree(value)[0]

    state_jacobian, input_jacobian = jax.jacfwd(flat_rhs, argnums=(0, 1))(flat_state, flat_control)
    output_jacobian, direct_input_jacobian = jax.jacfwd(flat_observable, argnums=(0, 1))(
        flat_state, flat_control
    )
    if flat_output.size == 0:
        raise ValueError("The local response observable cannot be empty")
    return LocalStateSpace(
        state_jacobian=state_jacobian,
        input_jacobian=input_jacobian,
        output_jacobian=output_jacobian,
        direct_input_jacobian=direct_input_jacobian,
    )


def transfer_matrix(model: LocalStateSpace, laplace_frequency) -> Array:
    """Return ``C (s I - A)^-1 B + D`` for one complex frequency ``s``."""

    frequency = jnp.asarray(laplace_frequency)
    dtype = jnp.result_type(model.state_jacobian, frequency)
    state_jacobian = jnp.asarray(model.state_jacobian, dtype=dtype)
    input_jacobian = jnp.asarray(model.input_jacobian, dtype=dtype)
    output_jacobian = jnp.asarray(model.output_jacobian, dtype=dtype)
    direct = jnp.asarray(model.direct_input_jacobian, dtype=dtype)
    identity = jnp.eye(state_jacobian.shape[0], dtype=dtype)
    response = jnp.linalg.solve(frequency * identity - state_jacobian, input_jacobian)
    return output_jacobian @ response + direct


def frequency_response(model: LocalStateSpace, angular_frequencies) -> Array:
    """Evaluate the transfer matrix along ``s = i omega``."""

    frequencies = jnp.asarray(angular_frequencies)
    return jax.vmap(lambda omega: transfer_matrix(model, 1j * omega))(frequencies)


def impulse_response(model: LocalStateSpace, times) -> Array:
    """Return the regular time-domain response ``C exp(A t) B``.

    A nonzero direct term represents an impulse at exactly ``t=0`` and is not
    included in the returned regular function.
    """

    times = jnp.asarray(times)

    def one_time(time):
        transition = jax.scipy.linalg.expm(model.state_jacobian * time)
        return model.output_jacobian @ transition @ model.input_jacobian

    return jax.vmap(one_time)(times)


def step_response(model: LocalStateSpace, times) -> Array:
    """Return the response to a unit step without requiring ``A`` to be invertible."""

    times = jnp.asarray(times)
    state_count = model.state_jacobian.shape[0]
    input_count = model.input_jacobian.shape[1]
    zero_input = jnp.zeros((input_count, state_count + input_count))
    augmented = jnp.concatenate(
        (
            jnp.concatenate((model.state_jacobian, model.input_jacobian), axis=1),
            zero_input,
        ),
        axis=0,
    )

    def one_time(time):
        transition = jax.scipy.linalg.expm(augmented * time)
        state_response = transition[:state_count, state_count:]
        return model.output_jacobian @ state_response + model.direct_input_jacobian

    return jax.vmap(one_time)(times)


def local_poles(model: LocalStateSpace) -> Array:
    """Return the eigenvalues of the frozen state Jacobian."""

    return jnp.linalg.eigvals(model.state_jacobian)
