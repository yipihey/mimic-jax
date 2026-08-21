"""Local linear-response tools for differentiable reservoir dynamics.

The functions in this module do not assume that the underlying model is
globally linear or time invariant.  They construct and interrogate the
frozen-coefficient linearization at one explicitly chosen state and forcing.
"""

from dataclasses import dataclass, replace
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class ResponseCoordinate:
    """One physically named coordinate in a local response calculation."""

    name: str
    label: str
    unit: str
    description: str = ""


@dataclass(frozen=True)
class LinearizationPoint:
    """Physical context at which nonlinear SAM dynamics were frozen."""

    model: str
    formulation: str
    time: float
    time_unit: str
    time_unit_in_gyr: float = 1.0
    redshift: Optional[float] = None
    halo_mass: Optional[float] = None
    halo_mass_unit: str = "unspecified"
    qualification: str = "local frozen-coefficient response"


@dataclass(frozen=True)
class AnnotatedStateSpace:
    """Local matrices plus the metadata needed for a scientific interpretation."""

    matrices: LocalStateSpace
    point: LinearizationPoint
    state_coordinates: Tuple[ResponseCoordinate, ...]
    input_coordinates: Tuple[ResponseCoordinate, ...]
    output_coordinates: Tuple[ResponseCoordinate, ...]
    derivative_method: str = "jax.jacfwd"
    state_scales: Optional[Array] = None

    @property
    def state_jacobian(self):
        """Return ``A = df/dx`` for compatibility with matrix-level tools."""

        return self.matrices.state_jacobian

    @property
    def input_jacobian(self):
        """Return ``B = df/du``."""

        return self.matrices.input_jacobian

    @property
    def output_jacobian(self):
        """Return ``C = dy/dx``."""

        return self.matrices.output_jacobian

    @property
    def direct_input_jacobian(self):
        """Return the direct input term ``D = dy/du``."""

        return self.matrices.direct_input_jacobian


@dataclass(frozen=True)
class CharacteristicModes:
    """Poles, response times, periods, and reservoir composition of local modes."""

    poles: Array
    response_times: Array
    oscillation_periods: Array
    response_times_gyr: Array
    oscillation_periods_gyr: Array
    right_eigenvectors: Array
    stable: Array
    neutral: Array
    unstable: Array
    time_unit: str
    time_unit_in_gyr: float
    state_coordinates: Tuple[ResponseCoordinate, ...]


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


def _coordinates(
    coordinates: Sequence[ResponseCoordinate], expected_size: int, coordinate_kind: str
) -> Tuple[ResponseCoordinate, ...]:
    resolved = tuple(coordinates)
    if len(resolved) != expected_size:
        raise ValueError(
            f"Expected {expected_size} {coordinate_kind} coordinates, received {len(resolved)}"
        )
    names = tuple(coordinate.name for coordinate in resolved)
    if len(set(names)) != len(names):
        raise ValueError(f"{coordinate_kind.capitalize()} coordinate names must be unique")
    return resolved


def annotate_state_space(
    matrices: LocalStateSpace,
    *,
    point: LinearizationPoint,
    state_coordinates: Sequence[ResponseCoordinate],
    input_coordinates: Sequence[ResponseCoordinate],
    output_coordinates: Sequence[ResponseCoordinate],
) -> AnnotatedStateSpace:
    """Attach model and physical-coordinate metadata to local response matrices."""

    return AnnotatedStateSpace(
        matrices=matrices,
        point=point,
        state_coordinates=_coordinates(
            state_coordinates, matrices.state_jacobian.shape[0], "state"
        ),
        input_coordinates=_coordinates(
            input_coordinates, matrices.input_jacobian.shape[1], "input"
        ),
        output_coordinates=_coordinates(
            output_coordinates, matrices.output_jacobian.shape[0], "output"
        ),
    )


def state_space_in_gyr(model: AnnotatedStateSpace) -> AnnotatedStateSpace:
    """Express a physically annotated local model in Gyr-based time units.

    If one native time unit equals ``q`` Gyr, ``dx/dt_Gyr = (1/q) dx/dt_native``;
    therefore both ``A`` and ``B`` are divided by ``q`` while ``C`` and ``D``
    are unchanged. This conversion is required before comparing response times
    or frequency responses from SAMs with different internal time units.
    """

    scale = model.point.time_unit_in_gyr
    if not scale > 0.0:
        raise ValueError("The native time unit must have a positive conversion to Gyr")
    matrices = LocalStateSpace(
        state_jacobian=model.state_jacobian / scale,
        input_jacobian=model.input_jacobian / scale,
        output_jacobian=model.output_jacobian,
        direct_input_jacobian=model.direct_input_jacobian,
    )
    return replace(
        model,
        matrices=matrices,
        point=replace(
            model.point,
            time=model.point.time * scale,
            time_unit="Gyr",
            time_unit_in_gyr=1.0,
        ),
    )


def scale_state_space(model: AnnotatedStateSpace, state_scales) -> AnnotatedStateSpace:
    """Apply a diagonal state similarity transform for numerical conditioning.

    With ``x = diag(q) z``, the scaled matrices are
    ``A_z = diag(q)^-1 A diag(q)``, ``B_z = diag(q)^-1 B``, and
    ``C_z = C diag(q)``. Poles and all input-output transfer functions are
    invariant. Explicit positive scales are required; no normalization is
    invented from units or current values.
    """

    scales = jnp.asarray(state_scales, dtype=jnp.float64)
    state_count = model.state_jacobian.shape[0]
    if scales.shape != (state_count,):
        raise ValueError(f"Expected {state_count} state scales, received shape {scales.shape}")
    if not bool(jnp.all(jnp.isfinite(scales) & (scales > 0.0))):
        raise ValueError("State scales must be finite and strictly positive")
    left = 1.0 / scales[:, None]
    right = scales[None, :]
    matrices = LocalStateSpace(
        state_jacobian=left * model.state_jacobian * right,
        input_jacobian=left * model.input_jacobian,
        output_jacobian=model.output_jacobian * right,
        direct_input_jacobian=model.direct_input_jacobian,
    )
    return replace(model, matrices=matrices, state_scales=scales)


def linearize_annotated(
    rhs: Callable,
    output: Callable,
    state,
    control,
    *,
    point: LinearizationPoint,
    state_coordinates: Sequence[ResponseCoordinate],
    input_coordinates: Sequence[ResponseCoordinate],
    output_coordinates: Sequence[ResponseCoordinate],
) -> AnnotatedStateSpace:
    """Linearize nonlinear dynamics and attach an explicit physical context."""

    return annotate_state_space(
        linearize_state_space(rhs, output, state, control),
        point=point,
        state_coordinates=state_coordinates,
        input_coordinates=input_coordinates,
        output_coordinates=output_coordinates,
    )


def _matrices(model) -> LocalStateSpace:
    return model.matrices if isinstance(model, AnnotatedStateSpace) else model


def transfer_matrix(model: LocalStateSpace, laplace_frequency) -> Array:
    """Return ``C (s I - A)^-1 B + D`` for one complex frequency ``s``."""

    model = _matrices(model)
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

    model = _matrices(model)
    times = jnp.asarray(times)

    def one_time(time):
        transition = jax.scipy.linalg.expm(model.state_jacobian * time)
        return model.output_jacobian @ transition @ model.input_jacobian

    return jax.vmap(one_time)(times)


def step_response(model: LocalStateSpace, times) -> Array:
    """Return the response to a unit step without requiring ``A`` to be invertible."""

    model = _matrices(model)
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

    return jnp.linalg.eigvals(_matrices(model).state_jacobian)


def characteristic_modes(model, *, neutral_tolerance: float = 1.0e-12) -> CharacteristicModes:
    """Classify local modes and express damping and oscillation in physical time.

    Stable modes have a finite e-folding response time ``-1/Re(s)``. Neutral
    and unstable modes receive an infinite damping time rather than a
    misleading negative value. A real mode has an infinite oscillation period.
    """

    matrices = _matrices(model)
    poles, eigenvectors = jnp.linalg.eig(matrices.state_jacobian)
    real = jnp.real(poles)
    imaginary = jnp.imag(poles)
    tolerance = jnp.asarray(neutral_tolerance, dtype=real.dtype)
    stable = real < -tolerance
    unstable = real > tolerance
    neutral = ~(stable | unstable)
    response_times = jnp.where(stable, -1.0 / real, jnp.inf)
    oscillatory = jnp.abs(imaginary) > tolerance
    oscillation_periods = jnp.where(oscillatory, 2.0 * jnp.pi / jnp.abs(imaginary), jnp.inf)
    if isinstance(model, AnnotatedStateSpace):
        time_unit = model.point.time_unit
        time_unit_in_gyr = model.point.time_unit_in_gyr
        state_coordinates = model.state_coordinates
    else:
        time_unit = "unspecified"
        time_unit_in_gyr = 1.0
        state_coordinates = tuple(
            ResponseCoordinate(f"state_{index}", f"state {index}", "unspecified")
            for index in range(matrices.state_jacobian.shape[0])
        )
    return CharacteristicModes(
        poles=poles,
        response_times=response_times,
        oscillation_periods=oscillation_periods,
        response_times_gyr=response_times * time_unit_in_gyr,
        oscillation_periods_gyr=oscillation_periods * time_unit_in_gyr,
        right_eigenvectors=eigenvectors,
        stable=stable,
        neutral=neutral,
        unstable=unstable,
        time_unit=time_unit,
        time_unit_in_gyr=time_unit_in_gyr,
        state_coordinates=state_coordinates,
    )
