"""Numerical diagnostics kept distinct from faithful SAGE16 physics kernels."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

Array = Any

FORWARD_EULER = "forward_euler"
HEUN_RK2 = "heun_rk2"
RK4 = "rk4"
FIXED_STEP_METHODS = (FORWARD_EULER, HEUN_RK2, RK4)

ADAPTIVE_SUCCESS = 0
ADAPTIVE_MAX_STEPS = 1
ADAPTIVE_MINIMUM_STEP = 2
ADAPTIVE_MAX_ATTEMPTS = 3


class FixedStepSolution(NamedTuple):
    """JAX-compatible fixed-step trajectory, including the initial state."""

    times: Array
    states: Any

    @property
    def final_state(self):
        return jax.tree_util.tree_map(lambda value: value[-1], self.states)


class AdaptiveSolution(NamedTuple):
    """Padded JAX trajectory and diagnostics from adaptive Dormand--Prince.

    Only entries through ``accepted_steps`` in ``times`` and ``states`` are
    physical trajectory points. Arrays are padded to static sizes so the
    solver remains compatible with ``jax.jit`` and reverse-mode AD on a fixed
    accept/reject branch.
    """

    times: Array
    states: Any
    accepted_step_sizes: Array
    accepted_error_norms: Array
    accepted_jacobian_norms: Array
    final_state: Any
    accepted_steps: Array
    rejected_steps: Array
    attempted_steps: Array
    rhs_evaluations: Array
    status: Array


class _AdaptiveCarry(NamedTuple):
    time: Array
    state: Any
    step_size: Array
    times: Array
    states: Any
    accepted_step_sizes: Array
    accepted_error_norms: Array
    accepted_jacobian_norms: Array
    accepted_steps: Array
    rejected_steps: Array
    attempted_steps: Array
    minimum_step_failed: Array


@dataclass(frozen=True)
class MethodConvergenceResult:
    """Convergence of several time-integration methods to an independent reference."""

    methods: Tuple[str, ...]
    step_counts: Array
    step_sizes: Array
    observable_values: Array
    reference_values: Array
    absolute_errors: Array
    relative_errors: Array
    observed_orders: Array
    rhs_evaluations: Array
    observable_names: Tuple[str, ...]
    observable_units: Tuple[str, ...]
    reference_method: str
    reference_steps: int
    forcing_interpolation: str

    def save(self, path) -> None:
        """Save convergence arrays and complete method metadata."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            methods=np.asarray(self.methods),
            step_counts=np.asarray(self.step_counts),
            step_sizes=np.asarray(self.step_sizes),
            observable_values=np.asarray(self.observable_values),
            reference_values=np.asarray(self.reference_values),
            absolute_errors=np.asarray(self.absolute_errors),
            relative_errors=np.asarray(self.relative_errors),
            observed_orders=np.asarray(self.observed_orders),
            rhs_evaluations=np.asarray(self.rhs_evaluations),
            observable_names=np.asarray(self.observable_names),
            observable_units=np.asarray(self.observable_units),
            reference_method=self.reference_method,
            reference_steps=self.reference_steps,
            forcing_interpolation=self.forcing_interpolation,
        )


@dataclass(frozen=True)
class TimestepRefinementResult:
    """Observable convergence under successively refined internal substeps."""

    substeps: Array
    observable_values: Array
    finest_values: Array
    absolute_errors: Array
    observed_orders: Array
    observable_names: Tuple[str, ...]
    observable_units: Tuple[str, ...]
    method: str
    forcing_interpolation: str

    def save(self, path) -> None:
        """Save values, errors, and numerical-method metadata."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            substeps=np.asarray(self.substeps),
            observable_values=np.asarray(self.observable_values),
            finest_values=np.asarray(self.finest_values),
            absolute_errors=np.asarray(self.absolute_errors),
            observed_orders=np.asarray(self.observed_orders),
            observable_names=np.asarray(self.observable_names),
            observable_units=np.asarray(self.observable_units),
            method=self.method,
            forcing_interpolation=self.forcing_interpolation,
        )


def _labels(labels: Optional[Sequence[str]], size: int, prefix: str) -> Tuple[str, ...]:
    values = tuple(labels) if labels is not None else tuple(f"{prefix}_{i}" for i in range(size))
    if len(values) != size:
        raise ValueError(f"Expected {size} {prefix} labels, received {len(values)}")
    return values


def fixed_step_rhs_evaluations(method: str) -> int:
    """Return right-hand-side evaluations required by one step."""

    evaluations = {FORWARD_EULER: 1, HEUN_RK2: 2, RK4: 4}
    try:
        return evaluations[method]
    except KeyError as error:
        raise ValueError(
            f"Unknown fixed-step method {method!r}; choose from {FIXED_STEP_METHODS}"
        ) from error


def _tree_add_scaled(state, *weighted_derivatives):
    weights = tuple(weight for weight, _ in weighted_derivatives)
    return jax.tree_util.tree_map(
        lambda value, *derivatives: value
        + sum(weight * derivative for weight, derivative in zip(weights, derivatives)),
        state,
        *(derivative for _, derivative in weighted_derivatives),
    )


def rhs_jacobian(rhs, time, state):
    """Return the dense Jacobian ``d rhs / d state`` in flattened PyTree order."""

    flat_state, unravel = ravel_pytree(state)

    def flat_rhs(flat_value):
        derivative = rhs(time, unravel(flat_value))
        flat_derivative, _ = ravel_pytree(derivative)
        return flat_derivative

    return jax.jacfwd(flat_rhs)(flat_state)


def jacobian_infinity_norm(rhs, time, state):
    """Conservative local rate scale from the dense RHS Jacobian.

    The matrix infinity norm bounds the spectral radius. Its reciprocal is a
    conservative local timescale for explicit stability control without
    assuming that a generally non-normal reservoir Jacobian is diagonalizable.
    """

    jacobian = rhs_jacobian(rhs, time, state)
    return jnp.max(jnp.sum(jnp.abs(jacobian), axis=1), initial=0.0)


def scaled_jacobian_infinity_norm(rhs, time, state, scale):
    """Return ``||D^-1 J D||_inf`` for positive component scales ``D``.

    Reservoirs and metals have different natural magnitudes. Scaling the
    Jacobian by the same component weights used for local-error control avoids
    a timestep recommendation that changes merely because a field's units or
    normalization convention changed.
    """

    jacobian = rhs_jacobian(rhs, time, state)
    flat_scale, _ = ravel_pytree(scale)
    safe_scale = jnp.maximum(flat_scale, jnp.finfo(flat_scale.dtype).tiny)
    scaled = jacobian * safe_scale[jnp.newaxis, :] / safe_scale[:, jnp.newaxis]
    return jnp.max(jnp.sum(jnp.abs(scaled), axis=1), initial=0.0)


def dormand_prince_step(rhs, time, state, step_size):
    """Return the fifth-order Dormand--Prince state and embedded error estimate."""

    first = rhs(time, state)
    second = rhs(
        time + step_size / 5.0,
        _tree_add_scaled(state, (step_size / 5.0, first)),
    )
    third = rhs(
        time + 3.0 * step_size / 10.0,
        _tree_add_scaled(
            state,
            (3.0 * step_size / 40.0, first),
            (9.0 * step_size / 40.0, second),
        ),
    )
    fourth = rhs(
        time + 4.0 * step_size / 5.0,
        _tree_add_scaled(
            state,
            (44.0 * step_size / 45.0, first),
            (-56.0 * step_size / 15.0, second),
            (32.0 * step_size / 9.0, third),
        ),
    )
    fifth = rhs(
        time + 8.0 * step_size / 9.0,
        _tree_add_scaled(
            state,
            (19372.0 * step_size / 6561.0, first),
            (-25360.0 * step_size / 2187.0, second),
            (64448.0 * step_size / 6561.0, third),
            (-212.0 * step_size / 729.0, fourth),
        ),
    )
    sixth = rhs(
        time + step_size,
        _tree_add_scaled(
            state,
            (9017.0 * step_size / 3168.0, first),
            (-355.0 * step_size / 33.0, second),
            (46732.0 * step_size / 5247.0, third),
            (49.0 * step_size / 176.0, fourth),
            (-5103.0 * step_size / 18656.0, fifth),
        ),
    )
    high_order = _tree_add_scaled(
        state,
        (35.0 * step_size / 384.0, first),
        (500.0 * step_size / 1113.0, third),
        (125.0 * step_size / 192.0, fourth),
        (-2187.0 * step_size / 6784.0, fifth),
        (11.0 * step_size / 84.0, sixth),
    )
    seventh = rhs(time + step_size, high_order)
    low_order = _tree_add_scaled(
        state,
        (5179.0 * step_size / 57600.0, first),
        (7571.0 * step_size / 16695.0, third),
        (393.0 * step_size / 640.0, fourth),
        (-92097.0 * step_size / 339200.0, fifth),
        (187.0 * step_size / 2100.0, sixth),
        (step_size / 40.0, seventh),
    )
    error = jax.tree_util.tree_map(lambda high, low: high - low, high_order, low_order)
    return high_order, error


def _absolute_tolerance_tree(absolute_tolerance, state):
    state_structure = jax.tree_util.tree_structure(state)
    tolerance_structure = jax.tree_util.tree_structure(absolute_tolerance)
    if tolerance_structure == state_structure:
        return jax.tree_util.tree_map(jnp.asarray, absolute_tolerance)
    tolerance = jnp.asarray(absolute_tolerance, dtype=jnp.float64)
    if tolerance.ndim != 0:
        raise ValueError("absolute_tolerance must be scalar or match the state PyTree")
    return jax.tree_util.tree_map(lambda value: jnp.full_like(value, tolerance), state)


def _weighted_error_norm(error, before, after, absolute_tolerance, relative_tolerance):
    contributions = jax.tree_util.tree_map(
        lambda difference, old, new, floor: jnp.sum(
            jnp.square(
                difference / (floor + relative_tolerance * jnp.maximum(jnp.abs(old), jnp.abs(new)))
            )
        ),
        error,
        before,
        after,
        absolute_tolerance,
    )
    element_counts = jax.tree_util.tree_map(lambda value: value.size, error)
    total = sum(jax.tree_util.tree_leaves(contributions))
    count = sum(jax.tree_util.tree_leaves(element_counts))
    return jnp.sqrt(total / count)


def _state_is_finite(state):
    return jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(state)])
    )


def _state_is_nonnegative(state):
    return jnp.all(jnp.stack([jnp.all(value >= 0.0) for value in jax.tree_util.tree_leaves(state)]))


def integrate_adaptive(
    rhs,
    initial_state,
    *,
    start_time=0.0,
    duration,
    relative_tolerance: float,
    absolute_tolerance,
    initial_step=None,
    minimum_step=None,
    maximum_step=None,
    max_steps: int = 4096,
    max_attempts: int = 16384,
    safety_factor: float = 0.9,
    minimum_factor: float = 0.2,
    maximum_factor: float = 5.0,
    jacobian_stability_factor=None,
    require_nonnegative: bool = False,
    state_is_valid=None,
) -> AdaptiveSolution:
    """Integrate a PyTree with adaptive Dormand--Prince 5(4).

    The embedded pair controls local error. If ``jacobian_stability_factor`` is
    provided, every attempted step is additionally bounded using the infinity
    norm of the tolerance-scaled Jacobian. Controller decisions and step sizes
    are stop-gradient quantities: derivatives follow the selected numerical
    path, while accept/reject boundaries remain explicitly piecewise
    non-differentiable.
    """

    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive Python integer")
    if not isinstance(max_attempts, int) or max_attempts < max_steps:
        raise ValueError("max_attempts must be a Python integer >= max_steps")
    if relative_tolerance <= 0.0 or not np.isfinite(relative_tolerance):
        raise ValueError("relative_tolerance must be finite and positive")
    if not 0.0 < safety_factor <= 1.0:
        raise ValueError("safety_factor must lie in (0, 1]")
    if not 0.0 < minimum_factor <= 1.0 <= maximum_factor:
        raise ValueError("step-size factors must satisfy 0 < minimum <= 1 <= maximum")
    if jacobian_stability_factor is not None and jacobian_stability_factor <= 0.0:
        raise ValueError("jacobian_stability_factor must be positive when provided")

    initial_state = jax.tree_util.tree_map(jnp.asarray, initial_state)
    start = jnp.asarray(start_time, dtype=jnp.float64)
    duration = jnp.asarray(duration, dtype=jnp.float64)
    end = start + duration
    maximum = duration if maximum_step is None else jnp.asarray(maximum_step, dtype=jnp.float64)
    minimum = (
        jnp.maximum(duration * 1.0e-12, jnp.finfo(jnp.float64).tiny)
        if minimum_step is None
        else jnp.asarray(minimum_step, dtype=jnp.float64)
    )
    initial = (
        jnp.minimum(duration / 16.0, maximum)
        if initial_step is None
        else jnp.asarray(initial_step, dtype=jnp.float64)
    )
    initial = jnp.clip(initial, minimum, maximum)
    relative = jnp.asarray(relative_tolerance, dtype=jnp.float64)
    tolerance = _absolute_tolerance_tree(absolute_tolerance, initial_state)

    times = jnp.full((max_steps + 1,), jnp.nan, dtype=jnp.float64).at[0].set(start)
    states = jax.tree_util.tree_map(
        lambda value: jnp.zeros((max_steps + 1,) + value.shape, dtype=value.dtype).at[0].set(value),
        initial_state,
    )
    accepted_step_sizes = jnp.full((max_steps,), jnp.nan, dtype=jnp.float64)
    accepted_error_norms = jnp.full((max_steps,), jnp.nan, dtype=jnp.float64)
    accepted_jacobian_norms = jnp.full((max_steps,), jnp.nan, dtype=jnp.float64)
    initial_carry = _AdaptiveCarry(
        time=start,
        state=initial_state,
        step_size=initial,
        times=times,
        states=states,
        accepted_step_sizes=accepted_step_sizes,
        accepted_error_norms=accepted_error_norms,
        accepted_jacobian_norms=accepted_jacobian_norms,
        accepted_steps=jnp.asarray(0, dtype=jnp.int32),
        rejected_steps=jnp.asarray(0, dtype=jnp.int32),
        attempted_steps=jnp.asarray(0, dtype=jnp.int32),
        minimum_step_failed=jnp.asarray(False),
    )

    def attempt_step(carry):
        remaining = end - carry.time
        trial_step = jnp.minimum(jnp.minimum(carry.step_size, maximum), remaining)
        if jacobian_stability_factor is None:
            jacobian_norm = jnp.asarray(jnp.nan, dtype=jnp.float64)
        else:
            component_scale = jax.tree_util.tree_map(
                lambda value, floor: floor + relative * jnp.abs(value),
                carry.state,
                tolerance,
            )
            jacobian_norm = scaled_jacobian_infinity_norm(
                rhs,
                carry.time,
                carry.state,
                component_scale,
            )
            stability_step = jnp.where(
                jacobian_norm > 0.0,
                jacobian_stability_factor / jacobian_norm,
                maximum,
            )
            trial_step = jnp.minimum(trial_step, stability_step)
        trial_step = jax.lax.stop_gradient(jnp.minimum(remaining, jnp.maximum(trial_step, minimum)))
        candidate, error = dormand_prince_step(rhs, carry.time, carry.state, trial_step)
        error_norm = _weighted_error_norm(
            error,
            carry.state,
            candidate,
            tolerance,
            relative,
        )
        valid = _state_is_finite(candidate) & jnp.isfinite(error_norm)
        if require_nonnegative:
            valid = valid & _state_is_nonnegative(candidate)
        if state_is_valid is not None:
            valid = valid & state_is_valid(candidate)
        accepted = valid & (error_norm <= 1.0)
        failed_at_minimum = (~accepted) & (trial_step <= minimum)

        safe_error = jnp.where((error_norm > 0.0) & jnp.isfinite(error_norm), error_norm, 1.0)
        factor = safety_factor * safe_error ** (-1.0 / 5.0)
        factor = jnp.where(error_norm == 0.0, maximum_factor, factor)
        factor = jnp.where(valid, factor, minimum_factor)
        factor = jnp.clip(factor, minimum_factor, maximum_factor)
        factor = jnp.where(accepted, factor, jnp.minimum(factor, 1.0))
        next_step = jax.lax.stop_gradient(jnp.clip(trial_step * factor, minimum, maximum))

        def record_accepted(current):
            next_time = current.time + trial_step
            output_index = current.accepted_steps + 1
            next_times = current.times.at[output_index].set(next_time)
            next_states = jax.tree_util.tree_map(
                lambda history, value: history.at[output_index].set(value),
                current.states,
                candidate,
            )
            step_index = current.accepted_steps
            return current._replace(
                time=next_time,
                state=candidate,
                times=next_times,
                states=next_states,
                accepted_step_sizes=current.accepted_step_sizes.at[step_index].set(trial_step),
                accepted_error_norms=current.accepted_error_norms.at[step_index].set(error_norm),
                accepted_jacobian_norms=current.accepted_jacobian_norms.at[step_index].set(
                    jacobian_norm
                ),
                accepted_steps=current.accepted_steps + 1,
            )

        updated = jax.lax.cond(accepted, record_accepted, lambda current: current, carry)
        return updated._replace(
            step_size=next_step,
            rejected_steps=updated.rejected_steps + (~accepted).astype(jnp.int32),
            attempted_steps=updated.attempted_steps + 1,
            minimum_step_failed=updated.minimum_step_failed | failed_at_minimum,
        )

    def scan_attempt(carry, _):
        active = (
            (carry.time < end) & (carry.accepted_steps < max_steps) & (~carry.minimum_step_failed)
        )
        updated = jax.lax.cond(active, attempt_step, lambda current: current, carry)
        return updated, None

    final, _ = jax.lax.scan(scan_attempt, initial_carry, xs=None, length=max_attempts)
    reached_end = final.time >= end
    status = jnp.where(
        reached_end,
        ADAPTIVE_SUCCESS,
        jnp.where(
            final.minimum_step_failed,
            ADAPTIVE_MINIMUM_STEP,
            jnp.where(
                final.accepted_steps >= max_steps,
                ADAPTIVE_MAX_STEPS,
                ADAPTIVE_MAX_ATTEMPTS,
            ),
        ),
    )
    return AdaptiveSolution(
        times=final.times,
        states=final.states,
        accepted_step_sizes=final.accepted_step_sizes,
        accepted_error_norms=final.accepted_error_norms,
        accepted_jacobian_norms=final.accepted_jacobian_norms,
        final_state=final.state,
        accepted_steps=final.accepted_steps,
        rejected_steps=final.rejected_steps,
        attempted_steps=final.attempted_steps,
        rhs_evaluations=7 * final.attempted_steps,
        status=status,
    )


def fixed_step_update(rhs, time, state, step_size, *, method: str):
    """Advance ``dy/dt = rhs(t, y)`` by one explicit fixed step.

    ``method`` is a Python-level numerical choice and should be closed over by
    callers that JIT-compile this function.
    """

    fixed_step_rhs_evaluations(method)
    if method == FORWARD_EULER:
        first = rhs(time, state)
        return _tree_add_scaled(state, (step_size, first))
    if method == HEUN_RK2:
        first = rhs(time, state)
        predictor = _tree_add_scaled(state, (step_size, first))
        second = rhs(time + step_size, predictor)
        return _tree_add_scaled(
            state,
            (0.5 * step_size, first),
            (0.5 * step_size, second),
        )

    first = rhs(time, state)
    second = rhs(time + 0.5 * step_size, _tree_add_scaled(state, (0.5 * step_size, first)))
    third = rhs(time + 0.5 * step_size, _tree_add_scaled(state, (0.5 * step_size, second)))
    fourth = rhs(time + step_size, _tree_add_scaled(state, (step_size, third)))
    return _tree_add_scaled(
        state,
        (step_size / 6.0, first),
        (step_size / 3.0, second),
        (step_size / 3.0, third),
        (step_size / 6.0, fourth),
    )


def integrate_fixed_step(
    rhs,
    initial_state,
    *,
    start_time=0.0,
    duration,
    num_steps: int,
    method: str,
) -> FixedStepSolution:
    """Integrate a PyTree state with Euler, Heun RK2, or classical RK4."""

    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive Python integer")
    fixed_step_rhs_evaluations(method)
    start_time = jnp.asarray(start_time, dtype=jnp.float64)
    duration = jnp.asarray(duration, dtype=jnp.float64)
    step_size = duration / num_steps
    step_starts = start_time + step_size * jnp.arange(num_steps, dtype=jnp.float64)

    def scan_step(state, time):
        next_state = fixed_step_update(rhs, time, state, step_size, method=method)
        return next_state, next_state

    final_state, evolved_states = jax.lax.scan(scan_step, initial_state, step_starts)
    del final_state
    states = jax.tree_util.tree_map(
        lambda initial, evolved: jnp.concatenate((initial[jnp.newaxis, ...], evolved), axis=0),
        initial_state,
        evolved_states,
    )
    times = start_time + step_size * jnp.arange(num_steps + 1, dtype=jnp.float64)
    return FixedStepSolution(times=times, states=states)


def method_convergence_study(
    run_method: Callable[[str, int], Any],
    observable_fn: Callable[[Any], Array],
    reference_result: Any,
    *,
    methods: Sequence[str],
    step_counts: Sequence[int] = (1, 2, 4, 8, 16, 32),
    observable_names: Optional[Sequence[str]] = None,
    observable_units: Optional[Sequence[str]] = None,
    rhs_evaluations_per_step: Optional[Mapping[str, int]] = None,
    reference_method: str,
    reference_steps: int,
    duration: float,
    forcing_interpolation: str = "piecewise_constant",
) -> MethodConvergenceResult:
    """Measure fixed-step convergence to a separately calculated reference solution."""

    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must contain unique method names")
    step_counts = tuple(int(value) for value in step_counts)
    if len(step_counts) < 3 or any(value <= 0 for value in step_counts):
        raise ValueError("step_counts must contain at least three positive levels")
    if any(right <= left for left, right in zip(step_counts, step_counts[1:])):
        raise ValueError("step_counts must be strictly increasing")
    if duration <= 0.0:
        raise ValueError("duration must be positive")

    reference = jnp.atleast_1d(jnp.asarray(observable_fn(reference_result)))
    if reference.ndim != 1:
        raise ValueError("observable_fn must return a scalar or one-dimensional array")
    method_values = []
    for method in methods:
        values = [
            jnp.atleast_1d(jnp.asarray(observable_fn(run_method(method, n)))) for n in step_counts
        ]
        if any(value.shape != reference.shape for value in values):
            raise ValueError("observable_fn returned inconsistent shapes across methods or levels")
        method_values.append(jnp.stack(values))
    values = jnp.stack(method_values)
    absolute_errors = jnp.abs(values - reference[jnp.newaxis, jnp.newaxis, :])
    safe_reference = jnp.where(jnp.abs(reference) > 0.0, jnp.abs(reference), 1.0)
    relative_errors = absolute_errors / safe_reference[jnp.newaxis, jnp.newaxis, :]
    relative_errors = jnp.where(
        jnp.abs(reference)[jnp.newaxis, jnp.newaxis, :] > 0.0,
        relative_errors,
        jnp.nan,
    )
    ratios = jnp.asarray(step_counts[1:], dtype=jnp.float64) / jnp.asarray(
        step_counts[:-1], dtype=jnp.float64
    )
    coarse = absolute_errors[:, :-1, :]
    fine = absolute_errors[:, 1:, :]
    valid = (coarse > 0.0) & (fine > 0.0)
    orders = jnp.where(
        valid,
        jnp.log(coarse / fine) / jnp.log(ratios[jnp.newaxis, :, jnp.newaxis]),
        jnp.nan,
    )
    evaluations = rhs_evaluations_per_step or {
        method: fixed_step_rhs_evaluations(method) for method in methods
    }
    try:
        evaluation_counts = jnp.asarray(
            [[evaluations[method] * count for count in step_counts] for method in methods],
            dtype=jnp.int32,
        )
    except KeyError as error:
        raise ValueError(f"Missing RHS evaluation count for method {error.args[0]!r}") from error
    names = _labels(observable_names, reference.shape[0], "observable")
    units = _labels(observable_units, reference.shape[0], "unspecified")
    return MethodConvergenceResult(
        methods=methods,
        step_counts=jnp.asarray(step_counts, dtype=jnp.int32),
        step_sizes=jnp.asarray(duration, dtype=jnp.float64)
        / jnp.asarray(step_counts, dtype=jnp.float64),
        observable_values=values,
        reference_values=reference,
        absolute_errors=absolute_errors,
        relative_errors=relative_errors,
        observed_orders=orders,
        rhs_evaluations=evaluation_counts,
        observable_names=names,
        observable_units=units,
        reference_method=reference_method,
        reference_steps=int(reference_steps),
        forcing_interpolation=forcing_interpolation,
    )


def timestep_refinement_study(
    run_with_substeps: Callable[[int], Any],
    observable_fn: Callable[[Any], Array],
    *,
    substeps: Sequence[int] = (1, 2, 4, 8),
    observable_names: Optional[Sequence[str]] = None,
    observable_units: Optional[Sequence[str]] = None,
    method: str = "upstream_sequential",
    forcing_interpolation: str = "piecewise_constant",
) -> TimestepRefinementResult:
    """Run a fixed-forcing refinement study and estimate empirical orders.

    The finest requested run is the provisional reference, not an assertion of
    an exact solution. Observed orders use consecutive error ratios and are
    reported as NaN wherever either error is zero or invalid.
    """

    substeps = tuple(int(value) for value in substeps)
    if len(substeps) < 3 or any(value <= 0 for value in substeps):
        raise ValueError("substeps must contain at least three positive levels")
    if any(right <= left for left, right in zip(substeps, substeps[1:])):
        raise ValueError("substeps must be strictly increasing")
    outputs = [jnp.atleast_1d(jnp.asarray(observable_fn(run_with_substeps(n)))) for n in substeps]
    if any(output.ndim != 1 for output in outputs):
        raise ValueError("observable_fn must return a scalar or one-dimensional array")
    if any(output.shape != outputs[0].shape for output in outputs[1:]):
        raise ValueError("observable_fn returned inconsistent shapes across refinement levels")

    values = jnp.stack(outputs)
    finest = values[-1]
    errors = jnp.abs(values - finest)
    coarse_errors = errors[:-2]
    fine_errors = errors[1:-1]
    refinement_ratios = jnp.asarray(substeps[1:-1], dtype=jnp.float64) / jnp.asarray(
        substeps[:-2], dtype=jnp.float64
    )
    valid = (coarse_errors > 0.0) & (fine_errors > 0.0)
    orders = jnp.where(
        valid,
        jnp.log(coarse_errors / fine_errors) / jnp.log(refinement_ratios[:, None]),
        jnp.nan,
    )
    names = _labels(observable_names, values.shape[1], "observable")
    units = _labels(observable_units, values.shape[1], "unspecified")
    return TimestepRefinementResult(
        substeps=jnp.asarray(substeps, dtype=jnp.int32),
        observable_values=values,
        finest_values=finest,
        absolute_errors=errors,
        observed_orders=orders,
        observable_names=names,
        observable_units=units,
        method=method,
        forcing_interpolation=forcing_interpolation,
    )


def conservation_residual(before, after, *, sources=0.0, sinks=0.0):
    """Return ``delta(total) - (sources - sinks)`` for an explicit ledger."""

    return jnp.asarray(after) - jnp.asarray(before) - (jnp.asarray(sources) - jnp.asarray(sinks))


def step_to_timescale_ratio(source_reservoir, transferred_mass):
    """Return ``dt/tau = |transfer|/source`` for a finite process step."""

    source = jnp.asarray(source_reservoir, dtype=jnp.float64)
    transfer = jnp.asarray(transferred_mass, dtype=jnp.float64)
    safe_source = jnp.where(source > 0.0, source, 1.0)
    ratio = jnp.abs(transfer) / safe_source
    return jnp.where(source > 0.0, ratio, jnp.where(transfer == 0.0, 0.0, jnp.inf))
