"""Numerical diagnostics kept distinct from faithful SAGE16 physics kernels."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

Array = Any

FORWARD_EULER = "forward_euler"
HEUN_RK2 = "heun_rk2"
RK4 = "rk4"
FIXED_STEP_METHODS = (FORWARD_EULER, HEUN_RK2, RK4)


class FixedStepSolution(NamedTuple):
    """JAX-compatible fixed-step trajectory, including the initial state."""

    times: Array
    states: Any

    @property
    def final_state(self):
        return jax.tree_util.tree_map(lambda value: value[-1], self.states)


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
