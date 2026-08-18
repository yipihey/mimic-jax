"""Numerical diagnostics kept distinct from faithful SAGE16 physics kernels."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

Array = Any


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
