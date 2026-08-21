"""Physically interpretable fractional-response tools built on JAX derivatives."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

Array = Any

LOG_ELASTICITY = "logarithmic_elasticity"
REFERENCE_SCALE = "reference_scale_sensitivity"
PROCESS_LOG_RESPONSE = "finite_epoch_log_process_response"


class InvalidNormalizationError(ValueError):
    """Raised when a requested logarithmic response has no positive scale."""


@dataclass(frozen=True)
class ParameterResponseMatrix:
    """Observable-by-parameter fractional responses and their scientific metadata."""

    values: Array
    raw_derivatives: Array
    valid: Array
    observable_values: Array
    parameter_values: Array
    observable_scales: Optional[Array]
    parameter_scales: Optional[Array]
    observable_names: Tuple[str, ...]
    parameter_names: Tuple[str, ...]
    observable_units: Tuple[str, ...]
    parameter_units: Tuple[str, ...]
    normalization: str
    derivative_method: str
    invalid_policy: str
    model: str = "unspecified"
    formulation: str = "unspecified"
    qualification: str = ""

    @property
    def sign(self):
        """Sign of each fractional response, preserving physical direction."""

        return jnp.sign(self.values)

    def save(self, path) -> None:
        """Save arrays and metadata in a portable compressed NumPy archive."""

        _save_archive(
            path,
            kind="parameter_response_matrix",
            values=self.values,
            raw_derivatives=self.raw_derivatives,
            valid=self.valid,
            observable_values=self.observable_values,
            input_values=self.parameter_values,
            observable_scales=(
                np.asarray([]) if self.observable_scales is None else self.observable_scales
            ),
            input_scales=(
                np.asarray([]) if self.parameter_scales is None else self.parameter_scales
            ),
            sign=self.sign,
            observable_names=self.observable_names,
            input_names=self.parameter_names,
            observable_units=self.observable_units,
            input_units=self.parameter_units,
            normalization=self.normalization,
            derivative_method=self.derivative_method,
            invalid_policy=self.invalid_policy,
            model=self.model,
            formulation=self.formulation,
            qualification=self.qualification,
        )


@dataclass(frozen=True)
class ParameterResponseValidation:
    """Automatic and symmetric finite-difference parameter responses."""

    automatic: ParameterResponseMatrix
    relative_steps: Array
    finite_difference: Array
    absolute_error: Array


@dataclass(frozen=True)
class HistoricalProcessResponse:
    """Observable-by-process-by-epoch fractional response tensor."""

    values: Array
    raw_derivatives: Array
    valid: Array
    observable_values: Array
    observable_scales: Optional[Array]
    observable_names: Tuple[str, ...]
    process_names: Tuple[str, ...]
    observable_units: Tuple[str, ...]
    ln_scale_factor_edges: Array
    redshift_edges: Array
    process_reference_values: Optional[Array]
    normalization: str
    derivative_method: str
    invalid_policy: str
    model: str = "unspecified"
    formulation: str = "unspecified"
    qualification: str = ""

    @property
    def sign(self):
        """Signed physical direction of every process response."""

        return jnp.sign(self.values)

    def save(self, path) -> None:
        """Save response arrays, epoch coordinates, and metadata."""

        extra = {}
        if self.process_reference_values is not None:
            extra["process_reference_values"] = np.asarray(self.process_reference_values)
        _save_archive(
            path,
            kind="historical_process_response",
            values=self.values,
            raw_derivatives=self.raw_derivatives,
            valid=self.valid,
            observable_values=self.observable_values,
            observable_scales=(
                np.asarray([]) if self.observable_scales is None else self.observable_scales
            ),
            input_values=np.asarray([], dtype=np.float64),
            sign=self.sign,
            observable_names=self.observable_names,
            input_names=self.process_names,
            observable_units=self.observable_units,
            input_units=tuple("fractional perturbation" for _ in self.process_names),
            normalization=self.normalization,
            derivative_method=self.derivative_method,
            invalid_policy=self.invalid_policy,
            model=self.model,
            formulation=self.formulation,
            qualification=self.qualification,
            ln_scale_factor_edges=np.asarray(self.ln_scale_factor_edges),
            redshift_edges=np.asarray(self.redshift_edges),
            **extra,
        )


@dataclass(frozen=True)
class HistoricalResponseValidation:
    """Automatic and symmetric finite-difference finite-epoch responses."""

    automatic: HistoricalProcessResponse
    log_rate_steps: Array
    finite_difference: Array
    absolute_error: Array


def _names(names: Optional[Sequence[str]], size: int, prefix: str) -> Tuple[str, ...]:
    resolved = tuple(names) if names is not None else tuple(f"{prefix}_{i}" for i in range(size))
    if len(resolved) != size:
        raise ValueError(f"Expected {size} {prefix} names, received {len(resolved)}")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{prefix.capitalize()} names must be unique")
    return resolved


def _units(units: Optional[Sequence[str]], size: int) -> Tuple[str, ...]:
    resolved = tuple(units) if units is not None else tuple("unspecified" for _ in range(size))
    if len(resolved) != size:
        raise ValueError(f"Expected {size} unit labels, received {len(resolved)}")
    return resolved


def _validate_invalid_policy(invalid: str) -> None:
    if invalid not in ("raise", "mask"):
        raise ValueError("invalid must be either 'raise' or 'mask'")


def _normalise_parameter_derivatives(
    raw_derivatives,
    observables,
    parameters,
    normalization,
    observable_scales,
    parameter_scales,
    invalid,
):
    _validate_invalid_policy(invalid)
    if normalization == LOG_ELASTICITY:
        if observable_scales is not None or parameter_scales is not None:
            raise ValueError("Reference scales cannot be supplied for logarithmic elasticity")
        valid = (observables[:, None] > 0.0) & (parameters[None, :] > 0.0)
        values = jnp.where(
            valid,
            raw_derivatives * parameters[None, :] / observables[:, None],
            jnp.nan,
        )
    elif normalization == REFERENCE_SCALE:
        if observable_scales is None or parameter_scales is None:
            raise ValueError(
                "reference_scale_sensitivity requires explicit observable_scales and "
                "parameter_scales"
            )
        observable_scales = jnp.asarray(observable_scales, dtype=jnp.float64)
        parameter_scales = jnp.asarray(parameter_scales, dtype=jnp.float64)
        if observable_scales.shape != observables.shape:
            raise ValueError("observable_scales must match the observable vector")
        if parameter_scales.shape != parameters.shape:
            raise ValueError("parameter_scales must match the selected parameter vector")
        valid = (
            jnp.isfinite(observable_scales[:, None])
            & jnp.isfinite(parameter_scales[None, :])
            & (observable_scales[:, None] > 0.0)
            & (parameter_scales[None, :] > 0.0)
        )
        values = jnp.where(
            valid,
            raw_derivatives * parameter_scales[None, :] / observable_scales[:, None],
            jnp.nan,
        )
    else:
        raise ValueError(f"Unknown response normalization: {normalization}")

    if invalid == "raise" and not bool(np.all(np.asarray(valid))):
        raise InvalidNormalizationError(
            "The requested fractional response has a zero, negative, or invalid scale. "
            "Use invalid='mask' or provide explicit positive reference scales."
        )
    return values, valid


def parameter_response_matrix(
    observable_fn: Callable[[Any], Array],
    parameters,
    *,
    parameter_names: Sequence[str],
    observable_names: Optional[Sequence[str]] = None,
    observable_units: Optional[Sequence[str]] = None,
    parameter_units: Optional[Sequence[str]] = None,
    normalization: str = LOG_ELASTICITY,
    observable_scales=None,
    parameter_scales=None,
    invalid: str = "raise",
    parameter_getter: Optional[Callable[[Any, str], Any]] = None,
    parameter_replacer: Optional[Callable[[Any, str, Any], Any]] = None,
) -> ParameterResponseMatrix:
    """Calculate ``d ln(observable) / d ln(parameter)`` for selected fields.

    ``observable_fn`` receives the same immutable parameter PyTree supplied in
    ``parameters`` and must return a scalar or one-dimensional array.
    """

    parameter_names = tuple(parameter_names)
    if not parameter_names:
        raise ValueError("At least one parameter name is required")
    if (parameter_getter is None) != (parameter_replacer is None):
        raise ValueError("parameter_getter and parameter_replacer must be supplied together")
    if parameter_getter is None:
        unknown = set(parameter_names) - set(parameters._fields)
        if unknown:
            raise ValueError(f"Unknown parameter fields: {sorted(unknown)}")
        parameter_getter = lambda record, name: getattr(record, name)
        parameter_replacer = lambda record, name, value: record._replace(**{name: value})
    try:
        selected_parameters = [
            jnp.asarray(parameter_getter(parameters, name)) for name in parameter_names
        ]
    except (AttributeError, KeyError) as error:
        raise ValueError(f"Unknown parameter path: {error}") from error
    if any(not jnp.issubdtype(value.dtype, jnp.inexact) for value in selected_parameters):
        raise TypeError("Selected parameters must have a differentiable floating-point dtype")
    parameter_values = jnp.stack(selected_parameters)

    def evaluate(selected_values):
        current = parameters
        for name, value in zip(parameter_names, selected_values):
            current = parameter_replacer(current, name, value)
        return jnp.atleast_1d(jnp.asarray(observable_fn(current)))

    observables = evaluate(parameter_values)
    if observables.ndim != 1:
        raise ValueError("observable_fn must return a scalar or one-dimensional array")
    raw_derivatives = jax.jacrev(evaluate)(parameter_values)
    return parameter_response_from_derivatives(
        raw_derivatives,
        observables,
        parameter_values,
        parameter_names=parameter_names,
        observable_names=observable_names,
        observable_units=observable_units,
        parameter_units=parameter_units,
        normalization=normalization,
        observable_scales=observable_scales,
        parameter_scales=parameter_scales,
        invalid=invalid,
        derivative_method="jax.jacrev",
    )


def parameter_response_from_derivatives(
    raw_derivatives,
    observable_values,
    parameter_values,
    *,
    parameter_names: Sequence[str],
    observable_names: Optional[Sequence[str]] = None,
    observable_units: Optional[Sequence[str]] = None,
    parameter_units: Optional[Sequence[str]] = None,
    normalization: str = LOG_ELASTICITY,
    observable_scales=None,
    parameter_scales=None,
    invalid: str = "raise",
    derivative_method: str = "supplied derivatives",
    model: str = "unspecified",
    formulation: str = "unspecified",
    qualification: str = "",
) -> ParameterResponseMatrix:
    """Normalize an already computed observable-by-parameter derivative matrix.

    This is the artifact/external-runtime counterpart to
    :func:`parameter_response_matrix`. It applies exactly the same explicit
    logarithmic or reference-scale rules without recomputing model physics.
    """

    observables = jnp.atleast_1d(jnp.asarray(observable_values))
    parameters = jnp.atleast_1d(jnp.asarray(parameter_values))
    derivatives = jnp.asarray(raw_derivatives)
    if observables.ndim != 1 or parameters.ndim != 1:
        raise ValueError("Observable and parameter values must be one-dimensional")
    if derivatives.shape != (observables.size, parameters.size):
        raise ValueError(
            "raw_derivatives must have shape (n_observable, n_parameter); "
            f"received {derivatives.shape}, expected {(observables.size, parameters.size)}"
        )
    values, valid = _normalise_parameter_derivatives(
        derivatives,
        observables,
        parameters,
        normalization,
        observable_scales,
        parameter_scales,
        invalid,
    )
    return ParameterResponseMatrix(
        values=values,
        raw_derivatives=derivatives,
        valid=valid,
        observable_values=observables,
        parameter_values=parameters,
        observable_scales=(
            None if observable_scales is None else jnp.asarray(observable_scales, dtype=jnp.float64)
        ),
        parameter_scales=(
            None if parameter_scales is None else jnp.asarray(parameter_scales, dtype=jnp.float64)
        ),
        observable_names=_names(observable_names, observables.size, "observable"),
        parameter_names=_names(parameter_names, parameters.size, "parameter"),
        observable_units=_units(observable_units, observables.size),
        parameter_units=_units(parameter_units, parameters.size),
        normalization=normalization,
        derivative_method=derivative_method,
        invalid_policy=invalid,
        model=model,
        formulation=formulation,
        qualification=qualification,
    )


def validate_parameter_response(
    response: ParameterResponseMatrix,
    observable_fn: Callable[[Any], Array],
    parameters,
    *,
    relative_steps: Sequence[float] = (1.0e-2, 3.0e-3, 1.0e-3),
    parameter_replacer: Optional[Callable[[Any, str, Any], Any]] = None,
) -> ParameterResponseValidation:
    """Compare an elasticity with symmetric multiplicative finite differences."""

    if response.normalization != LOG_ELASTICITY:
        raise ValueError("Multiplicative finite-difference validation requires log elasticity")
    steps = jnp.asarray(relative_steps, dtype=jnp.float64)
    if bool(np.any((np.asarray(steps) <= 0.0) | (np.asarray(steps) >= 1.0))):
        raise ValueError("relative_steps must lie strictly between zero and one")

    rows = []
    if parameter_replacer is None:
        parameter_replacer = lambda record, name, value: record._replace(**{name: value})
    for step in np.asarray(steps):
        columns = []
        denominator = np.log1p(step) - np.log1p(-step)
        for name, value in zip(response.parameter_names, response.parameter_values):
            upper = parameter_replacer(parameters, name, value * (1.0 + step))
            lower = parameter_replacer(parameters, name, value * (1.0 - step))
            upper_observable = np.atleast_1d(np.asarray(observable_fn(upper), dtype=np.float64))
            lower_observable = np.atleast_1d(np.asarray(observable_fn(lower), dtype=np.float64))
            if np.any(upper_observable <= 0.0) or np.any(lower_observable <= 0.0):
                raise InvalidNormalizationError(
                    "Finite-difference log responses require positive perturbed observables"
                )
            columns.append((np.log(upper_observable) - np.log(lower_observable)) / denominator)
        rows.append(np.stack(columns, axis=-1))
    finite_difference = jnp.asarray(np.stack(rows, axis=0))
    return ParameterResponseValidation(
        automatic=response,
        relative_steps=steps,
        finite_difference=finite_difference,
        absolute_error=jnp.abs(finite_difference - response.values[None, ...]),
    )


def process_response_tensor(
    observable_fn: Callable[[Array], Array],
    *,
    process_names: Sequence[str],
    ln_scale_factor_edges,
    observable_names: Optional[Sequence[str]] = None,
    observable_units: Optional[Sequence[str]] = None,
    observable_scales=None,
    process_reference_values=None,
    invalid: str = "raise",
) -> HistoricalProcessResponse:
    """Differentiate observables against finite-epoch log-rate perturbations.

    ``observable_fn`` receives an ``(n_process, n_epoch)`` matrix of epsilon
    values. A value ``epsilon[i, k]`` multiplies the faithful process transfer
    by ``exp(epsilon[i, k])`` in that finite epoch.
    """

    _validate_invalid_policy(invalid)
    process_names = tuple(process_names)
    edges = jnp.asarray(ln_scale_factor_edges, dtype=jnp.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("ln_scale_factor_edges must contain at least two edges")
    if not bool(np.all(np.diff(np.asarray(edges)) > 0.0)):
        raise ValueError("ln_scale_factor_edges must increase with cosmic time")
    epsilon = jnp.zeros((len(process_names), edges.size - 1), dtype=jnp.float64)

    def evaluate(values):
        return jnp.atleast_1d(jnp.asarray(observable_fn(values)))

    observables = evaluate(epsilon)
    if observables.ndim != 1:
        raise ValueError("observable_fn must return a scalar or one-dimensional array")
    raw_derivatives = jax.jacrev(evaluate)(epsilon)
    if observable_scales is None:
        valid_observable = observables > 0.0
        denominators = observables
        normalization = PROCESS_LOG_RESPONSE
    else:
        denominators = jnp.asarray(observable_scales, dtype=jnp.float64)
        if denominators.shape != observables.shape:
            raise ValueError("observable_scales must match the observable vector")
        valid_observable = jnp.isfinite(denominators) & (denominators > 0.0)
        normalization = REFERENCE_SCALE
    valid = jnp.broadcast_to(valid_observable[:, None, None], raw_derivatives.shape)
    values = jnp.where(valid, raw_derivatives / denominators[:, None, None], jnp.nan)
    if invalid == "raise" and not bool(np.all(np.asarray(valid))):
        raise InvalidNormalizationError(
            "Historical log responses require positive observables. Use invalid='mask' "
            "or supply explicit positive observable_scales."
        )
    if process_reference_values is not None:
        process_reference_values = jnp.asarray(process_reference_values)
        if process_reference_values.shape != epsilon.shape:
            raise ValueError("process_reference_values must have shape (n_process, n_epoch)")
    return HistoricalProcessResponse(
        values=values,
        raw_derivatives=raw_derivatives,
        valid=valid,
        observable_values=observables,
        observable_scales=(
            None if observable_scales is None else jnp.asarray(observable_scales, dtype=jnp.float64)
        ),
        observable_names=_names(observable_names, observables.size, "observable"),
        process_names=process_names,
        observable_units=_units(observable_units, observables.size),
        ln_scale_factor_edges=edges,
        redshift_edges=redshift_from_ln_scale_factor(edges),
        process_reference_values=process_reference_values,
        normalization=normalization,
        derivative_method="jax.jacrev",
        invalid_policy=invalid,
    )


def validate_process_response(
    response: HistoricalProcessResponse,
    observable_fn: Callable[[Array], Array],
    *,
    log_rate_steps: Sequence[float] = (1.0e-2, 3.0e-3, 1.0e-3),
) -> HistoricalResponseValidation:
    """Validate every finite-epoch response with symmetric log-rate perturbations."""

    if response.normalization != PROCESS_LOG_RESPONSE:
        raise ValueError("This validation requires positive-observable log process responses")
    steps = np.asarray(log_rate_steps, dtype=np.float64)
    if np.any(steps <= 0.0):
        raise ValueError("log_rate_steps must be positive")
    shape = (len(response.process_names), response.ln_scale_factor_edges.size - 1)
    rows = []
    for step in steps:
        derivative = np.empty(response.values.shape, dtype=np.float64)
        for process_index in range(shape[0]):
            for epoch_index in range(shape[1]):
                direction = np.zeros(shape, dtype=np.float64)
                direction[process_index, epoch_index] = step
                upper = np.atleast_1d(np.asarray(observable_fn(jnp.asarray(direction))))
                lower = np.atleast_1d(np.asarray(observable_fn(jnp.asarray(-direction))))
                if np.any(upper <= 0.0) or np.any(lower <= 0.0):
                    raise InvalidNormalizationError(
                        "Finite-difference log responses require positive perturbed observables"
                    )
                derivative[:, process_index, epoch_index] = (np.log(upper) - np.log(lower)) / (
                    2.0 * step
                )
        rows.append(derivative)
    finite_difference = jnp.asarray(np.stack(rows, axis=0))
    return HistoricalResponseValidation(
        automatic=response,
        log_rate_steps=jnp.asarray(steps),
        finite_difference=finite_difference,
        absolute_error=jnp.abs(finite_difference - response.values[None, ...]),
    )


def response_similarity(response: ParameterResponseMatrix):
    """Cosine similarity of parameter response fingerprints across observables."""

    pairwise_valid = response.valid[:, :, None] & response.valid[:, None, :]
    left = response.values[:, :, None]
    right = response.values[:, None, :]
    dot_products = jnp.sum(jnp.where(pairwise_valid, left * right, 0.0), axis=0)
    left_norms = jnp.sqrt(jnp.sum(jnp.where(pairwise_valid, left**2, 0.0), axis=0))
    right_norms = jnp.sqrt(jnp.sum(jnp.where(pairwise_valid, right**2, 0.0), axis=0))
    valid = (left_norms > 0.0) & (right_norms > 0.0)
    similarity = jnp.where(
        valid,
        dot_products / (left_norms * right_norms),
        jnp.nan,
    )
    return similarity, valid


def finite_epoch_magnitude_weights(response: HistoricalProcessResponse):
    """Normalize ``abs(R)`` across finite epochs without discarding signed ``R``."""

    magnitude = jnp.abs(response.values)
    total = jnp.sum(magnitude, axis=-1, keepdims=True)
    valid = total > 0.0
    return jnp.where(valid, magnitude / total, jnp.nan), jnp.squeeze(valid, axis=-1)


def ln_scale_factor(redshift):
    """Convert redshift to the monotonically increasing cosmic coordinate ``ln(a)``."""

    redshift = jnp.asarray(redshift, dtype=jnp.float64)
    return -jnp.log1p(redshift)


def redshift_from_ln_scale_factor(value):
    """Convert ``ln(a)`` back to the practitioner-facing redshift coordinate."""

    value = jnp.asarray(value, dtype=jnp.float64)
    return jnp.expm1(-value)


def uniform_ln_scale_factor_edges(redshift_start: float, redshift_end: float, num_epochs: int):
    """Construct finite epoch edges uniform in ``ln(a)`` and ordered in cosmic time."""

    if redshift_start <= redshift_end:
        raise ValueError("redshift_start must exceed redshift_end")
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    return jnp.linspace(
        ln_scale_factor(redshift_start),
        ln_scale_factor(redshift_end),
        num_epochs + 1,
        dtype=jnp.float64,
    )


def _save_archive(path, **contents) -> None:
    destination = Path(path)
    metadata_keys = {
        "kind",
        "observable_names",
        "input_names",
        "observable_units",
        "input_units",
        "normalization",
        "derivative_method",
        "invalid_policy",
        "model",
        "formulation",
        "qualification",
    }
    encoded = {}
    for key, value in contents.items():
        if key in metadata_keys:
            encoded[key] = np.asarray(value, dtype=np.str_)
        else:
            encoded[key] = np.asarray(value)
    np.savez_compressed(destination, **encoded)
