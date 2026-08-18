"""Adapters from canonical mimic-jax diagnostics to compact report summaries."""

from typing import Any, Mapping, Optional, Tuple

import numpy as np

from mimic_jax.reporting.model import (
    Artifact,
    Diagnostic,
    DiagnosticStatus,
    ParameterValue,
    ScalarMetric,
)
from mimic_jax.sensitivity import LOG_ELASTICITY, PROCESS_LOG_RESPONSE


def _optional_artifact(artifact: Optional[Artifact]) -> Tuple[Artifact, ...]:
    return () if artifact is None else (artifact,)


def _finite_max(values) -> Optional[float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return None if finite.size == 0 else float(np.max(finite))


def _response_word(value: float) -> str:
    if value > 0.0:
        return "increases"
    if value < 0.0:
        return "decreases"
    return "does not change"


def parameter_response_diagnostic(
    response,
    *,
    validation=None,
    validation_tolerance: Optional[float] = None,
    artifact: Optional[Artifact] = None,
    max_interpretations: int = 6,
) -> Diagnostic:
    """Summarize a ``ParameterResponseMatrix`` without duplicating its arrays."""

    values = np.asarray(response.values, dtype=np.float64)
    valid = np.asarray(response.valid, dtype=bool)
    finite_valid = valid & np.isfinite(values)
    notes = []
    if response.normalization == LOG_ELASTICITY:
        indices = np.argwhere(finite_valid)
        ranked = sorted(indices, key=lambda index: abs(values[tuple(index)]), reverse=True)
        for observable_index, parameter_index in ranked[:max_interpretations]:
            value = float(values[observable_index, parameter_index])
            notes.append(
                f"A 1% increase in `{response.parameter_names[parameter_index]}` "
                f"{_response_word(value)} `{response.observable_names[observable_index]}` by "
                f"approximately {abs(value):.3g}%."
            )
    else:
        notes.append(
            "These entries use explicit reference scales rather than logarithmic elasticity; "
            "the scales remain recorded in the referenced response product."
        )

    metrics = [
        ScalarMetric(
            "valid_entries",
            "Valid response entries",
            int(np.count_nonzero(finite_valid)),
            description=f"out of {values.size} observable-parameter entries",
        ),
        ScalarMetric(
            "maximum_absolute_response",
            "Largest absolute fractional response",
            _finite_max(np.abs(values[finite_valid])),
            description="largest magnitude in the response matrix",
        ),
    ]
    status = DiagnosticStatus.NOT_EVALUATED
    summary = (
        "Fractional responses were calculated, but finite-difference validation was not supplied."
    )
    tolerance = ""
    if validation is not None:
        errors = np.asarray(validation.absolute_error, dtype=np.float64)
        step_maximum = np.nanmax(errors.reshape(errors.shape[0], -1), axis=1)
        best_index = int(np.nanargmin(step_maximum))
        best_error = float(step_maximum[best_index])
        worst_error = float(np.nanmax(step_maximum))
        best_step = float(np.asarray(validation.relative_steps)[best_index])
        metrics.extend(
            [
                ScalarMetric(
                    "best_fd_max_absolute_error",
                    "Best finite-difference maximum absolute error",
                    best_error,
                    description=f"symmetric multiplicative step {best_step:.3g}",
                ),
                ScalarMetric(
                    "worst_fd_max_absolute_error",
                    "Worst finite-difference maximum absolute error",
                    worst_error,
                    description="largest error across all tested perturbation sizes",
                ),
            ]
        )
        if validation_tolerance is None:
            summary = (
                "Automatic and symmetric finite-difference responses were compared, but no "
                "acceptance tolerance was supplied."
            )
        else:
            status = (
                DiagnosticStatus.PASSED
                if best_error <= validation_tolerance
                else DiagnosticStatus.FAILED
            )
            tolerance = f"best tested maximum absolute error <= {validation_tolerance:.3g}"
            summary = (
                "At least one tested symmetric finite-difference step agrees with automatic "
                "differentiation within the stated tolerance."
                if status == DiagnosticStatus.PASSED
                else "No tested symmetric finite-difference step meets the stated tolerance."
            )
    return Diagnostic(
        key="parameter_responses",
        title="Fractional parameter responses",
        status=status,
        summary=summary,
        metrics=tuple(metrics),
        artifacts=_optional_artifact(artifact),
        notes=tuple(notes),
        method=response.derivative_method,
        tolerance=tolerance,
    )


def historical_response_diagnostic(
    response,
    *,
    validation=None,
    validation_tolerance: Optional[float] = None,
    artifact: Optional[Artifact] = None,
    max_interpretations: int = 6,
) -> Diagnostic:
    """Summarize finite-epoch process responses in practitioner-facing language."""

    values = np.asarray(response.values, dtype=np.float64)
    valid = np.asarray(response.valid, dtype=bool)
    finite_valid = valid & np.isfinite(values)
    notes = []
    if response.normalization == PROCESS_LOG_RESPONSE:
        indices = np.argwhere(finite_valid)
        ranked = sorted(indices, key=lambda index: abs(values[tuple(index)]), reverse=True)
        redshift_edges = np.asarray(response.redshift_edges, dtype=np.float64)
        for observable_index, process_index, epoch_index in ranked[:max_interpretations]:
            value = float(values[observable_index, process_index, epoch_index])
            high_redshift = redshift_edges[epoch_index]
            low_redshift = redshift_edges[epoch_index + 1]
            notes.append(
                f"A 1% increase in `{response.process_names[process_index]}` during "
                f"z={high_redshift:.3g}–{low_redshift:.3g} {_response_word(value)} today's "
                f"`{response.observable_names[observable_index]}` by approximately "
                f"{abs(value):.3g}%."
            )
    else:
        notes.append(
            "These historical entries use explicit observable reference scales; they are not "
            "logarithmic responses of a positive observable."
        )

    metrics = (
        ScalarMetric(
            "valid_entries",
            "Valid historical response entries",
            int(np.count_nonzero(finite_valid)),
            description=f"out of {values.size} observable-process-epoch entries",
        ),
        ScalarMetric(
            "maximum_absolute_response",
            "Largest absolute finite-epoch response",
            _finite_max(np.abs(values[finite_valid])),
            description="signed responses remain available in the referenced array product",
        ),
    )
    status = DiagnosticStatus.NOT_EVALUATED
    summary = (
        "Historical responses were calculated, but finite-difference validation was not supplied."
    )
    tolerance = ""
    if validation is not None:
        maximum_error = _finite_max(validation.absolute_error)
        if validation_tolerance is None:
            summary = (
                "Automatic and symmetric finite-difference historical responses were compared, "
                "but no acceptance tolerance was supplied."
            )
        else:
            status = (
                DiagnosticStatus.PASSED
                if maximum_error is not None and maximum_error <= validation_tolerance
                else DiagnosticStatus.FAILED
            )
            tolerance = f"maximum absolute error <= {validation_tolerance:.3g}"
            summary = (
                "Automatic and symmetric finite-difference historical responses agree within "
                "the stated tolerance."
                if status == DiagnosticStatus.PASSED
                else "Historical response validation exceeds the stated tolerance."
            )
    return Diagnostic(
        key="historical_process_responses",
        title="When do physical processes matter?",
        status=status,
        summary=summary,
        metrics=metrics,
        artifacts=_optional_artifact(artifact),
        notes=tuple(notes),
        method=response.derivative_method,
        tolerance=tolerance,
    )


def conservation_diagnostic(
    *,
    key: str,
    title: str,
    maximum_absolute_residual: Optional[float],
    tolerance: Optional[float],
    conserved_quantity: str,
    unit: str,
    method: str,
    artifact: Optional[Artifact] = None,
) -> Diagnostic:
    """Create a conservation status only when a residual and tolerance were evaluated."""

    status = DiagnosticStatus.NOT_EVALUATED
    summary = f"{conserved_quantity} conservation was not evaluated."
    tolerance_text = ""
    metrics = ()
    if maximum_absolute_residual is not None:
        metrics = (
            ScalarMetric(
                "maximum_absolute_residual",
                "Maximum absolute residual",
                float(maximum_absolute_residual),
                unit=unit,
                description="ledger delta minus explicit sources plus sinks",
            ),
        )
        if tolerance is None:
            summary = (
                f"A {conserved_quantity} residual was measured, but no acceptance tolerance "
                "was supplied."
            )
        else:
            status = (
                DiagnosticStatus.PASSED
                if abs(maximum_absolute_residual) <= tolerance
                else DiagnosticStatus.FAILED
            )
            tolerance_text = f"maximum absolute residual <= {tolerance:.3g} {unit}".rstrip()
            summary = (
                f"{conserved_quantity.capitalize()} conservation satisfies the stated ledger "
                "tolerance."
                if status == DiagnosticStatus.PASSED
                else f"{conserved_quantity.capitalize()} conservation exceeds the stated ledger "
                "tolerance."
            )
    return Diagnostic(
        key=key,
        title=title,
        status=status,
        summary=summary,
        metrics=metrics,
        artifacts=_optional_artifact(artifact),
        method=method,
        tolerance=tolerance_text,
    )


def equivalence_diagnostic(
    *,
    comparisons: Optional[int],
    mismatches: Optional[int],
    scope: str,
    tolerance: str,
    artifact: Optional[Artifact] = None,
) -> Diagnostic:
    """Summarize an upstream comparison without implying a broader tested scope."""

    if comparisons is None or mismatches is None:
        status = DiagnosticStatus.NOT_EVALUATED
        summary = f"Upstream equivalence was not evaluated for {scope}."
        metrics = ()
    else:
        status = DiagnosticStatus.PASSED if mismatches == 0 else DiagnosticStatus.FAILED
        summary = (
            f"All requested comparisons passed for {scope}."
            if mismatches == 0
            else f"{mismatches} requested comparisons failed for {scope}."
        )
        metrics = (
            ScalarMetric(
                "comparisons",
                "Field comparisons",
                int(comparisons),
                description=scope,
            ),
            ScalarMetric(
                "mismatches",
                "Comparisons outside tolerance",
                int(mismatches),
                description="zero is required for a passing equivalence check",
            ),
        )
    return Diagnostic(
        key="upstream_equivalence",
        title="Upstream equivalence",
        status=status,
        summary=summary,
        metrics=metrics,
        artifacts=_optional_artifact(artifact),
        method="field-by-field catalogue comparison",
        tolerance=tolerance,
    )


def timestep_refinement_diagnostic(
    study,
    *,
    maximum_allowed_relative_difference: Optional[float] = None,
    artifact: Optional[Artifact] = None,
) -> Diagnostic:
    """Summarize a canonical ``TimestepRefinementResult`` at an explicit threshold."""

    values = np.asarray(study.observable_values, dtype=np.float64)
    finest = np.asarray(study.finest_values, dtype=np.float64)
    differences = np.abs(values[0] - finest)
    valid_scale = np.abs(finest) > 0.0
    relative = np.where(valid_scale, differences / np.abs(finest), np.nan)
    maximum_relative = _finite_max(relative)
    maximum_absolute = _finite_max(differences)
    status = DiagnosticStatus.NOT_EVALUATED
    summary = "Timestep refinement was evaluated, but no acceptance threshold was supplied."
    tolerance = ""
    if maximum_allowed_relative_difference is not None:
        status = (
            DiagnosticStatus.PASSED
            if maximum_relative is not None
            and maximum_relative <= maximum_allowed_relative_difference
            else DiagnosticStatus.FAILED
        )
        tolerance = (
            "maximum coarsest-to-finest relative difference <= "
            f"{maximum_allowed_relative_difference:.3g}"
        )
        summary = (
            "The coarsest-to-finest observable differences satisfy the stated threshold."
            if status == DiagnosticStatus.PASSED
            else "At least one coarsest-to-finest observable difference exceeds the threshold."
        )
    return Diagnostic(
        key="timestep_refinement",
        title="Timestep refinement",
        status=status,
        summary=summary,
        metrics=(
            ScalarMetric(
                "maximum_relative_difference",
                "Maximum coarsest-to-finest relative difference",
                maximum_relative,
                description="reported only for observables with nonzero finest values",
            ),
            ScalarMetric(
                "maximum_absolute_difference",
                "Maximum coarsest-to-finest absolute difference",
                maximum_absolute,
                description="in the underlying observable units",
            ),
        ),
        artifacts=_optional_artifact(artifact),
        notes=(
            "The finest requested run is a provisional reference, not an exact solution.",
            f"Halo forcing interpolation: `{study.forcing_interpolation}`.",
        ),
        method=study.method,
        tolerance=tolerance,
    )


def benchmark_diagnostic(
    payload: Mapping[str, Any],
    *,
    status: DiagnosticStatus = DiagnosticStatus.NOT_EVALUATED,
    summary: str = "Runtime was measured; no speedup or performance acceptance claim is made.",
    artifact: Optional[Artifact] = None,
) -> Diagnostic:
    """Project benchmark JSON into report scalars while retaining it as the source artifact."""

    runs = tuple(payload.get("runs", ()))
    metrics = []
    if runs:
        metrics.append(
            ScalarMetric(
                "first_evolution_seconds",
                "First evolution time",
                float(runs[0]["evolution_seconds"]),
                unit="s",
                description="includes cold compilation and execution in this benchmark process",
            )
        )
    if len(runs) > 1:
        metrics.append(
            ScalarMetric(
                "best_warm_evolution_seconds",
                "Best warm evolution time",
                min(float(run["evolution_seconds"]) for run in runs[1:]),
                unit="s",
                description="best repeat after the first invocation",
            )
        )
    for key, label in (("tree_count", "Trees"), ("input_halos", "Input halos")):
        if key in payload:
            metrics.append(ScalarMetric(key, label, int(payload[key])))
    return Diagnostic(
        key="performance",
        title="Performance",
        status=status,
        summary=summary,
        metrics=tuple(metrics),
        artifacts=_optional_artifact(artifact),
        notes=("Compilation cost and warmed execution are reported separately.",),
        method="wall-clock benchmark",
    )


def parameters_from_namedtuple(
    parameters,
    *,
    units: Optional[Mapping[str, str]] = None,
    descriptions: Optional[Mapping[str, str]] = None,
) -> Tuple[ParameterValue, ...]:
    """Project an immutable model-parameter record into report metadata."""

    units = dict(units or {})
    descriptions = dict(descriptions or {})
    records = []
    for name in parameters._fields:
        value = np.asarray(getattr(parameters, name))
        if value.ndim != 0:
            raise ValueError(f"Report parameter {name!r} is not scalar")
        records.append(
            ParameterValue(
                name=name,
                value=value.item(),
                unit=units.get(name, "dimensionless"),
                description=descriptions.get(name, ""),
            )
        )
    return tuple(records)
