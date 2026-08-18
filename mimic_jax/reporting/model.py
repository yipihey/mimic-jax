"""Immutable, presentation-independent manifests for run and comparison reports."""

import math
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence, Tuple

REPORT_SCHEMA_VERSION = "mimic-jax-report/v1"
_KEY_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class DiagnosticStatus(str, Enum):
    """Outcome vocabulary shared by human- and machine-readable reports."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


def _validate_key(value: str, label: str) -> None:
    if not value or not _KEY_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must start with a lowercase letter or digit and contain only "
            "lowercase letters, digits, '.', '_', or '-'"
        )


def _validate_unique(items: Sequence[Any], label: str) -> None:
    keys = [item.key for item in items]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} keys must be unique")


def _validate_scalar(value: Any, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise TypeError(f"{label} must be a finite JSON scalar or None")


@dataclass(frozen=True)
class ReportLink:
    """A human-facing reference that is not owned as a report artifact."""

    title: str
    target: str

    def __post_init__(self) -> None:
        if not self.title or not self.target:
            raise ValueError("Report links require non-empty titles and targets")


@dataclass(frozen=True)
class Artifact:
    """A relative reference to a figure, array, configuration, or other durable file."""

    key: str
    title: str
    path: str
    media_type: str
    role: str
    description: str = ""
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        _validate_key(self.key, "Artifact key")
        if not self.title or not self.media_type or not self.role:
            raise ValueError("Artifacts require a title, media type, and role")
        candidate = PurePosixPath(self.path)
        if not self.path or candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("Artifact paths must be non-empty paths within the report directory")
        if self.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("Artifact sha256 must contain 64 lowercase hexadecimal characters")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("Artifact size_bytes cannot be negative")


@dataclass(frozen=True)
class ScalarMetric:
    """A compact scalar result with units and an explicit interpretation."""

    key: str
    label: str
    value: Any
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        _validate_key(self.key, "Metric key")
        if not self.label:
            raise ValueError("Metric labels cannot be empty")
        _validate_scalar(self.value, f"Metric {self.key!r}")


@dataclass(frozen=True)
class Diagnostic:
    """One evaluated, warned, failed, or unavailable scientific check."""

    key: str
    title: str
    status: DiagnosticStatus
    summary: str
    metrics: Tuple[ScalarMetric, ...] = ()
    artifacts: Tuple[Artifact, ...] = ()
    notes: Tuple[str, ...] = ()
    method: str = ""
    tolerance: str = ""

    def __post_init__(self) -> None:
        _validate_key(self.key, "Diagnostic key")
        if not self.title or not self.summary:
            raise ValueError("Diagnostics require a title and summary")
        if not isinstance(self.status, DiagnosticStatus):
            raise TypeError("Diagnostic status must be a DiagnosticStatus")
        _validate_unique(self.metrics, f"Diagnostic {self.key!r} metric")
        _validate_unique(self.artifacts, f"Diagnostic {self.key!r} artifact")


@dataclass(frozen=True)
class ReportSection:
    """An ordered, extensible scientific section in a run report."""

    key: str
    title: str
    summary: str
    diagnostics: Tuple[Diagnostic, ...] = ()
    artifacts: Tuple[Artifact, ...] = ()
    notes: Tuple[str, ...] = ()
    links: Tuple[ReportLink, ...] = ()

    def __post_init__(self) -> None:
        _validate_key(self.key, "Section key")
        if not self.title or not self.summary:
            raise ValueError("Report sections require a title and summary")
        _validate_unique(self.diagnostics, f"Section {self.key!r} diagnostic")
        _validate_unique(self.artifacts, f"Section {self.key!r} artifact")


@dataclass(frozen=True)
class RunIdentity:
    """The concise scientific identity shown at the top of a run report."""

    run_id: str
    title: str
    model: str
    dataset: str
    parameter_set: str
    integration_method: str
    summary: str

    def __post_init__(self) -> None:
        _validate_key(self.run_id, "Run ID")
        required = (
            self.title,
            self.model,
            self.dataset,
            self.parameter_set,
            self.integration_method,
            self.summary,
        )
        if any(not value for value in required):
            raise ValueError("Run identity fields cannot be empty")


@dataclass(frozen=True)
class ParameterValue:
    """One named parameter value recorded for reproducibility."""

    name: str
    value: Any
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Parameter names cannot be empty")
        _validate_scalar(self.value, f"Parameter {self.name!r}")


@dataclass(frozen=True)
class ProvenanceFile:
    """A checksummed input or configuration file used by a run."""

    path: str
    role: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.path or not self.role:
            raise ValueError("Provenance files require paths and roles")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("Provenance file sha256 must be lowercase hexadecimal")
        if self.size_bytes < 0:
            raise ValueError("Provenance file size_bytes cannot be negative")


@dataclass(frozen=True)
class Provenance:
    """Reproducibility metadata captured independently from report presentation."""

    generated_at: str
    git_commit: Optional[str]
    git_branch: Optional[str]
    git_dirty: Optional[bool]
    command: Tuple[str, ...] = ()
    files: Tuple[ProvenanceFile, ...] = ()
    software: Mapping[str, str] = None
    hardware: Mapping[str, Any] = None
    random_seeds: Mapping[str, Any] = None
    upstream_run: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        if not self.generated_at:
            raise ValueError("Provenance generated_at cannot be empty")
        object.__setattr__(self, "software", dict(self.software or {}))
        object.__setattr__(self, "hardware", dict(self.hardware or {}))
        object.__setattr__(self, "random_seeds", dict(self.random_seeds or {}))
        object.__setattr__(self, "upstream_run", dict(self.upstream_run or {}))


@dataclass(frozen=True)
class RunReport:
    """Complete presentation-independent manifest for one scientific run."""

    identity: RunIdentity
    provenance: Provenance
    health: Tuple[Diagnostic, ...]
    sections: Tuple[ReportSection, ...]
    overview_metrics: Tuple[ScalarMetric, ...] = ()
    headline_artifacts: Tuple[Artifact, ...] = ()
    parameters: Tuple[ParameterValue, ...] = ()
    links: Tuple[ReportLink, ...] = ()
    schema_version: str = REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported report schema version: {self.schema_version}")
        _validate_unique(self.health, "Health diagnostic")
        _validate_unique(self.sections, "Report section")
        _validate_unique(self.overview_metrics, "Overview metric")
        _validate_unique(self.headline_artifacts, "Headline artifact")
        parameter_names = [parameter.name for parameter in self.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Parameter names must be unique")

    def to_dict(self) -> Mapping[str, Any]:
        """Return the stable JSON-ready representation of this report."""

        payload = _to_json_value(self)
        payload["kind"] = "run"
        return payload


@dataclass(frozen=True)
class ComparedRun:
    """One run participating in a comparison report."""

    key: str
    label: str
    run_id: str
    report_path: Optional[str] = None
    git_commit: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_key(self.key, "Compared-run key")
        _validate_key(self.run_id, "Compared-run ID")
        if not self.label:
            raise ValueError("Compared-run labels cannot be empty")


@dataclass(frozen=True)
class ComparisonMetric:
    """Baseline/candidate values with safe absolute and fractional differences."""

    key: str
    label: str
    baseline: float
    candidate: float
    delta: float
    fractional_delta: Optional[float]
    unit: str = ""
    derivative_prediction: Optional[float] = None
    interpretation: str = ""

    def __post_init__(self) -> None:
        _validate_key(self.key, "Comparison metric key")
        if not self.label:
            raise ValueError("Comparison metric labels cannot be empty")
        for name in ("baseline", "candidate", "delta"):
            _validate_scalar(getattr(self, name), f"Comparison metric {name}")
        if self.fractional_delta is not None:
            _validate_scalar(self.fractional_delta, "Comparison metric fractional_delta")
        if self.derivative_prediction is not None:
            _validate_scalar(self.derivative_prediction, "Comparison metric derivative_prediction")

    @classmethod
    def from_values(
        cls,
        *,
        key: str,
        label: str,
        baseline: float,
        candidate: float,
        unit: str = "",
        derivative_prediction: Optional[float] = None,
        interpretation: str = "",
    ) -> "ComparisonMetric":
        """Construct a comparison without inventing a scale for a zero baseline."""

        baseline = float(baseline)
        candidate = float(candidate)
        delta = candidate - baseline
        fractional_delta = None if baseline == 0.0 else delta / baseline
        return cls(
            key=key,
            label=label,
            baseline=baseline,
            candidate=candidate,
            delta=delta,
            fractional_delta=fractional_delta,
            unit=unit,
            derivative_prediction=derivative_prediction,
            interpretation=interpretation,
        )


@dataclass(frozen=True)
class ComparisonReport:
    """First-class manifest for a scientific comparison between runs."""

    comparison_id: str
    title: str
    summary: str
    baseline: ComparedRun
    candidate: ComparedRun
    metrics: Tuple[ComparisonMetric, ...]
    provenance: Provenance
    health: Tuple[Diagnostic, ...] = ()
    sections: Tuple[ReportSection, ...] = ()
    links: Tuple[ReportLink, ...] = ()
    schema_version: str = REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_key(self.comparison_id, "Comparison ID")
        if not self.title or not self.summary:
            raise ValueError("Comparison reports require a title and summary")
        if self.baseline.key == self.candidate.key:
            raise ValueError("Baseline and candidate keys must differ")
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported report schema version: {self.schema_version}")
        keys = [metric.key for metric in self.metrics]
        if len(keys) != len(set(keys)):
            raise ValueError("Comparison metric keys must be unique")
        _validate_unique(self.health, "Comparison health diagnostic")
        _validate_unique(self.sections, "Comparison section")

    def to_dict(self) -> Mapping[str, Any]:
        """Return the stable JSON-ready representation of this comparison."""

        payload = _to_json_value(self)
        payload["kind"] = "comparison"
        return payload


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _to_json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item") and callable(value.item):
        return _to_json_value(value.item())
    return value
