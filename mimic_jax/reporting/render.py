"""Deterministic Markdown and JSON renderers for report manifests."""

import hashlib
import json
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Union

from mimic_jax.reporting.model import (
    Artifact,
    ComparedRun,
    ComparisonMetric,
    ComparisonReport,
    Diagnostic,
    DiagnosticStatus,
    Provenance,
    ReportLink,
    ReportSection,
    RunReport,
    ScalarMetric,
)

Report = Union[RunReport, ComparisonReport]

_STATUS_LABELS = {
    DiagnosticStatus.PASSED: "✅ Passed",
    DiagnosticStatus.WARNING: "⚠️ Warning",
    DiagnosticStatus.FAILED: "❌ Failed",
    DiagnosticStatus.NOT_EVALUATED: "⬚ Not evaluated",
}


@dataclass(frozen=True)
class WrittenReport:
    """Paths and checksum-resolved manifest returned by a report write."""

    report: Report
    markdown_path: Path
    manifest_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_artifact(artifact: Artifact, directory: Path, validate_assets: bool) -> Artifact:
    path = directory / artifact.path
    if not path.is_file():
        if validate_assets:
            raise FileNotFoundError(f"Report artifact does not exist: {path}")
        return artifact
    return replace(artifact, sha256=_sha256(path), size_bytes=path.stat().st_size)


def _resolve_diagnostic(
    diagnostic: Diagnostic, directory: Path, validate_assets: bool
) -> Diagnostic:
    return replace(
        diagnostic,
        artifacts=tuple(
            _resolve_artifact(artifact, directory, validate_assets)
            for artifact in diagnostic.artifacts
        ),
    )


def _resolve_section(
    section: ReportSection, directory: Path, validate_assets: bool
) -> ReportSection:
    return replace(
        section,
        diagnostics=tuple(
            _resolve_diagnostic(diagnostic, directory, validate_assets)
            for diagnostic in section.diagnostics
        ),
        artifacts=tuple(
            _resolve_artifact(artifact, directory, validate_assets)
            for artifact in section.artifacts
        ),
    )


def resolve_artifacts(report: Report, directory: Path, *, validate_assets: bool = True) -> Report:
    """Return a manifest with checksums and sizes for every available artifact."""

    if isinstance(report, RunReport):
        return replace(
            report,
            health=tuple(
                _resolve_diagnostic(diagnostic, directory, validate_assets)
                for diagnostic in report.health
            ),
            sections=tuple(
                _resolve_section(section, directory, validate_assets) for section in report.sections
            ),
            headline_artifacts=tuple(
                _resolve_artifact(artifact, directory, validate_assets)
                for artifact in report.headline_artifacts
            ),
        )
    return replace(
        report,
        health=tuple(
            _resolve_diagnostic(diagnostic, directory, validate_assets)
            for diagnostic in report.health
        ),
        sections=tuple(
            _resolve_section(section, directory, validate_assets) for section in report.sections
        ),
    )


def _table_text(value) -> str:
    if value is None:
        return "not defined"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _metric_value(metric: ScalarMetric) -> str:
    value = _table_text(metric.value)
    return f"{value} {metric.unit}".rstrip()


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _front_matter(title: str, report_id: str, kind: str, provenance: Provenance) -> List[str]:
    return [
        "---",
        f"title: {_yaml_string(title)}",
        f"report-id: {_yaml_string(report_id)}",
        f"report-kind: {_yaml_string(kind)}",
        f"date: {_yaml_string(provenance.generated_at)}",
        "toc: true",
        "---",
        "",
    ]


def _render_health(diagnostics: Sequence[Diagnostic], *, title: str = "Run health") -> List[str]:
    lines = [f"## {title}", ""]
    if not diagnostics:
        return lines + [
            "No health diagnostics were evaluated for this report.",
            "",
        ]
    lines.extend(["| Check | Status | Evidence |", "| --- | --- | --- |"])
    for diagnostic in diagnostics:
        lines.append(
            f"| {diagnostic.title} | {_STATUS_LABELS[diagnostic.status]} | "
            f"{_table_text(diagnostic.summary)} |"
        )
    lines.append("")
    return lines


def _render_metrics(metrics: Sequence[ScalarMetric]) -> List[str]:
    if not metrics:
        return []
    lines = ["| Quantity | Value | Interpretation |", "| --- | ---: | --- |"]
    for metric in metrics:
        lines.append(
            f"| {metric.label} | {_metric_value(metric)} | {_table_text(metric.description)} |"
        )
    return lines + [""]


def _render_artifacts(artifacts: Sequence[Artifact]) -> List[str]:
    lines = []
    for artifact in artifacts:
        if artifact.media_type.startswith("image/"):
            lines.extend(
                [
                    f"![{artifact.title}]({artifact.path})",
                    "",
                ]
            )
            if artifact.description:
                lines.extend([f"*{artifact.description}*", ""])
        else:
            description = f" — {artifact.description}" if artifact.description else ""
            lines.extend([f"[{artifact.title}]({artifact.path}){description}", ""])
    return lines


def _render_links(links: Sequence[ReportLink]) -> List[str]:
    if not links:
        return []
    return ["Related: " + " · ".join(f"[{link.title}]({link.target})" for link in links), ""]


def _render_diagnostic(diagnostic: Diagnostic, heading_level: int = 3) -> List[str]:
    prefix = "#" * heading_level
    lines = [
        f"{prefix} {diagnostic.title}",
        "",
        f"**Status:** {_STATUS_LABELS[diagnostic.status]}",
        "",
        diagnostic.summary,
        "",
    ]
    if diagnostic.method:
        lines.extend([f"**Method:** {diagnostic.method}", ""])
    if diagnostic.tolerance:
        lines.extend([f"**Acceptance criterion:** {diagnostic.tolerance}", ""])
    lines.extend(_render_metrics(diagnostic.metrics))
    if diagnostic.notes:
        lines.extend([f"- {note}" for note in diagnostic.notes])
        lines.append("")
    lines.extend(_render_artifacts(diagnostic.artifacts))
    return lines


def _render_section(section: ReportSection) -> List[str]:
    lines = [f"## {section.title}", "", section.summary, ""]
    lines.extend(_render_links(section.links))
    lines.extend(_render_artifacts(section.artifacts))
    for diagnostic in section.diagnostics:
        lines.extend(_render_diagnostic(diagnostic))
    if section.notes:
        lines.extend([f"- {note}" for note in section.notes])
        lines.append("")
    return lines


def _render_provenance(provenance: Provenance) -> List[str]:
    lines = ["## Provenance and reproducibility", ""]
    git_state = "not available"
    if provenance.git_commit:
        dirty = "dirty working tree" if provenance.git_dirty else "clean working tree"
        git_state = f"`{provenance.git_commit}` ({dirty})"
    lines.extend(
        [
            "| Item | Value |",
            "| --- | --- |",
            f"| Generated | {_table_text(provenance.generated_at)} |",
            f"| Git commit | {git_state} |",
            f"| Git branch | {_table_text(provenance.git_branch)} |",
            "",
        ]
    )
    if provenance.command:
        lines.extend(
            [
                "### Rerun command",
                "",
                "```shell",
                shlex.join(provenance.command),
                "```",
                "",
            ]
        )
    if provenance.files:
        lines.extend(
            [
                "### Configurations and inputs",
                "",
                "| Role | Path | SHA-256 | Bytes |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for item in provenance.files:
            lines.append(f"| {item.role} | `{item.path}` | `{item.sha256}` | {item.size_bytes} |")
        lines.append("")
    for title, values in (
        ("Software", provenance.software),
        ("Hardware and backend", provenance.hardware),
        ("Random seeds", provenance.random_seeds),
    ):
        if values:
            lines.extend([f"### {title}", "", "| Name | Value |", "| --- | --- |"])
            for key in sorted(values):
                lines.append(f"| {_table_text(key)} | {_table_text(values[key])} |")
            lines.append("")
    if provenance.upstream_run:
        lines.extend(
            [
                "### Upstream MIMIC run record",
                "",
                "```json",
                json.dumps(provenance.upstream_run, indent=2, sort_keys=True, ensure_ascii=False),
                "```",
                "",
            ]
        )
    return lines


def render_run_markdown(report: RunReport) -> str:
    """Render a run report as ordinary GitHub- and Obsidian-friendly Markdown."""

    lines = _front_matter(report.identity.title, report.identity.run_id, "run", report.provenance)
    lines.extend(
        [
            f"# {report.identity.title}",
            "",
            report.identity.summary,
            "",
            "[Machine-readable manifest](report.json)",
            "",
            "## Run overview",
            "",
            "| Item | Value |",
            "| --- | --- |",
            f"| Model | {report.identity.model} |",
            f"| Dataset / trees | {report.identity.dataset} |",
            f"| Parameter set | {report.identity.parameter_set} |",
            f"| Integration method | {report.identity.integration_method} |",
        ]
    )
    for metric in report.overview_metrics:
        lines.append(f"| {metric.label} | {_metric_value(metric)} |")
    lines.append("")
    lines.extend(_render_links(report.links))
    lines.extend(_render_health(report.health))
    if report.headline_artifacts:
        lines.extend(["## At a glance", ""])
        lines.extend(_render_artifacts(report.headline_artifacts))
    for section in report.sections:
        lines.extend(_render_section(section))
    if report.parameters:
        lines.extend(
            [
                "## Parameters",
                "",
                "| Parameter | Value | Units | Description |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for parameter in report.parameters:
            lines.append(
                f"| `{parameter.name}` | {_table_text(parameter.value)} | "
                f"{_table_text(parameter.unit)} | {_table_text(parameter.description)} |"
            )
        lines.append("")
    lines.extend(_render_provenance(report.provenance))
    return "\n".join(lines).rstrip() + "\n"


def _render_compared_run(run: ComparedRun) -> str:
    label = run.label
    if run.report_path:
        label = f"[{label}]({run.report_path})"
    commit = f" (`{run.git_commit}`)" if run.git_commit else ""
    return f"{label}{commit}"


def _fractional_text(value: Optional[float]) -> str:
    return "not defined" if value is None else f"{100.0 * value:.6g}%"


def _render_comparison_metrics(metrics: Sequence[ComparisonMetric]) -> List[str]:
    lines = [
        "| Observable | Baseline | Candidate | Difference | Fractional difference | "
        "Derivative prediction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in metrics:
        unit = f" {metric.unit}" if metric.unit else ""
        prediction = _fractional_text(metric.derivative_prediction)
        lines.append(
            f"| {metric.label} | {_table_text(metric.baseline)}{unit} | "
            f"{_table_text(metric.candidate)}{unit} | {_table_text(metric.delta)}{unit} | "
            f"{_fractional_text(metric.fractional_delta)} | {prediction} |"
        )
    return lines + [""]


def render_comparison_markdown(report: ComparisonReport) -> str:
    """Render a first-class baseline/candidate comparison as ordinary Markdown."""

    lines = _front_matter(
        report.title,
        report.comparison_id,
        "comparison",
        report.provenance,
    )
    lines.extend(
        [
            f"# {report.title}",
            "",
            report.summary,
            "",
            "[Machine-readable manifest](report.json)",
            "",
            "## Compared runs",
            "",
            "| Role | Run | Run ID |",
            "| --- | --- | --- |",
            f"| Baseline | {_render_compared_run(report.baseline)} | `{report.baseline.run_id}` |",
            f"| Candidate | {_render_compared_run(report.candidate)} | "
            f"`{report.candidate.run_id}` |",
            "",
        ]
    )
    lines.extend(_render_health(report.health, title="Comparison health"))
    lines.extend(["## Observable differences", ""])
    lines.extend(_render_comparison_metrics(report.metrics))
    for metric in report.metrics:
        if metric.interpretation:
            lines.append(f"- **{metric.label}:** {metric.interpretation}")
    if any(metric.interpretation for metric in report.metrics):
        lines.append("")
    for section in report.sections:
        lines.extend(_render_section(section))
    lines.extend(_render_links(report.links))
    lines.extend(_render_provenance(report.provenance))
    return "\n".join(lines).rstrip() + "\n"


def render_markdown(report: Report) -> str:
    """Render either report kind to Markdown."""

    if isinstance(report, RunReport):
        return render_run_markdown(report)
    return render_comparison_markdown(report)


def _atomic_write(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def write_report(
    report: Report,
    directory,
    *,
    validate_assets: bool = True,
) -> WrittenReport:
    """Write deterministic ``index.md`` and ``report.json`` artifacts for a report."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    resolved = resolve_artifacts(report, destination, validate_assets=validate_assets)
    markdown_path = destination / "index.md"
    manifest_path = destination / "report.json"
    manifest = json.dumps(
        resolved.to_dict(),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    _atomic_write(manifest_path, manifest + "\n")
    _atomic_write(markdown_path, render_markdown(resolved))
    return WrittenReport(resolved, markdown_path, manifest_path)
