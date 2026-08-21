"""Run-report manifests, renderers, provenance, and diagnostic adapters."""

import hashlib
import json

import jax.numpy as jnp
import pytest

from mimic_jax import parameter_response_matrix, validate_parameter_response
from mimic_jax.reporting import (
    Artifact,
    ComparedRun,
    ComparisonMetric,
    ComparisonReport,
    Diagnostic,
    DiagnosticStatus,
    ModelMetricValue,
    MultiModelComparisonReport,
    MultiModelMetric,
    Provenance,
    ReportSection,
    RunIdentity,
    RunReport,
    ScalarMetric,
    capture_provenance,
    conservation_diagnostic,
    parameter_response_diagnostic,
    parameters_from_namedtuple,
    write_report,
)
from mimic_jax.sage16 import fiducial_parameters


def _fixed_provenance():
    return Provenance(
        generated_at="2026-08-18T12:00:00Z",
        git_commit="a" * 40,
        git_branch="main",
        git_dirty=False,
        command=("mimic-jax", "run", "config.yaml", "--report"),
        software={"jax": "0.4.38", "python": "3.12.0"},
        hardware={"jax_backend": "cpu"},
    )


def _sample_report(artifact):
    missing_gradient = Diagnostic(
        key="gradient_validation",
        title="Gradient validation",
        status=DiagnosticStatus.NOT_EVALUATED,
        summary="No differentiable observable was requested for this run.",
    )
    baryons = conservation_diagnostic(
        key="baryon_conservation",
        title="Baryon conservation",
        maximum_absolute_residual=2.0e-10,
        tolerance=1.0e-9,
        conserved_quantity="baryon mass",
        unit="1e10 Msun/h",
        method="explicit source/sink ledger",
    )
    return RunReport(
        identity=RunIdentity(
            run_id="sage16-mini-reference",
            title="SAGE16 Mini-Millennium reference",
            model="SAGE16",
            dataset="Mini-Millennium",
            parameter_set="fiducial",
            integration_method="upstream_sequential",
            summary="A compact, evidence-limited reference run.",
        ),
        provenance=_fixed_provenance(),
        health=(baryons, missing_gradient),
        sections=(
            ReportSection(
                key="familiar_science",
                title="Familiar SAGE science",
                summary="The upstream-style stellar mass function is shown first.",
                body="The local relation is $\\delta \\dot{x}=A\\delta x$.",
                artifacts=(artifact,),
            ),
        ),
        overview_metrics=(ScalarMetric("galaxies", "Galaxies", 42),),
        headline_artifacts=(artifact,),
    )


def test_run_report_writes_deterministic_markdown_json_and_checksums(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    figure = assets / "stellar_mass_function.svg"
    figure.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
    artifact = Artifact(
        key="stellar_mass_function",
        title="Stellar mass function",
        path="assets/stellar_mass_function.svg",
        media_type="image/svg+xml",
        role="figure",
        description="Familiar upstream-style z=0 diagnostic.",
    )

    first = write_report(_sample_report(artifact), tmp_path)
    first_markdown = first.markdown_path.read_bytes()
    first_manifest = first.manifest_path.read_bytes()
    second = write_report(_sample_report(artifact), tmp_path)

    assert second.markdown_path.read_bytes() == first_markdown
    assert second.manifest_path.read_bytes() == first_manifest
    markdown = first_markdown.decode("utf-8")
    assert "✅ Passed" in markdown
    assert "⬚ Not evaluated" in markdown
    assert "![Stellar mass function](assets/stellar_mass_function.svg)" in markdown
    assert "The local relation is $\\delta \\dot{x}=A\\delta x$." in markdown
    assert "mimic-jax run config.yaml --report" in markdown

    manifest = json.loads(first_manifest)
    assert manifest["kind"] == "run"
    assert manifest["schema_version"] == "mimic-jax-report/v1"
    assert manifest["health"][1]["status"] == "not_evaluated"
    expected_digest = hashlib.sha256(figure.read_bytes()).hexdigest()
    assert manifest["headline_artifacts"][0]["sha256"] == expected_digest
    assert manifest["headline_artifacts"][0]["size_bytes"] == figure.stat().st_size
    assert manifest["sections"][0]["body"].startswith("The local relation")


def test_report_rejects_escaping_and_missing_artifacts(tmp_path):
    with pytest.raises(ValueError, match="within the report directory"):
        Artifact(
            key="outside",
            title="Outside",
            path="../outside.png",
            media_type="image/png",
            role="figure",
        )

    missing = Artifact(
        key="missing",
        title="Missing",
        path="assets/missing.png",
        media_type="image/png",
        role="figure",
    )
    with pytest.raises(FileNotFoundError, match="Report artifact does not exist"):
        write_report(_sample_report(missing), tmp_path)


def test_comparison_report_preserves_zero_baseline_and_derivative_prediction(tmp_path):
    zero_baseline = ComparisonMetric.from_values(
        key="quenched_count",
        label="Quenched galaxy count",
        baseline=0.0,
        candidate=3.0,
        derivative_prediction=0.12,
        interpretation="A fractional change is undefined because the baseline count is zero.",
    )
    stellar_mass = ComparisonMetric.from_values(
        key="stellar_mass",
        label="Final stellar mass",
        baseline=10.0,
        candidate=9.0,
        unit="1e10 Msun/h",
    )
    report = ComparisonReport(
        comparison_id="fiducial-vs-feedback",
        title="Fiducial versus stronger feedback",
        summary="Measured finite changes are shown beside derivative predictions where available.",
        baseline=ComparedRun("baseline", "Fiducial", "sage16-fiducial"),
        candidate=ComparedRun("candidate", "Stronger feedback", "sage16-feedback"),
        metrics=(zero_baseline, stellar_mass),
        provenance=_fixed_provenance(),
    )

    written = write_report(report, tmp_path)
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    markdown = written.markdown_path.read_text(encoding="utf-8")
    assert manifest["kind"] == "comparison"
    assert manifest["metrics"][0]["fractional_delta"] is None
    assert manifest["metrics"][1]["fractional_delta"] == pytest.approx(-0.1)
    assert "not defined" in markdown
    assert "12%" in markdown


def test_multi_model_report_renders_all_models_and_machine_readable_values(tmp_path):
    runs = (
        ComparedRun("sage16", "SAGE16", "sage16-local"),
        ComparedRun("shark", "SHARK", "shark-local"),
        ComparedRun("sapphire", "Sapphire", "sapphire-local"),
    )
    metric = MultiModelMetric(
        "stellar_mass",
        "Final stellar mass",
        (
            ModelMetricValue("sage16", 1.0),
            ModelMetricValue("shark", 1.1),
            ModelMetricValue("sapphire", 0.9),
        ),
        unit="1e10 Msun",
        interpretation="A controlled adapter test, not a population comparison.",
    )
    report = MultiModelComparisonReport(
        comparison_id="three-model-local",
        title="Three configured galaxy models",
        summary="All three are represented without forcing identical physics.",
        runs=runs,
        metrics=(metric,),
        provenance=_fixed_provenance(),
    )
    written = write_report(report, tmp_path)
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    markdown = written.markdown_path.read_text(encoding="utf-8")
    assert manifest["kind"] == "multi_model_comparison"
    assert [run["key"] for run in manifest["runs"]] == ["sage16", "shark", "sapphire"]
    assert manifest["metrics"][0]["values"][2]["value"] == 0.9
    assert "| Quantity | SAGE16 | SHARK | Sapphire |" in markdown
    assert "A controlled adapter test" in markdown

    missing = MultiModelMetric(
        "incomplete",
        "Incomplete",
        (ModelMetricValue("sage16", 1.0), ModelMetricValue("shark", 1.0)),
    )
    with pytest.raises(ValueError, match="exactly the report run keys"):
        MultiModelComparisonReport(
            comparison_id="invalid-three-model",
            title="Invalid",
            summary="Missing one model value.",
            runs=runs,
            metrics=(missing,),
            provenance=_fixed_provenance(),
        )


def test_provenance_capture_is_explicit_and_checksum_bound(tmp_path):
    configuration = tmp_path / "run.yaml"
    input_file = tmp_path / "tree.dat"
    configuration.write_text("model: sage16\n", encoding="utf-8")
    input_file.write_bytes(b"tree-input")

    provenance = capture_provenance(
        repository=tmp_path,
        command=("mimic-jax", "run", "run.yaml"),
        configuration_paths=(configuration,),
        input_paths=(input_file,),
        generated_at="2026-08-18T12:00:00Z",
        include_jax_runtime=False,
        package_names=(),
    )

    assert provenance.generated_at == "2026-08-18T12:00:00Z"
    assert provenance.git_commit is None
    assert provenance.git_dirty is None
    assert [item.path for item in provenance.files] == ["run.yaml", "tree.dat"]
    assert [item.role for item in provenance.files] == ["configuration", "input"]
    assert provenance.files[0].sha256 == hashlib.sha256(configuration.read_bytes()).hexdigest()


def test_parameter_response_adapter_reports_physical_interpretation_and_fd_status():
    parameters = fiducial_parameters()

    def observable(current):
        return jnp.asarray([current.SfrEfficiency**2])

    response = parameter_response_matrix(
        observable,
        parameters,
        parameter_names=("SfrEfficiency",),
        observable_names=("stellar_mass_proxy",),
    )
    validation = validate_parameter_response(
        response,
        observable,
        parameters,
        relative_steps=(1.0e-2, 1.0e-3),
    )
    diagnostic = parameter_response_diagnostic(
        response,
        validation=validation,
        validation_tolerance=1.0e-10,
    )

    assert diagnostic.status == DiagnosticStatus.PASSED
    assert "A 1% increase" in diagnostic.notes[0]
    assert "approximately 2%" in diagnostic.notes[0]
    assert parameters_from_namedtuple(parameters)[0].name == "GlobalBaryonFraction"
