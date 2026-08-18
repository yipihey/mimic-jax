#!/usr/bin/env python3
"""Build the canonical initial SAGE16 Mini-Millennium run report.

The expensive partition benchmark and equivalence check are separate commands
that write JSON into the report's ``assets`` directory. This builder consumes
those durable results, generates controlled differentiability/numerical
diagnostics, asks the existing MIMIC plot registry for familiar figures, and
writes the report manifest. It never reruns the Mini-Millennium model itself.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mimic_jax import (  # noqa: E402
    parameter_response_matrix,
    timestep_refinement_study,
    validate_parameter_response,
)
from mimic_jax.reporting import (  # noqa: E402
    Artifact,
    Diagnostic,
    DiagnosticStatus,
    ReportLink,
    ReportSection,
    RunIdentity,
    RunReport,
    ScalarMetric,
    benchmark_diagnostic,
    capture_provenance,
    conservation_diagnostic,
    equivalence_diagnostic,
    parameter_response_diagnostic,
    parameters_from_namedtuple,
    timestep_refinement_diagnostic,
    write_report,
)
from mimic_jax.sage16 import (  # noqa: E402
    baryonic_mass,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    load_cooling_tables,
    quiescent_disk_step,
    sage16_units,
    step_context,
    subcycle_upstream_sequential_central,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPOSITORY / "reports/mini-millennium-sage16-initial"
DEFAULT_RUN_FILE = REPOSITORY / "models/sage16/input/sage16_mini-millennium.yaml"
DEFAULT_SCALE_FACTORS = REPOSITORY / "simulations/mini-millennium/mini-millennium.a_list"
DEFAULT_UPSTREAM_OUTPUT = REPOSITORY / "output/sage16-mini-millennium"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--run-file", type=Path, default=DEFAULT_RUN_FILE)
    parser.add_argument(
        "--upstream-output",
        type=Path,
        default=DEFAULT_UPSTREAM_OUTPUT,
        help="existing upstream MIMIC output used by the familiar plot registry",
    )
    parser.add_argument(
        "--equivalence-json",
        type=Path,
        help="selected-tree equivalence JSON (default: <output>/assets/equivalence.json)",
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        help="partition benchmark JSON (default: <output>/assets/benchmark.json)",
    )
    parser.add_argument(
        "--skip-familiar-plots",
        action="store_true",
        help="omit upstream-style figures when MIMIC output is unavailable",
    )
    return parser.parse_args()


def load_json(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read required diagnostic {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"Diagnostic JSON must contain an object: {path}")
    return payload


def stage_json(source: Path, destination: Path) -> Path:
    """Place one diagnostic JSON inside the shareable report directory."""

    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def controlled_parameter_response(output_path: Path):
    parameters = fiducial_parameters()

    def observables(current):
        state = initial_galaxy_state(
            ColdGas=10.0,
            HotGas=5.0,
            StellarMass=2.0,
            DiskScaleRadius=0.01,
        )
        halo = initial_halo_forcing(Vvir=150.0, dT=1.0e-4)
        result = quiescent_disk_step(
            state,
            state,
            halo,
            halo,
            step_context(time_interval=1.0e-4),
            current,
            sage16_units(),
        )
        return jnp.asarray([result.galaxy.StellarMass, result.galaxy.ColdGas])

    response = parameter_response_matrix(
        observables,
        parameters,
        parameter_names=("SfrEfficiency", "FeedbackReheatingEpsilon"),
        observable_names=("final_stellar_mass", "final_cold_gas"),
        observable_units=("1e10 Msun/h", "1e10 Msun/h"),
        parameter_units=("dimensionless", "dimensionless"),
    )
    validation = validate_parameter_response(
        response,
        observables,
        parameters,
        relative_steps=(1.0e-2, 3.0e-3, 1.0e-3),
    )
    response.save(output_path)
    return response, validation


def controlled_refinement(output_path: Path):
    state = initial_galaxy_state(
        ColdGas=2.0,
        HotGas=10.0,
        EjectedGas=1.0,
        StellarMass=1.0,
        MetalsColdGas=0.04,
        MetalsHotGas=0.2,
        MetalsEjectedGas=0.02,
        MetalsStellarMass=0.02,
        BlackHoleMass=0.01,
        DiskScaleRadius=0.01,
    )
    halo = initial_halo_forcing(Mvir=100.0, Rvir=0.2, Vvir=200.0, dT=0.01)
    context = step_context(time_interval=0.01)
    parameters = fiducial_parameters()
    units = sage16_units()
    tables = load_cooling_tables()

    def run(num_substeps):
        return subcycle_upstream_sequential_central(
            state,
            halo,
            context,
            parameters,
            units,
            tables,
            num_substeps=num_substeps,
        )

    def observables(result):
        final = result.final_state
        return jnp.asarray([final.StellarMass, final.ColdGas, final.HotGas, final.BlackHoleMass])

    study = timestep_refinement_study(
        run,
        observables,
        substeps=(1, 2, 4, 8),
        observable_names=("stellar_mass", "cold_gas", "hot_gas", "black_hole_mass"),
        observable_units=("1e10 Msun/h",) * 4,
    )
    study.save(output_path)
    residuals = [
        float(baryonic_mass(run(num_substeps).final_state) - baryonic_mass(state))
        for num_substeps in (1, 2, 4, 8)
    ]
    return study, max(abs(residual) for residual in residuals)


def generate_familiar_plots(arguments, assets: Path):
    if arguments.skip_familiar_plots:
        return ()
    upstream_config = arguments.upstream_output / "metadata/sage16_mini-millennium.yaml"
    if not upstream_config.is_file():
        raise SystemExit(
            f"Upstream MIMIC metadata not found at {upstream_config}. Run MIMIC first or use "
            "--skip-familiar-plots."
        )
    command = [
        sys.executable,
        str(REPOSITORY / "plot/mimic-plot/mimic-plot.py"),
        f"--param-file={upstream_config}",
        "--snapshot=63",
        "--snapshot-plots",
        "--plots=stellar_mass_function,black_hole_bulge_relation",
        f"--output-dir={assets}",
        "--format=.svg",
        "--quiet",
    ]
    cache_root = REPOSITORY / "build/report-matplotlib-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(cache_root / "matplotlib")
    environment["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    subprocess.run(command, cwd=REPOSITORY, check=True, env=environment)
    expected = (assets / "StellarMassFunction.svg", assets / "BlackHoleBulgeRelation.svg")
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise SystemExit(f"Plot registry did not produce expected figures: {missing}")
    return (
        Artifact(
            key="stellar_mass_function",
            title="Upstream MIMIC z=0 stellar mass function",
            path="assets/StellarMassFunction.svg",
            media_type="image/svg+xml",
            role="figure",
            description=(
                "The familiar SAGE diagnostic is sourced from the upstream MIMIC catalogue. "
                "It is context, not a claim of full mimic-jax population equivalence."
            ),
        ),
        Artifact(
            key="black_hole_bulge_relation",
            title="Upstream MIMIC black-hole–bulge relation",
            path="assets/BlackHoleBulgeRelation.svg",
            media_type="image/svg+xml",
            role="figure",
            description="An existing model-local SAGE16 plot generated without report-specific logic.",
        ),
    )


def main():
    arguments = parse_arguments()
    output = arguments.output_dir.resolve()
    assets = output / "assets"
    equivalence_source = (
        arguments.equivalence_json.resolve()
        if arguments.equivalence_json is not None
        else assets / "equivalence.json"
    )
    benchmark_source = (
        arguments.benchmark_json.resolve()
        if arguments.benchmark_json is not None
        else assets / "benchmark.json"
    )
    equivalence = load_json(equivalence_source)
    benchmark = load_json(benchmark_source)
    upstream_partitions = tuple(
        sorted(arguments.upstream_output.glob("model_[0-9][0-9][0-9].hdf5"))
    )
    if not upstream_partitions:
        raise SystemExit(
            f"No upstream MIMIC partition files found under {arguments.upstream_output}"
        )

    # Capture source state before generating or updating report artifacts.
    provenance = capture_provenance(
        repository=REPOSITORY,
        command=(sys.executable, *sys.argv),
        configuration_paths=(arguments.run_file, DEFAULT_SCALE_FACTORS),
        input_paths=(
            arguments.upstream_output / "model.hdf5",
            arguments.upstream_output / "metadata/version_info.json",
            *upstream_partitions,
        ),
        upstream_version_info=arguments.upstream_output / "metadata/version_info.json",
    )

    assets.mkdir(parents=True, exist_ok=True)
    equivalence_path = stage_json(equivalence_source, assets / "equivalence.json")
    benchmark_path = stage_json(benchmark_source, assets / "benchmark.json")
    familiar_artifacts = generate_familiar_plots(arguments, assets)
    response_artifact = Artifact(
        key="controlled_parameter_response",
        title="Controlled fractional parameter response arrays",
        path="assets/controlled_parameter_response.npz",
        media_type="application/x-npz",
        role="scientific_array",
        description="Values, validity mask, normalization, names, units, and derivative method.",
    )
    response, response_validation = controlled_parameter_response(
        assets / "controlled_parameter_response.npz"
    )
    response_diagnostic = parameter_response_diagnostic(
        response,
        validation=response_validation,
        validation_tolerance=2.0e-5,
        artifact=response_artifact,
    )

    refinement_artifact = Artifact(
        key="controlled_timestep_refinement",
        title="Controlled timestep-refinement arrays",
        path="assets/controlled_timestep_refinement.npz",
        media_type="application/x-npz",
        role="scientific_array",
        description="Substeps, observables, provisional errors, and empirical orders.",
    )
    refinement, maximum_baryon_residual = controlled_refinement(
        assets / "controlled_timestep_refinement.npz"
    )
    refinement_diagnostic = timestep_refinement_diagnostic(
        refinement,
        artifact=refinement_artifact,
    )
    baryon_diagnostic = conservation_diagnostic(
        key="baryon_conservation",
        title="Baryon conservation",
        maximum_absolute_residual=maximum_baryon_residual,
        tolerance=3.0e-6,
        conserved_quantity="baryon mass",
        unit="1e10 Msun/h",
        method="controlled central source/sink ledger over 1, 2, 4, and 8 substeps",
    )
    metal_diagnostic = Diagnostic(
        key="metal_conservation",
        title="Metal conservation",
        status=DiagnosticStatus.NOT_EVALUATED,
        summary="A report-level Mini-Millennium metal ledger was not evaluated for this run.",
    )

    equivalence_artifact = Artifact(
        key="selected_tree_equivalence",
        title="Selected-tree equivalence JSON",
        path=equivalence_path.relative_to(output).as_posix(),
        media_type="application/json",
        role="diagnostic",
        description="Exact evaluated scope, comparison count, tolerances, and residual summary.",
    )
    selected_equivalence = equivalence_diagnostic(
        comparisons=int(equivalence["field_comparisons"]),
        mismatches=int(equivalence["mismatches"]),
        scope=(
            f"Mini-Millennium trees {equivalence['tree_start']}–"
            f"{int(equivalence['tree_end']) - 1}"
        ),
        tolerance="float32/Cooling/Heating rtol=atol=2e-6; other float64 "
        "rtol=atol=2e-12; integers exact",
        artifact=equivalence_artifact,
    )
    equivalence_health = Diagnostic(
        key="upstream_equivalence",
        title="Upstream equivalence",
        status=DiagnosticStatus.WARNING,
        summary=(
            "The selected 100-tree control passes, but full-population equivalence is not yet "
            "established and the separate complex tree-0 gate retains one known mismatch."
        ),
        metrics=selected_equivalence.metrics,
        artifacts=(equivalence_artifact,),
        notes=(
            "This warning prevents a selected-tree pass from being presented as Mini-Millennium "
            "population equivalence.",
        ),
        method=selected_equivalence.method,
        tolerance=selected_equivalence.tolerance,
    )

    benchmark_artifact = Artifact(
        key="partition_benchmark",
        title="Selected-tree benchmark JSON",
        path=benchmark_path.relative_to(output).as_posix(),
        media_type="application/json",
        role="benchmark",
        description="Cold/warm timing, backend, device, memory, shapes, and catalogue digest.",
    )
    performance = benchmark_diagnostic(
        benchmark,
        status=DiagnosticStatus.WARNING,
        summary=(
            "Warmed execution is much faster than the first call, but the cold catalogue path is "
            "currently much slower than upstream MIMIC; no JAX speedup is claimed."
        ),
        artifact=benchmark_artifact,
    )

    first_run = benchmark["runs"][0]
    warm_seconds = min(run["evolution_seconds"] for run in benchmark["runs"][1:])
    sections = (
        ReportSection(
            key="familiar_science",
            title="Familiar SAGE science",
            summary=(
                "These figures come directly from the existing SAGE16 plot registry and the "
                "upstream MIMIC catalogue. They establish the practitioner-facing context before "
                "new diagnostics are introduced."
            ),
            artifacts=familiar_artifacts,
            links=(ReportLink("SAGE16 plotting manual", "../../plot/mimic-plot/README.md"),),
        ),
        ReportSection(
            key="equivalence",
            title="What has been matched upstream?",
            summary=(
                "The evaluated sample compares catalogue fields by `UniqueGalaxyID` over every "
                "configured output snapshot. Its scope is stated explicitly."
            ),
            diagnostics=(selected_equivalence,),
            links=(
                ReportLink(
                    "Mini-Millennium equivalence evidence",
                    "../../docs/mini_millennium_equivalence.md",
                ),
            ),
        ),
        ReportSection(
            key="conservation",
            title="Conservation",
            summary=(
                "Executable ledgers make closed transfers and explicit sources or sinks visible. "
                "This first report includes a controlled baryon check only."
            ),
            diagnostics=(baryon_diagnostic, metal_diagnostic),
            links=(ReportLink("Conservation contract", "../../docs/conservation.md"),),
        ),
        ReportSection(
            key="numerical_integration",
            title="Numerical integration",
            summary=(
                "The faithful upstream-sequential method is refined on a controlled fixed-forcing "
                "central. This is API evidence, not Mini-Millennium convergence."
            ),
            diagnostics=(refinement_diagnostic,),
            links=(
                ReportLink("Numerical integration contract", "../../docs/numerical_integration.md"),
            ),
        ),
        ReportSection(
            key="parameter_responses",
            title="How does familiar SAGE physics change the result?",
            summary=(
                "The first validated derivative is shown as a fractional response: percentage "
                "change in the observable per 1% parameter change. The example is a controlled "
                "quiescent disk step, not a population response."
            ),
            diagnostics=(response_diagnostic,),
            links=(ReportLink("Fractional-response API", "../../docs/sensitivity.md"),),
        ),
        ReportSection(
            key="performance",
            title="Performance",
            summary=(
                "Compilation, first execution, warmed execution, host work, catalogue conversion, "
                "and memory are kept distinct."
            ),
            diagnostics=(performance,),
            links=(ReportLink("Current performance evidence", "../../docs/performance.md"),),
        ),
    )

    report = RunReport(
        identity=RunIdentity(
            run_id="mini-millennium-sage16-initial",
            title="SAGE16 Mini-Millennium: initial mimic-jax run report",
            model="fiducial SAGE16",
            dataset="Mini-Millennium, selected trees 1500–1599",
            parameter_set="sage16_mini-millennium fiducial",
            integration_method="upstream_sequential, 10 configured substeps",
            summary=(
                "This report combines familiar upstream MIMIC figures with the current selected-"
                "tree mimic-jax equivalence gate and controlled conservation, timestep, gradient, "
                "and performance diagnostics. It deliberately does not claim full-population "
                "equivalence."
            ),
        ),
        provenance=provenance,
        health=(
            equivalence_health,
            baryon_diagnostic,
            metal_diagnostic,
            response_diagnostic,
            refinement_diagnostic,
        ),
        sections=sections,
        overview_metrics=(
            ScalarMetric("trees", "Trees in selected gate", int(equivalence["tree_count"])),
            ScalarMetric("input_halos", "Input halos", int(equivalence["input_halos"])),
            ScalarMetric(
                "catalogue_records",
                "Catalogue records compared",
                int(equivalence["records_compared"]),
            ),
            ScalarMetric(
                "cold_evolution_seconds",
                "First evolution call",
                float(first_run["evolution_seconds"]),
                unit="s",
            ),
            ScalarMetric(
                "warm_evolution_seconds",
                "Best warm evolution call",
                float(warm_seconds),
                unit="s",
            ),
            ScalarMetric("backend", "JAX backend", str(benchmark["backend"])),
        ),
        headline_artifacts=familiar_artifacts[:1],
        parameters=parameters_from_namedtuple(fiducial_parameters()),
        links=(
            ReportLink("Report architecture", "../../docs/reporting.md"),
            ReportLink(
                "Scientific application program", "../../docs/mimic_jax_scientific_program.md"
            ),
        ),
    )
    written = write_report(report, output)
    print(f"Wrote Markdown report: {written.markdown_path}")
    print(f"Wrote machine-readable manifest: {written.manifest_path}")


if __name__ == "__main__":
    main()
