#!/usr/bin/env python3
"""Generate the first native SAGE16/SHARK/Sapphire comparison report."""

import argparse
import json
import shutil
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import generate_common_sam_response_report as established
import matplotlib.pyplot as plt

from mimic_jax import (
    AnnotatedStateSpace,
    LocalStateSpace,
    ResponseCoordinate,
    characteristic_modes,
    frequency_response,
    load_model,
    scale_state_space,
)
from mimic_jax.reporting import (
    Artifact,
    ComparedRun,
    Diagnostic,
    DiagnosticStatus,
    ModelMetricValue,
    MultiModelComparisonReport,
    MultiModelMetric,
    ReportLink,
    ReportSection,
    ScalarMetric,
    capture_provenance,
    write_report,
)
from mimic_jax.sapphire import SapphireNativeArtifact

COLORS = {"sage16": "#3567a9", "shark": "#d05a3a", "sapphire": "#4b8b62"}


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/three-model-response-foundation")
    )
    parser.add_argument(
        "--sapphire-artifact",
        type=Path,
        default=Path("tests/data/sapphire/native-v0.130-controlled"),
    )
    return parser.parse_args()


def _sapphire_response(artifact, periods):
    model = load_model("sapphire")
    full = model.local_response(artifact=artifact)
    sfr_index = artifact.observable_names.index("star_formation_rate")
    sfr = artifact.observable("star_formation_rate")
    matrices = LocalStateSpace(
        state_jacobian=full.state_jacobian,
        input_jacobian=full.input_jacobian,
        output_jacobian=full.output_jacobian[sfr_index : sfr_index + 1] / sfr,
        direct_input_jacobian=full.direct_input_jacobian[sfr_index : sfr_index + 1] / sfr,
    )
    fractional = AnnotatedStateSpace(
        matrices=matrices,
        point=full.point,
        state_coordinates=full.state_coordinates,
        input_coordinates=full.input_coordinates,
        output_coordinates=(
            ResponseCoordinate(
                "fractional_sfr",
                "fractional star-formation rate",
                "dimensionless",
                "SFR divided by its value at the native Sapphire operating point.",
            ),
        ),
        derivative_method=full.derivative_method,
    )
    fractional = scale_state_space(fractional, artifact.state)
    frequencies = 2.0 * np.pi / periods
    transfer = np.asarray(frequency_response(fractional, frequencies))[:, 0, :]
    modes = characteristic_modes(fractional)
    stable_times = np.sort(
        np.asarray(modes.response_times_gyr)[np.asarray(modes.stable, dtype=bool)]
    )
    return {
        "model": model,
        "artifact": artifact,
        "space": fractional,
        "transfer": transfer,
        "stable_times": stable_times,
    }


def _plot_supply_response(path, periods, established_responses, sapphire):
    figure, axis = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    for name in ("sage16", "shark"):
        values = established_responses[name]
        index = values["model"].metadata.process_control_names.index("cooling")
        axis.loglog(
            periods,
            np.abs(values["transfer"][:, index]),
            color=COLORS[name],
            linewidth=2.3,
            label=f"{values['model'].metadata.label}: cooling supply",
        )
    input_index = sapphire["artifact"].input_names.index("Mdot_in_dm")
    axis.loglog(
        periods,
        np.abs(sapphire["transfer"][:, input_index]),
        color=COLORS["sapphire"],
        linewidth=2.3,
        label="Sapphire Pandya23: halo accretion supply",
    )
    axis.axhline(1.0, color="0.55", linewidth=0.8)
    axis.set_xlabel("Variation period [Gyr]")
    axis.set_ylabel("Fractional SFR response / fractional supply change")
    axis.set_title("How strongly does SFR follow a changing supply?")
    axis.legend(frameon=False, fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def _plot_modes(path, responses):
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    labels = {"sage16": "SAGE16", "shark": "SHARK Lagos23", "sapphire": "Sapphire Pandya23"}
    for axis, name in zip(axes, ("sage16", "shark", "sapphire")):
        times = responses[name]["stable_times"]
        axis.barh(np.arange(times.size), times, color=COLORS[name])
        axis.set_xscale("log")
        finite = times[np.isfinite(times) & (times > 0.0)]
        if finite.size:
            axis.set_xlim(max(1.0e-3, finite.min() / 2.0), finite.max() * 2.0)
        axis.set_xlabel("Local damping time [Gyr]")
        axis.set_title(labels[name])
        axis.set_yticks(np.arange(times.size), [f"mode {index + 1}" for index in range(times.size)])
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Stable coupled mode (fast to slow)")
    figure.suptitle("The coupled reservoir systems remember perturbations on several timescales")
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def _plot_sapphire_trajectory(path, artifact):
    times = artifact.arrays["trajectory_times_gyr"]
    states = artifact.arrays["trajectory_state"]
    observables = artifact.arrays["trajectory_observables"]
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), constrained_layout=True)
    for index, label, color in (
        (0, "stars", "#7b4fa3"),
        (1, "ISM", "#2f8fb3"),
        (2, "CGM", "#d28d30"),
    ):
        axes[0].plot(times, states[:, index] / 1.0e10, label=label, color=color, linewidth=2.2)
    axes[0].set_xlabel("Cosmic time [Gyr]")
    axes[0].set_ylabel(r"Mass [$10^{10}\,M_\odot$]")
    axes[0].set_title("Where are the baryons in the native run?")
    axes[0].legend(frameon=False)
    axes[1].plot(times, observables[:, 3], color=COLORS["sapphire"], linewidth=2.2)
    axes[1].set_xlabel("Cosmic time [Gyr]")
    axes[1].set_ylabel(r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]")
    axes[1].set_title("How does the star-formation rate evolve?")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def _plot_native_validation(path, artifact, relative_budget_residuals):
    derivative = artifact.derivative_validation
    labels = (
        "state\nRHS",
        "halo input\nRHS",
        "local\nparameter",
        "full-history\nparameter",
    )
    values = (
        derivative["state_jacobian_relative_l2_error"],
        derivative["input_jacobian_relative_l2_error"],
        derivative["parameter_output_jacobian_relative_l2_error"],
        derivative["trajectory_parameter_output_jacobian_relative_l2_error"],
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), constrained_layout=True)
    axes[0].bar(labels, values, color=COLORS["sapphire"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Relative L2 error")
    axes[0].set_title("Automatic derivatives agree with finite differences")
    axes[1].bar(
        ("solver\nrefinement", "baryon\nbudget", "metal\nbudget"),
        (
            np.max(np.abs(artifact.convergence_fraction)),
            relative_budget_residuals[0],
            relative_budget_residuals[1],
        ),
        color=("#756bb1", "#4b8b62", "#4b8b62"),
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Maximum fractional discrepancy")
    axes[1].set_title("Native trajectory and budgets close")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, facecolor="white")
    plt.close(figure)


def _figure(key, title, filename, description):
    return Artifact(key, title, f"assets/{filename}", "image/svg+xml", "figure", description)


def main():
    arguments = _arguments()
    repository = Path(__file__).resolve().parents[1]
    output = arguments.output
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    artifact = SapphireNativeArtifact.load(arguments.sapphire_artifact)

    periods, established_responses = established._responses(established._cases())
    sapphire = _sapphire_response(artifact, periods)
    responses = {**established_responses, "sapphire": sapphire}

    model = sapphire["model"]
    balances = model.conservation_balances(artifact)
    relative_budget_residuals = tuple(
        abs(float(balance.residual))
        / max(float(balance.source_rate), float(balance.sink_rate), 1.0)
        for balance in balances
    )
    normalized_conservation_residuals = {"sapphire": max(relative_budget_residuals)}
    for name in ("sage16", "shark"):
        local_balances = responses[name]["model"].conservation_balances(responses[name]["result"])
        normalized_conservation_residuals[name] = max(
            abs(float(balance.residual))
            / max(abs(float(balance.source_rate)), abs(float(balance.sink_rate)), 1.0)
            for balance in local_balances
        )
    local_validation_max = max(
        value
        for name, value in artifact.derivative_validation.items()
        if not name.startswith("trajectory_")
    )
    trajectory_validation = artifact.derivative_validation[
        "trajectory_parameter_output_jacobian_relative_l2_error"
    ]
    convergence_max = float(np.max(np.abs(artifact.convergence_fraction)))

    supply_figure = _figure(
        "supply-response",
        "Gas-supply response across three models",
        "three-model-supply-response.svg",
        "SAGE16 and SHARK are perturbed at the cooling boundary; Sapphire is perturbed at the upstream dark-matter accretion boundary. The distinction is part of the result, not hidden normalization.",
    )
    modes_figure = _figure(
        "response-times",
        "Coupled response times across three models",
        "three-model-response-times.svg",
        "Stable local damping times at each explicitly recorded operating point; direct numerical values should not be interpreted as a same-halo population comparison.",
    )
    trajectory_figure = _figure(
        "sapphire-trajectory",
        "Native Sapphire controlled trajectory",
        "native-sapphire-trajectory.svg",
        "Pandya23 is run by Sapphire v0.130 with native Diffrax Tsit5 integration under constant smooth halo forcing.",
    )
    validation_figure = _figure(
        "sapphire-validation",
        "Native Sapphire adapter validation",
        "native-sapphire-validation.svg",
        "AD/finite-difference, tolerance-refinement, and open-system conservation diagnostics are computed from the pinned native run.",
    )
    _plot_supply_response(
        assets / supply_figure.path.split("/")[-1], periods, established_responses, sapphire
    )
    _plot_modes(assets / modes_figure.path.split("/")[-1], responses)
    _plot_sapphire_trajectory(assets / trajectory_figure.path.split("/")[-1], artifact)
    _plot_native_validation(
        assets / validation_figure.path.split("/")[-1], artifact, relative_budget_residuals
    )

    shutil.copy2(arguments.sapphire_artifact / "artifact.json", assets / "sapphire-artifact.json")
    shutil.copy2(arguments.sapphire_artifact / "arrays.npz", assets / "sapphire-arrays.npz")
    native_manifest = Artifact(
        "sapphire-native-manifest",
        "Native Sapphire artifact manifest",
        "assets/sapphire-artifact.json",
        "application/json",
        "metadata",
        "Pinned revision, case, coordinates, solver, finite-difference validation, cooling-table checksum, software, and hardware.",
    )
    native_arrays = Artifact(
        "sapphire-native-arrays",
        "Native Sapphire trajectory and response arrays",
        "assets/sapphire-arrays.npz",
        "application/x-npz",
        "data",
        "Physical trajectory, rates, state/input/parameter Jacobians, finite differences, and convergence arrays.",
    )

    protocol_path = assets / "three-model-protocols.json"
    protocol_path.write_text(
        json.dumps(
            {name: load_model(name).metadata.to_dict() for name in ("sage16", "shark", "sapphire")},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    protocol_artifact = Artifact(
        "three-model-protocols",
        "Three-model semantic manifests",
        "assets/three-model-protocols.json",
        "application/json",
        "metadata",
        "State, forcing, parameters, processes, capabilities, qualifications, and upstream revisions.",
    )

    period_index = int(np.argmin(np.abs(periods - 10.0)))
    gains = {}
    for name in ("sage16", "shark"):
        index = responses[name]["model"].metadata.process_control_names.index("cooling")
        gains[name] = float(abs(responses[name]["transfer"][period_index, index]))
    sapphire_input = artifact.input_names.index("Mdot_in_dm")
    gains["sapphire"] = float(abs(sapphire["transfer"][period_index, sapphire_input]))

    comparison_matrix = {
        "schema_version": "mimic-jax-three-model-comparison/v1",
        "models": ["sage16", "shark", "sapphire"],
        "domains": {
            "configured_model_metadata": {
                "status": "evaluated",
                "qualification": "All three load through mimic_jax.load_model.",
            },
            "continuous_local_rhs": {
                "status": "evaluated",
                "qualification": "SAGE16 and SHARK run in-process; Sapphire is delegated to its pinned native runtime.",
            },
            "local_conservation": {
                "status": "evaluated",
                "qualification": "SAGE16 and SHARK use structural ledgers; Sapphire uses native rates with explicit open-system boundaries.",
            },
            "local_state_jacobian_and_modes": {
                "status": "evaluated",
                "qualification": "Each result belongs to its recorded operating point and state coordinates.",
            },
            "supply_to_sfr_response": {
                "status": "qualified",
                "qualification": "SAGE16/SHARK perturb cooling; Sapphire perturbs halo accretion upstream of its CGM.",
            },
            "parameter_response": {
                "status": "evaluated_with_distinct_coordinates",
                "qualification": "One normalization API is available, but parameters are not presumed to be physically interchangeable.",
            },
            "same_halo_history_population": {
                "status": "not_evaluated",
                "qualification": "Sapphire smooth-central forcing has not been matched to SAGE16/SHARK branch topology and population weights.",
            },
            "common_event_topology": {
                "status": "not_applicable",
                "qualification": "The audited Sapphire model has no merger/satellite event system.",
            },
        },
        "reservoir_correspondence": {
            "long_lived_stars": {
                "sage16": "StellarMass",
                "shark": "stellar_mass",
                "sapphire": "M_star",
                "comparison": "direct at the continuous-state level",
            },
            "star_forming_gas": {
                "sage16": "ColdGas",
                "shark": "cold_gas",
                "sapphire": "M_ism",
                "comparison": "qualified because phase and aperture definitions remain model-owned",
            },
            "halo_atmosphere": {
                "sage16": "HotGas",
                "shark": "cold_halo_gas + hot_halo_gas",
                "sapphire": "M_cgm + Eth_cgm",
                "comparison": "qualified; these are not synonymous reservoirs",
            },
            "ejected_gas": {
                "sage16": "EjectedGas",
                "shark": "ejected_gas + lost_gas",
                "sapphire": None,
                "comparison": "unavailable as a three-model reservoir",
            },
            "tracked_metals": {
                "sage16": "four metal reservoirs",
                "shark": "six metal reservoirs plus formed-metal tracker",
                "sapphire": "stellar, ISM, and CGM metal reservoirs",
                "comparison": "shared total ledgers; reservoir-level mapping is qualified",
            },
        },
        "shared_observables": {
            "stellar_mass": "direct local quantity in all three",
            "star_formation_rate": "direct local quantity in all three",
            "star_forming_gas_mass": "qualified cold-gas/ISM comparison",
            "stellar_metallicity": "available in all three model ecosystems; only Sapphire is exercised in this controlled native artifact",
            "gas_metallicity": "qualified phase-definition comparison",
            "stellar_mass_function": "requires a weighted common population and is not evaluated for Sapphire here",
        },
        "normalized_conservation_residuals": normalized_conservation_residuals,
        "ten_gyr_supply_to_sfr_gain": gains,
        "slowest_stable_mode_gyr": {
            name: float(responses[name]["stable_times"][-1])
            for name in ("sage16", "shark", "sapphire")
        },
    }
    comparison_matrix_path = assets / "three-model-comparison-matrix.json"
    comparison_matrix_path.write_text(
        json.dumps(comparison_matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    comparison_matrix_artifact = Artifact(
        "three-model-comparison-matrix",
        "Three-model comparison matrix",
        "assets/three-model-comparison-matrix.json",
        "application/json",
        "data",
        "Evaluated, qualified, unavailable, and not-evaluated comparison domains with reservoir and observable semantics.",
    )

    health = (
        Diagnostic(
            "configured-models",
            "Three configured model boundaries",
            DiagnosticStatus.PASSED,
            "SAGE16, SHARK Lagos23, and native Sapphire Pandya23 load through one registry with state, forcing, parameter, process, observable, and capability metadata.",
            method="mimic_jax.load_model",
        ),
        Diagnostic(
            "sapphire-native-derivatives",
            "Native Sapphire local derivative validation",
            (DiagnosticStatus.PASSED if local_validation_max < 1.0e-7 else DiagnosticStatus.FAILED),
            "State, fractional halo-input, and fixed-state parameter-to-observable Jacobians are checked against symmetric finite differences.",
            metrics=(
                ScalarMetric(
                    "maximum-relative-l2-error",
                    "Maximum relative L2 error",
                    local_validation_max,
                    "fraction",
                ),
            ),
            artifacts=(validation_figure,),
            tolerance="maximum relative L2 error < 1e-7",
        ),
        Diagnostic(
            "sapphire-trajectory-parameter-derivatives",
            "End-to-end Sapphire parameter derivatives",
            (
                DiagnosticStatus.PASSED
                if trajectory_validation < 5.0e-3
                else DiagnosticStatus.FAILED
            ),
            "Final-observable derivatives pass through the native adaptive Diffrax trajectory and are compared with symmetric finite differences across five perturbation sizes. The looser gate records the accept/reject controller's discrete path rather than pretending it is a smooth local RHS.",
            metrics=(
                ScalarMetric(
                    "trajectory-relative-l2-error",
                    "Relative L2 error at the declared reference step",
                    trajectory_validation,
                    "fraction",
                ),
            ),
            artifacts=(validation_figure,),
            tolerance="relative L2 error < 5e-3 at parameter-coordinate step 1e-4",
        ),
        Diagnostic(
            "sapphire-native-integration",
            "Native Sapphire integration refinement",
            DiagnosticStatus.PASSED if convergence_max < 1.0e-4 else DiagnosticStatus.FAILED,
            "The requested 1e-8 adaptive solve is compared with a 1e-10 solve for the same physical case.",
            metrics=(
                ScalarMetric(
                    "maximum-final-state-difference",
                    "Maximum final-state difference",
                    convergence_max,
                    "fraction",
                ),
            ),
            tolerance="maximum absolute fractional final-state difference < 1e-4",
        ),
        Diagnostic(
            "three-model-local-conservation",
            "Three-model local conservation",
            (
                DiagnosticStatus.PASSED
                if max(normalized_conservation_residuals.values()) < 1.0e-12
                else DiagnosticStatus.FAILED
            ),
            "Each evaluated continuous calculation closes its declared mass, metal, and where present angular-momentum ledger after external boundaries are included.",
            metrics=tuple(
                ScalarMetric(
                    f"{name}-normalized-residual",
                    f"{load_model(name).metadata.label} normalized residual",
                    residual,
                    "fraction of max(source, sink, 1 native rate unit)",
                )
                for name, residual in normalized_conservation_residuals.items()
            ),
            tolerance="maximum normalized residual < 1e-12",
        ),
        Diagnostic(
            "shared-process-controls",
            "Identical process-control surface",
            DiagnosticStatus.WARNING,
            "SAGE16 and SHARK expose explicit cooling/feedback process perturbations. The unmodified Sapphire closure currently exposes halo-input and parameter perturbations; mimic-jax does not copy or rewrite its RHS to fabricate process hooks.",
        ),
        Diagnostic(
            "common-population",
            "Same-history population comparison",
            DiagnosticStatus.NOT_EVALUATED,
            "Sapphire's native smooth central-halo histories are not yet matched to the full branch/event topology used by SAGE16 and SHARK.",
        ),
    )

    report = MultiModelComparisonReport(
        comparison_id="three-model-response-foundation",
        title="What becomes comparable when SAGE16, SHARK, and Sapphire share one analysis language?",
        summary=(
            "Sapphire is now a native third configured model rather than an architectural footnote. "
            "Its Pandya23 equations remain owned and executed by Sapphire in an isolated modern-JAX "
            "environment; mimic-jax imports versioned scientific artifacts and asks the same questions "
            "about reservoirs, forcing, conservation, parameter derivatives, and local response."
        ),
        runs=(
            ComparedRun("sage16", "MIMIC/SAGE16", "sage16-common-local-response"),
            ComparedRun("shark", "SHARK Lagos23", "shark-common-local-response"),
            ComparedRun("sapphire", "Sapphire Pandya23", "sapphire-v0.130-native-controlled"),
        ),
        metrics=(
            MultiModelMetric(
                "state-dimension",
                "Continuous state coordinates",
                tuple(
                    ModelMetricValue(name, len(load_model(name).metadata.state_variables))
                    for name in ("sage16", "shark", "sapphire")
                ),
                interpretation="Different dimensions reflect different model physics; they are not padded into a universal state.",
            ),
            MultiModelMetric(
                "parameter-dimension",
                "Exposed parameter coordinates",
                tuple(
                    ModelMetricValue(name, len(load_model(name).metadata.parameter_variables))
                    for name in ("sage16", "shark", "sapphire")
                ),
                interpretation="Parameter counts describe the audited formulation, not comparable degrees of freedom.",
            ),
            MultiModelMetric(
                "perturbable-processes",
                "Named fractional process controls",
                tuple(
                    ModelMetricValue(name, len(load_model(name).metadata.process_control_names))
                    for name in ("sage16", "shark", "sapphire")
                ),
                interpretation="Sapphire exposes native halo-input and parameter derivatives but no copied process-control hooks.",
            ),
            MultiModelMetric(
                "slowest-stable-mode",
                "Slowest finite stable local mode",
                tuple(
                    ModelMetricValue(name, float(responses[name]["stable_times"][-1]))
                    for name in ("sage16", "shark", "sapphire")
                ),
                unit="Gyr",
                interpretation="Each value belongs to its declared local operating point and closure boundary.",
            ),
            MultiModelMetric(
                "supply-sfr-gain",
                "Supply-to-SFR response at a 10 Gyr period",
                tuple(
                    ModelMetricValue(name, gains[name]) for name in ("sage16", "shark", "sapphire")
                ),
                unit="fraction/fraction",
                interpretation="SAGE16/SHARK inputs are cooling; Sapphire's input is upstream halo accretion, so the latter includes CGM filtering.",
            ),
            MultiModelMetric(
                "event-topology",
                "Event/topology capability",
                tuple(
                    ModelMetricValue(name, load_model(name).metadata.capability("events").status)
                    for name in ("sage16", "shark", "sapphire")
                ),
                interpretation="Unavailable is a model-scope statement, not a failed numerical test.",
            ),
            MultiModelMetric(
                "normalized-conservation-residual",
                "Maximum normalized local conservation residual",
                tuple(
                    ModelMetricValue(name, normalized_conservation_residuals[name])
                    for name in ("sage16", "shark", "sapphire")
                ),
                unit="fraction of max(source, sink, 1 native rate unit)",
                interpretation="Each ledger includes its declared external source and sink boundary before normalization.",
            ),
        ),
        provenance=capture_provenance(
            repository=repository,
            command=(
                "scripts/generate_three_model_response_report.py",
                "--output",
                output.as_posix(),
                "--sapphire-artifact",
                arguments.sapphire_artifact.as_posix(),
            ),
            configuration_paths=(repository / "docs/dev/MIMIC-JAX-SAPPHIRE-INTEGRATION-PLAN.md",),
            input_paths=(
                arguments.sapphire_artifact / "artifact.json",
                arguments.sapphire_artifact / "arrays.npz",
            ),
        ),
        health=health,
        sections=(
            ReportSection(
                "comparison-boundary",
                "What can be compared today?",
                "The three models now share an analysis language, while every scientific comparison retains an explicit evidence state.",
                artifacts=(comparison_matrix_artifact,),
                body=r"""| Question | Status | What the comparison means |
|---|---|---|
| Can all three expose named state, forcing, parameters, processes, and observables? | **Evaluated** | One semantic registry and machine-readable protocol, without padding states into a universal vector. |
| Can all three expose a local continuous RHS and state Jacobian? | **Evaluated** | SAGE16/SHARK run in-process; Sapphire is evaluated by its pinned native Pandya23 runtime. |
| Can all three close baryon and metal budgets? | **Evaluated** | SAGE16/SHARK use structural ledgers; Sapphire includes halo inflow, CGM outflow, yield, and enriched-flow boundaries. |
| Can all three be asked how SFR responds to changing supply? | **Qualified** | SAGE16/SHARK perturb cooling; Sapphire perturbs dark-matter accretion before CGM filtering. |
| Can all three expose parameter responses? | **Evaluated with distinct coordinates** | The normalization and metadata API is shared; parameter identities and derivative horizons are not silently equated. |
| Can all three be compared on the same full merger-tree population? | **Not evaluated** | A smooth-main-branch forcing adapter, population weights, and aligned selections remain required. |
| Can merger and satellite event maps be compared across all three? | **Not applicable** | The audited Sapphire independent-central model has no such topology. |
""",
            ),
            ReportSection(
                "native-sapphire",
                "Is this actually Sapphire?",
                "Yes: the controlled trajectory and derivatives come from the pinned upstream Pandya23 closure and native Diffrax solver.",
                artifacts=(trajectory_figure, validation_figure, native_manifest, native_arrays),
                body=(
                    "The bridge runs Sapphire v0.130 at revision `ee50e858e3427de50368c32205001248849b8be0` with its official SD93 cooling table. "
                    r"It converts Sapphire's internal $d\log_{10}x/d\log_{10}t$ derivative back to physical $dx/dt$ before exporting the state Jacobian. "
                    "The complete case, solver tolerances, cooling-table checksum, software versions, device, trajectory, rates, and finite-difference arrays accompany this page."
                ),
            ),
            ReportSection(
                "same-language",
                "What is genuinely common across the three models?",
                "The commonality is semantic and mathematical, not an invented claim that the reservoirs or prescriptions are identical.",
                artifacts=(protocol_artifact,),
                body=r"""| Concept | SAGE16 | SHARK Lagos23 | Sapphire Pandya23 |
|---|---|---|---|
| Continuous state | Cold/hot/ejected/stars + metals | Disk/halo/ejected/lost + metals, trackers, angular momentum | Stars/ISM/CGM + CGM thermal energy + metals |
| Halo forcing | Merger-tree virial properties | Native SHARK tree/halo interval data | Smooth $\dot M_h$, $M_h$, $R_\mathrm{vir}$, $V_\mathrm{vir}$, concentration |
| Events | Mergers, instability, stripping/topology maps | Mergers, instability, disruption and native topology | Not present in the audited independent-central model |
| Differentiable inputs here | Named process controls and parameters | Named process controls and nested parameters | Fractional halo inputs and native Pandya23 parameters |
| Common observables available now | Stellar/gas masses and SFR | Stellar/gas masses and SFR | Stellar/ISM/CGM masses, SFR and metallicities |
""",
            ),
            ReportSection(
                "reservoir-correspondence",
                "Which reservoirs really correspond?",
                "Several physical roles overlap, but the report preserves phase boundaries and model-specific memory rather than equating names.",
                body=r"""| Physical role | SAGE16 | SHARK Lagos23 | Sapphire Pandya23 | Comparison status |
|---|---|---|---|---|
| Long-lived stars | `StellarMass` | `stellar_mass` | `M_star` | Direct local mass quantity |
| Star-forming gas | `ColdGas` | `cold_gas` | `M_ism` | Qualified by phase and aperture conventions |
| Halo atmosphere | `HotGas` | `cold_halo_gas` + `hot_halo_gas` | `M_cgm` plus `Eth_cgm` | Qualified; not synonymous state coordinates |
| Ejected material | `EjectedGas` | `ejected_gas` + `lost_gas` | No separate reservoir | Unavailable as a three-model reservoir comparison |
| Metals | Cold/hot/ejected/stellar | Six gas/stellar reservoirs plus trackers | Stellar/ISM/CGM | Total metal budgets compare; individual reservoirs remain qualified |
| Dynamical memory | Upstream history/event state outside this local vector | Angular momentum, formed-mass trackers, AGN memory in wider model | CGM thermal energy | Model-specific; mode composition must use named coordinates |
""",
            ),
            ReportSection(
                "process-correspondence",
                "Which baryon-cycle processes overlap?",
                "All three contain supply, cooling, star formation, feedback transport, and enrichment, but their closure boundaries differ.",
                body=r"""| Physical process | SAGE16 | SHARK Lagos23 | Sapphire Pandya23 | Safe comparison today |
|---|---|---|---|---|
| Cosmological supply | Finite tree-driven infall budget | Native halo-interval infall preparation | Smooth halo accretion forcing | Forcing metadata and open-system budget |
| Cooling | Hot-to-cold transfer | Halo-to-disk cooling supply | CGM energy loss plus CGM-to-ISM transfer | Local response, with input boundary stated |
| Star formation | Thresholded disk law with recycling | Pressure-based molecular disk law with recycling | ISM depletion law with recycling | SFR and local derivatives, not recipe identity |
| Stellar feedback | Reheating and ejection | Reheating, ejection, QSO channels | ISM wind plus energy injection and CGM outflow | Flux/budget roles; process controls are not yet identical |
| Reincorporation | Explicit ejected-to-hot flow | Explicit ejected return | No separate ejected reservoir | Two-model process comparison only |
| Metal enrichment | Yield plus advective flows | Yield plus multi-reservoir transport | Yield plus stellar/ISM/CGM transport | Total metal conservation and qualified metallicities |
| AGN/BH regulation | Present in full hybrid SAGE16 | Present in full SHARK | Absent in audited Pandya23 model | Not a three-model comparison |
""",
            ),
            ReportSection(
                "supply-response",
                "On what timescales does star formation follow its supply?",
                "The same response machinery can be applied without pretending the input boundary is the same.",
                artifacts=(supply_figure,),
                body=(
                    "For SAGE16 and SHARK the experiment perturbs the cooling transfer directly. For Sapphire it perturbs dark-matter accretion, which changes baryonic halo inflow and must propagate through the CGM before reaching the ISM. "
                    "The different boundary is scientifically useful: it separates a recipe-level cooling response from the full atmosphere's filtering of halo supply."
                ),
            ),
            ReportSection(
                "memory-times",
                "Do the three baryon cycles forget perturbations on the same timescales?",
                "Each nonlinear model generates several coupled local damping times rather than inheriting one recipe timescale.",
                artifacts=(modes_figure,),
                body=(
                    "The poles are calculated from each physical state Jacobian after an input-output-invariant state scaling. Neutral integrated-mass/tracker modes are excluded. "
                    "The current figure compares local mathematical structure, not matched galaxies: a same-history mass/redshift grid is the next evidence gate."
                ),
            ),
            ReportSection(
                "conservation-comparison",
                "Do the three local baryon cycles close their budgets?",
                "Yes for the evaluated local calculations, once every model's external boundary is included explicitly.",
                body=(
                    "The maximum normalized residuals are "
                    f"`{normalized_conservation_residuals['sage16']:.3e}` for SAGE16, "
                    f"`{normalized_conservation_residuals['shark']:.3e}` for SHARK, and "
                    f"`{normalized_conservation_residuals['sapphire']:.3e}` for Sapphire. "
                    "For closed ledgers the denominator floor is one native rate unit; for open ledgers it is the larger declared source or sink. This is a numerical closure test, not a claim that the models place baryons in equivalent reservoirs."
                ),
            ),
            ReportSection(
                "observable-overlap",
                "Which familiar outputs can eventually be compared?",
                "Stellar mass, SFR, star-forming gas, and metallicity provide the strongest three-model overlap; abundance statistics require one more population-level gate.",
                body=r"""| Observable | Three-model status | Required qualification |
|---|---|---|
| Stellar mass | Direct local overlap | Align IMF, units, aperture, selection, and population weights for catalogues |
| Star-formation rate | Direct local overlap | Align instantaneous/averaged definitions before observational comparison |
| Cold gas / ISM mass | Qualified overlap | Preserve phase definitions; Sapphire's ISM is not automatically a SAGE/SHARK cold-gas aperture |
| Stellar metallicity | Available | Align mass weighting, yield convention, and solar normalization |
| Gas metallicity | Qualified overlap | Align gas phase and distinguish metal mass fraction from observational oxygen calibration |
| Stellar mass function | Not evaluated for three models | Sapphire needs a number-density-complete or explicitly weighted population on compatible halo histories |
| Black-hole and AGN observables | Unavailable across all three | Pandya23 has no BH/AGN state; retain the established SAGE16--SHARK comparison separately |
| Satellite/environment/clustering statistics | Not applicable to current Sapphire model | Requires an extended topology-owning model rather than an invented adapter |
""",
            ),
            ReportSection(
                "parameter-response-boundary",
                "What does a common parameter response mean?",
                "The API is common; the parameters and time horizons remain physical properties of each model.",
                body=(
                    r"Mimic-jax exposes dimensionless fractional responses when the observable and parameter have meaningful positive scales, $E_{O,\theta}=\partial\ln O/\partial\ln\theta$, and explicit reference-scale responses otherwise. "
                    "SAGE16 and SHARK can differentiate their in-process configured subsets. Sapphire now exports both a fixed-state local parameter Jacobian and the derivative of final observables through its complete adaptive native trajectory. "
                    "Sapphire's `A_*` coordinates are base-10 logarithmic normalizations and several slope parameters are zero or signed, so the report does not label raw coordinate derivatives as elasticities or match them one-to-one with SAGE/SHARK parameters."
                ),
            ),
            ReportSection(
                "limits",
                "What remains before a population-level three-model science claim?",
                "The adapter is executable and validated, but the forcing and topology domains are not yet identical.",
                body=(
                    "Sapphire currently models independent central histories and familiar scaling relations, whereas full SAGE16 and SHARK include branch topology, satellites, mergers, black holes, and additional observables. "
                    "The next comparison must construct a documented main-progenitor forcing adapter, preserve each model's event scope, attach population weights, and then compare SMHM, gas fractions, metallicity, SFMS, and any number-density statistic that is actually defined for the sample."
                ),
            ),
        ),
        links=(
            ReportLink(
                "SAGE16–SHARK response foundation",
                "../sage16-shark-response-foundation/index.md",
            ),
            ReportLink("Sapphire source", "https://github.com/virajpandya/sapphire"),
            ReportLink(
                "Common protocol plan", "../../docs/dev/MIMIC-JAX-COMMON-SAM-PROTOCOL-PLAN.md"
            ),
        ),
    )
    write_report(report, output)


if __name__ == "__main__":
    main()
