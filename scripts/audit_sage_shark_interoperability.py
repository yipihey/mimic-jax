#!/usr/bin/env python3
"""Audit SAGE16/SHARK catalogues, shared observables, and tree portability."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from mimic_jax.catalogue import COMMON_OBSERVABLES, observable_capabilities  # noqa: E402
from mimic_jax.io import open_lhalo_partition  # noqa: E402
from mimic_jax.observables import (  # noqa: E402
    catalogue_black_hole_bulge_relation,
    catalogue_cold_gas_fraction_relation,
    catalogue_cosmic_sfr_density,
    catalogue_mass_function,
    catalogue_quenched_fraction,
)
from mimic_jax.observations import load_baldry2008_stellar_mass_function  # noqa: E402
from mimic_jax.reporting import (  # noqa: E402
    Artifact,
    ComparedRun,
    ComparisonMetric,
    ComparisonReport,
    Diagnostic,
    DiagnosticStatus,
    ReportLink,
    ReportSection,
    ScalarMetric,
    capture_provenance,
    write_report,
)
from mimic_jax.sage16 import (  # noqa: E402
    load_sage_comparison_catalogue,
    load_scale_factors,
    snapshot_timing,
)
from mimic_jax.shark import (  # noqa: E402
    load_shark_catalogue,
    load_shark_tree,
    shark_comparison_catalogue,
)
from mimic_jax.trees import (  # noqa: E402
    SAGE16_TREE_REQUIREMENTS,
    SHARK_LAGOS23_TREE_REQUIREMENTS,
    assess_tree_compatibility,
    canonical_tree_from_lhalo,
    canonical_tree_from_shark,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage-catalogues", type=Path, nargs="+", required=True)
    parser.add_argument("--sage-snapshot", type=int, default=63)
    parser.add_argument("--sage-redshift", type=float, default=0.0)
    parser.add_argument("--sage-hubble-h", type=float, default=0.73)
    parser.add_argument("--sage-box-size", type=float, default=62.5)
    parser.add_argument("--lhalo-tree", type=Path, required=True)
    parser.add_argument("--lhalo-tree-index", type=int, default=1575)
    parser.add_argument("--scale-factors", type=Path, required=True)
    parser.add_argument("--particle-mass", type=float, default=0.0860657)
    parser.add_argument("--shark-catalogue", type=Path, required=True)
    parser.add_argument("--shark-snapshot", type=int, default=199)
    parser.add_argument("--shark-tree", type=Path, required=True)
    parser.add_argument(
        "--shark-tree-index",
        type=int,
        default=-1,
        help="tree row, or -1 for the largest tree in the file",
    )
    parser.add_argument(
        "--observation",
        type=Path,
        default=REPOSITORY / "data/observations/baldry2008_stellar_mass_function.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _serialise_field(field):
    return {
        "unit": field.unit,
        "description": field.description,
        "source_fields": list(field.source_fields),
        "projection": field.projection,
        "qualification": field.qualification,
    }


def _serialise_tree_compatibility(result):
    payload = asdict(result)
    payload["fully_runnable"] = result.fully_runnable
    return payload


def _plot_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
        }
    )
    return plt


def plot_common_observables(path, sage, shark, observation):
    plt = _plot_style()
    sage_colour = "#2b6cb0"
    shark_colour = "#d97706"
    edges = np.arange(8.0, 12.6, 0.25)
    relation_edges = np.arange(8.0, 12.2, 0.35)
    bh_edges = np.arange(7.5, 12.1, 0.4)
    sage_smf = catalogue_mass_function(sage, "stellar_mass", bin_edges=edges)
    shark_smf = catalogue_mass_function(shark, "stellar_mass", bin_edges=edges)
    sage_quenched = catalogue_quenched_fraction(sage, bin_edges=relation_edges)
    shark_quenched = catalogue_quenched_fraction(shark, bin_edges=relation_edges)
    sage_gas = catalogue_cold_gas_fraction_relation(
        sage, bin_edges=relation_edges, centrals_only=True
    )
    shark_gas = catalogue_cold_gas_fraction_relation(
        shark, bin_edges=relation_edges, centrals_only=True
    )
    sage_bh = catalogue_black_hole_bulge_relation(sage, bin_edges=bh_edges)
    shark_bh = catalogue_black_hole_bulge_relation(shark, bin_edges=bh_edges)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    ax = axes[0, 0]
    for result, label, colour in (
        (sage_smf, "SAGE16 / Mini-Millennium", sage_colour),
        (shark_smf, "SHARK Lagos23 / mini-SURFS", shark_colour),
    ):
        valid = (result.counts > 0) & (result.number_density > 0.0)
        ax.plot(
            result.bin_centres[valid],
            np.log10(result.number_density[valid]),
            marker="o",
            markersize=3,
            color=colour,
            label=label,
        )
    observation_lower = observation.values - observation.lower_errors
    valid_observation_band = observation_lower > 0.0
    ax.plot(
        observation.coordinate,
        np.log10(observation.values),
        color="#7c3aed",
        marker=".",
        markersize=3,
        lw=0.8,
        label="Baldry et al. (2008), h=0.7",
    )
    ax.fill_between(
        observation.coordinate,
        np.log10(np.maximum(observation_lower, 1.0e-30)),
        np.log10(observation.values + observation.upper_errors),
        where=valid_observation_band,
        color="#7c3aed",
        alpha=0.16,
    )
    ax.set(
        xlabel=r"$\log_{10}(M_\star/M_\odot)$",
        ylabel=r"$\log_{10}\phi$ [Mpc$^{-3}$ dex$^{-1}$]",
        title="One definition of the stellar mass function",
    )
    ax.legend(frameon=False)
    ax.set_ylim(-5.5, -0.5)

    ax = axes[0, 1]
    for result, label, colour in (
        (sage_quenched, "SAGE16", sage_colour),
        (shark_quenched, "SHARK Lagos23", shark_colour),
    ):
        valid = result.counts >= 5
        ax.plot(
            result.bin_centres[valid], result.fraction[valid], marker="o", color=colour, label=label
        )
    ax.axhline(0.5, color="0.75", lw=0.8)
    ax.set(
        xlabel=r"$\log_{10}(M_\star/M_\odot)$",
        ylabel="fraction with sSFR < $10^{-11}$ yr$^{-1}$",
        title="The same quenching selection",
    )
    ax.set_ylim(-0.03, 1.03)

    ax = axes[1, 0]
    for result, label, colour in (
        (sage_gas, "SAGE16", sage_colour),
        (shark_gas, "SHARK Lagos23", shark_colour),
    ):
        valid = result.counts >= 5
        ax.plot(
            result.bin_centres[valid], result.median[valid], marker="o", color=colour, label=label
        )
    ax.set(
        xlabel=r"$\log_{10}(M_\star/M_\odot)$",
        ylabel=r"median $M_{\rm cold}/(M_{\rm cold}+M_\star)$",
        title="Common cold-gas projection (qualified)",
    )
    ax.set_ylim(bottom=0.0)

    ax = axes[1, 1]
    for result, label, colour in (
        (sage_bh, "SAGE16", sage_colour),
        (shark_bh, "SHARK Lagos23", shark_colour),
    ):
        valid = result.counts >= 5
        ax.plot(
            result.bin_centres[valid], result.median[valid], marker="o", color=colour, label=label
        )
        ax.fill_between(
            result.bin_centres[valid],
            result.lower[valid],
            result.upper[valid],
            color=colour,
            alpha=0.12,
        )
    ax.set(
        xlabel=r"$\log_{10}(M_{\rm bulge}/M_\odot)$",
        ylabel=r"median $\log_{10}(M_{\rm BH}/M_\odot)$",
        title="The same BH--bulge reduction",
    )

    for ax in axes.flat:
        ax.grid(alpha=0.18)
        ax.tick_params(direction="in", top=True, right=True)
    figure.suptitle(
        "Shared observable code on native outputs\n"
        "(different trees and cosmologies: this is an adapter demonstration, not a model-only test)",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return {
        "smf_edges": edges,
        "sage_smf": sage_smf,
        "shark_smf": shark_smf,
        "relation_edges": relation_edges,
        "sage_quenched": sage_quenched,
        "shark_quenched": shark_quenched,
    }


def plot_capability_matrix(path, capabilities):
    plt = _plot_style()
    labels = [spec.label for spec in COMMON_OBSERVABLES]
    models = list(capabilities)
    encoding = {"unavailable": 0, "qualified": 1, "direct": 2}
    values = np.asarray(
        [
            [encoding[capabilities[model][spec.key].status] for model in models]
            for spec in COMMON_OBSERVABLES
        ]
    )
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    figure, ax = plt.subplots(figsize=(8.2, max(6.3, 0.42 * len(labels) + 1.8)))
    cmap = ListedColormap(["#e5e7eb", "#fbbf24", "#34d399"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(models)), labels=models)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    symbols = {0: "—", 1: "qualified", 2: "direct"}
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column, row, symbols[int(values[row, column])], ha="center", va="center", fontsize=8
            )
    ax.set_title("Which observables can both models predict through one definition?")
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.legend(
        handles=[
            Patch(color="#34d399", label="direct"),
            Patch(color="#fbbf24", label="available with a visible qualification"),
            Patch(color="#e5e7eb", label="unavailable"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.08),
        frameon=False,
        ncol=1,
    )
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_tree_matrix(path, matrix):
    plt = _plot_style()
    source_labels = ["L-Halo / Mini-Millennium", "VELOCIraptor / mini-SURFS"]
    model_labels = ["SAGE16 JAX", "SHARK Lagos23 JAX"]
    values = np.zeros((2, 2), dtype=int)
    text = np.empty((2, 2), dtype=object)
    for row, source in enumerate(("lhalo_binary", "shark_velociraptor_hdf5")):
        for column, model in enumerate(("SAGE16", "SHARK Lagos23")):
            result = matrix[(source, model)]
            if result.fully_runnable:
                values[row, column] = 2
                text[row, column] = "runnable"
            elif result.field_ready:
                values[row, column] = 1
                text[row, column] = "fields mapped\ndriver open"
            else:
                values[row, column] = 0
                text[row, column] = f"blocked\n{len(result.missing_fields)} fields"
    from matplotlib.colors import BoundaryNorm, ListedColormap

    figure, ax = plt.subplots(figsize=(7.0, 3.5))
    cmap = ListedColormap(["#fca5a5", "#fbbf24", "#34d399"])
    ax.imshow(values, cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), aspect="auto")
    ax.set_xticks(np.arange(2), labels=model_labels)
    ax.set_yticks(np.arange(2), labels=source_labels)
    for row in range(2):
        for column in range(2):
            ax.text(column, row, text[row, column], ha="center", va="center", fontweight="bold")
    ax.set_title("Can each JAX population driver run each native tree format today?")
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _capability_table(capabilities):
    lines = [
        "| Observable | SAGE16 | SHARK Lagos23 | Shared interpretation |",
        "| --- | --- | --- | --- |",
    ]
    specs = {spec.key: spec for spec in COMMON_OBSERVABLES}
    for key in specs:
        spec = specs[key]
        sage = capabilities["SAGE16"][key]
        shark = capabilities["SHARK Lagos23"][key]
        qualification = spec.qualification or "One canonical definition and unit convention."
        lines.append(f"| {spec.label} | {sage.status} | {shark.status} | {qualification} |")
    return "\n".join(lines)


def _observation_registry(observation):
    return {
        "stellar_mass_function": {
            "source": observation.source,
            "doi": observation.doi,
            "target_hubble_h": 0.7,
            "imf": "Chabrier",
            "status": "registered",
            "next_step": "ready for both canonical catalogues",
        },
        "cosmic_sfr_density": {
            "source": "multi-survey compilation in legacy SAGE plotting code",
            "status": "legacy_embedded",
            "next_step": "extract individual citations, units, selection, and covariance",
        },
        "gas_mass_functions": {
            "source": "Zwaan et al. (2005); Obreschkow & Rawlings (2009)",
            "status": "legacy_embedded",
            "next_step": "extract tables and keep HI/H2 versus total-cold definitions separate",
        },
        "gas_mass_metallicity": {
            "source": "Tremonti et al. relation in legacy SAGE plotting code",
            "status": "legacy_analytic_relation",
            "next_step": "record calibration, IMF transform, aperture, and total-Z conversion",
        },
        "black_hole_bulge": {
            "source": "Haring & Rix (2004) relation in legacy SAGE plotting code",
            "status": "legacy_analytic_relation",
            "next_step": "record scatter, IMF/bulge definition, sample selection, and covariance",
        },
        "stellar_mass_density_evolution": {
            "source": "Marchesini et al. (2009) compilation in legacy SAGE plotting code",
            "status": "legacy_embedded",
            "next_step": "extract redshift bins, IMF/h convention, uncertainties, and covariance",
        },
        "baryonic_tully_fisher": {
            "source": "legacy SAGE plotting module",
            "status": "not_audited",
            "next_step": "define observed and model velocity proxies before extracting data",
        },
        "quenched_fraction": {
            "source": "no durable shared observational product",
            "status": "unregistered",
            "next_step": "choose aperture, sSFR threshold, redshift, and sample selection",
        },
    }


def main():
    arguments = parse_arguments()
    output = arguments.output
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    sage = load_sage_comparison_catalogue(
        arguments.sage_catalogues,
        snapshot=arguments.sage_snapshot,
        hubble_h=arguments.sage_hubble_h,
        effective_volume_mpc_over_h_cubed=arguments.sage_box_size**3,
        redshift=arguments.sage_redshift,
        dataset="Mini-Millennium",
    )
    native_shark = load_shark_catalogue(arguments.shark_catalogue)
    shark = shark_comparison_catalogue(
        native_shark,
        dataset="mini-SURFS public CI tree",
        snapshot=arguments.shark_snapshot,
    )

    capabilities = {
        catalogue.model: {item.key: item for item in observable_capabilities(catalogue)}
        for catalogue in (sage, shark)
    }
    observation = load_baldry2008_stellar_mass_function(arguments.observation, hubble_h=0.7)
    observation_registry = _observation_registry(observation)
    common_figure = assets / "native-common-observables.svg"
    arrays = plot_common_observables(common_figure, sage, shark, observation)
    capability_figure = assets / "observable-capability-matrix.svg"
    plot_capability_matrix(capability_figure, capabilities)

    partition = open_lhalo_partition(arguments.lhalo_tree)
    lhalo = canonical_tree_from_lhalo(
        partition.read_tree(arguments.lhalo_tree_index),
        snapshot_timing(load_scale_factors(arguments.scale_factors)),
        source_path=arguments.lhalo_tree,
        tree_index=arguments.lhalo_tree_index,
        particle_mass_1e10_msun_over_h=arguments.particle_mass,
    )
    shark_tree_data = load_shark_tree(arguments.shark_tree)
    shark_tree_index = arguments.shark_tree_index
    if shark_tree_index < 0:
        shark_tree_index = int(np.argmax(shark_tree_data.number_of_nodes))
    shark_tree = canonical_tree_from_shark(shark_tree_data, shark_tree_index)
    tree_matrix = {}
    for tree in (lhalo, shark_tree):
        for requirements in (SAGE16_TREE_REQUIREMENTS, SHARK_LAGOS23_TREE_REQUIREMENTS):
            tree_matrix[(tree.source_format, requirements.model)] = assess_tree_compatibility(
                tree, requirements
            )
    tree_figure = assets / "tree-portability-matrix.svg"
    plot_tree_matrix(tree_figure, tree_matrix)

    shared = [
        spec.key
        for spec in COMMON_OBSERVABLES
        if all(capabilities[model][spec.key].status != "unavailable" for model in capabilities)
    ]
    qualified = [
        key
        for key in shared
        if any(capabilities[model][key].status == "qualified" for model in capabilities)
    ]
    direct = [key for key in shared if key not in qualified]
    foreign_tree_runs = sum(
        result.fully_runnable for result in tree_matrix.values() if not result.native_run
    )

    array_path = assets / "native-common-observables.npz"
    np.savez_compressed(
        array_path,
        smf_edges=arrays["smf_edges"],
        sage_smf=arrays["sage_smf"].number_density,
        sage_smf_counts=arrays["sage_smf"].counts,
        shark_smf=arrays["shark_smf"].number_density,
        shark_smf_counts=arrays["shark_smf"].counts,
        relation_edges=arrays["relation_edges"],
        sage_quenched=arrays["sage_quenched"].fraction,
        shark_quenched=arrays["shark_quenched"].fraction,
        observation_mass=observation.coordinate,
        observation_smf=observation.values,
    )
    audit = {
        "schema": "mimic-jax-model-comparison-audit/v1",
        "catalogues": {
            catalogue.model: {
                "dataset": catalogue.dataset,
                "snapshot": catalogue.snapshot,
                "redshift": catalogue.redshift,
                "galaxy_count": catalogue.galaxy_count,
                "hubble_h": catalogue.hubble_h,
                "effective_volume_mpc_over_h_cubed": catalogue.effective_volume_mpc_over_h_cubed,
                "fields": {
                    name: _serialise_field(field) for name, field in catalogue.fields.items()
                },
                "unavailable_fields": dict(catalogue.unavailable_fields),
                "observable_capabilities": {
                    key: asdict(value) for key, value in capabilities[catalogue.model].items()
                },
            }
            for catalogue in (sage, shark)
        },
        "tree_compatibility": {
            f"{source}__{model.replace(' ', '_')}": _serialise_tree_compatibility(result)
            for (source, model), result in tree_matrix.items()
        },
        "shared_observables": shared,
        "direct_shared_observables": direct,
        "qualified_shared_observables": qualified,
        "observation_registry": observation_registry,
        "claims": {
            "native_catalogue_common_projection": True,
            "same_forcing_population_comparison": False,
            "sage_on_shark_tree": False,
            "shark_on_lhalo_tree": False,
            "native_outputs_isolate_model_physics": False,
        },
    }
    audit_path = assets / "model-comparison-audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sage_sfr = catalogue_cosmic_sfr_density(sage, minimum_stellar_mass_msun=1.0e8)
    shark_sfr = catalogue_cosmic_sfr_density(shark, minimum_stellar_mass_msun=1.0e8)
    health = (
        Diagnostic(
            key="canonical_catalogues",
            title="Canonical catalogue projection",
            status=DiagnosticStatus.PASSED,
            summary=(
                f"{sage.galaxy_count:,} SAGE16 and {shark.galaxy_count:,} SHARK rows were "
                "projected into explicit physical units with field provenance."
            ),
            metrics=(
                ScalarMetric("shared_observables", "Shared observable definitions", len(shared)),
                ScalarMetric("direct_observables", "Unqualified shared definitions", len(direct)),
                ScalarMetric(
                    "qualified_observables", "Qualified shared definitions", len(qualified)
                ),
            ),
        ),
        Diagnostic(
            key="foreign_tree_execution",
            title="Cross-tree population execution",
            status=DiagnosticStatus.WARNING,
            summary=(
                f"{foreign_tree_runs} of 2 foreign-tree model paths are runnable today. "
                "Both formats now project into one audited forcing contract, but required "
                "halo conventions and population drivers remain open."
            ),
            tolerance="Both foreign-tree paths must have complete forcing semantics and a topology-owning JAX driver.",
        ),
        Diagnostic(
            key="shark_population_driver",
            title="Independent SHARK JAX topology driver",
            status=DiagnosticStatus.WARNING,
            summary=(
                "The native SHARK population physics shadow replay is fully evaluated, but "
                "native SHARK still owns variable-cardinality topology and event scheduling."
            ),
        ),
        Diagnostic(
            key="observation_registry",
            title="Shared observation comparison",
            status=DiagnosticStatus.WARNING,
            summary=(
                "The Baldry et al. (2008) stellar mass function has a durable shared loader; "
                "other legacy SAGE observation compilations are not yet model-neutral datasets."
            ),
        ),
    )

    metrics = (
        ComparisonMetric.from_values(
            key="galaxy_count",
            label="z=0 catalogue rows",
            baseline=sage.galaxy_count,
            candidate=shark.galaxy_count,
            interpretation=(
                "The counts are not a model-only comparison because the simulations, volumes, "
                "resolution, cosmology, and tree construction differ."
            ),
        ),
        ComparisonMetric.from_values(
            key="cosmic_sfr_density",
            label="SFR density above 1e8 Msun",
            baseline=sage_sfr,
            candidate=shark_sfr,
            unit="Msun yr^-1 Mpc^-3",
            interpretation=(
                "This verifies one common reduction and selection; it must not be interpreted "
                "as an isolated SAGE--SHARK physics difference until same-forcing runs exist."
            ),
        ),
    )

    capability_body = _capability_table(capabilities)
    observation_lines = [
        "| Observable | Current status | Source / next gate |",
        "| --- | --- | --- |",
    ]
    for key, entry in observation_registry.items():
        observation_lines.append(
            f"| {key.replace('_', ' ')} | {entry['status']} | {entry['source']}; "
            f"{entry['next_step']} |"
        )
    tree_lines = [
        "| Source tree | Target model | Field contract | JAX population driver | Missing fields |",
        "| --- | --- | --- | --- | --- |",
    ]
    for (source, model), result in tree_matrix.items():
        missing = ", ".join(f"`{field}`" for field in result.missing_fields) or "none"
        tree_lines.append(
            f"| {source} | {model} | {'ready' if result.field_ready else 'incomplete'} | "
            f"{'ready' if result.population_driver_ready else 'open'} | {missing} |"
        )
    finding_body = (
        f"- **{len(shared)} of {len(COMMON_OBSERVABLES)}** reviewed observable definitions are "
        f"available for both catalogues; {len(qualified)} carry a visible physical qualification.\n"
        f"- **{len(direct)}** are direct under the current definition.\n"
        f"- **{foreign_tree_runs} of 2** foreign-tree execution paths are runnable: common data "
        "structures are no substitute for missing halo semantics or topology drivers.\n"
        "- **1 observational product** is currently registered in the shared layer; expanding "
        "this is now a data/provenance task rather than a model-specific plotting task."
    )
    report = ComparisonReport(
        comparison_id="sage16-shark-interoperability-audit",
        title="Can SAGE16 and SHARK be compared without hidden conventions?",
        summary=(
            "A science-facing interoperability audit: both native catalogues now feed the same "
            "observable definitions, while merger-tree portability is measured honestly rather "
            "than inferred from similar field names."
        ),
        baseline=ComparedRun(
            key="sage16",
            label="SAGE16 / Mini-Millennium",
            run_id="mini-millennium-sage16-science-program",
            report_path="../mini-millennium-sage16-science-program/index.md",
        ),
        candidate=ComparedRun(
            key="shark",
            label="SHARK Lagos23 / mini-SURFS",
            run_id="shark-continuous-foundation",
            report_path="../shark-continuous-foundation/index.md",
        ),
        metrics=metrics,
        provenance=capture_provenance(
            repository=REPOSITORY,
            command=sys.argv,
            configuration_paths=(REPOSITORY / "docs/dev/MIMIC-JAX-SHARK-INTEGRATION-PLAN.md",),
            input_paths=tuple(arguments.sage_catalogues)
            + (
                arguments.shark_catalogue,
                arguments.lhalo_tree,
                arguments.shark_tree,
                arguments.scale_factors,
                arguments.observation,
            ),
            random_seeds={"SHARK native reference": native_shark.seed},
        ),
        health=health,
        sections=(
            ReportSection(
                key="findings",
                title="What the audit establishes",
                summary="The useful overlap is already substantial, but common forcing is not yet solved.",
                body=finding_body,
            ),
            ReportSection(
                key="native_observables",
                title="Can both catalogues answer the same questions?",
                summary=(
                    "Yes for the principal mass, SFR, gas, metallicity, BH, and halo relations. "
                    "The figure uses exactly the same reductions on both native catalogues."
                ),
                body=(
                    "The curves are deliberately **not** presented as a controlled model comparison: "
                    "Mini-Millennium and mini-SURFS differ in tree finder, cosmology, volume, and "
                    "resolution. Their purpose here is to prove that the observable boundary no "
                    "longer changes code between models."
                ),
                artifacts=(
                    Artifact(
                        "native_common_observables",
                        "Shared observables on native catalogues",
                        "assets/native-common-observables.svg",
                        "image/svg+xml",
                        "figure",
                        "Identical selections, bins, physical units, and zero handling; native forcing differs.",
                    ),
                    Artifact(
                        "native_common_arrays",
                        "Shared observable arrays",
                        "assets/native-common-observables.npz",
                        "application/octet-stream",
                        "data",
                        "Compact numerical arrays behind the figure.",
                    ),
                ),
            ),
            ReportSection(
                key="observable_contract",
                title="What overlaps—and what remains model-specific?",
                summary=(
                    "Unavailable quantities remain unavailable; mimic-jax does not invent an HI/H2 "
                    "split for SAGE or relabel different disk-radius definitions as identical."
                ),
                body=capability_body,
                artifacts=(
                    Artifact(
                        "observable_capability_matrix",
                        "Observable capability matrix",
                        "assets/observable-capability-matrix.svg",
                        "image/svg+xml",
                        "figure",
                        "Direct, qualified, and unavailable outputs under one reviewed contract.",
                    ),
                ),
            ),
            ReportSection(
                key="tree_portability",
                title="Can either model run the other model's trees?",
                summary=(
                    "Not yet. Both native formats now project into a canonical tree-local forcing "
                    "record, but cross-running still lacks scientifically validated conventions and drivers."
                ),
                body="\n".join(tree_lines),
                artifacts=(
                    Artifact(
                        "tree_portability_matrix",
                        "Tree portability matrix",
                        "assets/tree-portability-matrix.svg",
                        "image/svg+xml",
                        "figure",
                        "Field completeness and topology-driver readiness are evaluated separately.",
                    ),
                ),
                notes=(
                    "For SAGE on SHARK trees, the L-Halo first-progenitor ordering, velocity dispersion, virial radius, and Spin-vector convention remain unresolved.",
                    "For SHARK on L-Halo trees, concentration, interpolation/DHalo flags, main-progenitor semantics, and halo-spin/size conventions remain unresolved.",
                    "SHARK's exhaustive JAX RHS replay validates physics evaluations; it does not yet replace native SHARK's population topology/event scheduler.",
                ),
            ),
            ReportSection(
                key="observations",
                title="Can observations be compared once rather than model by model?",
                summary=(
                    "The Baldry et al. stellar mass function is the first shared observational "
                    "product. Other legacy arrays must be extracted with citations, IMF, aperture, "
                    "h, and calibration conventions before becoming common data."
                ),
                body=(
                    "The plotted Baldry curve uses one declared target convention (`h=0.7`, "
                    "Chabrier IMF) for both models. A model-specific cosmology must not silently "
                    "move the observation between panels.\n\n" + "\n".join(observation_lines)
                ),
            ),
            ReportSection(
                key="machine_contract",
                title="Reusable products for the next model",
                summary=(
                    "The complete field provenance, observable capabilities, tree blockers, and "
                    "claim boundaries are machine-readable. A third model can implement the same "
                    "adapters instead of adding another pairwise comparison path."
                ),
                artifacts=(
                    Artifact(
                        "model_comparison_audit",
                        "Model comparison audit",
                        "assets/model-comparison-audit.json",
                        "application/json",
                        "data",
                        "Canonical fields, observable support, tree readiness, and explicit claims.",
                    ),
                ),
            ),
        ),
        links=(
            ReportLink("SHARK implementation guide", "../../docs/shark_lagos23.md"),
            ReportLink(
                "SAGE--SHARK integration plan",
                "../../docs/dev/MIMIC-JAX-SHARK-INTEGRATION-PLAN.md",
            ),
        ),
    )
    written = write_report(report, output)
    print(written.markdown_path)


if __name__ == "__main__":
    main()
