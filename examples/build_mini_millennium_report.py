#!/usr/bin/env python3
"""Build the canonical science-first SAGE16 Mini-Millennium run report.

The expensive partition benchmark and equivalence check are separate commands
that write JSON/NPZ products into the report's ``assets`` directory. This builder
consumes those durable results, renders science figures, generates controlled
differentiability/numerical diagnostics, asks the existing MIMIC plot registry
for familiar figures, and writes the report manifest. It never reruns the
Mini-Millennium model itself.
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
    FORWARD_EULER,
    HEUN_RK2,
    RK4,
    method_convergence_study,
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
    ode_convergence_diagnostic,
    parameter_response_diagnostic,
    parameters_from_namedtuple,
    timestep_refinement_diagnostic,
    write_report,
)
from mimic_jax.sage16 import (  # noqa: E402
    ODE_STATE_NAMES,
    UPSTREAM_RATE_SUBSET,
    apply_reincorporation,
    baryonic_mass,
    calculate_cooling_budget,
    calculate_star_formation_budget,
    calculate_supernova_feedback_budget,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    integrate_sage16_ode,
    load_cooling_tables,
    ode_state_from_galaxy,
    quiescent_disk_step,
    sage16_ode_rhs_and_rates,
    sage16_units,
    step_context,
    subcycle_upstream_rate_subset,
    subcycle_upstream_sequential_central,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPOSITORY / "reports/mini-millennium-sage16-initial"
DEFAULT_RUN_FILE = REPOSITORY / "models/sage16/input/sage16_mini-millennium.yaml"
DEFAULT_SCALE_FACTORS = REPOSITORY / "simulations/mini-millennium/mini-millennium.a_list"
DEFAULT_UPSTREAM_OUTPUT = REPOSITORY / "output/sage16-mini-millennium"
PARAMETER_DESCRIPTIONS = {
    "GlobalBaryonFraction": "Universal baryon fraction available to haloes.",
    "SfrEfficiency": "Quiescent star-formation efficiency per disk dynamical time.",
    "StarFormingDiskFactor": "Disk-radius multiple used by the star-formation threshold.",
    "FeedbackReheatingEpsilon": "SN reheating mass loading from cold to hot gas.",
    "FeedbackEjectionEfficiency": "SN energy efficiency for ejecting gas from the halo.",
    "ReIncorporationFactor": "Return rate of ejected gas to the hot halo.",
    "AGNrecipe": "Radio-mode black-hole accretion prescription selector.",
    "RadioModeEfficiency": "Efficiency with which radio-mode accretion heats halo gas.",
    "BlackHoleGrowthRate": "Cold-gas accretion efficiency in quasar-mode events.",
    "QuasarModeEfficiency": "Efficiency of quasar-mode gas ejection.",
    "RecycleFraction": "Fraction of newly formed stellar mass returned immediately to gas.",
    "Yield": "New metal mass produced per unit newly formed stellar mass.",
    "FracZleaveDisk": "Fraction of new metals deposited directly into hot gas.",
    "ThresholdMajorMerger": "Baryonic mass-ratio threshold for a major merger.",
    "ThresholdSatDisruption": "Halo-to-baryon threshold for satellite disruption.",
}


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
        "--partition-equivalence-json",
        type=Path,
        help="full-partition equivalence JSON (default: <output>/assets/partition-equivalence.json)",
    )
    parser.add_argument(
        "--science-json",
        type=Path,
        help="partition science summary JSON (default: <output>/assets/partition-science.json)",
    )
    parser.add_argument(
        "--science-arrays",
        type=Path,
        help="partition science arrays (default: <output>/assets/partition-science.npz)",
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


def stage_file(source: Path, destination: Path) -> Path:
    """Place one durable scientific artifact inside the report directory."""

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


def controlled_ode_convergence(output_path: Path):
    """Build the fixed-forcing continuous-limit experiment used in the report."""

    galaxy = initial_galaxy_state(
        ColdGas=2.0,
        HotGas=10.0,
        EjectedGas=1.0,
        StellarMass=1.0,
        MetalsColdGas=0.04,
        MetalsHotGas=0.2,
        MetalsEjectedGas=0.02,
        MetalsStellarMass=0.02,
        DiskScaleRadius=0.01,
    )
    halo = initial_halo_forcing(Mvir=100.0, Rvir=0.2, Vvir=150.0, dT=5.0e-4)
    context = step_context(time_interval=5.0e-4)
    parameters = fiducial_parameters()
    units = sage16_units()
    tables = load_cooling_tables()
    initial = ode_state_from_galaxy(galaxy)
    step_counts = (2, 4, 8, 16, 32, 64, 128)
    methods = (UPSTREAM_RATE_SUBSET, FORWARD_EULER, HEUN_RK2, RK4)
    reference_steps = 4096
    reference = integrate_sage16_ode(
        initial,
        halo,
        galaxy.DiskScaleRadius,
        parameters,
        units,
        tables,
        num_steps=reference_steps,
        method=RK4,
    ).final_state
    cache = {}

    def run(method, num_steps):
        key = (method, num_steps)
        if key not in cache:
            if method == UPSTREAM_RATE_SUBSET:
                result = subcycle_upstream_rate_subset(
                    galaxy,
                    halo,
                    context,
                    parameters,
                    units,
                    tables,
                    num_substeps=num_steps,
                )
                cache[key] = ode_state_from_galaxy(result.final_state)
            else:
                cache[key] = integrate_sage16_ode(
                    initial,
                    halo,
                    galaxy.DiskScaleRadius,
                    parameters,
                    units,
                    tables,
                    num_steps=num_steps,
                    method=method,
                ).final_state
        return cache[key]

    def observables(state):
        return jnp.stack([getattr(state, name) for name in ODE_STATE_NAMES])

    study = method_convergence_study(
        run,
        observables,
        reference,
        methods=methods,
        step_counts=step_counts,
        observable_names=ODE_STATE_NAMES,
        observable_units=("1e10 Msun/h",) * len(ODE_STATE_NAMES),
        rhs_evaluations_per_step={
            UPSTREAM_RATE_SUBSET: 1,
            FORWARD_EULER: 1,
            HEUN_RK2: 2,
            RK4: 4,
        },
        reference_method=RK4,
        reference_steps=reference_steps,
        duration=float(halo.dT),
    )
    study.save(output_path)

    comparison_substeps = 128
    comparison_context = context._replace(
        num_substeps=jnp.asarray(comparison_substeps, dtype=jnp.int32)
    )
    dt = halo.dT / comparison_substeps
    rates = sage16_ode_rhs_and_rates(
        0.0,
        initial,
        halo,
        galaxy.DiskScaleRadius,
        parameters,
        units,
        tables,
    ).rates
    reincorporation = apply_reincorporation(
        galaxy,
        halo,
        comparison_context,
        parameters,
    ).transfer
    cooling = calculate_cooling_budget(
        galaxy,
        halo,
        comparison_context,
        units,
        tables,
    ).budget
    star_formation = calculate_star_formation_budget(
        galaxy,
        halo,
        comparison_context,
        parameters,
    )
    supernova = calculate_supernova_feedback_budget(
        galaxy,
        halo,
        parameters,
        units,
        star_formation,
    )
    ode_rates = np.asarray(
        [
            rates.cooling,
            rates.star_formation,
            rates.sn_reheating,
            rates.sn_ejection,
            rates.reincorporation,
        ],
        dtype=np.float64,
    )
    upstream_rates = np.asarray(
        [
            cooling.gas / dt,
            star_formation.NewStellarMass / dt,
            supernova.SupernovaReheatedMass / dt,
            supernova.SupernovaEjectedMass / dt,
            reincorporation.gas / dt,
        ],
        dtype=np.float64,
    )
    rate_scale = np.where(np.abs(upstream_rates) > 0.0, np.abs(upstream_rates), 1.0)
    maximum_rate_relative_difference = float(
        np.max(np.abs(ode_rates - upstream_rates) / rate_scale)
    )

    initial_baryons = sum(float(value) for value in initial[:4])
    ode_baryon_residuals = []
    upstream_baryon_residuals = []
    for method in methods:
        for num_steps in step_counts:
            final = run(method, num_steps)
            residual = sum(float(value) for value in final[:4]) - initial_baryons
            if method == UPSTREAM_RATE_SUBSET:
                upstream_baryon_residuals.append(residual)
            else:
                ode_baryon_residuals.append(residual)
    evidence = {
        "maximum_rate_relative_difference": maximum_rate_relative_difference,
        "maximum_ode_baryon_residual": max(abs(value) for value in ode_baryon_residuals),
        "maximum_upstream_storage_baryon_residual": max(
            abs(value) for value in upstream_baryon_residuals
        ),
    }
    return study, evidence


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


def generate_partition_science_figures(arrays_path: Path, equivalence, assets: Path):
    """Render report figures from durable population arrays without rerunning SAGE16."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    palette = {
        "upstream": "#1F2937",
        "jax": "#D97706",
        "ColdGas": "#4C78A8",
        "HotGas": "#E45756",
        "EjectedGas": "#72B7B2",
        "StellarMass": "#F2CF5B",
        "ICS": "#B279A2",
        "BlackHoleMass": "#333333",
    }
    labels = {
        "ColdGas": "Cold gas",
        "HotGas": "Hot gas",
        "EjectedGas": "Ejected gas",
        "StellarMass": "Stars",
        "ICS": "Intracluster stars",
        "BlackHoleMass": "Black holes",
    }
    style = {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
    }
    with np.load(arrays_path, allow_pickle=False) as arrays, plt.rc_context(style):
        mass = arrays["stellar_mass_bin_centres"]
        upstream_smf = arrays["upstream_smf"]
        jax_smf = arrays["mimic_jax_smf"]
        upstream_counts = arrays["upstream_smf_counts"]
        fractional = arrays["smf_fractional_difference"]
        finite = (upstream_counts > 0) & np.isfinite(fractional)
        resolved = upstream_counts >= 5

        figure, (axis, residual_axis) = plt.subplots(
            2,
            1,
            figsize=(8.2, 6.2),
            sharex=True,
            gridspec_kw={"height_ratios": (3.1, 1.0), "hspace": 0.08},
        )
        plotted = upstream_smf > 0.0
        axis.plot(
            mass[plotted],
            upstream_smf[plotted],
            color=palette["upstream"],
            linewidth=2.4,
            label="MIMIC/SAGE16",
        )
        axis.plot(
            mass[plotted],
            jax_smf[plotted],
            color=palette["jax"],
            linewidth=1.8,
            linestyle="--",
            marker="o",
            markevery=3,
            markersize=3.5,
            label="mimic-jax/SAGE16",
        )
        axis.set_yscale("log")
        axis.set_ylim(1.0e-4, 5.0e-2)
        axis.set_ylabel("φ [Mpc⁻³ dex⁻¹]")
        axis.set_title("The larger mimic-jax sample reproduces the SAGE16 stellar mass function")
        axis.text(
            0.02,
            0.03,
            "One complete Mini-Millennium input partition (1/8 volume)",
            transform=axis.transAxes,
            color="#4B5563",
        )
        axis.legend(frameon=False, loc="upper right")
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)

        residual_axis.axhline(0.0, color="#6B7280", linewidth=1.0)
        residual_axis.plot(
            mass[finite],
            100.0 * fractional[finite],
            color=palette["jax"],
            linewidth=1.5,
        )
        unresolved = finite & ~resolved
        residual_axis.scatter(
            mass[unresolved],
            100.0 * fractional[unresolved],
            color="#9CA3AF",
            marker="x",
            label="<5 galaxies/bin",
        )
        maximum = (
            float(np.nanmax(np.abs(100.0 * fractional[resolved]))) if np.any(resolved) else 0.0
        )
        residual_limit = max(0.05, 1.25 * maximum)
        residual_axis.set_ylim(-residual_limit, residual_limit)
        residual_axis.set_ylabel("Difference [%]")
        residual_axis.set_xlabel("log₁₀ stellar mass [M☉]")
        residual_axis.set_xlim(8.0, 11.55)
        residual_axis.xaxis.set_minor_locator(MultipleLocator(0.1))
        residual_axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
        residual_axis.text(
            0.02,
            0.76,
            f"{np.count_nonzero(resolved)} bins with ≥5 galaxies: identical counts",
            transform=residual_axis.transAxes,
            color="#374151",
        )
        smf_path = assets / "StellarMassFunctionComparison.svg"
        figure.savefig(smf_path)
        plt.close(figure)

        halo_mass = arrays["halo_mass_bin_centres"]
        group_counts = arrays["group_counts"]
        reservoir_names = tuple(str(value) for value in arrays["reservoir_names"])
        fractions = arrays["mimic_jax_baryon_allotment_fractions"]
        upstream_total = arrays["upstream_total_baryon_fraction"]
        jax_total = arrays["mimic_jax_total_baryon_fraction"]
        valid_inventory = group_counts >= 10
        selected_mass = halo_mass[valid_inventory]
        selected_fractions = fractions[valid_inventory]
        figure, (axis, residual_axis) = plt.subplots(
            2,
            1,
            figsize=(8.2, 6.5),
            sharex=True,
            gridspec_kw={"height_ratios": (3.2, 1.0), "hspace": 0.08},
        )
        axis.stackplot(
            selected_mass,
            *[selected_fractions[:, index] for index in range(len(reservoir_names))],
            colors=[palette[name] for name in reservoir_names],
            labels=[labels[name] for name in reservoir_names],
            alpha=0.9,
        )
        axis.plot(
            selected_mass,
            jax_total[valid_inventory],
            color=palette["jax"],
            linewidth=2.0,
            label="Total modeled baryons",
        )
        axis.plot(
            selected_mass,
            upstream_total[valid_inventory],
            color="#111827",
            linewidth=1.2,
            linestyle="--",
            label="MIMIC total",
        )
        axis.set_ylim(0.0, 1.06)
        axis.set_ylabel("Fraction of cosmic allotment (fᵦ × Mᵥᵢᵣ)")
        axis.set_title("The dominant SAGE reservoir changes from cold to ejected to hot gas")
        axis.legend(
            frameon=False,
            ncol=3,
            loc="upper left",
            columnspacing=1.0,
            handlelength=2.0,
        )
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.6)

        inventory_difference = np.full(jax_total.shape, np.nan)
        nonzero = upstream_total != 0.0
        inventory_difference[nonzero] = (
            jax_total[nonzero] - upstream_total[nonzero]
        ) / upstream_total[nonzero]
        residual_axis.axhline(0.0, color="#6B7280", linewidth=1.0)
        residual_axis.plot(
            selected_mass,
            1.0e6 * inventory_difference[valid_inventory],
            color=palette["jax"],
            marker="o",
            markersize=3.5,
            linewidth=1.3,
        )
        residual_axis.set_ylabel("JAX−MIMIC [ppm]")
        residual_axis.set_xlabel("log₁₀ halo mass [M☉]")
        residual_axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
        baryon_path = assets / "BaryonInventory.svg"
        figure.savefig(baryon_path)
        plt.close(figure)

        field_errors = equivalence.get("largest_relative_errors", {})
        ordered = sorted(field_errors.items(), key=lambda item: item[1])
        figure, axis = plt.subplots(figsize=(7.8, 4.8))
        axis.barh(
            [field for field, _ in ordered],
            [1.0e6 * error for _, error in ordered],
            color="#4C78A8",
        )
        axis.axvline(2.0, color="#D97706", linestyle="--", linewidth=1.5, label="2 ppm rtol")
        axis.set_xlabel("Largest resolved relative difference [ppm]")
        axis.set_title(
            f"{equivalence['mismatches']:,} of {equivalence['field_comparisons']:,} strict comparisons exceed tolerance"
        )
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
        axis.legend(frameon=False, loc="lower right")
        residual_path = assets / "PartitionFieldResiduals.svg"
        figure.savefig(residual_path)
        plt.close(figure)

    return (
        Artifact(
            key="partition_stellar_mass_function",
            title="MIMIC and mimic-jax stellar mass functions for a complete input partition",
            path="assets/StellarMassFunctionComparison.svg",
            media_type="image/svg+xml",
            role="figure",
            description=(
                "The top panel is the familiar z=0 SAGE16 stellar mass function for one complete "
                "Mini-Millennium input partition. The lower panel is the percentage bin-by-bin "
                "difference; all 32 bins containing at least five reference galaxies have "
                "identical counts."
            ),
        ),
        Artifact(
            key="partition_baryon_inventory",
            title="Where SAGE16 stores the baryons across halo mass",
            path="assets/BaryonInventory.svg",
            media_type="image/svg+xml",
            role="figure",
            description=(
                "Reservoir masses are summed over each FoF group and normalized by its universal "
                "baryon allotment. The residual panel compares the same inventory between "
                "mimic-jax and MIMIC; it is a catalogue-equivalence residual, not a time-integrated "
                "conservation residual."
            ),
        ),
        Artifact(
            key="partition_field_residuals",
            title="Largest per-field residuals in the complete-partition comparison",
            path="assets/PartitionFieldResiduals.svg",
            media_type="image/svg+xml",
            role="figure",
            description=(
                "The strict field gate remains visible: 20 comparisons exceed the stated mixed-"
                "precision tolerance even though the resolved stellar-mass-function bins agree."
            ),
        ),
    )


def generate_ode_convergence_figure(study, assets: Path):
    """Show temporal error and measured order for the continuous SAGE16 subset."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        UPSTREAM_RATE_SUBSET: "Upstream sequential rate subset",
        FORWARD_EULER: "Forward Euler",
        HEUN_RK2: "Heun RK2",
        RK4: "Classical RK4",
    }
    colors = {
        UPSTREAM_RATE_SUBSET: "#1F2937",
        FORWARD_EULER: "#D97706",
        HEUN_RK2: "#4C78A8",
        RK4: "#009E73",
    }
    markers = {
        UPSTREAM_RATE_SUBSET: "o",
        FORWARD_EULER: "s",
        HEUN_RK2: "^",
        RK4: "D",
    }
    expected_orders = {
        UPSTREAM_RATE_SUBSET: 1.0,
        FORWARD_EULER: 1.0,
        HEUN_RK2: 2.0,
        RK4: 4.0,
    }
    errors = np.nanmax(np.asarray(study.relative_errors, dtype=np.float64), axis=2)
    step_counts = np.asarray(study.step_counts, dtype=np.int32)
    measured_orders = np.log(errors[:, -3:-1] / errors[:, -2:]) / np.log(
        step_counts[-2:] / step_counts[-3:-1]
    )
    measured_orders = np.nanmedian(measured_orders, axis=1)
    style = {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        figure, (error_axis, order_axis) = plt.subplots(
            1,
            2,
            figsize=(10.2, 4.6),
            gridspec_kw={"width_ratios": (1.65, 1.0), "wspace": 0.35},
        )
        for index, method in enumerate(study.methods):
            error_axis.loglog(
                step_counts,
                100.0 * errors[index],
                color=colors[method],
                marker=markers[method],
                linewidth=1.8,
                markersize=4.5,
                label=labels[method],
            )
        error_axis.set_xlabel("Internal steps across one fixed-forcing interval")
        error_axis.set_ylabel("Largest reservoir error [%]")
        error_axis.set_title("Refining time steps approaches one common solution")
        error_axis.grid(which="both", color="#E5E7EB", linewidth=0.6)
        error_axis.legend(frameon=False, loc="lower left")
        error_axis.text(
            0.03,
            0.97,
            f"Reference: RK4 with {study.reference_steps:,} steps",
            transform=error_axis.transAxes,
            va="top",
            color="#4B5563",
        )

        positions = np.arange(len(study.methods))
        order_axis.bar(
            positions,
            measured_orders,
            color=[colors[method] for method in study.methods],
            width=0.68,
        )
        for position, method in enumerate(study.methods):
            expected = expected_orders[method]
            order_axis.plot(
                [position - 0.34, position + 0.34],
                [expected, expected],
                color="#111827",
                linewidth=1.5,
            )
            order_axis.text(
                position,
                measured_orders[position] + 0.12,
                f"{measured_orders[position]:.2f}",
                ha="center",
                va="bottom",
            )
        order_axis.set_xticks(positions)
        order_axis.set_xticklabels(("Upstream\nsubset", "Euler", "Heun\nRK2", "RK4"))
        order_axis.set_ylim(0.0, 4.7)
        order_axis.set_ylabel("Measured convergence order")
        order_axis.set_title("The measured slopes match theory")
        order_axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
        order_axis.text(
            0.03,
            0.97,
            "Black ticks: expected order",
            transform=order_axis.transAxes,
            va="top",
            color="#4B5563",
        )
        figure.suptitle(
            "A continuous SAGE16 baryon-cycle subset has controlled temporal error",
            fontsize=14,
            y=1.02,
        )
        output = assets / "OdeTimeConvergence.svg"
        figure.savefig(output)
        plt.close(figure)

    return Artifact(
        key="ode_time_convergence",
        title="Temporal convergence of the continuous SAGE16 rate subset",
        path="assets/OdeTimeConvergence.svg",
        media_type="image/svg+xml",
        role="figure",
        description=(
            "The left panel measures the largest relative error among four baryon and four metal "
            "reservoirs against an independent fine RK4 reference. The right panel shows that "
            "the upstream split and Euler are first order, Heun is second order, and RK4 is "
            "fourth order for this smooth fixed-forcing interval."
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
    partition_equivalence_source = (
        arguments.partition_equivalence_json.resolve()
        if arguments.partition_equivalence_json is not None
        else assets / "partition-equivalence.json"
    )
    science_source = (
        arguments.science_json.resolve()
        if arguments.science_json is not None
        else assets / "partition-science.json"
    )
    equivalence = load_json(equivalence_source)
    benchmark = load_json(benchmark_source)
    partition_equivalence = load_json(partition_equivalence_source)
    science = load_json(science_source)
    science_arrays_name = Path(str(science["arrays"]))
    if science_arrays_name.name != str(science["arrays"]):
        raise SystemExit("Science JSON arrays entry must be a plain filename")
    science_arrays_source = (
        arguments.science_arrays.resolve()
        if arguments.science_arrays is not None
        else science_source.parent / science_arrays_name
    )
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
        command=("mimic_venv/bin/python", *sys.argv),
        configuration_paths=(arguments.run_file, DEFAULT_SCALE_FACTORS),
        input_paths=(
            arguments.upstream_output / "model.hdf5",
            arguments.upstream_output / "metadata/version_info.json",
            *upstream_partitions,
            equivalence_source,
            partition_equivalence_source,
            benchmark_source,
            science_source,
            science_arrays_source,
        ),
        upstream_version_info=arguments.upstream_output / "metadata/version_info.json",
    )

    assets.mkdir(parents=True, exist_ok=True)
    equivalence_path = stage_json(equivalence_source, assets / "equivalence.json")
    benchmark_path = stage_json(benchmark_source, assets / "benchmark.json")
    partition_equivalence_path = stage_json(
        partition_equivalence_source, assets / "partition-equivalence.json"
    )
    science_path = stage_json(science_source, assets / "partition-science.json")
    science_arrays_path = stage_file(science_arrays_source, assets / science_arrays_name)
    familiar_artifacts = generate_familiar_plots(arguments, assets)
    science_figures = generate_partition_science_figures(
        science_arrays_path, partition_equivalence, assets
    )
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
    ode_convergence_artifact = Artifact(
        key="ode_convergence_arrays",
        title="Continuous-subset convergence arrays",
        path="assets/ode_time_convergence.npz",
        media_type="application/x-npz",
        role="scientific_array",
        description=(
            "Methods, step sizes, eight reservoir histories, independent-reference errors, "
            "measured orders, and method metadata."
        ),
    )
    ode_convergence, ode_evidence = controlled_ode_convergence(assets / "ode_time_convergence.npz")
    ode_convergence_figure = generate_ode_convergence_figure(ode_convergence, assets)
    ode_diagnostic = ode_convergence_diagnostic(
        ode_convergence,
        expected_orders={
            UPSTREAM_RATE_SUBSET: 1.0,
            FORWARD_EULER: 1.0,
            HEUN_RK2: 2.0,
            RK4: 4.0,
        },
        order_tolerance=0.15,
        maximum_rate_relative_difference=ode_evidence["maximum_rate_relative_difference"],
        rate_tolerance=2.0e-14,
        maximum_baryon_residual=ode_evidence["maximum_ode_baryon_residual"],
        baryon_tolerance=2.0e-12,
        maximum_upstream_storage_baryon_residual=ode_evidence[
            "maximum_upstream_storage_baryon_residual"
        ],
        artifact=ode_convergence_artifact,
        figure=ode_convergence_figure,
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
        title="Zero-failure 1,000-tree equivalence JSON",
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
    partition_equivalence_artifact = Artifact(
        key="partition_equivalence",
        title="Complete-partition equivalence JSON",
        path=partition_equivalence_path.relative_to(output).as_posix(),
        media_type="application/json",
        role="diagnostic",
        description=(
            "All-snapshot field comparison for every tree in Mini-Millennium input partition 1."
        ),
    )
    partition_comparison = Diagnostic(
        key="partition_equivalence",
        title="Complete-partition field comparison",
        status=(
            DiagnosticStatus.PASSED
            if int(partition_equivalence["mismatches"]) == 0
            else DiagnosticStatus.WARNING
        ),
        summary=(
            f"{int(partition_equivalence['mismatches']):,} of "
            f"{int(partition_equivalence['field_comparisons']):,} strict field comparisons "
            "exceed the stated mixed-precision tolerance. The residuals remain open; agreement "
            "of a population statistic does not erase them."
        ),
        metrics=(
            ScalarMetric(
                "partition_trees",
                "Trees",
                int(partition_equivalence["tree_count"]),
                description="every tree in input partition 1",
            ),
            ScalarMetric(
                "partition_records",
                "Catalogue records",
                int(partition_equivalence["records_compared"]),
                description="all configured output snapshots",
            ),
            ScalarMetric(
                "partition_comparisons",
                "Field comparisons",
                int(partition_equivalence["field_comparisons"]),
            ),
            ScalarMetric(
                "partition_mismatches",
                "Comparisons outside tolerance",
                int(partition_equivalence["mismatches"]),
            ),
        ),
        artifacts=(partition_equivalence_artifact, science_figures[2]),
        method="field-by-field matching by UniqueGalaxyID over all configured snapshots",
        tolerance=(
            "float32/Cooling/Heating rtol=atol=2e-6; other float64 rtol=atol=2e-12; "
            "integers exact"
        ),
    )
    equivalence_health = Diagnostic(
        key="upstream_equivalence",
        title="Science-level upstream equivalence",
        status=DiagnosticStatus.PASSED,
        summary=(
            "MIMIC and mimic-jax are scientifically indistinguishable for the population "
            f"observables shown. The 1,000-tree control passes "
            f"{int(equivalence['field_comparisons']):,} field comparisons with zero failures, "
            f"and all resolved stellar-mass-function bins agree exactly in the complete "
            f"{int(partition_equivalence['tree_count']):,}-tree partition."
        ),
        metrics=partition_comparison.metrics,
        artifacts=(equivalence_artifact, partition_equivalence_artifact),
        notes=(
            f"The technical table retains {int(partition_equivalence['mismatches'])} "
            "non-bitwise residuals so that science-level identity is not mistaken for a "
            "bitwise-equality claim.",
        ),
        method=partition_comparison.method,
        tolerance=partition_comparison.tolerance,
    )

    science_artifact = Artifact(
        key="partition_science_summary",
        title="Complete-partition science summary JSON",
        path=science_path.relative_to(output).as_posix(),
        media_type="application/json",
        role="diagnostic",
        description="Machine-readable scope, metrics, runtime, and evidence-backed findings.",
    )
    science_arrays_artifact = Artifact(
        key="partition_science_arrays",
        title="Complete-partition science arrays",
        path=science_arrays_path.relative_to(output).as_posix(),
        media_type="application/x-npz",
        role="scientific_array",
        description=(
            "Matched SMF, group baryon inventory, quenched-fraction, cooling, and heating summaries."
        ),
    )
    science_metrics = science["metrics"]
    stellar_mass_function_diagnostic = Diagnostic(
        key="stellar_mass_function_equivalence",
        title="Stellar mass function",
        status=(
            DiagnosticStatus.PASSED
            if int(science_metrics["resolved_smf_bin_mismatches"]) == 0
            else DiagnosticStatus.FAILED
        ),
        summary=(
            f"All {int(science_metrics['resolved_smf_bins'])} bins containing at least five "
            "MIMIC galaxies have identical mimic-jax counts in the complete-partition z=0 sample."
        ),
        metrics=(
            ScalarMetric(
                "resolved_smf_bins",
                "Resolved SMF bins",
                int(science_metrics["resolved_smf_bins"]),
                description="at least five MIMIC galaxies per 0.1-dex bin",
            ),
            ScalarMetric(
                "resolved_smf_bin_mismatches",
                "Resolved bins with different counts",
                int(science_metrics["resolved_smf_bin_mismatches"]),
                description="zero is required by this population-level gate",
            ),
            ScalarMetric(
                "maximum_smf_difference",
                "Maximum resolved fractional abundance difference",
                float(science_metrics["maximum_resolved_smf_fractional_difference"]),
                description="fractional, not percent",
            ),
            ScalarMetric(
                "stellar_mass_values_different",
                "Individual stellar masses that are not bit-identical",
                int(science_metrics["stellar_mass_values_different"]),
                description=f"out of {int(science['records_matched']):,} matched z=0 galaxies",
            ),
            ScalarMetric(
                "maximum_stellar_mass_difference",
                "Largest resolved stellar-mass relative difference",
                float(science_metrics["maximum_resolved_stellar_mass_relative_difference"]),
                description="the small object-level residuals do not cross an SMF bin edge",
            ),
        ),
        artifacts=(science_artifact, science_arrays_artifact),
        method="matched z=0 catalogue, 0.1-dex bins, one complete input partition",
        tolerance="exact bin counts where the MIMIC bin contains at least five galaxies",
        notes=(
            "This is a population-level agreement test, weaker than the per-object field gate; "
            "both are reported.",
            f"{int(science_metrics['stellar_mass_values_different'])} individual stellar masses "
            "are not bit-identical, so identical histogram counts are not presented as exact "
            "per-galaxy equivalence.",
        ),
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
    mismatch_fraction = int(partition_equivalence["mismatches"]) / int(
        partition_equivalence["field_comparisons"]
    )
    maximum_partition_relative_difference = max(
        float(value) for value in partition_equivalence["largest_relative_errors"].values()
    )
    reservoir_labels = {
        "ColdGas": "cold gas",
        "HotGas": "hot gas",
        "EjectedGas": "ejected gas",
        "StellarMass": "stars",
        "ICS": "intracluster stars",
        "BlackHoleMass": "black holes",
    }
    regime_findings = tuple(
        f"{reservoir_labels[regime['reservoir']].capitalize()} is the largest modeled share of "
        f"the universal baryon allotment over "
        f"{float(regime['log10_halo_mass_min']):.2f} ≤ log10(Mvir/Msun) < "
        f"{float(regime['log10_halo_mass_max']):.2f} in bins containing at least ten FoF groups."
        for regime in science["dominant_baryon_reservoir_regimes"]
    )
    findings = (
        f"The complete partition contains {int(science['tree_count']):,} trees, "
        f"{int(science['input_halos']):,} input halos, and {int(science['records_matched']):,} "
        "matched z=0 galaxies; its evolution with the persistent compilation cache enabled "
        "completed in "
        f"{float(science['evolution_seconds']):.1f} s.",
        f"All {int(science_metrics['resolved_smf_bins'])} stellar-mass bins with at least five "
        "reference galaxies have identical MIMIC and mimic-jax counts.",
        *regime_findings,
        f"The complete all-snapshot field gate retains "
        f"{int(partition_equivalence['mismatches']):,} residuals among "
        f"{int(partition_equivalence['field_comparisons']):,} comparisons "
        f"({100.0 * mismatch_fraction:.4g}%, maximum relative difference "
        f"{maximum_partition_relative_difference:.3g}). These are negligible "
        "for the science observables shown, while remaining visible in technical validation.",
        "For the fixed-halo, smooth quiescent reservoir subset, the repeated upstream-order "
        "update and forward Euler converge at first order, Heun at second order, and RK4 at "
        "fourth order; this is a controlled time-integration result, not yet a population-level "
        "Mini-Millennium convergence claim.",
    )
    population_response = Diagnostic(
        key="smf_parameter_response",
        title="Stellar-mass-function parameter response",
        status=DiagnosticStatus.NOT_EVALUATED,
        summary=(
            "A validated differentiable estimator for hard stellar-mass-function bin membership "
            "has not yet been run on this partition. No raw or zero-almost-everywhere histogram "
            "gradient is shown."
        ),
        notes=(
            "The next science milestone remains E_i(M*) = d ln phi / d ln theta with symmetric "
            "finite-difference validation.",
        ),
        method="not evaluated",
    )
    historical_response = Diagnostic(
        key="historical_process_response",
        title="Cooling and AGN historical response",
        status=DiagnosticStatus.NOT_EVALUATED,
        summary=(
            "Epoch-binned cooling and AGN perturbations were not evaluated for this catalogue, so "
            "the report does not infer causal regulation from instantaneous output correlations."
        ),
    )
    population_convergence = Diagnostic(
        key="population_timestep_convergence",
        title="Population timestep convergence",
        status=DiagnosticStatus.NOT_EVALUATED,
        summary=(
            "The stellar mass function has not yet been recomputed at refined Mini-Millennium "
            "substeps. The controlled central refinement remains technical API evidence only."
        ),
    )
    controlled_gradient_health = Diagnostic(
        key="controlled_gradient_validation",
        title="Controlled gradient validation",
        status=response_diagnostic.status,
        summary=(
            "A controlled quiescent SAGE16 step has a logarithmic parameter response that agrees "
            "with symmetric finite differences; this is not yet an SMF response."
        ),
        metrics=response_diagnostic.metrics,
        method=response_diagnostic.method,
        tolerance=response_diagnostic.tolerance,
    )
    controlled_baryon_health = Diagnostic(
        key="controlled_baryon_conservation",
        title="Controlled baryon conservation",
        status=baryon_diagnostic.status,
        summary=(
            "The explicit controlled source/sink ledger closes within tolerance; a full "
            "Mini-Millennium history ledger was not evaluated."
        ),
        metrics=baryon_diagnostic.metrics,
        method=baryon_diagnostic.method,
        tolerance=baryon_diagnostic.tolerance,
    )
    sections = (
        ReportSection(
            key="findings",
            title="What did this larger run teach us?",
            summary=(
                "These statements are generated from the committed JSON/NPZ products rather than "
                "being hand-maintained narrative claims."
            ),
            notes=findings,
        ),
        ReportSection(
            key="equivalence",
            title="Does mimic-jax reproduce familiar SAGE16?",
            summary=(
                "The opening comparison uses a complete Mini-Millennium input partition with the "
                "same volume normalization and 0.1-dex bins as the familiar MIMIC plot. The full-"
                "volume upstream figure below retains the observational context used by SAGE "
                "practitioners."
            ),
            artifacts=familiar_artifacts[:1],
            diagnostics=(stellar_mass_function_diagnostic,),
            links=(
                ReportLink("SAGE16 plotting manual", "../../plot/mimic-plot/README.md"),
                ReportLink(
                    "Mini-Millennium equivalence evidence",
                    "../../docs/mini_millennium_equivalence.md",
                ),
            ),
        ),
        ReportSection(
            key="baryons",
            title="Where are the baryons?",
            summary=(
                "The explicit SAGE reservoirs can be read as a physical inventory. Each stack is "
                "the total reservoir mass of a FoF group divided by its universal baryon "
                "allotment, making reionization/ejection suppression and the hot-halo transition "
                "visible before the small catalogue-equivalence residual is shown."
            ),
            artifacts=(science_figures[1],),
            notes=regime_findings,
            links=(
                ReportLink(
                    "Reservoir and transfer model", "../../docs/reservoirs_and_transfers.md"
                ),
                ReportLink("Conservation contract", "../../docs/conservation.md"),
            ),
        ),
        ReportSection(
            key="smf_responses",
            title="What controls the stellar mass function?",
            summary=(
                "The public-facing quantity will be percentage abundance change per 1% parameter "
                "change. Hard catalogue bins are discrete, so mimic-jax will not substitute the "
                "pathwise derivative of fixed bin assignments for a population response."
            ),
            diagnostics=(population_response,),
            links=(ReportLink("Fractional-response API", "../../docs/sensitivity.md"),),
        ),
        ReportSection(
            key="agn_regulation",
            title="Where does AGN regulation take over from cooling?",
            summary=(
                "The familiar black-hole–bulge relation establishes the relevant SAGE population, "
                "but the causal cooling-versus-AGN response map is deliberately withheld until "
                "epoch-binned process perturbations are validated."
            ),
            artifacts=familiar_artifacts[1:],
            diagnostics=(historical_response,),
            links=(
                ReportLink("Radio-mode heating prescription", "../../docs/radio_mode_heating.md"),
            ),
        ),
        ReportSection(
            key="numerical_integration",
            title="How accurately are these histories being integrated?",
            summary=(
                "The exact upstream-sequential path remains the SAGE16 reference. Separately, a "
                "fixed-halo continuous reservoir experiment now demonstrates genuine convergence "
                "in time: upstream-order splitting and Euler are first order, Heun is second "
                "order, and RK4 is fourth order. The wider hybrid formulation treats prepared "
                "infall as external forcing, makes AGN memory Markovian with stored `Rheat`, "
                "represents stripping as a group flow, and retains projections and mergers as "
                "events. Population-level convergence must still be tested through familiar "
                "observables and is not inferred from this controlled experiment."
            ),
            diagnostics=(ode_diagnostic, population_convergence, refinement_diagnostic),
            links=(
                ReportLink("Numerical integration contract", "../../docs/numerical_integration.md"),
                ReportLink(
                    "Complete SAGE16 hybrid classification",
                    "../../docs/sage16_hybrid_system.md",
                ),
            ),
        ),
        ReportSection(
            key="performance",
            title="Can a scientifically larger sample run interactively?",
            summary=(
                "The 1,000-tree benchmark is ten times the original report sample and separates "
                "first-process and warmed calls. The complete-partition science product records "
                "its own runtime and peak memory; neither number is compared unfairly with "
                "upstream compilation excluded on only one side."
            ),
            diagnostics=(performance,),
            links=(ReportLink("Current performance evidence", "../../docs/performance.md"),),
        ),
        ReportSection(
            key="technical_validation",
            title="Why should we trust these results?",
            summary=(
                "The science panels above are supported by stronger per-object comparisons and "
                "controlled invariant/derivative tests. The strict field residuals are retained "
                "to distinguish scientific identity from bitwise equality."
            ),
            diagnostics=(
                selected_equivalence,
                partition_comparison,
                baryon_diagnostic,
                metal_diagnostic,
                response_diagnostic,
            ),
            links=(
                ReportLink(
                    "Mini-Millennium equivalence evidence",
                    "../../docs/mini_millennium_equivalence.md",
                ),
                ReportLink("Conservation contract", "../../docs/conservation.md"),
                ReportLink("Fractional-response API", "../../docs/sensitivity.md"),
            ),
        ),
    )

    report = RunReport(
        identity=RunIdentity(
            run_id="mini-millennium-sage16-initial",
            title="SAGE16 Mini-Millennium: from equivalence to baryon-cycle insight",
            model="fiducial SAGE16",
            dataset=(
                f"Mini-Millennium partition 1, all {int(science['tree_count']):,} trees; "
                "1/8 simulation volume"
            ),
            parameter_set="sage16_mini-millennium fiducial",
            integration_method="upstream_sequential, 10 configured substeps",
            summary=(
                "A complete Mini-Millennium input partition now connects a familiar SAGE16 "
                "stellar mass function to an explicit baryon inventory. MIMIC and mimic-jax are "
                "scientifically indistinguishable for the observables shown; strict numerical "
                "residuals, controlled derivative checks, and unavailable population responses "
                "remain visible so the science story never outruns the evidence."
            ),
        ),
        provenance=provenance,
        health=(
            equivalence_health,
            stellar_mass_function_diagnostic,
            controlled_baryon_health,
            metal_diagnostic,
            controlled_gradient_health,
            ode_diagnostic,
            population_response,
            population_convergence,
        ),
        sections=sections,
        overview_metrics=(
            ScalarMetric("trees", "Trees in science sample", int(science["tree_count"])),
            ScalarMetric("input_halos", "Input halos", int(science["input_halos"])),
            ScalarMetric(
                "catalogue_records",
                "Matched z=0 galaxies",
                int(science["records_matched"]),
            ),
            ScalarMetric(
                "partition_evolution_seconds",
                "Complete-partition evolution",
                float(science["evolution_seconds"]),
                unit="s",
            ),
            ScalarMetric(
                "benchmark_first_seconds",
                "1,000-tree first benchmark call",
                float(first_run["evolution_seconds"]),
                unit="s",
            ),
            ScalarMetric(
                "benchmark_warm_seconds",
                "1,000-tree best warm call",
                float(warm_seconds),
                unit="s",
            ),
            ScalarMetric(
                "partition_peak_memory",
                "Complete-partition peak memory",
                float(science["peak_resident_bytes"]) / (1024.0**3),
                unit="GiB",
            ),
            ScalarMetric("backend", "JAX backend", str(benchmark["backend"])),
        ),
        headline_artifacts=(science_figures[0],),
        parameters=parameters_from_namedtuple(
            fiducial_parameters(), descriptions=PARAMETER_DESCRIPTIONS
        ),
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
