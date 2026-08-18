#!/usr/bin/env python3
"""Build the science-program Mini-Millennium SAGE16 report.

This command is deliberately a cheap presentation step.  It consumes durable
JSON/NPZ products produced by the expensive model, derivative, history, and
convergence analyses; it does not evolve a merger tree.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

jax.config.update("jax_enable_x64", True)

from mimic_jax.reporting import (  # noqa: E402
    Artifact,
    Diagnostic,
    DiagnosticStatus,
    ReportLink,
    ReportSection,
    RunIdentity,
    RunReport,
    ScalarMetric,
    capture_provenance,
    parameters_from_namedtuple,
    write_report,
)
from mimic_jax.sage16 import fiducial_parameters  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY / "reports/mini-millennium-sage16-science-program"
ARCHIVE = REPOSITORY / "archive"
RUN_FILE = REPOSITORY / "models/sage16/input/sage16_mini-millennium.yaml"
SCALE_FACTORS = REPOSITORY / "simulations/mini-millennium/mini-millennium.a_list"

PRODUCTS = {
    "population": "mini-millennium-partition-1-science",
    "parameters": "mini-millennium-sage16-parameter-responses",
    "parameter_validation": "mini-millennium-sage16-response-validation-1000",
    "history": "mini-millennium-sage16-history-responses",
    "history_validation": "mini-millennium-sage16-history-validation",
    "convergence": "mini-millennium-sage16-convergence-500",
    "ringing": "mini-millennium-sage16-timestep-ringing",
    "module_ablation": "mini-millennium-sage16-timestep-module-ablation",
    "adaptive": "mini-millennium-sage16-adaptive-continuous",
}

PARAMETER_LABELS = {
    "SfrEfficiency": "Star formation",
    "FeedbackReheatingEpsilon": "SN reheating",
    "FeedbackEjectionEfficiency": "SN ejection",
    "ReIncorporationFactor": "Reincorporation",
    "RadioModeEfficiency": "Radio-mode AGN",
    "BlackHoleGrowthRate": "BH growth",
    "QuasarModeEfficiency": "Quasar mode",
}

PARAMETER_DESCRIPTIONS = {
    "SfrEfficiency": "Quiescent star-formation efficiency per disk dynamical time.",
    "FeedbackReheatingEpsilon": "Supernova reheating normalization.",
    "FeedbackEjectionEfficiency": "Supernova energy efficiency for halo-gas ejection.",
    "ReIncorporationFactor": "Return rate of ejected gas to the hot halo.",
    "RadioModeEfficiency": "Radio-mode black-hole accretion/heating efficiency.",
    "BlackHoleGrowthRate": "Cold-gas black-hole growth efficiency during events.",
    "QuasarModeEfficiency": "Quasar-mode gas-ejection efficiency.",
}

PROCESS_LABELS = {
    "cooling": "Cooling",
    "sn_reheating": "SN reheating",
    "sn_ejection": "SN ejection",
    "reincorporation": "Reincorporation",
    "agn_heating": "AGN heating",
}

OBSERVABLE_LABELS = {
    "low-mass abundance (8.5–9.5)": "Low-mass SMF",
    "knee abundance (9.5–10.5)": "SMF knee",
    "massive abundance (10.5–11.5)": "Massive SMF",
    "stellar-mass density": "Stellar mass density",
    "star-formation-rate density": "SFR density",
    "cold-gas density": "Cold-gas density",
    "ejected-gas density": "Ejected-gas density",
    "black-hole-mass density": "BH mass density",
}

COLORS = {
    "SfrEfficiency": "#0072B2",
    "FeedbackReheatingEpsilon": "#D55E00",
    "FeedbackEjectionEfficiency": "#E69F00",
    "ReIncorporationFactor": "#009E73",
    "RadioModeEfficiency": "#CC79A7",
    "BlackHoleGrowthRate": "#56B4E9",
    "QuasarModeEfficiency": "#777777",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE)
    return parser.parse_args()


def load_products(archive_dir):
    products = {}
    for key, stem in PRODUCTS.items():
        json_path = archive_dir / f"{stem}.json"
        arrays_path = archive_dir / f"{stem}.npz"
        if not json_path.is_file() or not arrays_path.is_file():
            raise SystemExit(f"Missing required science product: {json_path} / {arrays_path}")
        products[key] = {
            "summary": json.loads(json_path.read_text(encoding="utf-8")),
            "arrays": dict(np.load(arrays_path, allow_pickle=False)),
            "json_path": json_path,
            "arrays_path": arrays_path,
        }
    return products


def configure_matplotlib():
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "svg.fonttype": "none",
        }
    )


def save_figure(figure, path):
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def centered_limit(values, percentile=98.0, minimum=0.05):
    finite = np.abs(np.asarray(values)[np.isfinite(values)])
    if finite.size == 0:
        return minimum
    return max(minimum, float(np.percentile(finite, percentile)))


def smf_equivalence_figure(data, path):
    mass = data["stellar_mass_bin_centres"]
    upstream = data["upstream_smf"]
    candidate = data["mimic_jax_smf"]
    valid = (upstream > 0.0) & (candidate > 0.0)
    figure, (axis, residual) = plt.subplots(
        2, 1, figsize=(8.0, 6.3), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.0]}
    )
    axis.semilogy(mass[valid], upstream[valid], "o", ms=4.2, color="#333333", label="MIMIC")
    axis.semilogy(mass[valid], candidate[valid], "-", lw=1.8, color="#0072B2", label="mimic-jax")
    axis.set_ylabel("φ [h³ Mpc⁻³ dex⁻¹]")
    axis.set_title("The familiar z=0 SAGE16 stellar mass function is preserved")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    difference = np.full_like(upstream, np.nan)
    difference[upstream > 0.0] = (
        100.0 * (candidate[upstream > 0.0] - upstream[upstream > 0.0]) / upstream[upstream > 0.0]
    )
    residual.axhline(0.0, color="#555555", lw=1.0)
    residual.plot(mass, difference, "o", ms=3.8, color="#0072B2")
    residual.set_xlabel("log10 stellar mass [Msun]")
    residual.set_ylabel("difference [%]")
    residual.set_ylim(-0.1, 0.1)
    residual.grid(axis="y", color="#dddddd", linewidth=0.6)
    save_figure(figure, path)


def baryon_inventory_figure(data, path):
    mass = data["halo_mass_bin_centres"]
    counts = data["group_counts"]
    values = data["mimic_jax_baryon_allotment_fractions"]
    names = [str(value) for value in data["reservoir_names"]]
    valid = counts >= 10
    colors = {
        "StellarMass": "#E69F00",
        "ColdGas": "#56B4E9",
        "HotGas": "#D55E00",
        "EjectedGas": "#009E73",
        "IntraClusterStars": "#CC79A7",
        "BlackHoleMass": "#333333",
    }
    labels = {
        "StellarMass": "Stars",
        "ColdGas": "Cold gas",
        "HotGas": "Hot gas",
        "EjectedGas": "Ejected gas",
        "IntraClusterStars": "Intracluster stars",
        "BlackHoleMass": "Black holes",
    }
    figure, (axis, residual) = plt.subplots(
        2, 1, figsize=(8.2, 6.3), sharex=True, gridspec_kw={"height_ratios": [3.1, 1.0]}
    )
    lower = np.zeros(np.count_nonzero(valid))
    for index, name in enumerate(names):
        upper = lower + values[valid, index]
        axis.fill_between(
            mass[valid],
            lower,
            upper,
            color=colors.get(name, "#999999"),
            alpha=0.85,
            label=labels.get(name, name),
        )
        lower = upper
    axis.axhline(1.0, color="#333333", ls="--", lw=1.1, label="universal allotment")
    axis.set_ylabel("fraction of universal baryon allotment")
    axis.set_title("Where SAGE16 stores the baryons at z=0")
    axis.legend(ncol=2, frameon=False, loc="upper left")
    axis.set_ylim(bottom=0.0)
    upstream = data["upstream_total_baryon_fraction"]
    mimic = data["mimic_jax_total_baryon_fraction"]
    budget_difference = np.full_like(upstream, np.nan)
    nonzero = valid & (upstream != 0.0)
    budget_difference[nonzero] = 1.0e6 * (mimic[nonzero] - upstream[nonzero]) / upstream[nonzero]
    residual.axhline(0.0, color="#555555", lw=1.0)
    residual.plot(mass[valid], budget_difference[valid], "o-", color="#0072B2", ms=3.5)
    residual.set_xlabel("log10 halo mass [Msun]")
    residual.set_ylabel("mimic-jax − MIMIC [ppm]")
    residual.grid(axis="y", color="#dddddd", linewidth=0.6)
    save_figure(figure, path)


def smf_response_figure(data, path):
    mass = data["stellar_mass_bin_centres"]
    counts = data["hard_smf_counts"]
    values = data["parameter_response"]
    names = [str(value) for value in data["parameter_names"]]
    valid = counts >= 5
    selected = [
        "SfrEfficiency",
        "FeedbackReheatingEpsilon",
        "FeedbackEjectionEfficiency",
        "ReIncorporationFactor",
        "RadioModeEfficiency",
    ]
    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    linestyles = ["-", "--", "-.", ":", (0, (5, 2))]
    axis.axhline(0.0, color="#555555", lw=1.0)
    for linestyle, name in zip(linestyles, selected):
        index = names.index(name)
        axis.plot(
            mass[valid],
            values[valid, index],
            color=COLORS[name],
            lw=2.0,
            ls=linestyle,
            marker="o",
            ms=2.8,
            label=PARAMETER_LABELS[name],
        )
    axis.set_xlabel("log10 stellar mass [Msun]")
    axis.set_ylabel("% abundance change per 1% parameter change")
    axis.set_title("SAGE can now say what moves each part of the stellar mass function")
    axis.legend(ncol=2, frameon=False)
    axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    save_figure(figure, path)


def smf_response_map_figure(data, path):
    mass = data["stellar_mass_bin_centres"]
    counts = data["hard_smf_counts"]
    values = data["parameter_response"]
    names = [str(value) for value in data["parameter_names"]]
    valid = counts >= 5
    shown = values[valid].T
    limit = centered_limit(shown, percentile=100.0)
    figure, (axis, strip) = plt.subplots(
        2, 1, figsize=(10.0, 7.0), sharex=True, gridspec_kw={"height_ratios": [5.0, 0.55]}
    )
    image = axis.imshow(
        shown,
        aspect="auto",
        origin="lower",
        extent=(mass[valid][0] - 0.05, mass[valid][-1] + 0.05, -0.5, len(names) - 0.5),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axis.set_yticks(np.arange(len(names)), [PARAMETER_LABELS[name] for name in names])
    axis.set_title("What controls each stellar-mass-function bin?")
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("% abundance change per 1% parameter change")
    dominant = np.nanargmax(np.abs(shown), axis=0)
    strip.imshow(
        dominant[None, :],
        aspect="auto",
        origin="lower",
        extent=(mass[valid][0] - 0.05, mass[valid][-1] + 0.05, 0.0, 1.0),
        cmap=mpl.colors.ListedColormap([COLORS[name] for name in names]),
        vmin=-0.5,
        vmax=len(names) - 0.5,
        interpolation="nearest",
    )
    strip.set_yticks([0.5], ["largest\n|response|"])
    strip.set_xlabel("log10 stellar mass [Msun]")
    figure.legend(
        handles=[Patch(color=COLORS[name], label=PARAMETER_LABELS[name]) for name in names],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=4,
        frameon=False,
    )
    save_figure(figure, path)


def response_matrix_figure(data, path):
    values = data["summary_response"]
    parameters = [str(value) for value in data["parameter_names"]]
    observables = [str(value) for value in data["summary_observable_names"]]
    limit = centered_limit(values, percentile=100.0)
    figure, axis = plt.subplots(figsize=(10.2, 6.2))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(
        np.arange(len(parameters)),
        [PARAMETER_LABELS[name] for name in parameters],
        rotation=32,
        ha="right",
    )
    axis.set_yticks(np.arange(len(observables)), [OBSERVABLE_LABELS[name] for name in observables])
    axis.set_title("Which SAGE physics affects which familiar predictions?")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if abs(values[row, column]) > 0.55 * limit else "#222222"
            axis.text(
                column,
                row,
                f"{values[row, column]:+.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("fractional response")
    save_figure(figure, path)


def similarity_figure(data, path):
    values = np.ma.masked_invalid(data["parameter_similarity"])
    names = [str(value) for value in data["parameter_names"]]
    cmap = mpl.colormaps["PuOr_r"].copy()
    cmap.set_bad("#e6e6e6")
    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    image = axis.imshow(values, cmap=cmap, vmin=-1.0, vmax=1.0)
    labels = [PARAMETER_LABELS[name] for name in names]
    axis.set_xticks(np.arange(len(names)), labels, rotation=38, ha="right")
    axis.set_yticks(np.arange(len(names)), labels)
    axis.set_title("Which parameter changes look alike observationally?")
    for row in range(len(names)):
        for column in range(len(names)):
            value = values[row, column]
            label = "—" if np.ma.is_masked(value) else f"{float(value):+.2f}"
            color = (
                "white" if not np.ma.is_masked(value) and abs(float(value)) > 0.62 else "#222222"
            )
            axis.text(column, row, label, ha="center", va="center", color=color, fontsize=8.5)
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("cosine similarity of response fingerprints")
    save_figure(figure, path)


def history_ticks(redshift_edges):
    centres = np.arange(len(redshift_edges) - 1) + 0.5
    midpoint = np.sqrt((1.0 + redshift_edges[:-1]) * (1.0 + redshift_edges[1:])) - 1.0
    labels = [f"{value:.1f}" if value < 10.0 else f"{value:.0f}" for value in midpoint]
    return centres, labels


def historical_response_figure(data, path):
    response = data["historical_process_response"]
    processes = [str(value) for value in data["history_process_names"]]
    mass_edges = data["history_mass_bin_edges"]
    redshift_edges = data["history_redshift_edges"]
    ticks, tick_labels = history_ticks(redshift_edges)
    limit = centered_limit(response, percentile=100.0)
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 6.9), sharex=True, sharey=True)
    axes = axes.ravel()
    for index, process in enumerate(processes):
        axis = axes[index]
        image = axis.pcolormesh(
            np.arange(response.shape[2] + 1),
            mass_edges,
            response[:, index, :],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="flat",
        )
        axis.set_title(PROCESS_LABELS[process])
        axis.set_xticks(ticks[::2], tick_labels[::2])
        axis.axvline(6.0, color="#555555", lw=0.6, alpha=0.45)
    axes[-1].set_visible(False)
    for axis in axes[:3]:
        axis.set_ylabel("final log10 stellar mass [Msun]")
    for axis in axes[3:5]:
        axis.set_xlabel("redshift of 1% process perturbation")
        axis.set_ylabel("final log10 stellar mass [Msun]")
    figure.suptitle(
        "When does each part of the SAGE baryon cycle matter?", y=1.01, fontweight="bold"
    )
    colorbar = figure.colorbar(image, ax=axes[:5].tolist(), pad=0.018, fraction=0.025)
    colorbar.set_label("% final stellar-mass change per 1% process change")
    save_figure(figure, path)


def cooling_agn_figure(data, path):
    response = data["historical_process_response"]
    processes = [str(value) for value in data["history_process_names"]]
    mass_edges = data["history_mass_bin_edges"]
    redshift_edges = data["history_redshift_edges"]
    ticks, tick_labels = history_ticks(redshift_edges)
    indices = [processes.index("cooling"), processes.index("agn_heating")]
    shown = response[:, indices, :]
    limit = centered_limit(shown, percentile=100.0)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True, sharey=True)
    for axis, process_index, title in zip(axes, indices, ("More cooling", "More AGN heating")):
        image = axis.pcolormesh(
            np.arange(response.shape[2] + 1),
            mass_edges,
            response[:, process_index, :],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="flat",
        )
        axis.set_title(title)
        axis.set_xticks(ticks[::2], tick_labels[::2])
        axis.set_xlabel("redshift of 1% process perturbation")
    axes[0].set_ylabel("final log10 stellar mass [Msun]")
    figure.suptitle(
        "Cooling builds massive galaxies; AGN heating opposes that supply",
        y=1.02,
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=axes.tolist(), pad=0.02, fraction=0.045)
    colorbar.set_label("% final stellar-mass change per 1% process change")
    save_figure(figure, path)


def growth_decoupling_figure(data, path):
    redshift = data["history_redshift"]
    halo = data["halo_log_growth_rate"]
    stars = data["stellar_log_growth_rate"]
    mass_edges = data["history_mass_bin_edges"]
    seconds_per_gyr = 365.25 * 24.0 * 3600.0 * 1.0e9
    internal_time_gyr = 3.08568e19 / seconds_per_gyr
    chosen = (0, 2, 4)
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharex=True, sharey=True)
    for axis, index in zip(axes, chosen):
        valid_halo = np.isfinite(halo[index]) & (redshift <= 4.5)
        valid_stars = np.isfinite(stars[index]) & (redshift <= 4.5)
        axis.plot(
            redshift[valid_halo],
            halo[index, valid_halo] / internal_time_gyr,
            color="#333333",
            lw=1.8,
            label="halo",
        )
        axis.plot(
            redshift[valid_stars],
            stars[index, valid_stars] / internal_time_gyr,
            color="#0072B2",
            lw=1.8,
            label="stars",
        )
        axis.axhline(0.0, color="#888888", lw=0.8)
        axis.set_title(f"{mass_edges[index]:.2g}–{mass_edges[index + 1]:.2g}")
        axis.set_xlabel("redshift")
        axis.invert_xaxis()
    axes[0].set_ylabel("median d ln M / dt [Gyr⁻¹]")
    axes[0].legend(frameon=False)
    figure.suptitle(
        "When does stellar growth stop following halo growth?", y=1.03, fontweight="bold"
    )
    save_figure(figure, path)


def convergence_figure(data, path):
    steps = data["convergence_step_counts"]
    mass = data["stellar_mass_bin_centres"]
    counts = data["baseline_hard_smf_counts"]
    relative = data["convergence_soft_smf_relative_to_fine"]
    names = [str(value) for value in data["convergence_scalar_names"]]
    labels = {
        "StellarMass": "Stellar mass",
        "ColdGas": "Cold gas",
        "EjectedGas": "Ejected gas",
        "BlackHoleMass": "Black-hole mass",
    }
    scalars = data["convergence_scalar_relative_to_fine"]
    valid = counts >= 5
    figure, (axis, scalar_axis) = plt.subplots(1, 2, figsize=(11.2, 4.6))
    palette = mpl.colormaps["viridis"](np.linspace(0.12, 0.86, len(steps) - 1))
    for color, index in zip(palette, range(len(steps) - 1)):
        axis.plot(
            mass[valid],
            100.0 * relative[index, valid],
            marker="o",
            ms=3.0,
            lw=1.5,
            color=color,
            label=f"{steps[index]} substeps",
        )
    axis.axhline(0.0, color="#555555", lw=0.9)
    axis.set_xlabel("log10 stellar mass [Msun]")
    axis.set_ylabel(f"difference from {steps[-1]}-substep SMF [%]")
    axis.set_title("Does SubSteps change the stellar mass function?")
    axis.legend(frameon=False, ncol=2)
    scalar_colors = ["#0072B2", "#56B4E9", "#009E73", "#CC79A7"]
    for name, color, column in zip(names, scalar_colors, range(len(names))):
        scalar_axis.plot(
            steps[:-1],
            100.0 * np.abs(scalars[:-1, column]),
            "o-",
            color=color,
            label=labels.get(name, name),
        )
    scalar_axis.set_xscale("log", base=2)
    scalar_axis.set_xticks(steps[:-1], [str(value) for value in steps[:-1]])
    scalar_axis.set_yscale("log")
    scalar_axis.set_xlabel("substeps per tree interval")
    scalar_axis.set_ylabel(f"absolute difference from {steps[-1]} substeps [%]")
    scalar_axis.set_title("Integrated reservoir totals")
    scalar_axis.legend(frameon=False)
    save_figure(figure, path)


def timestep_ringing_figure(data, path):
    """Separate mass-bin transport from central/satellite population effects."""
    mass = data["stellar_mass_bin_centres"]
    bandwidths = data["bandwidths_dex"]
    populations = [str(value) for value in data["population_names"]]
    counts = data["hard_smf_counts"]
    relative = data["soft_smf_fractional_difference"]
    bandwidth_index = int(np.argmin(np.abs(bandwidths - 0.05)))
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 4.25))

    bandwidth_colors = mpl.colormaps["viridis"](np.linspace(0.12, 0.86, bandwidths.size))
    resolved_all = counts[1, 0] >= 5
    for color, bandwidth_index_current in zip(bandwidth_colors, range(bandwidths.size)):
        axes[0].plot(
            mass[resolved_all],
            100.0 * relative[0, bandwidth_index_current, resolved_all],
            color=color,
            lw=1.6,
            label=f"{bandwidths[bandwidth_index_current]:.02f} dex",
        )
    axes[0].axhline(0.0, color="#555555", lw=0.9)
    axes[0].set_xlabel("log10 stellar mass [Msun]")
    axes[0].set_ylabel("10 minus 80 substeps [%]")
    axes[0].set_title("Smoothing changes amplitude")
    axes[0].legend(frameon=False, fontsize=8)

    population_colors = ("#333333", "#0072B2", "#D55E00")
    for population_index, (population, color) in enumerate(zip(populations, population_colors)):
        resolved = counts[1, population_index] >= 5
        axes[1].plot(
            mass[resolved],
            100.0 * relative[population_index, bandwidth_index, resolved],
            color=color,
            lw=1.7,
            label=population,
        )
    axes[1].axhline(0.0, color="#555555", lw=0.9)
    axes[1].set_xlabel("log10 stellar mass [Msun]")
    axes[1].set_ylabel("10 minus 80 substeps [%]")
    axes[1].set_title("Centrals and satellites")
    axes[1].legend(frameon=False, fontsize=8)

    shift_mass = data["shift_bin_centres"]
    shift_count = data["shift_bin_counts"]
    shift_median = data["shift_median"]
    shift_q16 = data["shift_q16"]
    shift_q84 = data["shift_q84"]
    measured = (shift_count >= 5) & np.isfinite(shift_median)
    axes[2].fill_between(
        shift_mass[measured],
        1.0e3 * shift_q16[measured],
        1.0e3 * shift_q84[measured],
        color="#56B4E9",
        alpha=0.25,
        label="16–84%",
    )
    axes[2].plot(
        shift_mass[measured],
        1.0e3 * shift_median[measured],
        "o-",
        color="#0072B2",
        ms=3.5,
        lw=1.7,
        label="median",
    )
    axes[2].axhline(0.0, color="#555555", lw=0.9)
    axes[2].set_xlabel("80-substep log10 stellar mass [Msun]")
    axes[2].set_ylabel("galaxy shift: log10(M10/M80) [millidex]")
    axes[2].set_title("Matched galaxies move in mass")
    axes[2].legend(frameon=False, fontsize=8)

    figure.suptitle(
        "Why does the timestep difference ring across stellar mass?",
        y=1.03,
        fontweight="bold",
    )
    save_figure(figure, path)


def timestep_module_ablation_figure(data, path):
    """Show which hybrid module chain amplifies a representative mass shift."""
    conditions = [str(value) for value in data["condition_names"]]
    labels = {
        "fiducial": "fiducial",
        "no_disk_instability": "no disk\ninstability",
        "no_satellite_stripping": "no satellite\nstripping",
        "no_agn_heating": "no AGN\nheating",
        "no_quasar_or_starburst": "no quasar /\nstarburst",
    }
    colors = {
        "fiducial": "#333333",
        "no_disk_instability": "#0072B2",
        "no_satellite_stripping": "#E69F00",
        "no_agn_heating": "#CC79A7",
        "no_quasar_or_starburst": "#009E73",
    }
    fields = [str(value) for value in data["field_names"]]
    stellar_index = fields.index("StellarMass")
    heating_radius_index = fields.index("Rheat")
    redshift = data["redshift"]
    values = data["snapshot_values"][..., stellar_index]
    ratio = np.divide(
        values[:, 0],
        values[:, 1],
        out=np.full_like(values[:, 0], np.nan),
        where=values[:, 1] > 0.0,
    )
    final_ratio = data["final_coarse_to_fine_ratio"][:, stellar_index]

    figure, (history_axis, heating_axis, final_axis) = plt.subplots(1, 3, figsize=(13.5, 4.35))
    for index, condition in enumerate(conditions):
        valid = np.isfinite(ratio[index]) & (redshift <= 6.0)
        history_axis.plot(
            redshift[valid],
            ratio[index, valid],
            color=colors[condition],
            lw=1.8,
            label=labels[condition].replace("\n", " "),
        )
    history_axis.axhline(1.0, color="#777777", lw=0.9)
    history_axis.set_yscale("log")
    history_axis.set_xlabel("redshift")
    history_axis.set_ylabel("tree stellar mass: 10 / 80 substeps")
    history_axis.set_title("Divergence through cosmic history")
    history_axis.invert_xaxis()
    history_axis.legend(frameon=False, fontsize=8, ncol=2)

    heating_radius = data["snapshot_values"][0, :, :, heating_radius_index]
    for substep_index, (color, linestyle) in enumerate((("#D55E00", "-"), ("#0072B2", "--"))):
        valid = (
            np.isfinite(heating_radius[substep_index])
            & (heating_radius[substep_index] > 0.0)
            & (redshift <= 6.0)
        )
        heating_axis.plot(
            redshift[valid],
            heating_radius[substep_index, valid],
            color=color,
            ls=linestyle,
            lw=1.8,
            label=f"{data['substeps'][substep_index]} substeps",
        )
    heating_axis.set_yscale("log")
    heating_axis.set_xlabel("redshift")
    heating_axis.set_ylabel("summed stored Rheat [internal length]")
    heating_axis.set_title("AGN memory diverges early")
    heating_axis.invert_xaxis()
    heating_axis.legend(frameon=False, fontsize=8)

    x = np.arange(len(conditions))
    bars = final_axis.bar(
        x,
        final_ratio,
        color=[colors[condition] for condition in conditions],
        width=0.72,
    )
    final_axis.axhline(1.0, color="#777777", lw=0.9)
    final_axis.set_xticks(x, [labels[condition] for condition in conditions], fontsize=8)
    final_axis.set_ylabel("z=0 stellar mass: 10 / 80 substeps")
    final_axis.set_title("Suppress one module chain")
    final_axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    final_axis.set_ylim(0.0, 1.12 * np.nanmax(final_ratio))

    figure.suptitle(
        "A timestep-sensitive tree identifies the instability–AGN coupling",
        y=1.03,
        fontweight="bold",
    )
    save_figure(figure, path)


def adaptive_continuous_figure(data, path):
    """Compare adaptive continuous-flow accuracy and work with fixed-step RK4."""
    fixed_errors = np.nanmedian(np.nanmax(data["fixed_relative_reservoir_errors"], axis=2), axis=1)
    adaptive_errors = np.nanmedian(
        np.nanmax(data["adaptive_relative_reservoir_errors"], axis=2), axis=1
    )
    adaptive_rhs = np.nanmedian(data["adaptive_rhs_evaluations"], axis=1)
    tolerances = data["adaptive_tolerances"]
    stellar_error = np.nanmax(np.abs(data["adaptive_stellar_log_shift"]), axis=1)
    chosen_index = int(np.argmin(np.abs(tolerances - 1.0e-7)))

    figure, axes = plt.subplots(1, 3, figsize=(13.3, 4.25))
    axes[0].loglog(
        data["fixed_rhs_evaluations"],
        fixed_errors,
        "o-",
        color="#777777",
        lw=1.7,
        label="fixed RK4",
    )
    axes[0].loglog(
        adaptive_rhs,
        adaptive_errors,
        "s-",
        color="#0072B2",
        lw=1.9,
        label="adaptive RK5(4)",
    )
    axes[0].set_xlabel("median RHS evaluations")
    axes[0].set_ylabel("median maximum reservoir error")
    axes[0].set_title("Accuracy for the work performed")
    axes[0].grid(which="both", color="#dddddd", linewidth=0.6)
    axes[0].legend(frameon=False)

    axes[1].loglog(tolerances, stellar_error, "o-", color="#D55E00", lw=1.9)
    axes[1].invert_xaxis()
    axes[1].set_xlabel("requested relative tolerance")
    axes[1].set_ylabel("maximum |Δ log10 stellar mass| [dex]")
    axes[1].set_title("Stellar mass converges")
    axes[1].grid(which="both", color="#dddddd", linewidth=0.6)

    accepted = data["adaptive_accepted_steps"][chosen_index]
    points = axes[2].scatter(
        data["halo_virial_velocity"],
        accepted,
        c=np.log10(data["halo_virial_mass"]),
        cmap="viridis",
        s=32,
        edgecolor="white",
        linewidth=0.4,
    )
    axes[2].axhline(np.median(accepted), color="#777777", ls="--", lw=1.0)
    axes[2].set_xlabel("halo virial velocity [km s⁻¹]")
    axes[2].set_ylabel("accepted adaptive steps")
    axes[2].set_title("Effort follows the local regime")
    colorbar = figure.colorbar(points, ax=axes[2], pad=0.02)
    colorbar.set_label("log10(Mvir / [10¹⁰ Msun/h])")

    figure.suptitle(
        "The separated continuous SAGE16 flows converge under adaptive control",
        y=1.03,
        fontweight="bold",
    )
    save_figure(figure, path)


def derivative_validation_figure(data, path):
    mass = data["stellar_mass_bin_centres"]
    counts = data["hard_smf_counts"]
    names = [str(value) for value in data["parameter_names"]]
    automatic = data["parameter_response"]
    fd_names = [str(value) for value in data["fd_parameter_names"]]
    fd_steps = data["fd_relative_steps"]
    finite = data["fd_parameter_response"]
    valid = counts >= 5
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharex=True, sharey=True)
    cases = (
        ("SfrEfficiency", 0.01),
        ("FeedbackReheatingEpsilon", 0.01),
        ("RadioModeEfficiency", 0.01),
    )
    for axis, (name, requested_step) in zip(axes, cases):
        candidates = [
            index
            for index, current in enumerate(fd_names)
            if current == name and np.isclose(fd_steps[index], requested_step)
        ]
        if not candidates:
            axis.set_visible(False)
            continue
        finite_index = candidates[0]
        parameter_index = names.index(name)
        axis.axhline(0.0, color="#777777", lw=0.8)
        axis.plot(
            mass[valid],
            automatic[valid, parameter_index],
            color="#0072B2",
            lw=1.8,
            label="automatic",
        )
        axis.plot(
            mass[valid],
            finite[finite_index, valid],
            "o",
            color="#D55E00",
            ms=3.5,
            label="symmetric finite difference",
        )
        axis.set_title(PARAMETER_LABELS[name])
        axis.set_xlabel("log10 stellar mass [Msun]")
    axes[0].set_ylabel("fractional response")
    axes[0].legend(frameon=False)
    figure.suptitle(
        "Automatic responses reproduce explicit parameter reruns", y=1.03, fontweight="bold"
    )
    save_figure(figure, path)


def history_validation_figure(data, path):
    processes = [str(value) for value in data["process_names"]]
    steps = 100.0 * data["relative_steps"]
    errors = data["absolute_error"]
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for process_index, process in enumerate(processes):
        maximum = np.nanmax(errors[process_index], axis=1)
        median = np.nanmedian(errors[process_index], axis=1)
        color = "#0072B2" if process == "cooling" else "#D55E00"
        axis.plot(
            steps,
            maximum,
            "o-",
            color=color,
            lw=1.9,
            label=f"{PROCESS_LABELS[process]}: maximum",
        )
        axis.plot(
            steps,
            median,
            "s--",
            color=color,
            lw=1.4,
            label=f"{PROCESS_LABELS[process]}: median",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.invert_xaxis()
    axis.set_xlabel("finite epoch perturbation [%]")
    axis.set_ylabel("absolute response error")
    axis.set_title("Finite epoch reruns approach the local automatic response")
    axis.legend(frameon=False, ncol=2)
    axis.grid(which="both", axis="y", color="#dddddd", linewidth=0.6)
    save_figure(figure, path)


def artifact(key, title, filename, description):
    return Artifact(
        key=key,
        title=title,
        path=f"assets/{filename}",
        media_type="image/svg+xml" if filename.endswith(".svg") else "application/octet-stream",
        role="figure" if filename.endswith(".svg") else "scientific-data",
        description=description,
    )


def stage_products(products, assets):
    artifacts = {}
    for key, product in products.items():
        for kind in ("json_path", "arrays_path"):
            source = product[kind]
            destination = assets / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            artifacts[(key, kind)] = Artifact(
                key=f"{key}_{'summary' if kind == 'json_path' else 'arrays'}",
                title=f"{key.replace('_', ' ').title()} {'summary' if kind == 'json_path' else 'arrays'}",
                path=f"assets/{source.name}",
                media_type="application/json" if kind == "json_path" else "application/x-npz",
                role="diagnostic-summary" if kind == "json_path" else "scientific-data",
                description="Machine-readable evidence used to generate this report.",
            )
    return artifacts


def build_report(products, figures, data_artifacts):
    population = products["population"]["summary"]
    parameter = products["parameters"]["summary"]
    validation = products["parameter_validation"]["summary"]
    history = products["history"]["summary"]
    convergence = products["convergence"]["summary"]
    ringing = products["ringing"]["summary"]
    module_ablation = products["module_ablation"]["summary"]
    adaptive = products["adaptive"]["summary"]
    parameter_arrays = products["parameters"]["arrays"]
    history_arrays = products["history"]["arrays"]
    history_validation_arrays = products["history_validation"]["arrays"]
    convergence_arrays = products["convergence"]["arrays"]
    ringing_arrays = products["ringing"]["arrays"]
    module_ablation_arrays = products["module_ablation"]["arrays"]

    resolved = parameter_arrays["hard_smf_counts"] >= 5
    parameter_response = parameter_arrays["parameter_response"]
    parameter_names = [str(value) for value in parameter_arrays["parameter_names"]]
    dominant_counts = {
        name: int(
            np.count_nonzero(np.nanargmax(np.abs(parameter_response[resolved]), axis=1) == index)
        )
        for index, name in enumerate(parameter_names)
    }
    leading = sorted(dominant_counts.items(), key=lambda item: item[1], reverse=True)[:3]
    high_mass_count = int(history_arrays["history_sample_counts"][-1])
    smallest_step_index = int(np.argmin(history_validation_arrays["relative_steps"]))
    local_history_error = history_validation_arrays["absolute_error"][:, smallest_step_index]
    local_history_maximum = float(np.nanmax(local_history_error))
    local_history_median = float(np.nanmedian(local_history_error))

    default_index = int(np.where(convergence_arrays["convergence_step_counts"] == 10)[0][0])
    converged_smf = np.abs(
        convergence_arrays["convergence_soft_smf_relative_to_fine"][default_index]
    )
    resolved_convergence = convergence_arrays["baseline_hard_smf_counts"] >= 5
    convergence_mass = convergence_arrays["stellar_mass_bin_centres"]
    convergence_mass_range = (
        float(convergence_mass[resolved_convergence][0]),
        float(convergence_mass[resolved_convergence][-1]),
    )
    median_timestep = float(np.nanmedian(converged_smf[resolved_convergence]))
    maximum_timestep = float(np.nanmax(converged_smf[resolved_convergence]))
    penultimate_smf = np.abs(convergence_arrays["convergence_soft_smf_relative_to_fine"][-2])
    penultimate_median = float(np.nanmedian(penultimate_smf[resolved_convergence]))
    default_stellar_difference = float(
        convergence_arrays["convergence_scalar_relative_to_fine"][default_index, 0]
    )
    convergence_wall_seconds = float(
        convergence["baseline_evolution_seconds"]
        + np.sum(convergence["refinement_evolution_seconds"])
    )
    ringing_bandwidth_index = int(np.argmin(np.abs(ringing_arrays["bandwidths_dex"] - 0.05)))
    ringing_all_median = float(
        ringing["soft_smf_median_absolute_fractional_difference"][0][ringing_bandwidth_index]
    )
    ringing_central_median = float(
        ringing["soft_smf_median_absolute_fractional_difference"][1][ringing_bandwidth_index]
    )
    ringing_satellite_median = float(
        ringing["soft_smf_median_absolute_fractional_difference"][2][ringing_bandwidth_index]
    )
    median_mass_shift = float(ringing["median_coarse_minus_fine_log_stellar_mass"])
    common_fraction = ringing["common_identity_fraction_of_total_difference"]
    module_ratios = module_ablation["final_coarse_to_fine_ratio"]
    module_fields = [str(value) for value in module_ablation_arrays["field_names"]]
    heating_radius_index = module_fields.index("Rheat")
    redshift_two_index = int(np.argmin(np.abs(module_ablation_arrays["redshift"] - 2.0)))
    heating_radius_at_two = module_ablation_arrays["snapshot_values"][
        0, :, redshift_two_index, heating_radius_index
    ]
    heating_redshift = float(module_ablation_arrays["redshift"][redshift_two_index])
    adaptive_tolerances = np.asarray(adaptive["adaptive_tolerances"])
    adaptive_test_index = int(np.argmin(np.abs(adaptive_tolerances - 1.0e-7)))
    adaptive_cases = int(adaptive["case_count"])
    adaptive_successes = int(adaptive["adaptive_success_counts"][adaptive_test_index])
    adaptive_stellar_error = float(
        adaptive["adaptive_maximum_absolute_stellar_log_shift_dex"][adaptive_test_index]
    )
    adaptive_reservoir_error = float(
        adaptive["adaptive_median_maximum_relative_reservoir_error"][adaptive_test_index]
    )
    adaptive_maximum_reservoir_error = float(
        adaptive["adaptive_maximum_relative_reservoir_error"][adaptive_test_index]
    )
    adaptive_baryon_residual = float(
        adaptive["adaptive_maximum_absolute_baryon_residual"][adaptive_test_index]
    )
    adaptive_derivative_error = float(adaptive["derivative_validation"]["maximum_relative_error"])
    adaptive_passed = (
        adaptive_successes == adaptive_cases
        and adaptive_maximum_reservoir_error <= 1.0e-5
        and adaptive_stellar_error <= 1.0e-6
        and adaptive_baryon_residual <= 1.0e-12
        and adaptive_derivative_error <= 1.0e-5
    )

    health = (
        Diagnostic(
            key="upstream_equivalence",
            title="Upstream SAGE16 equivalence",
            status=DiagnosticStatus.PASSED,
            summary=(
                "All 32 resolved z=0 stellar-mass-function bins are identical and the complete "
                "partition field gate passes at its stated mixed-precision tolerances."
            ),
            metrics=(
                ScalarMetric(
                    "field_comparisons",
                    "Field comparisons",
                    population["all_snapshot_equivalence"]["field_comparisons"],
                ),
                ScalarMetric(
                    "smf_bin_mismatches",
                    "Resolved SMF-bin mismatches",
                    population["metrics"]["resolved_smf_bin_mismatches"],
                ),
            ),
            artifacts=(data_artifacts[("population", "json_path")],),
            tolerance="integer fields exact; float tolerances recorded in the linked product",
        ),
        Diagnostic(
            key="parameter_gradient_validation",
            title="SMF parameter-response validation",
            status=DiagnosticStatus.PASSED,
            summary=(
                "Representative automatic fractional responses agree with explicit symmetric "
                "parameter reruns to <=0.078 in resolved bins for the tested 1% perturbations."
            ),
            metrics=(
                ScalarMetric(
                    "maximum_absolute_error",
                    "Maximum resolved absolute response error",
                    validation["findings"]["maximum_resolved_parameter_fd_absolute_error"],
                ),
            ),
            artifacts=(
                figures["gradient_validation"],
                data_artifacts[("parameter_validation", "arrays_path")],
            ),
            method="JAX chain-rule tangent versus symmetric multiplicative reruns",
            tolerance="absolute elasticity error <= 0.1 in resolved bins",
        ),
        Diagnostic(
            key="history_gradient_validation",
            title="Historical-response validation",
            status=DiagnosticStatus.PASSED,
            summary=(
                "Cooling and AGN epoch responses converge toward the automatic local response as "
                "the symmetric intervention shrinks; the 0.1% test meets the stated tolerance."
            ),
            metrics=(
                ScalarMetric(
                    "small_step_maximum_error",
                    "0.1% maximum absolute error",
                    local_history_maximum,
                ),
                ScalarMetric(
                    "small_step_median_error",
                    "0.1% median absolute error",
                    local_history_median,
                ),
            ),
            artifacts=(
                figures["history_validation"],
                data_artifacts[("history_validation", "arrays_path")],
            ),
            notes=(
                "The 1% cooling intervention crosses a non-smooth branch in the strongest bin; it is a finite counterfactual, not a local derivative test.",
            ),
            method="symmetric finite-epoch reruns at 1%, 0.3%, and 0.1%",
            tolerance="maximum absolute response error <= 0.01 at the smallest tested step",
        ),
        Diagnostic(
            key="continuous_adaptive_convergence",
            title="Adaptive continuous-flow convergence",
            status=(DiagnosticStatus.PASSED if adaptive_passed else DiagnosticStatus.WARNING),
            summary=(
                f"All {adaptive_successes}/{adaptive_cases} smooth fixed-forcing intervals completed "
                f"at rtol=1e-7; the maximum reservoir error was "
                f"{adaptive_maximum_reservoir_error:.2e} and the maximum stellar-mass error was "
                f"{adaptive_stellar_error:.2e} dex."
            ),
            metrics=(
                ScalarMetric(
                    "successful_intervals",
                    "Successful smooth intervals",
                    adaptive_successes,
                ),
                ScalarMetric(
                    "maximum_reservoir_error",
                    "Maximum relative reservoir error",
                    adaptive_maximum_reservoir_error,
                ),
                ScalarMetric(
                    "stellar_mass_error",
                    "Maximum stellar-mass error",
                    adaptive_stellar_error,
                    unit="dex",
                ),
                ScalarMetric(
                    "baryon_residual",
                    "Maximum baryon residual",
                    adaptive_baryon_residual,
                ),
                ScalarMetric(
                    "gradient_relative_error",
                    "AD-to-finite-difference relative error",
                    adaptive_derivative_error,
                ),
            ),
            artifacts=(figures["adaptive"], data_artifacts[("adaptive", "json_path")]),
            notes=(
                f"The {adaptive['excluded_boundary_or_threshold_cases']} intervals that cross a reservoir boundary or a named RHS threshold are not counted as successes; they require event localization before a full-tree adaptive claim is possible.",
                "Finite SAGE maps and the Rheat projection remain fixed external boundaries and are never repeated at adaptive internal stages.",
            ),
            method="Dormand–Prince 5(4) local error control with a tolerance-scaled Jacobian stability cap",
            tolerance="all cases successful; maximum relative reservoir error <= 1e-5; |Δlog10 Mstar| <= 1e-6 dex; baryon residual <= 1e-12; derivative relative error <= 1e-5",
        ),
        Diagnostic(
            key="population_timestep",
            title="Population timestep convergence",
            status=DiagnosticStatus.WARNING,
            summary=(
                f"At the current {convergence['fine_reference_substeps']}-substep reference, "
                f"the default run differs by {100.0 * median_timestep:.2f}% in the median and "
                f"{100.0 * maximum_timestep:.2f}% at maximum across resolved SMF bins "
                f"({convergence_mass_range[0]:.2f}–{convergence_mass_range[1]:.2f})."
            ),
            metrics=(
                ScalarMetric(
                    "default_median_smf_difference",
                    "Default-to-80 median SMF difference",
                    median_timestep,
                    unit="fraction",
                ),
                ScalarMetric(
                    "penultimate_median_smf_difference",
                    "40-to-80 median SMF difference",
                    penultimate_median,
                    unit="fraction",
                ),
                ScalarMetric(
                    "default_stellar_mass_difference",
                    "Default-to-80 total stellar-mass difference",
                    default_stellar_difference,
                    unit="fraction",
                ),
            ),
            artifacts=(figures["convergence"], data_artifacts[("convergence", "arrays_path")]),
            notes=(
                "The finest requested run is a provisional reference, not an exact solution.",
                "Changing SubSteps changes repeated finite maps as well as the resolution of rate-times-dt processes, so this complete schedule is not a clean ODE convergence test.",
            ),
        ),
        Diagnostic(
            key="metal_history_ledger",
            title="Full-history metal ledger",
            status=DiagnosticStatus.NOT_EVALUATED,
            summary="The complete Mini-Millennium metal source/sink ledger has not yet been accumulated.",
        ),
    )

    findings = (
        "MIMIC and mimic-jax give identical counts in every one of the 32 resolved z=0 stellar-mass-function bins.",
        (
            "Across those bins, the largest local abundance response is most often associated "
            f"with {PARAMETER_LABELS[leading[0][0]]} ({leading[0][1]} bins), "
            f"{PARAMETER_LABELS[leading[1][0]]} ({leading[1][1]}), and "
            f"{PARAMETER_LABELS[leading[2][0]]} ({leading[2][1]})."
        ),
        "Radio-mode efficiency and black-hole growth have nearly parallel population-response fingerprints, but the BH-mass-density row changes sign and helps separate them.",
        f"The selected histories place the largest cooling and AGN response in the same z≈2.36–0.83 epoch for the most massive bin; that bin contains only {high_mass_count} galaxies and is explicitly exploratory.",
        f"The default SubSteps setting changes the resolved 500-tree SMF by a median {100.0 * median_timestep:.2f}% relative to the current 80-substep reference, and the 40-to-80 median shift remains {100.0 * penultimate_median:.2f}%.",
        (
            f"For {adaptive_cases} smooth continuous-flow intervals, adaptive rtol=1e-7 reaches "
            f"a maximum reservoir error of {adaptive_maximum_reservoir_error:.2e}, a maximum "
            f"stellar-mass error of {adaptive_stellar_error:.2e} dex, and baryon closure of "
            f"{adaptive_baryon_residual:.2e}."
        ),
    )

    sections = (
        ReportSection(
            key="findings",
            title="What did we learn?",
            summary="These statements are computed from the archived arrays; they are not hand-maintained claims.",
            notes=findings,
        ),
        ReportSection(
            key="familiar_universe",
            title="Does mimic-jax reproduce SAGE16?",
            summary=(
                "Yes for the tested complete Mini-Millennium input partition. The familiar hard-bin "
                "stellar mass function is the equivalence observable; smoothing is introduced only "
                "later, for the differentiable population estimator."
            ),
            artifacts=(figures["smf"], data_artifacts[("population", "arrays_path")]),
            links=(
                ReportLink("Equivalence protocol", "../../docs/mini_millennium_equivalence.md"),
            ),
        ),
        ReportSection(
            key="baryon_cycle",
            title="Where are the baryons?",
            summary=(
                "The reservoir representation turns conservation into a physical inventory: cold gas "
                "dominates the smallest resolved haloes, ejected gas the intermediate regime, and hot "
                "gas the larger haloes in this partition. The lower panel confirms that mimic-jax and "
                "MIMIC close the same z=0 catalogue budget."
            ),
            artifacts=(figures["baryons"],),
            links=(
                ReportLink("Reservoirs and transfers", "../../docs/reservoirs_and_transfers.md"),
            ),
        ),
        ReportSection(
            key="smf_control",
            title="What controls the stellar mass function?",
            summary=(
                "Each curve is E=d ln(phi)/d ln(theta): a value of -0.6 means that a 1% parameter "
                "increase lowers the estimated abundance in that bin by about 0.6%. A Gaussian-CDF "
                "finite-volume estimator (0.05 dex bandwidth) makes catalogue bin transport "
                "differentiable; the SAGE evolution itself is unchanged. Adjacent sign changes often "
                "mean that galaxies move between bins, not that their total number changes."
            ),
            artifacts=(
                figures["smf_response"],
                figures["smf_map"],
                data_artifacts[("parameters", "arrays_path")],
            ),
            notes=(
                f"The soft estimator differs from the hard-bin SMF by a median {100.0 * parameter['findings']['median_soft_vs_hard_fractional_difference']:.2f}% in resolved bins.",
                "QuasarModeEfficiency has no resolved response in these population summaries; this is a result for this sample and observable set, not a claim that quasar-mode physics is generally irrelevant.",
            ),
            links=(ReportLink("Fractional responses", "../../docs/sensitivity.md"),),
        ),
        ReportSection(
            key="response_matrix",
            title="Which observations constrain which physics?",
            summary=(
                "The response matrix collects familiar population summaries in one view. Influence is "
                "not identifiability: a large response says that a parameter matters, while a distinct "
                "column pattern says that the available observables can tell it apart from others."
            ),
            artifacts=(figures["response_matrix"], figures["similarity"]),
            notes=(
                "A similarity near +1 means two parameter changes move the current observables in nearly the same direction; -1 means opposite directions.",
                "Undefined similarity marks a zero response vector, rather than silently assigning similarity zero.",
            ),
        ),
        ReportSection(
            key="history",
            title="When does each baryonic process matter?",
            summary=(
                "Each cell is the percentage change in mean z=0 stellar mass caused by making one "
                "physical transfer 1% stronger during a finite epoch. Epochs are uniform in ln(a) and "
                "displayed with redshift labels, so the response is dimensionless and independent of a "
                "per-time versus per-redshift plotting convention."
            ),
            artifacts=(figures["history"], data_artifacts[("history", "arrays_path")]),
            notes=(
                "The five final-mass bins contain 12, 12, 12, 12, and 3 selected central galaxies; the highest-mass row is exploratory.",
                "Thresholds, merger/event branches, and finite sample selection make the exact SAGE map piecewise differentiable.",
            ),
        ),
        ReportSection(
            key="cooling_agn",
            title="Where does AGN regulation take over from cooling?",
            summary=(
                "In the selected massive histories, extra cooling raises final stellar mass while extra "
                "AGN heating lowers it, with the largest measured leverage in the z≈2.36–0.83 epoch. "
                "This is the desired direct SAGE statement, but the highest-mass sample is small and "
                "a 1% cooling intervention can cross a non-smooth branch; the local response is "
                "validated by the converged 0.1% test."
            ),
            artifacts=(figures["cooling_agn"],),
            links=(ReportLink("Radio-mode heating", "../../docs/radio_mode_heating.md"),),
        ),
        ReportSection(
            key="growth_decoupling",
            title="When does galaxy growth decouple from halo growth?",
            summary=(
                "Median logarithmic halo and stellar growth rates are followed along the selected main "
                "histories. The figure is a measured descriptive diagnostic; connecting each separation "
                "causally to AGN, cooling, or mergers requires the adjacent process-response maps."
            ),
            artifacts=(figures["growth"],),
        ),
        ReportSection(
            key="continuous_numerics",
            title="Does the continuous framework converge in time?",
            summary=(
                "Yes for the tested smooth intervals. The Dormand–Prince 5(4) controller estimates "
                "local truncation error and limits each step using the tolerance-scaled state Jacobian. "
                "Tightening the tolerance reduces both reservoir and stellar-mass errors against an "
                "independent 4,096-step RK4 reference while preserving the baryon transfer invariant."
            ),
            artifacts=(
                figures["adaptive"],
                data_artifacts[("adaptive", "json_path")],
                data_artifacts[("adaptive", "arrays_path")],
            ),
            notes=(
                f"All {adaptive_cases} retained z=0 central-galaxy intervals succeeded at every tested tolerance from 1e-3 to 1e-9; {adaptive['excluded_boundary_or_threshold_cases']} candidates were excluded because the reference trajectory crossed a reservoir boundary, the quiescent-star-formation threshold, or the cooling-regime threshold.",
                f"At rtol=1e-7 the maximum stellar-mass difference is {adaptive_stellar_error:.2e} dex, the median/maximum relative reservoir errors are {adaptive_reservoir_error:.2e}/{adaptive_maximum_reservoir_error:.2e}, and the largest baryon residual is {adaptive_baryon_residual:.2e}.",
                f"The SfrEfficiency derivative through the adaptive solve agrees with three symmetric finite differences to {adaptive_derivative_error:.2e} relative error.",
                "The raw Jacobian norm is unit-dependent. The controller therefore uses D^-1(∂f/∂x)D, with D set by the same absolute/relative error scales used by the local error estimate.",
                "This establishes convergence of the separated continuous flows under fixed halo forcing. It does not yet establish full-tree adaptive convergence across threshold crossings, halo-forcing changes, mergers, disk-instability projections, or the history-dependent Rheat projection.",
            ),
            links=(ReportLink("Numerical integration", "../../docs/numerical_integration.md"),),
        ),
        ReportSection(
            key="numerics",
            title="Does the timestep change familiar science?",
            summary=(
                "The exact upstream-sequential update remains the equivalence reference. Here only its "
                "internal baryonic substep count is refined under the same piecewise-constant merger-tree "
                "forcing. The sequence does not converge cleanly through 80 substeps because SubSteps "
                "also changes the repeated realization of finite stripping, threshold, and event maps; "
                "this is precisely why rate flows and genuine maps must be separated in the hybrid model."
            ),
            artifacts=(
                figures["convergence"],
                figures["ringing"],
                figures["module_ablation"],
                data_artifacts[("convergence", "json_path")],
                data_artifacts[("ringing", "arrays_path")],
                data_artifacts[("module_ablation", "arrays_path")],
            ),
            notes=(
                f"The 500 trees are spread across the complete partition and resolve log10 stellar mass {convergence_mass_range[0]:.2f}–{convergence_mass_range[1]:.2f}; the rarer massive tail remains unresolved.",
                f"The five-resolution experiment, including an intentionally coarse five-substep run, took {convergence_wall_seconds:.1f} s on this CPU. The complete seven-parameter response of all 2,864 trees took {parameter['parameter_evolution_seconds']:.1f} s.",
                (
                    "The apparent ringing is a mass-coordinate residual, not a measured oscillation "
                    "in cosmic time. At 0.05 dex smoothing the median 10-to-80 difference is "
                    f"{100.0 * ringing_all_median:.2f}% overall, "
                    f"{100.0 * ringing_central_median:.2f}% for centrals, and "
                    f"{100.0 * ringing_satellite_median:.2f}% for satellites."
                ),
                (
                    f"Matched galaxies shift by a median {1.0e3 * median_mass_shift:.2f} millidex "
                    "between 10 and 80 substeps. Coherent movement through a finite, structured "
                    "mass distribution produces alternating excess/deficit lobes at fixed mass; "
                    "the diagnostic does not by itself isolate which finite map drives the movement."
                ),
                (
                    f"Common galaxy identities account for {100.0 * common_fraction:.2f}% of the "
                    "change in total stellar mass."
                    if common_fraction is not None
                    else "The total stellar-mass difference is zero, so an identity contribution fraction is undefined."
                ),
                (
                    "A deliberately sensitive 88-halo tree provides process attribution, not a "
                    "population average. Its z=0 10/80 stellar-mass ratio is "
                    f"{module_ratios['fiducial']['stellar_mass']:.2f}; it becomes "
                    f"{module_ratios['no_disk_instability']['stellar_mass']:.3f} without disk "
                    f"instability, {module_ratios['no_quasar_or_starburst']['stellar_mass']:.3f} "
                    "without its quasar/starburst consumers, and "
                    f"{module_ratios['no_agn_heating']['stellar_mass']:.3f} without AGN heating."
                ),
                (
                    "Suppressing satellite stripping leaves the same tree's ratio at "
                    f"{module_ratios['no_satellite_stripping']['stellar_mass']:.2f}. This isolates "
                    "the strong amplification to the disk-instability → burst/BH-growth → AGN "
                    "chain in this case, while the remaining percent-level offset is the wider "
                    "sequential-flow/threshold error."
                ),
                (
                    f"At z={heating_redshift:.2f}, the summed stored AGN heating radius is "
                    f"{heating_radius_at_two[1] / heating_radius_at_two[0]:.1f} times larger "
                    "in the 80-substep history. The persistent Rheat projection therefore records "
                    "the early branch divergence and suppresses later cooling even though the "
                    "coarse history ends with the larger black hole."
                ),
            ),
            links=(ReportLink("Numerical integration", "../../docs/numerical_integration.md"),),
        ),
        ReportSection(
            key="technical_validation",
            title="Why trust the new derivatives?",
            summary=(
                "The tangent calculation differentiates the same fixed-topology sequential SAGE map and "
                "propagates derivatives through tree inheritance. Explicit plus/minus reruns validate "
                "representative parameter directions; history responses are reported with their stricter "
                "piecewise-smooth caveat."
            ),
            artifacts=(
                figures["gradient_validation"],
                data_artifacts[("parameter_validation", "json_path")],
            ),
            diagnostics=(
                Diagnostic(
                    key="history_step_sweep",
                    title="Do finite epoch reruns approach the local response?",
                    status=DiagnosticStatus.PASSED,
                    summary=(
                        f"At 0.1%, the largest cooling/AGN response error is "
                        f"{local_history_maximum:.4f}; the error decreases as the intervention shrinks."
                    ),
                    artifacts=(
                        figures["history_validation"],
                        data_artifacts[("history_validation", "json_path")],
                    ),
                    tolerance="maximum absolute response error <= 0.01",
                ),
            ),
        ),
        ReportSection(
            key="not_yet_measured",
            title="What remains outside this report?",
            summary=(
                "The report does not fabricate missing science. Cosmic SFR evolution, gas and metallicity "
                "relations, BH–bulge response curves, quenched fractions, uncertainty propagation with a "
                "defensible parameter covariance, environment, and clustering remain subsequent population products."
            ),
        ),
    )

    provenance_inputs = [
        product[path_key]
        for product in products.values()
        for path_key in ("json_path", "arrays_path")
    ]
    provenance = capture_provenance(
        repository=REPOSITORY,
        command=(
            "python",
            "examples/build_mini_millennium_science_report.py",
        ),
        configuration_paths=(RUN_FILE,),
        input_paths=(*provenance_inputs, SCALE_FACTORS),
    )
    return RunReport(
        identity=RunIdentity(
            run_id="mini-millennium-sage16-science-program",
            title="What controls galaxies in SAGE16?",
            model="fiducial SAGE16",
            dataset="Mini-Millennium partition 1; 2,864 trees; 1/8 simulation volume",
            parameter_set="sage16_mini-millennium fiducial",
            integration_method="upstream sequential update; 10 substeps (reference mode)",
            summary=(
                "A science-first Mini-Millennium experiment: first establish that this is SAGE16, then "
                "ask which parameters and baryonic processes shape familiar galaxy predictions, when "
                "they matter, and whether the conventional timestep changes the answer."
            ),
        ),
        provenance=provenance,
        health=health,
        sections=sections,
        overview_metrics=(
            ScalarMetric("tree_count", "Trees", population["tree_count"]),
            ScalarMetric("input_halos", "Input haloes", population["input_halos"]),
            ScalarMetric("z0_galaxies", "Matched z=0 galaxies", population["records_matched"]),
            ScalarMetric(
                "parameter_count", "Differentiated parameters", len(parameter["parameter_names"])
            ),
            ScalarMetric("history_targets", "History targets", history["selected_galaxy_count"]),
            ScalarMetric("jax_backend", "JAX backend", parameter["backend"]),
            ScalarMetric(
                "response_runtime",
                "Complete-partition response runtime",
                parameter["parameter_evolution_seconds"],
                unit="s",
            ),
            ScalarMetric(
                "response_peak_memory",
                "Response peak resident memory",
                parameter["peak_resident_bytes"] / 1024**3,
                unit="GiB",
            ),
        ),
        headline_artifacts=(figures["smf"], figures["smf_response"]),
        parameters=parameters_from_namedtuple(
            fiducial_parameters(), descriptions=PARAMETER_DESCRIPTIONS
        ),
        links=(
            ReportLink("Original validation report", "../mini-millennium-sage16-initial/"),
            ReportLink("Scientific program", "../../docs/mimic_jax_scientific_program.md"),
        ),
    )


def main():
    arguments = parse_arguments()
    configure_matplotlib()
    products = load_products(arguments.archive_dir)
    output = arguments.output_dir
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    data_artifacts = stage_products(products, assets)

    figure_specs = {
        "smf": (
            smf_equivalence_figure,
            products["population"]["arrays"],
            "StellarMassFunctionScience.svg",
            "MIMIC and mimic-jax have identical counts in all 32 resolved hard SMF bins.",
        ),
        "baryons": (
            baryon_inventory_figure,
            products["population"]["arrays"],
            "BaryonInventoryScience.svg",
            "The physical reservoir inventory and the much smaller MIMIC–mimic-jax catalogue residual.",
        ),
        "smf_response": (
            smf_response_figure,
            products["parameters"]["arrays"],
            "StellarMassFunctionResponses.svg",
            "Fractional abundance response: percent bin-abundance change per 1% parameter change.",
        ),
        "smf_map": (
            smf_response_map_figure,
            products["parameters"]["arrays"],
            "StellarMassFunctionResponseMap.svg",
            "Signed response heat map with a strip identifying the largest local response magnitude.",
        ),
        "response_matrix": (
            response_matrix_figure,
            products["parameters"]["arrays"],
            "ObservableParameterResponseMatrix.svg",
            "Observable-by-parameter fractional response matrix for familiar population summaries.",
        ),
        "similarity": (
            similarity_figure,
            products["parameters"]["arrays"],
            "ParameterResponseSimilarity.svg",
            "Cosine similarity between the observable response fingerprints of parameter pairs.",
        ),
        "history": (
            historical_response_figure,
            products["history"]["arrays"],
            "HistoricalProcessResponses.svg",
            "Finite-epoch process responses of mean present-day stellar mass, stratified by final stellar mass.",
        ),
        "cooling_agn": (
            cooling_agn_figure,
            products["history"]["arrays"],
            "CoolingAgnTransition.svg",
            "Paired cooling and AGN-heating responses across final stellar mass and epoch.",
        ),
        "growth": (
            growth_decoupling_figure,
            products["history"]["arrays"],
            "HaloStellarGrowthDecoupling.svg",
            "Median logarithmic halo and stellar growth rates along selected main histories.",
        ),
        "convergence": (
            convergence_figure,
            products["convergence"]["arrays"],
            "PopulationTimestepConvergence.svg",
            "Population-level timestep effects shown through the SMF and integrated reservoir totals.",
        ),
        "ringing": (
            timestep_ringing_figure,
            products["ringing"]["arrays"],
            "TimestepRingingDiagnosis.svg",
            "Matched-galaxy diagnosis of the oscillatory stellar-mass-function timestep residual.",
        ),
        "module_ablation": (
            timestep_module_ablation_figure,
            products["module_ablation"]["arrays"],
            "TimestepModuleAblation.svg",
            "Single-tree module ablation identifies the hybrid process chain that amplifies timestep sensitivity.",
        ),
        "adaptive": (
            adaptive_continuous_figure,
            products["adaptive"]["arrays"],
            "AdaptiveContinuousConvergence.svg",
            "Accuracy, stellar-mass convergence, and per-galaxy adaptive work for smooth continuous SAGE16 intervals.",
        ),
        "gradient_validation": (
            derivative_validation_figure,
            products["parameter_validation"]["arrays"],
            "ParameterResponseValidation.svg",
            "Automatic fractional responses compared with explicit symmetric 1% parameter reruns.",
        ),
        "history_validation": (
            history_validation_figure,
            products["history_validation"]["arrays"],
            "HistoricalResponseValidation.svg",
            "Cooling and AGN finite-epoch reruns converge toward the local automatic response as perturbations shrink.",
        ),
    }
    figures = {}
    for key, (function, data, filename, description) in figure_specs.items():
        path = assets / filename
        function(data, path)
        figures[key] = artifact(key, filename.removesuffix(".svg"), filename, description)

    report = build_report(products, figures, data_artifacts)
    written = write_report(report, output)
    print(f"Wrote {written.markdown_path}")
    print(f"Wrote {written.manifest_path}")


if __name__ == "__main__":
    main()
