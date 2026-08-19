#!/usr/bin/env python3
"""Build the standalone SAGE16 galaxy-memory and linear-response report."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
DEFAULT_OUTPUT = REPOSITORY / "reports/sage16-linear-response"
DEFAULT_JSON = REPOSITORY / "archive/mini-millennium-sage16-linear-response.json"
DEFAULT_ARRAYS = REPOSITORY / "archive/mini-millennium-sage16-linear-response.npz"
RUN_FILE = REPOSITORY / "models/sage16/input/sage16_mini-millennium.yaml"
TREE_FILE = REPOSITORY / "simulations/mini-millennium/snapshots/trees_063.1"
SCALE_FACTORS = REPOSITORY / "simulations/mini-millennium/mini-millennium.a_list"

COLORS = {
    "hot": "#D55E00",
    "cold": "#0072B2",
    "ejected": "#009E73",
    "stars": "#E69F00",
    "black_hole": "#6A3D9A",
    "linear": "#333333",
    "agn": "#CC79A7",
    "no_agn": "#777777",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--input-arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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
            "svg.hashsalt": "mimic-jax-sage16-linear-response",
        }
    )


def save_figure(figure, path):
    metadata = {"Date": None} if path.suffix.lower() == ".svg" else None
    figure.savefig(path, bbox_inches="tight", facecolor="white", metadata=metadata)
    plt.close(figure)


def markdown(*blocks):
    return "\n\n".join(block.strip() for block in blocks if block.strip())


def figure_artifact(key, title, description):
    return Artifact(
        key=key,
        title=title,
        path=f"assets/{key}.svg",
        media_type="image/svg+xml",
        role="figure",
        description=description,
    )


def add_box(axis, centre, label, color, width=1.6, height=0.72):
    x_value, y_value = centre
    patch = FancyBboxPatch(
        (x_value - width / 2.0, y_value - height / 2.0),
        width,
        height,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        linewidth=1.6,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.13),
    )
    axis.add_patch(patch)
    axis.text(x_value, y_value, label, ha="center", va="center", weight="bold", color="#222222")


def add_arrow(axis, start, end, label, color="#555555", style="-", curve=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.5,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={curve}",
    )
    axis.add_patch(arrow)
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    axis.text(
        midpoint[0],
        midpoint[1] + (0.14 if curve >= 0.0 else -0.18),
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9},
    )


def baryon_cycle_figure(path):
    figure, axis = plt.subplots(figsize=(9.0, 5.3))
    axis.set_xlim(-0.4, 9.4)
    axis.set_ylim(-0.4, 5.4)
    axis.axis("off")
    positions = {
        "hot": (4.0, 4.2),
        "cold": (4.0, 2.3),
        "ejected": (1.2, 4.2),
        "stars": (6.7, 2.3),
        "black_hole": (8.2, 4.2),
    }
    add_box(axis, positions["hot"], r"Hot halo  $M_{\rm hot}$", COLORS["hot"])
    add_box(axis, positions["cold"], r"Cold disk  $M_{\rm cold}$", COLORS["cold"])
    add_box(axis, positions["ejected"], r"Ejected  $M_{\rm eject}$", COLORS["ejected"])
    add_box(axis, positions["stars"], r"Stars  $M_\star$", COLORS["stars"])
    add_box(axis, positions["black_hole"], r"Black hole  $M_{\rm BH}$", COLORS["black_hole"])
    add_arrow(axis, (2.45, 5.05), (3.55, 4.55), "infall / halo forcing")
    add_arrow(axis, (4.0, 3.78), (4.0, 2.72), "cooling", COLORS["cold"])
    add_arrow(axis, (4.9, 2.3), (5.8, 2.3), "star formation", COLORS["stars"])
    add_arrow(axis, (3.15, 2.55), (1.8, 3.85), "SN ejection", COLORS["ejected"], curve=-0.08)
    add_arrow(axis, (1.9, 4.2), (3.1, 4.2), "reincorporation", COLORS["ejected"])
    add_arrow(axis, (6.9, 2.7), (7.8, 3.82), "BH growth", COLORS["black_hole"])
    add_arrow(axis, (7.4, 4.2), (4.9, 4.2), "AGN heating memory", COLORS["agn"], style="--")
    add_arrow(axis, (5.9, 2.55), (4.85, 3.85), "recycling / reheating", "#666666", curve=0.12)
    axis.text(
        4.5,
        0.45,
        "Continuous transfers between events  •  explicit maps at mergers and projections",
        ha="center",
        fontsize=10.5,
        color="#333333",
    )
    axis.set_title("The familiar SAGE baryon cycle is the dynamical system", pad=8)
    save_figure(figure, path)


def pulse_figure(data, summary, path):
    times = data["pulse_times_gyr"]
    amplitudes = data["pulse_amplitudes"]
    selected = int(np.argmin(np.abs(amplitudes - 0.01)))
    errors = 100.0 * data["pulse_normalized_rmse"]
    figure = plt.figure(figsize=(10.4, 6.3))
    grid = figure.add_gridspec(2, 2, width_ratios=(3.3, 1.35), hspace=0.12, wspace=0.35)
    cold_axis = figure.add_subplot(grid[0, 0])
    sfr_axis = figure.add_subplot(grid[1, 0], sharex=cold_axis)
    error_axis = figure.add_subplot(grid[:, 1])
    cold_axis.plot(
        times,
        100.0 * data["pulse_nonlinear_cold_fraction"][selected],
        color=COLORS["cold"],
        lw=2.2,
        label="full nonlinear SAGE flow",
    )
    cold_axis.plot(
        times,
        100.0 * data["pulse_linear_cold_fraction"][selected],
        color=COLORS["linear"],
        lw=1.7,
        ls="--",
        label="local linear prediction",
    )
    sfr_axis.plot(
        times,
        100.0 * data["pulse_nonlinear_sfr_fraction"][selected],
        color=COLORS["stars"],
        lw=2.2,
    )
    sfr_axis.plot(
        times,
        100.0 * data["pulse_linear_sfr_fraction"][selected],
        color=COLORS["linear"],
        lw=1.7,
        ls="--",
    )
    pulse_duration = summary["pulse_case"]["pulse_duration_gyr"]
    response_time = summary["pulse_case"]["dominant_cooling_memory_gyr"]
    for axis in (cold_axis, sfr_axis):
        axis.axvspan(0.0, pulse_duration, color="#56B4E9", alpha=0.12)
        axis.axvline(response_time, color="#777777", ls=":", lw=1.0)
        axis.axhline(0.0, color="#777777", lw=0.8)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    cold_axis.set_ylabel("cold-gas change [%]")
    cold_axis.set_title("A brief cooling enhancement rises, overshoots, then relaxes")
    cold_axis.legend(frameon=False, loc="upper right")
    cold_axis.tick_params(labelbottom=False)
    sfr_axis.set_ylabel("SFR change [%]")
    sfr_axis.set_xlabel("time after perturbation [Gyr]")
    error_axis.semilogx(
        100.0 * amplitudes, errors[:, 0], "o-", color=COLORS["cold"], label="cold gas"
    )
    error_axis.semilogx(100.0 * amplitudes, errors[:, 1], "s--", color=COLORS["stars"], label="SFR")
    error_axis.axhline(5.0, color="#777777", ls=":", label="5% criterion")
    error_axis.set_xlabel("cooling pulse amplitude [%]")
    error_axis.set_ylabel("linear-prediction normalized RMSE [%]")
    error_axis.set_title("Where is the local model valid?")
    error_axis.legend(frameon=False)
    error_axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    save_figure(figure, path)


def filter_figure(data, summary, path):
    timescale = data["inverse_angular_frequency_gyr"]
    response = np.abs(data["fractional_frequency_response"])
    tau_equilibrium = summary["local_modes"]["recipe_times_gyr"]["cold-gas regulator"]
    slow_gain = float(response[-1, 1])
    peak_index = int(np.argmax(response[:, 1]))
    simple = slow_gain / np.sqrt(1.0 + (tau_equilibrium / timescale) ** 2)
    figure, axis = plt.subplots(figsize=(8.7, 5.4))
    axis.loglog(
        timescale, response[:, 0], color=COLORS["cold"], lw=2.0, label="cold gas: coupled SAGE"
    )
    axis.loglog(timescale, response[:, 1], color=COLORS["stars"], lw=2.3, label="SFR: coupled SAGE")
    axis.loglog(
        timescale, simple, color="#555555", lw=1.6, ls="--", label="one-reservoir regulator shape"
    )
    axis.axvline(tau_equilibrium, color="#777777", ls=":", lw=1.2)
    axis.text(tau_equilibrium * 1.08, axis.get_ylim()[0] * 1.5, r"$\tau_{\rm eq}$", color="#555555")
    axis.set_xlabel("inverse angular frequency [Gyr]")
    axis.set_ylabel("fractional response amplitude")
    axis.plot(timescale[peak_index], response[peak_index, 1], "o", color=COLORS["stars"], ms=5.5)
    axis.annotate(
        f"maximum response near {timescale[peak_index]:.2f} Gyr",
        xy=(timescale[peak_index], response[peak_index, 1]),
        xytext=(18, -32),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=8.5,
    )
    axis.set_title("Coupled SAGE filters rapid changes but responds most near 0.5 Gyr")
    axis.grid(which="both", color="#e4e4e4", linewidth=0.55)
    axis.legend(frameon=False, loc="lower right")
    save_figure(figure, path)


def mode_figure(data, path):
    coupled_times = np.asarray(data["selected_mode_times_gyr"])
    poles = np.asarray(data["selected_mode_poles_per_gyr"])
    recipes = [str(value) for value in data["recipe_names"]]
    recipe_times = np.asarray(data["recipe_times_gyr"])
    compositions = np.asarray(data["selected_mode_composition"])
    state_names = [str(value) for value in data["response_state_names"]]
    mode_labels = []
    for index, pole in enumerate(poles):
        qualifier = "damped oscillatory" if abs(np.imag(pole)) > 1.0e-8 else "relaxation"
        mode_labels.append(f"coupled mode {index + 1}\n({qualifier})")
    figure, (time_axis, composition_axis) = plt.subplots(
        1, 2, figsize=(11.0, 5.4), gridspec_kw={"width_ratios": (1.05, 1.4)}
    )
    labels = mode_labels + recipes
    values = np.concatenate((coupled_times, recipe_times))
    colors = ["#0072B2"] * len(mode_labels) + ["#B8B8B8"] * len(recipes)
    positions = np.arange(len(labels))[::-1]
    time_axis.barh(positions, values, color=colors, edgecolor="white")
    time_axis.set_yticks(positions, labels)
    time_axis.set_xscale("log")
    time_axis.set_xlabel("timescale [Gyr]")
    time_axis.set_title("Coupling creates response times, not just recipe times")
    time_axis.grid(axis="x", which="both", color="#dddddd", linewidth=0.6)
    bottoms = np.zeros(len(mode_labels))
    component_colors = [COLORS["cold"], COLORS["hot"], COLORS["ejected"], COLORS["stars"]]
    component_labels = {
        "ColdGas": "cold gas",
        "HotGas": "hot gas",
        "EjectedGas": "ejected gas",
        "StellarMass": "stars",
    }
    for index, name in enumerate(state_names):
        composition_axis.bar(
            np.arange(len(mode_labels)),
            compositions[:, index],
            bottom=bottoms,
            color=component_colors[index],
            label=component_labels.get(name, name),
        )
        bottoms += compositions[:, index]
    composition_axis.set_xticks(
        np.arange(len(mode_labels)), [f"mode {index + 1}" for index in range(len(mode_labels))]
    )
    composition_axis.set_ylabel("absolute eigenvector share")
    composition_axis.set_ylim(0.0, 1.0)
    composition_axis.set_title("Which reservoirs participate in each response?")
    composition_axis.legend(frameon=False, ncol=2, loc="upper center")
    save_figure(figure, path)


def feedback_response_figure(data, path):
    timescale = np.asarray(data["inverse_angular_frequency_gyr"])
    names = [str(value) for value in data["feedback_variant_names"]]
    responses = np.abs(data["feedback_variant_response"])
    mode_times = np.asarray(data["feedback_variant_mode_times_gyr"])
    colors = [COLORS["stars"], "#D55E00", "#009E73"]
    linestyles = ["-", "--", "-."]
    figure, (response_axis, time_axis) = plt.subplots(
        1, 2, figsize=(10.7, 4.9), gridspec_kw={"width_ratios": (1.8, 1.0)}
    )
    for name, response, color, linestyle in zip(names, responses, colors, linestyles):
        response_axis.loglog(timescale, response, color=color, ls=linestyle, lw=2.1, label=name)
        response_axis.set_xlabel("inverse angular frequency [Gyr]")
    response_axis.set_ylabel("cooling-to-SFR fractional response")
    response_axis.set_title("Feedback changes which cooling variations reach SFR")
    response_axis.grid(which="both", color="#e4e4e4", linewidth=0.55)
    response_axis.legend(frameon=False)
    positions = np.arange(len(names))
    time_axis.barh(positions, mode_times, color=colors)
    time_axis.set_yticks(
        positions, [name.replace(" locally removed", "\nremoved") for name in names]
    )
    time_axis.invert_yaxis()
    time_axis.set_xlabel("dominant response time [Gyr]")
    time_axis.set_title("Local memory time")
    time_axis.grid(axis="x", color="#dddddd", linewidth=0.6)
    save_figure(figure, path)


def response_map_figure(data, path):
    values = np.asarray(data["memory_timescale_gyr"])
    counts = np.asarray(data["memory_counts"])
    mass_edges = np.asarray(data["memory_mass_edges"])
    redshifts = np.asarray(data["memory_redshifts"])
    finite = values[np.isfinite(values) & (values > 0.0)]
    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    image = axis.imshow(
        values,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=max(0.03, float(np.min(finite))), vmax=float(np.max(finite))),
        extent=(mass_edges[0], mass_edges[-1], -0.5, len(redshifts) - 0.5),
    )
    axis.set_yticks(np.arange(len(redshifts)), [f"{value:.2g}" for value in redshifts])
    axis.set_xlabel("log10 halo mass [solar masses]")
    axis.set_ylabel("redshift")
    axis.set_title("SAGE remembers cooling perturbations longer in massive haloes")
    for redshift_index in range(values.shape[0]):
        for mass_index in range(values.shape[1]):
            if np.isfinite(values[redshift_index, mass_index]):
                axis.text(
                    (mass_edges[mass_index] + mass_edges[mass_index + 1]) / 2.0,
                    redshift_index,
                    f"{values[redshift_index, mass_index]:.2g}\n(n={counts[redshift_index, mass_index]})",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=(
                        "white"
                        if values[redshift_index, mass_index] < np.median(finite)
                        else "#111111"
                    ),
                )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("dominant cooling-to-cold-gas response time [Gyr]")
    save_figure(figure, path)


def agn_response_figure(data, summary, path):
    timescale = data["agn_inverse_angular_frequency_gyr"]
    fiducial = np.abs(data["agn_fiducial_fractional_response"])
    removed = np.abs(data["agn_removed_fractional_response"])
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), sharex=True)
    for output_index, (axis, label, color) in enumerate(
        ((axes[0], "cold gas", COLORS["cold"]), (axes[1], "SFR", COLORS["stars"]))
    ):
        axis.loglog(
            timescale,
            fiducial[:, output_index],
            color=color,
            lw=2.3,
            label="fiducial local AGN coupling",
        )
        axis.loglog(
            timescale,
            removed[:, output_index],
            color=COLORS["no_agn"],
            lw=1.8,
            ls="--",
            label="AGN coupling locally removed",
        )
        axis.set_xlabel("inverse angular frequency [Gyr]")
        axis.set_title(f"Cooling supply → {label}")
        axis.grid(which="both", color="#e4e4e4", linewidth=0.55)
    axes[0].set_ylabel("fractional response amplitude")
    axes[1].legend(frameon=False, loc="lower right")
    figure.suptitle(
        "Stored AGN heating makes a massive SAGE galaxy less responsive to new cooling supply",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.02,
        f"Local fiducial-background comparison at z={summary['agn_case']['redshift']:.2f}; this is not a rerun of the full history.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    save_figure(figure, path)


def agn_map_figure(data, path):
    values = np.asarray(data["agn_suppression"])
    counts = np.asarray(data["agn_counts"])
    mass_edges = np.asarray(data["agn_mass_edges"])
    redshifts = np.asarray(data["agn_redshifts"])
    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    image = axis.imshow(
        values,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        extent=(mass_edges[0], mass_edges[-1], -0.5, len(redshifts) - 0.5),
    )
    axis.set_yticks(np.arange(len(redshifts)), [f"{value:.2g}" for value in redshifts])
    axis.set_xlabel("log10 halo mass [solar masses]")
    axis.set_ylabel("redshift")
    axis.set_title("The stored SAGE heating radius suppresses cooling first in massive haloes")
    for redshift_index in range(values.shape[0]):
        for mass_index in range(values.shape[1]):
            if np.isfinite(values[redshift_index, mass_index]):
                axis.text(
                    (mass_edges[mass_index] + mass_edges[mass_index + 1]) / 2.0,
                    redshift_index,
                    f"{100.0 * values[redshift_index, mass_index]:.0f}%\n(n={counts[redshift_index, mass_index]})",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if values[redshift_index, mass_index] < 0.55 else "#111111",
                )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("instantaneous cooling suppression from prior AGN heating")
    save_figure(figure, path)


def stochastic_figure(data, path):
    timescale = data["inverse_angular_frequency_gyr"]
    omega = 1.0 / timescale
    response = np.abs(data["fractional_frequency_response"][:, 1])
    input_power = 1.0 / (1.0 + (0.5 * omega) ** 2)
    output_power = response**2 * input_power
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.loglog(
        timescale,
        input_power,
        color="#777777",
        lw=1.8,
        ls="--",
        label="illustrative cooling variability",
    )
    axis.loglog(
        timescale, output_power, color=COLORS["stars"], lw=2.3, label="predicted SFR variability"
    )
    axis.set_xlabel("inverse angular frequency [Gyr]")
    axis.set_ylabel("illustrative power [arbitrary normalization]")
    axis.set_title("The measured SAGE response predicts which stochastic variability survives")
    axis.grid(which="both", color="#e4e4e4", linewidth=0.55)
    axis.legend(frameon=False)
    save_figure(figure, path)


def build_report(summary, data, figures, products):
    one_percent_error = float(summary["linear_validation"]["one_percent_maximum_normalized_rmse"])
    pulse_time = float(summary["pulse_case"]["dominant_cooling_memory_gyr"])
    mode_times = np.asarray(data["selected_mode_times_gyr"])
    regulator_time = float(summary["local_modes"]["recipe_times_gyr"]["cold-gas regulator"])
    memory = np.asarray(data["memory_timescale_gyr"])
    finite_memory = memory[np.isfinite(memory)]
    agn_fiducial = np.abs(data["agn_fiducial_fractional_response"][-1, 1])
    agn_removed = np.abs(data["agn_removed_fractional_response"][-1, 1])
    agn_gain_reduction = 1.0 - agn_fiducial / agn_removed
    sfr_gain = np.abs(data["fractional_frequency_response"][:, 1])
    response_timescales = data["inverse_angular_frequency_gyr"]
    peak_index = int(np.argmax(sfr_gain))
    peak_timescale = float(response_timescales[peak_index])
    peak_to_slow_gain = float(sfr_gain[peak_index] / sfr_gain[-1])
    feedback_names = [str(value) for value in data["feedback_variant_names"]]
    feedback_times = np.asarray(data["feedback_variant_mode_times_gyr"])
    feedback_time_by_name = dict(zip(feedback_names, feedback_times))
    no_sn_time = float(feedback_time_by_name["SN feedback locally removed"])
    no_reincorporation_time = float(feedback_time_by_name["reincorporation locally removed"])
    validation_passed = bool(summary["linear_validation"]["passed"])
    health = (
        Diagnostic(
            key="upstream_equivalence",
            title="Upstream SAGE16 equivalence",
            status=DiagnosticStatus.NOT_EVALUATED,
            summary="Not rerun by this local-response analysis; the same physics core is validated in the linked Mini-Millennium report.",
        ),
        Diagnostic(
            key="local_linear_validity",
            title="Local nonlinear-response validation",
            status=DiagnosticStatus.PASSED if validation_passed else DiagnosticStatus.WARNING,
            summary=f"For the representative 1% cooling pulse, the maximum cold-gas/SFR normalized RMSE is {100.0 * one_percent_error:.2f}%.",
            metrics=(
                ScalarMetric(
                    "one_percent_rmse", "1% pulse maximum normalized RMSE", one_percent_error
                ),
            ),
            artifacts=(figures["cooling_pulse"], products["arrays"]),
            method="full nonlinear fixed-forcing SAGE flow versus exact matrix-exponential response of the frozen local Jacobian",
            tolerance="maximum normalized RMSE <= 0.05 at a 1% cooling pulse",
        ),
        Diagnostic(
            key="hybrid_scope",
            title="Hybrid event boundary",
            status=DiagnosticStatus.WARNING,
            summary="The local AGN flow reads the stored heating radius, but its monotone Rheat projection and genuine mergers remain explicit maps outside each frozen transfer function.",
        ),
        Diagnostic(
            key="conservation",
            title="Conservation in this analysis",
            status=DiagnosticStatus.NOT_EVALUATED,
            summary="This report reuses the structurally conservative RHS but does not repeat the separate baryon/metal conservation campaign.",
        ),
    )
    findings = (
        f"A 1% cooling pulse in the representative z={summary['pulse_case']['redshift']:.2f} central is reproduced by the local model to {100.0 * one_percent_error:.2f}% normalized RMSE and has a dominant response time of {pulse_time:.2f} Gyr.",
        f"The local cold-gas-regulator recipe estimate is {regulator_time:.2f} Gyr, close to the strongest coupled mode at {mode_times[0]:.2f} Gyr; a weaker {mode_times[-1]:.1f} Gyr collective mode is also present in this operating point.",
        f"The coupled cooling-to-SFR response peaks near inverse angular frequency {peak_timescale:.2f} Gyr at {peak_to_slow_gain:.1f} times its very-slow gain, so the actual local baryon cycle is more structured than a one-reservoir low-pass regulator.",
        f"At the same fixed SAGE state, locally suppressing SN reheating/ejection changes the dominant response time from {pulse_time:.2f} to {no_sn_time:.2f} Gyr, while suppressing reincorporation changes it to {no_reincorporation_time:.2f} Gyr.",
        f"Across the stratified trajectory survey, the median dominant cooling-memory time spans {np.min(finite_memory):.2f}–{np.max(finite_memory):.2f} Gyr across the resolved mass/redshift cells.",
        f"At the selected massive fiducial state, retained AGN heating suppresses the slow cooling-to-SFR gain by {100.0 * agn_gain_reduction:.2f}% relative to locally removing the AGN coupling.",
    )
    sections = (
        ReportSection(
            key="findings",
            title="What did SAGE tell us?",
            summary="These findings are computed from the archived response arrays and are deliberately phrased as galaxy-formation results, not as claims about mathematical technique.",
            notes=findings,
        ),
        ReportSection(
            key="baryon_cycle",
            title="A familiar SAGE galaxy",
            summary="The starting point is the SAGE16 baryon cycle: familiar reservoirs connected by familiar prescriptions, with the merger tree supplying the halo history.",
            body=markdown(
                "Between genuine events, the implemented continuous subset can be written schematically as",
                r"$$\dot{x}=f(x,h(t),\theta),$$",
                r"where $x$ contains the hot, cold, ejected, stellar, and metal reservoirs and $h(t)$ is the halo/tree forcing. Mergers, topology changes, threshold projections, and the stored AGN-heating-radius update remain explicit maps $x^+=J(x^-)$. Nothing in this notation replaces a SAGE prescription; it only makes the transfers explicit.",
            ),
            artifacts=(figures["baryon_cycle"],),
            links=(
                ReportLink("Hybrid SAGE16 structure", "../../docs/sage16_hybrid_system.md"),
                ReportLink(
                    "Mini-Millennium science report",
                    "../mini-millennium-sage16-science-program/index.md",
                ),
            ),
        ),
        ReportSection(
            key="cooling_perturbation",
            title="How long does a SAGE galaxy remember extra cooling?",
            summary=f"A temporary 1% increase in the cooling transfer raises cold gas and SFR, then the coupled baryon cycle erases the perturbation on a measured {pulse_time:.2f} Gyr dominant timescale.",
            body=markdown(
                r"We perturb the faithful cooling rate as $\dot M_{\rm cool}\rightarrow\dot M_{\rm cool}\exp(\epsilon)$ for a finite interval. The solid curves are two full nonlinear SAGE-flow evolutions—perturbed minus unperturbed—under the same frozen halo forcing. The dashed curves are predicted from the Jacobian at the initial fiducial state.",
                r"The impulse response, or Green function, is $g(t)=\mathcal{L}^{-1}\{H(s)\}$: it says how the influence of an infinitesimal parcel-like perturbation propagates into later observables. A sustained step integrates that response. The finite rectangular pulse plotted here is the difference between a step beginning at $t=0$ and the same step beginning when the cooling enhancement ends.",
                "The right panel is the quantitative validity test. A 1% intervention is small enough for the local model here; this is measured rather than assumed. The decreasing relative error at 5% occurs for this finite pulse and normalization and must not be read as a general extension of the linear regime.",
            ),
            artifacts=(figures["cooling_pulse"],),
        ),
        ReportSection(
            key="simple_regulator",
            title="Which variations in gas supply reach the SFR?",
            summary=f"Rapid cooling variations are attenuated, but the actual coupled response is not a featureless low-pass filter: the SFR gain reaches a maximum near inverse angular frequency {peak_timescale:.2f} Gyr, {peak_to_slow_gain:.1f} times its very-slow value.",
            body=markdown(
                r"The simplest local cold-gas regulator is $\dot M_{\rm cold}=\dot M_{\rm cool}-(1-R+\eta)\psi$ with $\psi\simeq M_{\rm cold}/\tau_\star$. Its equilibration time is $\tau_{\rm eq}=\tau_\star/(1-R+\eta)$. Taking the Laplace transform replaces the time derivative by multiplication by $s$ and gives the cooling-to-SFR response",
                r"$$\frac{\Psi(s)}{\dot M_{\rm cool}(s)}=\frac{1/\tau_\star}{s+1/\tau_{\rm eq}}.$$",
                r"The horizontal coordinate is $1/\omega$ in Gyr, not abstract angular frequency. Moving right means asking about progressively slower supply variations. The dashed one-reservoir curve is shape-matched at low frequency; the solid curves come from the actual coupled SAGE Jacobian. Their intermediate-timescale maximum and the pulse undershoot are the signature of the damped coupled mode measured in the next section. This is a result SAGE supplies; it is not inserted by the transform.",
            ),
            artifacts=(figures["filter"],),
        ),
        ReportSection(
            key="coupled_response",
            title="Which reservoirs set the galaxy response time?",
            summary="The coupled SAGE timescales need not equal any one timescale written into a recipe. The strongest local response closely follows the regulator estimate here, while a much slower, weakly coupled collective mode also appears.",
            body=markdown(
                r"Around the actual nonlinear trajectory $x_0(t)$, write $x=x_0+\delta x$. On a branch where the prescriptions are differentiable,",
                r"$$\delta\dot{x}=A(t)\,\delta x+B(t)\,\delta u,\qquad A=\left.\frac{\partial f}{\partial x}\right|_{x_0},\quad B=\left.\frac{\partial f}{\partial u}\right|_{x_0}.$$",
                r"For coefficients frozen at one epoch, stable poles $s_k=\operatorname{eig}(A)$ give decay times $\tau_k=-1/\Re(s_k)$. Two neutral directions are present in this eight-state representation and are not assigned finite forgetting times. The mode bars show the absolute shares of the mass-reservoir components; metal components enter the eigenproblem but are omitted from the composition display so the baryonic transport remains readable.",
            ),
            artifacts=(figures["modes"],),
        ),
        ReportSection(
            key="cosmic_history",
            title="How does galaxy memory change with halo mass and epoch?",
            summary="The dominant local cooling-to-cold-gas response generally becomes longer toward larger halo mass and later cosmic time in the cells resolved by this stratified survey.",
            body=markdown(
                "Each cell is the median decay time of the stable mode with the largest cooling-to-cold-gas residue. The sample deliberately combines uniform tree coverage with the largest trees and caps each mass/snapshot cell; it is a trajectory diagnostic, not an abundance-weighted Mini-Millennium statistic. Cells require at least three retained centrals, and counts are printed in the map.",
                r"Because $A=A[x(t),h(t)]$, this is a sequence of local frozen-coefficient measurements, not one transfer function for a galaxy from high redshift to $z=0$.",
            ),
            artifacts=(figures["memory_map"], products["arrays"]),
        ),
        ReportSection(
            key="feedback_memory",
            title="How do SN feedback and reincorporation alter galaxy memory?",
            summary=f"At the representative fixed state, removing the local SN reheating/ejection flow changes the dominant response time from {pulse_time:.2f} to {no_sn_time:.2f} Gyr; removing local reincorporation changes it to {no_reincorporation_time:.2f} Gyr.",
            body=markdown(
                "These are local coupling-removal experiments around one already formed fiducial SAGE galaxy. The SN-reheating/ejection or reincorporation rate is multiplied by `exp(-50)` while the state, halo, parameters, cooling input, and all other flows are held fixed. This isolates how each link changes the local response; it is not a new self-consistent feedback-off history.",
                "The response curves show whether each flow changes only the overall gain or also shifts the fluctuation timescales that reach star formation. The adjacent mode-time bars report the strongest cooling-to-SFR pole after each local intervention.",
            ),
            artifacts=(figures["feedback_response"],),
        ),
        ReportSection(
            key="agn_response",
            title="What changes dynamically when AGN regulation becomes important?",
            summary=f"At the selected massive state, prior radio-mode heating has already reduced instantaneous cooling by {100.0 * summary['agn_case']['instantaneous_cooling_suppression']:.2f}% and lowers the long-timescale cooling-to-SFR gain by {100.0 * agn_gain_reduction:.2f}%.",
            body=markdown(
                "The solid response uses the actual fiducial trajectory state, including its stored heating radius. The dashed response holds that same state and halo fixed but sets `AGNrecipe=0` only in the local flow. This asks what the existing AGN coupling changes dynamically; it is not an AGN-off rerun and does not erase the history that produced the background galaxy.",
                r"The SAGE16 heating radius is Markov state but advances through a monotone projection on the prescribed schedule. It therefore supplies genuine memory to the hybrid model without becoming an ordinary continuous pole in this frozen interval. Treating that projection as if it were a smooth global feedback equation would manufacture dynamics that SAGE does not implement.",
            ),
            artifacts=(figures["agn_response"],),
            links=(
                ReportLink("Radio-mode heating prescription", "../../docs/radio_mode_heating.md"),
            ),
        ),
        ReportSection(
            key="agn_transition",
            title="Where does stored AGN heating take over from cooling supply?",
            summary="The direct SAGE quantity shown here is the fraction of raw local cooling suppressed by the previously accumulated heating radius. In the sampled trajectories, strong suppression appears first in the higher-mass halo bins.",
            body=markdown(
                r"The map shows $1-\dot M_{\rm cool,after\ prior\ heating}/\dot M_{\rm cool,raw}$. It is an instantaneous regulation diagnostic, not by itself a causal statement about the final stellar mass. The existing finite-epoch response report provides that complementary historical question.",
            ),
            artifacts=(figures["agn_map"],),
            links=(
                ReportLink(
                    "Historical process responses",
                    "../mini-millennium-sage16-science-program/index.md#when-does-each-baryonic-process-matter",
                ),
            ),
        ),
        ReportSection(
            key="mathematical_connection",
            title="How are all these SAGE questions connected?",
            summary="The same derivatives already used for parameter and historical responses also predict the response to time-dependent physical inputs.",
            body=markdown(
                r"For a selected input $B$ and observable $C$, the locally frozen response is",
                r"$$H(s)=C(sI-A)^{-1}B.$$",
                "Here `B` says where extra cooling or another fractional process perturbation enters, `A` says how SAGE transports it among reservoirs, and `C` says whether we read out cold gas, SFR, metallicity, or another property. The inverse transform gives impulse/step responses; the eigenvalues give local response times; evaluating at $s=i\omega$ asks which fluctuation timescales propagate.",
                "Parameter response asks: what happens if I change a SAGE parameter? Dynamical response asks: what happens if a physical input varies on this timescale? Historical response asks: when in the past did a process matter for today’s observable? They are complementary projections of the same explicit model, but none requires a practitioner to use the word Jacobian to interpret the result.",
            ),
            links=(
                ReportLink(
                    "Fractional parameter and process responses", "../../docs/sensitivity.md"
                ),
            ),
        ),
        ReportSection(
            key="stochastic_variability",
            title="Which stochastic fluctuations survive the baryon cycle?",
            summary="Once the deterministic response is measured, an assumed spectrum of cooling/accretion variability can be propagated into an SFR-variability prediction without rerunning one realization per fluctuation.",
            body=markdown(
                r"For a scalar input, $P_y(\omega)=|H(i\omega)|^2P_u(\omega)$. For multivariate fluctuations, $S_y(\omega)=H(i\omega)S_u(\omega)H^\dagger(i\omega)$. The figure uses an explicitly illustrative input spectrum with arbitrary normalization; it is a demonstration of propagation, not a calibrated stochastic halo-accretion model.",
            ),
            artifacts=(figures["stochastic"],),
        ),
        ReportSection(
            key="validity",
            title="When is this analysis valid?",
            summary="SAGE is nonlinear, cosmologically time dependent, thresholded, tree-forced, and interrupted by events. There is no single global transfer function for an entire galaxy history.",
            body=markdown(
                r"This report freezes $A(t)$, $B(t)$, and $C(t)$ at a real trajectory point. The approximation is useful while the perturbation remains small, stays on the same piecewise-smooth branch, and evolves on a timescale short enough that the background state and halo forcing do not move substantially. The nonlinear pulse comparison measures that regime for one representative state.",
                r"For a genuinely time-dependent linear perturbation, the correct object is the propagator $\Phi(t,t')$, not one resolvent. For mergers and other jumps, sensitivities pass through the derivative of the explicit event map. For the heating-radius projection, the response includes the projection schedule. Larger perturbations must be checked against the full nonlinear hybrid evolution.",
            ),
            diagnostics=(health[1], health[2]),
        ),
        ReportSection(
            key="practical_value",
            title="What does this buy a SAGE practitioner?",
            summary="The new capability is not the Laplace transform itself. It is the ability to ask SAGE directly how long galaxies remember gas supply, which reservoirs carry that memory, which variability reaches star formation, and how feedback changes those answers.",
            notes=(
                "Measure a galaxy response time without defining it from an arbitrary rerun spacing.",
                "Distinguish timescales written into recipes from collective timescales of the coupled reservoirs.",
                "Locate where cooling supply ceases to control the response and stored AGN heating suppresses it.",
                "Connect deterministic responses to parameter sensitivity, historical response, and eventually stochastic variability using the same differentiated SAGE implementation.",
            ),
        ),
    )
    provenance = capture_provenance(
        repository=REPOSITORY,
        command=(
            sys.executable,
            "examples/build_sage16_linear_response_report.py",
            "--input-json",
            str(DEFAULT_JSON.relative_to(REPOSITORY)),
            "--input-arrays",
            str(DEFAULT_ARRAYS.relative_to(REPOSITORY)),
        ),
        configuration_paths=(RUN_FILE,),
        input_paths=(TREE_FILE, SCALE_FACTORS, DEFAULT_JSON, DEFAULT_ARRAYS),
        random_seeds={},
    )
    return RunReport(
        identity=RunIdentity(
            run_id="sage16-galaxy-memory",
            title="How long does SAGE remember?",
            model="fiducial SAGE16",
            dataset=f"Mini-Millennium partition 1; {summary['tree_count']} stratified trees at {len(summary['snapshots'])} epochs",
            parameter_set="fiducial",
            integration_method="upstream-sequential trajectory; frozen local continuous RHS",
            summary="A practitioner-facing measurement of how SAGE16 responds to changes in gas supply, which reservoirs set its response times, and how stored AGN heating changes that response.",
        ),
        provenance=provenance,
        health=health,
        sections=sections,
        overview_metrics=(
            ScalarMetric(
                "trajectory_points", "Local trajectory points", summary["local_state_count"]
            ),
            ScalarMetric("epochs", "Sampled epochs", len(summary["snapshots"])),
            ScalarMetric("response_time", "Representative cooling memory", pulse_time, unit="Gyr"),
            ScalarMetric(
                "analysis_runtime",
                "Science analysis wall time",
                summary["total_analysis_seconds"],
                unit="s",
            ),
        ),
        headline_artifacts=(
            figures["baryon_cycle"],
            figures["cooling_pulse"],
            figures["filter"],
            figures["agn_response"],
        ),
        parameters=parameters_from_namedtuple(fiducial_parameters()),
        links=(
            ReportLink(
                "Mini-Millennium science program",
                "../mini-millennium-sage16-science-program/index.md",
            ),
            ReportLink("Initial equivalence report", "../mini-millennium-sage16-initial/index.md"),
            ReportLink(
                "Machine-readable response arrays",
                "assets/mini-millennium-sage16-linear-response.npz",
            ),
        ),
    )


def main():
    arguments = parse_arguments()
    if not arguments.input_json.is_file() or not arguments.input_arrays.is_file():
        raise SystemExit("The durable linear-response JSON/NPZ products are required")
    summary = json.loads(arguments.input_json.read_text(encoding="utf-8"))
    data = dict(np.load(arguments.input_arrays, allow_pickle=False))
    output = arguments.output_dir
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    products = {
        "summary": Artifact(
            key="linear_response_summary",
            title="Linear-response analysis summary",
            path="assets/mini-millennium-sage16-linear-response.json",
            media_type="application/json",
            role="data",
            description="Sampling, validation, scope, runtime, and selected-case metadata.",
        ),
        "arrays": Artifact(
            key="linear_response_arrays",
            title="Linear-response scientific arrays",
            path="assets/mini-millennium-sage16-linear-response.npz",
            media_type="application/x-npz",
            role="data",
            description="Pulse responses, transfer functions, poles, mode compositions, and mass-redshift maps.",
        ),
    }
    shutil.copy2(arguments.input_json, assets / "mini-millennium-sage16-linear-response.json")
    shutil.copy2(arguments.input_arrays, assets / "mini-millennium-sage16-linear-response.npz")

    figure_definitions = {
        "baryon_cycle": (
            "A familiar SAGE baryon cycle",
            "SAGE16 reservoirs and transfers, with continuous flows distinguished from hybrid maps.",
        ),
        "cooling_pulse": (
            "How long does extra cooling matter?",
            "A 1% nonlinear cooling pulse compared with the frozen local prediction and an amplitude-sweep validity test.",
        ),
        "filter": (
            "Which gas-supply variations reach star formation?",
            "Fractional cold-gas and SFR response versus inverse angular frequency in Gyr.",
        ),
        "modes": (
            "Which reservoirs set the response times?",
            "Coupled local modes compared with prescription timescales and decomposed by reservoir participation.",
        ),
        "memory_map": (
            "Galaxy memory across halo mass and redshift",
            "Median dominant cooling-to-cold-gas response time; every displayed cell includes its retained sample count.",
        ),
        "feedback_response": (
            "How SN feedback and reincorporation change memory",
            "Local cooling-to-SFR responses and dominant times with fiducial flows or one feedback link suppressed.",
        ),
        "agn_response": (
            "How AGN regulation changes the response",
            "Cooling-to-cold-gas and cooling-to-SFR gain at one fiducial massive state, with the local AGN coupling retained or removed.",
        ),
        "agn_map": (
            "Where prior AGN heating suppresses cooling",
            "Instantaneous fraction of raw cooling suppressed by the stored SAGE heating radius.",
        ),
        "stochastic": (
            "How SAGE filters illustrative stochastic supply",
            "Illustrative variability propagated through the measured cooling-to-SFR response.",
        ),
    }
    figures = {
        key: figure_artifact(key, title, description)
        for key, (title, description) in figure_definitions.items()
    }
    baryon_cycle_figure(assets / "baryon_cycle.svg")
    pulse_figure(data, summary, assets / "cooling_pulse.svg")
    filter_figure(data, summary, assets / "filter.svg")
    mode_figure(data, assets / "modes.svg")
    response_map_figure(data, assets / "memory_map.svg")
    feedback_response_figure(data, assets / "feedback_response.svg")
    agn_response_figure(data, summary, assets / "agn_response.svg")
    agn_map_figure(data, assets / "agn_map.svg")
    stochastic_figure(data, assets / "stochastic.svg")
    report = build_report(summary, data, figures, products)
    written = write_report(report, output)
    print(written.markdown_path)
    print(written.manifest_path)


if __name__ == "__main__":
    main()
