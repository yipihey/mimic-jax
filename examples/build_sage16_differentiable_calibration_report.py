#!/usr/bin/env python3
"""Build the practitioner-facing differentiable SAGE16 calibration report."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from mimic_jax.reporting import (
    Artifact,
    Diagnostic,
    DiagnosticStatus,
    ReportLink,
    ReportSection,
    RunIdentity,
    RunReport,
    ScalarMetric,
    capture_provenance,
    write_report,
)
from mimic_jax.sage16 import fiducial_parameters

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_JSON = REPOSITORY / "archive/mini-millennium-sage16-differentiable-calibration.json"
DEFAULT_ARRAYS = REPOSITORY / "archive/mini-millennium-sage16-differentiable-calibration.npz"
DEFAULT_OUTPUT = REPOSITORY / "reports/sage16-differentiable-calibration"
OBSERVATION_FILE = REPOSITORY / "data/observations/baldry2008_stellar_mass_function.csv"
TREE_FILE = REPOSITORY / "simulations/mini-millennium/snapshots/trees_063.1"
BASELINE_RESPONSE = REPOSITORY / "archive/mini-millennium-sage16-parameter-responses.npz"

COLORS = {
    "fiducial": "#222222",
    "fit": "#0072B2",
    "observation": "#6A3D9A",
    "reheating": "#D55E00",
    "reincorporation": "#009E73",
    "warning": "#E69F00",
    "failure": "#B2182B",
    "muted": "#777777",
}

PARAMETER_LABELS = {
    "FeedbackReheatingEpsilon": "SN reheating mass loading",
    "ReIncorporationFactor": "ejected-gas reincorporation",
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
            "legend.fontsize": 8.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "mimic-jax-differentiable-calibration",
        }
    )


def save_figure(figure, path):
    figure.savefig(path, bbox_inches="tight", facecolor="white", metadata={"Date": None})
    plt.close(figure)


def figure_artifact(key, title, description):
    return Artifact(
        key=key,
        title=title,
        path=f"assets/{key}.svg",
        media_type="image/svg+xml",
        role="figure",
        description=description,
    )


def markdown(*blocks):
    return "\n\n".join(block.strip() for block in blocks if block.strip())


def chi_square(values, data):
    mask = data["fit_mask"]
    residual = np.log(values[mask]) - np.log(data["observation_interpolated"][mask])
    return float(residual @ np.linalg.inv(data["observation_log_covariance"]) @ residual)


def stellar_mass_function_figure(data, summary, path):
    observation_mass = data["observation_mass"]
    observation = data["observation_smf"]
    lower = np.maximum(observation - data["observation_error_lower"], 1.0e-8)
    upper = observation + data["observation_error_upper"]
    centres = data["stellar_mass_bin_centres"]
    fiducial = data["baseline_hard_smf"]
    fitted = data["optimum_exact_hard_smf"]
    mask = data["fit_mask"]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.08},
    )
    axis, residual_axis = axes
    axis.fill_between(
        observation_mass,
        lower,
        upper,
        color=COLORS["observation"],
        alpha=0.2,
        label="Baldry et al. (2008) quoted range",
    )
    axis.plot(
        centres,
        fiducial,
        color=COLORS["fiducial"],
        linewidth=2.1,
        label="fiducial SAGE16",
    )
    axis.plot(
        centres,
        fitted,
        color=COLORS["fit"],
        linestyle="--",
        linewidth=2.3,
        label="exact SAGE at tested response-selected point",
    )
    axis.set_yscale("log")
    axis.set_ylim(2.0e-4, 6.0e-2)
    axis.set_ylabel(r"$\phi(M_\star)$ [Mpc$^{-3}$ dex$^{-1}$]")
    axis.set_title("A real SAGE calibration plot, before and after following its response")
    axis.legend(loc="lower left", ncol=1)
    axis.grid(alpha=0.18)

    observed_at_bins = data["observation_interpolated"]
    valid = mask & (fiducial > 0.0) & (fitted > 0.0)
    residual_axis.plot(
        centres[valid],
        np.log10(fiducial[valid] / observed_at_bins[valid]),
        color=COLORS["fiducial"],
        linewidth=1.8,
    )
    residual_axis.plot(
        centres[valid],
        np.log10(fitted[valid] / observed_at_bins[valid]),
        color=COLORS["fit"],
        linestyle="--",
        linewidth=2.0,
    )
    residual_axis.axhline(0.0, color="#555555", linewidth=1.0)
    residual_axis.fill_between(centres[valid], -0.05, 0.05, color=COLORS["observation"], alpha=0.08)
    residual_axis.set_ylabel("model / data\n[dex]")
    residual_axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    residual_axis.grid(alpha=0.18)
    residual_axis.text(
        0.02,
        0.06,
        (
            rf"working $\chi^2$: {summary['fit']['fiducial_chi_square']:.2f} $\rightarrow$ "
            rf"{summary['fit']['exact_sage_chi_square']:.2f}"
        ),
        transform=residual_axis.transAxes,
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    residual_axis.text(
        0.98,
        0.94,
        "purple band: ±0.05 dex visual guide",
        transform=residual_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=COLORS["observation"],
    )
    save_figure(figure, path)


def response_figure(data, path):
    centres = data["stellar_mass_bin_centres"]
    response = data["baseline_response"]
    mask = data["fit_mask"]
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    styles = (
        (COLORS["reheating"], "-", "SN reheating mass loading"),
        (COLORS["reincorporation"], "--", "ejected-gas reincorporation"),
    )
    for index, (color, linestyle, label) in enumerate(styles):
        axis.plot(
            centres[mask],
            response[mask, index],
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
            marker="o",
            markersize=3.2,
            label=label,
        )
    axis.axhline(0.0, color="#555555", linewidth=1.0)
    axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    axis.set_ylabel("% abundance change per\n1% parameter increase")
    axis.set_title("One differentiated SAGE run says which way every mass bin moves")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    axis.text(
        0.02,
        0.04,
        "Example: +0.5 means a 1% parameter increase raises this bin by about 0.5%.",
        transform=axis.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    save_figure(figure, path)


def fit_path_figure(data, summary, path):
    candidate_chi_square = chi_square(data["initial_candidate_hard_smf"], data)
    values = np.asarray(
        [
            summary["fit"]["fiducial_chi_square"],
            summary["initial_local_fit"]["chi_square_predicted"],
            candidate_chi_square,
            summary["fit"]["emulator_chi_square"],
            summary["fit"]["exact_sage_chi_square"],
        ]
    )
    labels = (
        "fiducial\nexact SAGE",
        "linear\nprediction",
        "linear point\nexact SAGE",
        "quadratic\nprediction",
        "quadratic point\nexact SAGE",
    )
    colors = (
        COLORS["fiducial"],
        COLORS["warning"],
        COLORS["reheating"],
        COLORS["warning"],
        COLORS["fit"],
    )
    figure, axis = plt.subplots(figsize=(9.4, 4.8))
    bars = axis.bar(np.arange(values.size), values, color=colors, alpha=0.9)
    axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axis.set_xticks(np.arange(values.size), labels)
    axis.set_ylabel(r"working $\chi^2$ (lower is better)")
    axis.set_title("Gradients find a useful direction; exact SAGE decides whether to trust it")
    axis.set_ylim(0.0, max(values) * 1.22)
    axis.grid(axis="y", alpha=0.2)
    axis.annotate(
        "prediction ≠ validation",
        xy=(2, values[2]),
        xytext=(1.3, max(values) * 1.08),
        arrowprops={"arrowstyle": "->", "color": COLORS["failure"]},
        color=COLORS["failure"],
        weight="bold",
    )
    save_figure(figure, path)


def response_surface_figure(data, summary, path):
    mask = data["fit_mask"]
    response = data["baseline_response"][mask]
    coefficients = data["quadratic_coefficients"]
    baseline = data["baseline_hard_smf"][mask]
    observed = data["observation_interpolated"][mask]
    precision = np.linalg.inv(data["observation_log_covariance"])
    q_reheat = np.linspace(-0.18, 0.0, 140)
    q_reinc = np.linspace(-0.24, 0.0, 140)
    chi_square_grid = np.empty((q_reinc.size, q_reheat.size))
    for row, right in enumerate(q_reinc):
        points = np.column_stack((q_reheat, np.full_like(q_reheat, right)))
        features = np.column_stack(
            (0.5 * points[:, 0] ** 2, points[:, 0] * points[:, 1], 0.5 * points[:, 1] ** 2)
        )
        changes = points @ response.T + features @ coefficients
        residual = np.log(baseline)[None, :] + changes - np.log(observed)[None, :]
        chi_square_grid[row] = np.einsum("bi,ij,bj->b", residual, precision, residual)
    minimum = np.min(chi_square_grid)
    x_percent = 100.0 * np.expm1(q_reheat)
    y_percent = 100.0 * np.expm1(q_reinc)

    figure, axis = plt.subplots(figsize=(8.2, 6.4))
    filled = axis.contourf(
        x_percent,
        y_percent,
        chi_square_grid - minimum,
        levels=np.linspace(0, 12, 13),
        cmap="Blues_r",
        extend="max",
    )
    axis.contour(
        x_percent,
        y_percent,
        chi_square_grid - minimum,
        levels=(2.30, 6.17),
        colors=("#222222", "#555555"),
        linewidths=(1.5, 1.0),
    )
    training = data["training_points"]
    held_out = data["held_out_points"]
    optimum = data["optimum_log_parameter_ratios"]
    axis.scatter(
        100.0 * np.expm1(training[:, 0]),
        100.0 * np.expm1(training[:, 1]),
        marker="s",
        facecolors="white",
        edgecolors=COLORS["fiducial"],
        s=45,
        label="exact SAGE training",
    )
    axis.scatter(
        100.0 * np.expm1(held_out[:, 0]),
        100.0 * np.expm1(held_out[:, 1]),
        marker="o",
        color=COLORS["warning"],
        s=42,
        label="held-out exact SAGE",
    )
    axis.scatter(
        100.0 * np.expm1(optimum[0]),
        100.0 * np.expm1(optimum[1]),
        marker="*",
        color=COLORS["failure"],
        edgecolors="white",
        linewidths=0.6,
        s=180,
        label="emulator-selected; exact-tested",
        zorder=5,
    )
    axis.set_xlabel("SN reheating parameter change [%]")
    axis.set_ylabel("reincorporation parameter change [%]")
    axis.set_title("The real-data fit runs into the edge of the measured emulator domain")
    axis.set_ylim(100.0 * np.expm1(-0.255), 0.2)
    axis.legend(loc="upper right", framealpha=0.92)
    colorbar = figure.colorbar(filled, ax=axis, pad=0.02)
    colorbar.set_label(r"emulator $\Delta\chi^2$")
    axis.text(
        0.02,
        0.72,
        (
            "Diagnostic surface only: hard-bin validation failed\n"
            f"(worst error {summary['emulator']['held_out_maximum_absolute_error_dex']:.3f} dex)."
        ),
        transform=axis.transAxes,
        fontsize=9,
        color=COLORS["failure"],
        bbox={"facecolor": "white", "edgecolor": COLORS["failure"], "alpha": 0.9},
    )
    save_figure(figure, path)


def emulator_validation_figure(data, summary, path):
    linear = np.max(np.abs(data["held_out_linear_errors_dex"]), axis=1)
    quadratic = np.max(np.abs(data["held_out_quadratic_errors_dex"]), axis=1)
    optimum = float(np.max(np.abs(data["optimum_error_dex"])))
    labels = [f"cell {index + 1}" for index in range(linear.size)]
    x_values = np.arange(linear.size)
    width = 0.34
    figure, axis = plt.subplots(figsize=(9.2, 5.0))
    axis.bar(
        x_values - width / 2,
        linear,
        width,
        color=COLORS["muted"],
        label="local linear response",
    )
    axis.bar(
        x_values + width / 2,
        quadratic,
        width,
        color=COLORS["fit"],
        label="gradient-constrained quadratic",
    )
    axis.scatter(
        [linear.size],
        [optimum],
        marker="*",
        s=150,
        color=COLORS["failure"],
        label="quadratic optimum",
        zorder=5,
    )
    axis.axhline(
        summary["fit_definition"]["surrogate_gate_maximum_error_dex"],
        color=COLORS["failure"],
        linestyle="--",
        linewidth=1.6,
        label="predeclared 0.05-dex gate",
    )
    axis.set_xticks(np.arange(linear.size + 1), labels + ["optimum"])
    axis.set_ylabel("worst fitted-bin error [dex]")
    axis.set_title("A small emulator helps, but does not yet earn parameter error bars")
    axis.set_ylim(0.0, max(np.max(linear), np.max(quadratic), optimum) * 1.24)
    axis.legend(loc="upper left", ncol=2)
    axis.grid(axis="y", alpha=0.2)
    save_figure(figure, path)


def cost_figure(summary, path):
    primal = summary["runtime"]["baseline_primal_seconds"]
    tangent = summary["runtime"]["baseline_linearized_seconds"]
    finite_two = 4.0 * primal
    finite_seven = 14.0 * primal
    design = summary["runtime"]["exact_sage_seconds"]
    labels = (
        "one exact\nSAGE run",
        "7-parameter\nJAX tangent",
        "2-parameter\ncentral differences",
        "7-parameter\ncentral differences",
        "14-run emulator\ndesign + validation",
    )
    values = np.asarray((primal, tangent, finite_two, finite_seven, design))
    colors = (
        COLORS["fiducial"],
        COLORS["fit"],
        COLORS["warning"],
        COLORS["warning"],
        COLORS["reincorporation"],
    )
    figure, axis = plt.subplots(figsize=(10.0, 5.1))
    bars = axis.bar(np.arange(values.size), values, color=colors)
    axis.bar_label(bars, labels=[f"{value / 60.0:.1f} min" for value in values], padding=3)
    axis.set_xticks(np.arange(values.size), labels)
    axis.set_ylabel("measured or derived wall time [s]")
    axis.set_yscale("log")
    axis.set_title(
        "Differentiate once for direction; emulate only when repeated likelihoods justify it"
    )
    axis.grid(axis="y", which="both", alpha=0.2)
    axis.text(
        0.02,
        0.94,
        "Compilation and first-run costs are retained. Finite-difference bars are derived from the measured primal time.",
        transform=axis.transAxes,
        va="top",
        fontsize=8.8,
    )
    save_figure(figure, path)


def build_report(summary, data, figures, products):
    fiducial = fiducial_parameters()
    parameter_names = tuple(str(value) for value in data["parameter_names"])
    candidate_chi_square = chi_square(data["initial_candidate_hard_smf"], data)
    fiducial_chi_square = summary["fit"]["fiducial_chi_square"]
    exact_chi_square = summary["fit"]["exact_sage_chi_square"]
    improvement = 100.0 * (fiducial_chi_square - exact_chi_square) / fiducial_chi_square
    local_sigma = np.sqrt(np.diag(data["initial_covariance"]))
    local_correlation = data["initial_covariance"] / np.sqrt(
        np.diag(data["initial_covariance"])[:, None] * np.diag(data["initial_covariance"])[None, :]
    )
    tangent_speedup = (
        14.0
        * summary["runtime"]["baseline_primal_seconds"]
        / summary["runtime"]["baseline_linearized_seconds"]
    )
    interval_status = (
        DiagnosticStatus.PASSED if summary["emulator"]["passed"] else DiagnosticStatus.FAILED
    )
    health = (
        Diagnostic(
            key="reference_equivalence",
            title="Reference SAGE16 status",
            status=DiagnosticStatus.PASSED,
            summary="The underlying complete-partition path is the previously validated upstream-sequential SAGE16 map; this application changes parameters, not the physics implementation.",
            metrics=(
                ScalarMetric("trees", "Merger trees", summary["tree_count"]),
                ScalarMetric("records", "z=0 galaxies", summary["records"]),
            ),
            notes=(
                "See the linked Mini-Millennium report for field-level and hard-bin equivalence evidence.",
            ),
        ),
        Diagnostic(
            key="gradient_validation",
            title="Parameter-response validation",
            status=DiagnosticStatus.PASSED,
            summary="The baseline seven-parameter SAGE response used here was previously checked against symmetric full-tree reruns to <=0.078 absolute elasticity error in resolved bins.",
            tolerance="absolute elasticity error <= 0.1 in resolved bins",
            method="JAX chain-rule tangent versus symmetric multiplicative SAGE reruns",
        ),
        Diagnostic(
            key="exact_fit_improvement",
            title="Exact SAGE fit improvement",
            status=DiagnosticStatus.PASSED,
            summary=f"The exact evaluated SAGE point lowers the stated working chi-square by {improvement:.1f}% relative to fiducial.",
            metrics=(
                ScalarMetric("chi2_fiducial", "Fiducial chi-square", fiducial_chi_square),
                ScalarMetric("chi2_exact", "Exact selected-point chi-square", exact_chi_square),
            ),
            notes=(
                "This is a fit to one observational relation under a deliberately simplified diagonal working likelihood, not a new SAGE calibration.",
            ),
        ),
        Diagnostic(
            key="emulator_validation",
            title="Surrogate validation",
            status=interval_status,
            summary=(
                "The gradient-constrained quadratic emulator failed the predeclared hard-SMF validation gate, so it is not accepted for scientific parameter intervals."
                if not summary["emulator"]["passed"]
                else "The emulator passed every held-out hard-SMF validation check."
            ),
            metrics=(
                ScalarMetric(
                    "maximum_error",
                    "Worst held-out bin error",
                    summary["emulator"]["held_out_maximum_absolute_error_dex"],
                    unit="dex",
                ),
                ScalarMetric(
                    "gate",
                    "Acceptance gate",
                    summary["fit_definition"]["surrogate_gate_maximum_error_dex"],
                    unit="dex",
                ),
            ),
            tolerance="maximum absolute hard-SMF error <= 0.05 dex at all held-out points",
        ),
        Diagnostic(
            key="parameter_intervals",
            title="Final parameter error bars",
            status=interval_status,
            summary=(
                "Unavailable: the optimum reaches the reincorporation design boundary and the emulator fails validation. Local curvature numbers are retained only as a forecast/diagnostic."
                if not summary["emulator"]["passed"]
                else "Working-likelihood parameter intervals are available within the validated emulator domain."
            ),
            notes=(
                "No unstated Gaussian prior was added to manufacture two-sided constraints.",
                "The saved MCMC chain samples the rejected emulator and is retained only to diagnose the proposed workflow.",
            ),
        ),
        Diagnostic(
            key="continuous_population_path",
            title="Continuous full-tree inference",
            status=DiagnosticStatus.NOT_EVALUATED,
            summary="The adaptive continuous/hybrid RHS is not yet a complete validated population driver through all merger-tree events; this report therefore uses the exact differentiable reference tree map.",
            notes=(
                "Using the continuous RHS here would make the implementation request sound complete at the cost of changing the scientific model being fitted.",
            ),
        ),
    )

    parameter_lines = []
    for index, name in enumerate(parameter_names):
        candidate_ratio = summary["initial_local_fit"]["parameter_ratios"][index]
        selected_ratio = summary["fit"]["parameter_ratios"][index]
        parameter_lines.append(
            f"| `{name}` | {PARAMETER_LABELS[name]} | {float(getattr(fiducial, name)):.4g} | "
            f"{candidate_ratio:.3f}× fiducial | {selected_ratio:.3f}× fiducial | "
            f"{100.0 * local_sigma[index]:.1f}% |"
        )
    parameter_table = "\n".join(
        (
            "| SAGE parameter | Physical role | Fiducial | First gradient proposal | Best evaluated direction | Fiducial local 1σ forecast |",
            "|---|---|---:|---:|---:|---:|",
            *parameter_lines,
        )
    )

    sections = (
        ReportSection(
            key="findings",
            title="What did we learn?",
            summary="The evidence supports a useful fit direction and local precision forecast, but not final observational parameter intervals.",
            notes=(
                f"The exact SAGE point selected by the small response model reduces the working chi-square from {fiducial_chi_square:.2f} to {exact_chi_square:.2f} ({improvement:.1f}%).",
                f"The first derivative-only proposal already reaches chi-square {candidate_chi_square:.2f}, showing that one tangent calculation identifies a useful physical direction.",
                f"At fiducial, the diagonal working likelihood has local log-parameter widths of {100.0 * local_sigma[0]:.1f}% and {100.0 * local_sigma[1]:.1f}%, with response correlation {local_correlation[0, 1]:+.2f}; these are forecasts, not final error bars.",
                f"The seven-parameter JAX tangent cost {summary['runtime']['baseline_linearized_seconds'] / 60.0:.1f} min, about {tangent_speedup:.1f}× less than fourteen central-difference SAGE runs at the measured primal time.",
                f"The emulator's worst held-out hard-SMF error is {summary['emulator']['held_out_maximum_absolute_error_dex']:.3f} dex, above the 0.05-dex gate, so final error bars remain unavailable.",
            ),
        ),
        ReportSection(
            key="real_observation",
            title="Can SAGE move closer to a real calibration observation?",
            summary="Yes. The exact selected SAGE run moves the familiar z≈0 stellar mass function substantially closer to the Baldry et al. observational band under the stated working likelihood.",
            body=markdown(
                "The observation is the same Baldry, Glazebrook & Driver (2008) table used by the upstream-style SAGE plot. We fit 27 bins from log10(M*/Msun)=8.5 to 11.15, require at least ten fiducial model galaxies per bin, and freeze the covariance before fitting.",
                "The covariance is deliberately simple: the quoted observational width is treated as diagonal Gaussian uncertainty in ln(phi), with a fixed 1/N Mini-Millennium counting term. It omits observational covariance, stellar-mass systematics, cosmic variance beyond Poisson noise, and model discrepancy. Those omissions make this a workflow demonstration, not a publishable SAGE recalibration.",
            ),
            artifacts=(figures["stellar_mass_function"],),
            links=(
                ReportLink("Baldry et al. 2008", "https://arxiv.org/abs/0804.2892"),
                ReportLink("SAGE16 calibration paper", "https://arxiv.org/abs/1601.04709"),
            ),
        ),
        ReportSection(
            key="one_gradient",
            title="What does one differentiated SAGE run buy?",
            summary="It returns the fractional movement of every fitted mass bin with respect to every selected parameter, so the optimizer receives direction and scale simultaneously.",
            body=markdown(
                r"The plotted quantity is $E_{\alpha i}=\partial\ln\phi_\alpha/\partial\ln\theta_i$. A value of −0.6 means that increasing the parameter by 1% lowers abundance in that mass bin by about 0.6% near the fiducial model.",
                "A two-sided finite difference needs two complete SAGE runs per parameter. The existing seven-parameter tangent pass carries all seven directions through one tree traversal. It is not free—forward tangents cost more than one primal run—but the measured cost is still far below fourteen separate reruns.",
            ),
            artifacts=(figures["responses"], figures["cost"]),
        ),
        ReportSection(
            key="follow_gradient",
            title="Can we simply follow the gradient to the best fit?",
            summary="It gives a good first proposal, but exact SAGE validation is essential because thresholds, events, and hard population bins make the finite move nonlinear.",
            body=markdown(
                "The first local solve proposes 0.947× the fiducial SN reheating parameter and 0.885× the reincorporation factor. That exact SAGE run improves the fit, but individual hard-SMF bins differ from the linear prediction by as much as 0.114 dex, failing the predeclared 0.05-dex gate.",
                "This is not a failure of automatic differentiation: the derivative is local and was separately finite-difference validated. It is a failure of treating that local derivative as a global surrogate across a finite parameter step.",
            ),
            artifacts=(figures["fit_path"],),
            diagnostics=(health[1],),
        ),
        ReportSection(
            key="emulator",
            title="Should we build an emulator as part of the same workflow?",
            summary="Yes in principle, because repeated posterior likelihoods are where emulation helps most—but this first small emulator is not yet accurate enough for the conventional hard-bin SMF.",
            body=markdown(
                "The test uses a fixed 3×3 design in the two logarithmic parameters. Eight new exact SAGE runs supply curvature; the fiducial value and JAX elasticity are imposed exactly. Four cell centers and the optimizer-selected point are held out from training.",
                "The quadratic emulator halves the worst error relative to a purely local response, but its 0.115-dex worst bin still exceeds the 0.05-dex contract. A smoother finite-volume SMF behaves better, suggesting that much of the remaining difficulty is hard-bin migration, but that post-hoc observation does not override the declared gate.",
                "The next emulator should use a denser/adaptive design, multiple Mini-Millennium partitions, and a validation metric matched to the explicitly differentiable population estimator. A neural network is not required for this two-parameter problem; the validation design matters more than model fashion.",
            ),
            artifacts=(figures["emulator_validation"], products["arrays"]),
            diagnostics=(health[3],),
            links=(
                ReportLink(
                    "GALFORM emulator example (Elliott et al. 2021)",
                    "https://arxiv.org/abs/2103.01072",
                ),
                ReportLink(
                    "Meraxes/PRISM emulator example (van der Velden et al. 2020)",
                    "https://arxiv.org/abs/2011.14530",
                ),
            ),
        ),
        ReportSection(
            key="parameters",
            title="What parameter values and error bars can we report?",
            summary="We can report the direction and exact evaluated improvement. We cannot yet report validated final two-sided observational error bars.",
            body=markdown(
                parameter_table,
                "The best evaluated direction corresponds to SN reheating 2.59 rather than 3.0 and reincorporation 0.118 rather than 0.15. It is shown because exact SAGE was run there, not because the emulator earned a global optimum claim.",
                "The final response surface reaches the lower reincorporation boundary, and the surrogate fails held-out validation. The MCMC percentiles and local Hessian stored in the NPZ are therefore diagnostic only. Adding a narrow prior would produce neat contours, but those contours would describe the prior as much as the stellar mass function.",
            ),
            artifacts=(figures["response_surface"],),
            diagnostics=(health[4],),
        ),
        ReportSection(
            key="mcmc",
            title="Does differentiability replace MCMC?",
            summary="No. It changes how efficiently we find modes, diagnose degeneracies, construct proposals, and build emulators; MCMC still answers the global posterior question when the likelihood is nonlinear or multimodal.",
            body=markdown(
                "The saved 50,000-step random-walk chain samples exactly the same bounded emulator likelihood as the optimizer. It is retained to make the comparison familiar, but it cannot rescue an invalid emulator.",
                "The practical hybrid workflow is: use JAX gradients to find influential directions and local curvature; run exact SAGE at a designed set of points; validate an emulator; then use MCMC or another sampler on that emulator, with occasional exact checks. Gradient-based optimization and MCMC answer different questions.",
                "For context, MCMC calibration has a long history in SAMs; Henriques et al. (2009) used it to expose strong parameter correlations. Differentiability adds information per SAGE run, not permission to skip posterior validation.",
            ),
            links=(
                ReportLink(
                    "Henriques et al. 2009 MCMC SAM calibration",
                    "https://arxiv.org/abs/0810.2548",
                ),
            ),
        ),
        ReportSection(
            key="next_observations",
            title="Which observations should constrain the next parameters?",
            summary="The stellar mass function alone should not be asked to identify every SAGE process. The next application should add the real observables SAGE16 was calibrated against.",
            body=markdown(
                "Croton et al. (2016) show the z≈0 stellar mass function together with the baryonic Tully–Fisher relation, the mass–metallicity relation, and the black-hole–bulge relation. These observables have complementary physical leverage:",
                "- baryonic Tully–Fisher and gas statistics constrain star formation and feedback without using abundance alone;\n- mass–metallicity adds leverage on metal production and outflows;\n- black-hole–bulge data are required before expecting radio/quasar parameters to be identifiable;\n- cosmic SFR history tests whether a z=0 fit achieved the right assembly path rather than only the right endpoint.",
                "The response-matrix API already supplies the mathematical object needed to quantify that complementarity. The missing work is a defensible joint covariance and the corresponding differentiable population summaries—not a larger optimizer.",
            ),
        ),
        ReportSection(
            key="continuous_scope",
            title="Why is this not yet the adaptive continuous full-tree fit?",
            summary="The current adaptive continuous/hybrid formulation is validated on smooth fixed-forcing intervals, but not yet as a complete population driver through every merger, threshold, and projection.",
            body=markdown(
                "For observational fitting, changing the tree evolution scheme would mix two questions: parameter calibration and numerical reformulation. This first application therefore uses the exact upstream-sequential differentiable map whose Mini-Millennium outputs are already equivalent to MIMIC.",
                "The inference and report APIs are agnostic to that choice. Once the continuous/hybrid full-tree driver passes the same population-equivalence and event-localization gates, it can be substituted as a second model and compared under the identical likelihood. Until then, calling it the fitted SAGE population would overstate what has been validated.",
            ),
            diagnostics=(health[5],),
        ),
        ReportSection(
            key="reproducibility",
            title="Technical validation and reproducibility",
            summary="Every fit decision, failure gate, exact SAGE point, sampler draw, and observational conversion is retained in durable JSON/NPZ products.",
            artifacts=(products["summary"], products["arrays"]),
            diagnostics=health,
            notes=(
                "The design and validation points were fixed before the final exact checks.",
                "The observational table is now a single repository data product shared with the upstream-style plotting routine.",
                "The report generator does not rerun SAGE; it consumes the archived scientific products.",
            ),
        ),
    )
    provenance = capture_provenance(
        repository=REPOSITORY,
        command=(
            sys.executable,
            "examples/build_sage16_differentiable_calibration_report.py",
            "--input-json",
            str(DEFAULT_JSON.relative_to(REPOSITORY)),
            "--input-arrays",
            str(DEFAULT_ARRAYS.relative_to(REPOSITORY)),
        ),
        input_paths=(DEFAULT_JSON, DEFAULT_ARRAYS, OBSERVATION_FILE, TREE_FILE, BASELINE_RESPONSE),
        random_seeds={"mcmc": 481516},
    )
    return RunReport(
        identity=RunIdentity(
            run_id="sage16-differentiable-calibration",
            title="Fit SAGE with gradients: what one stellar mass function can—and cannot—constrain",
            model="fiducial SAGE16 with two varied physical parameters",
            dataset="Mini-Millennium partition 1 + Baldry et al. (2008) z≈0 stellar mass function",
            parameter_set="SN reheating and reincorporation varied in log space",
            integration_method="exact upstream-sequential differentiable tree map",
            summary="A practitioner-facing comparison of JAX response fitting, exact SAGE validation, local curvature, MCMC, and a deliberately failed first surrogate.",
        ),
        provenance=provenance,
        health=health,
        sections=sections,
        overview_metrics=(
            ScalarMetric("fit_bins", "Fitted SMF bins", summary["fit_definition"]["fitted_bins"]),
            ScalarMetric("trees", "Merger trees", summary["tree_count"]),
            ScalarMetric("chi2_fiducial", "Fiducial working chi-square", fiducial_chi_square),
            ScalarMetric("chi2_exact", "Selected exact-run chi-square", exact_chi_square),
            ScalarMetric(
                "emulator_error",
                "Worst held-out emulator error",
                summary["emulator"]["held_out_maximum_absolute_error_dex"],
                unit="dex",
            ),
        ),
        headline_artifacts=(
            figures["stellar_mass_function"],
            figures["responses"],
            figures["fit_path"],
            figures["emulator_validation"],
        ),
        links=(
            ReportLink(
                "Mini-Millennium science program",
                "../mini-millennium-sage16-science-program/index.md",
            ),
            ReportLink("Fractional response API", "../../docs/sensitivity.md"),
            ReportLink("Report architecture", "../../docs/reporting.md"),
        ),
    )


def main():
    arguments = parse_arguments()
    if not arguments.input_json.is_file() or not arguments.input_arrays.is_file():
        raise SystemExit("The durable calibration JSON/NPZ products are required")
    summary = json.loads(arguments.input_json.read_text(encoding="utf-8"))
    data = dict(np.load(arguments.input_arrays, allow_pickle=False))
    output = arguments.output_dir
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    products = {
        "summary": Artifact(
            key="calibration_summary",
            title="Differentiable calibration summary",
            path="assets/mini-millennium-sage16-differentiable-calibration.json",
            media_type="application/json",
            role="data",
            description="Likelihood, fit, validation, runtime, and failure-gate metadata.",
        ),
        "arrays": Artifact(
            key="calibration_arrays",
            title="Differentiable calibration arrays",
            path="assets/mini-millennium-sage16-differentiable-calibration.npz",
            media_type="application/x-npz",
            role="data",
            description="Observation, response, exact SAGE design, held-out residuals, fit geometry, and MCMC samples.",
        ),
    }
    shutil.copy2(
        arguments.input_json,
        assets / "mini-millennium-sage16-differentiable-calibration.json",
    )
    shutil.copy2(
        arguments.input_arrays,
        assets / "mini-millennium-sage16-differentiable-calibration.npz",
    )
    figures = {
        "stellar_mass_function": figure_artifact(
            "stellar_mass_function",
            "A real SAGE calibration plot",
            "Baldry et al. stellar mass function, fiducial SAGE16, and the exact evaluated response-selected point.",
        ),
        "responses": figure_artifact(
            "responses",
            "What one differentiated run tells us",
            "Fractional bin-abundance response to SN reheating and reincorporation.",
        ),
        "fit_path": figure_artifact(
            "fit_path",
            "Prediction versus exact validation",
            "Working chi-square for fiducial, response predictions, and their exact SAGE evaluations.",
        ),
        "response_surface": figure_artifact(
            "response_surface",
            "Diagnostic parameter-response surface",
            "Training design, held-out points, and the boundary-selected fit on the rejected quadratic emulator.",
        ),
        "emulator_validation": figure_artifact(
            "emulator_validation",
            "Does the emulator earn scientific use?",
            "Worst hard-bin residual at every predeclared held-out point and at the proposed optimum.",
        ),
        "cost": figure_artifact(
            "cost",
            "Where differentiability saves SAGE reruns",
            "Measured tangent/primal times and transparently derived finite-difference costs.",
        ),
    }
    stellar_mass_function_figure(data, summary, assets / "stellar_mass_function.svg")
    response_figure(data, assets / "responses.svg")
    fit_path_figure(data, summary, assets / "fit_path.svg")
    response_surface_figure(data, summary, assets / "response_surface.svg")
    emulator_validation_figure(data, summary, assets / "emulator_validation.svg")
    cost_figure(summary, assets / "cost.svg")
    report = build_report(summary, data, figures, products)
    written = write_report(report, output)
    print(written.markdown_path)
    print(written.manifest_path)


if __name__ == "__main__":
    main()
