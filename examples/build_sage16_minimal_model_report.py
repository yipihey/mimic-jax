#!/usr/bin/env python3
"""Build the report for the held-out SAGE16 minimal-model experiment."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mimic_jax.reporting import (
    Artifact,
    Diagnostic,
    DiagnosticStatus,
    ParameterValue,
    ReportLink,
    ReportSection,
    RunIdentity,
    RunReport,
    ScalarMetric,
    capture_provenance,
    write_report,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_JSON = REPOSITORY / "archive/mini-millennium-sage16-minimal.json"
DEFAULT_ARRAYS = REPOSITORY / "archive/mini-millennium-sage16-minimal.npz"
DEFAULT_VALIDATION_JSON = REPOSITORY / "archive/mini-millennium-sage16-reduction-p4.json"
DEFAULT_REJECTED_CANDIDATE_JSON = (
    REPOSITORY / "archive/mini-millennium-sage16-reduction-delayed.json"
)
DEFAULT_OUTPUT = REPOSITORY / "reports/sage16-minimal-model"
HUBBLE_H = 0.73
PARTITION_VOLUME = 62.5**3 / 8.0

COLORS = {
    "sage": "#222222",
    "static": "#999999",
    "reduced": "#0072B2",
    "cold": "#009E73",
    "stars": "#E69F00",
    "black_hole": "#6A3D9A",
    "failed": "#D55E00",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--input-arrays", type=Path, default=DEFAULT_ARRAYS)
    parser.add_argument("--validation-json", type=Path, default=DEFAULT_VALIDATION_JSON)
    parser.add_argument(
        "--rejected-candidate-json", type=Path, default=DEFAULT_REJECTED_CANDIDATE_JSON
    )
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
            "svg.fonttype": "none",
            "svg.hashsalt": "mimic-jax-sage16-minimal-model",
        }
    )


def save_figure(figure, path):
    figure.savefig(
        path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None},
    )
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


def add_box(axis, centre, label, color, width=2.0):
    x_value, y_value = centre
    patch = FancyBboxPatch(
        (x_value - width / 2.0, y_value - 0.38),
        width,
        0.76,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        linewidth=1.6,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.14),
    )
    axis.add_patch(patch)
    axis.text(x_value, y_value, label, ha="center", va="center", weight="bold")


def add_arrow(axis, start, end, label, color="#555555", curve=0.0):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.5,
            color=color,
            connectionstyle=f"arc3,rad={curve}",
        )
    )
    axis.text(
        (start[0] + end[0]) / 2.0,
        (start[1] + end[1]) / 2.0 + 0.16,
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
    )


def structure_figure(path):
    figure, axis = plt.subplots(figsize=(10.4, 5.0))
    axis.set_xlim(-0.5, 10.5)
    axis.set_ylim(-0.5, 5.1)
    axis.axis("off")
    positions = {
        "circumgalactic": (3.1, 3.8),
        "cold": (3.1, 1.8),
        "stars": (6.2, 1.8),
        "black_hole": (7.5, 3.8),
    }
    add_box(
        axis,
        positions["circumgalactic"],
        "Circumgalactic gas\n(hot + ejected)",
        COLORS["failed"],
        width=2.4,
    )
    add_box(axis, positions["cold"], "Cold gas", COLORS["cold"], width=1.7)
    add_box(axis, positions["stars"], "Long-lived stars", COLORS["stars"], width=2.0)
    add_box(
        axis,
        positions["black_hole"],
        "Black-hole\nregulation proxy",
        COLORS["black_hole"],
        width=2.2,
    )
    add_arrow(axis, (0.5, 4.65), (2.0, 4.05), "halo infall")
    add_arrow(axis, (3.5, 3.38), (3.5, 2.22), "cooling", COLORS["reduced"])
    add_arrow(axis, (4.0, 1.8), (5.15, 1.8), "star formation", COLORS["stars"])
    add_arrow(axis, (2.4, 2.2), (2.4, 3.35), "SN feedback", COLORS["failed"])
    add_arrow(axis, (6.7, 2.2), (7.2, 3.35), "merger growth", COLORS["black_hole"])
    add_arrow(axis, (6.35, 3.8), (4.35, 3.8), "quenches cooling", COLORS["black_hole"])
    axis.text(
        5.0,
        0.45,
        "Full merger-tree topology retained  •  metals, bulge, disk radius, merger clocks, and detailed AGN history removed",
        ha="center",
        color="#444444",
    )
    axis.set_title("The smallest candidate that passed the stellar-mass contract")
    save_figure(figure, path)


def stellar_mass_function_figure(data, path):
    centres = data["fine_stellar_mass_bin_centres"]
    width = float(np.diff(data["fine_stellar_mass_bin_edges"][:2])[0])
    factor = 1.0 / (PARTITION_VOLUME * width)
    figure = plt.figure(figsize=(9.4, 6.8))
    grid = figure.add_gridspec(2, 1, height_ratios=(2.4, 1.0), hspace=0.08)
    top = figure.add_subplot(grid[0, 0])
    bottom = figure.add_subplot(grid[1, 0], sharex=top)
    top.step(
        centres,
        data["fine_sage_smf_counts"] * factor,
        where="mid",
        color=COLORS["sage"],
        lw=2.3,
        label="SAGE16 teacher",
    )
    top.step(
        centres,
        data["fine_static_smf_counts"] * factor,
        where="mid",
        color=COLORS["static"],
        lw=1.7,
        ls="--",
        label="static halo-efficiency fit",
    )
    top.step(
        centres,
        data["fine_reduced_autonomous_smf_counts"] * factor,
        where="mid",
        color=COLORS["reduced"],
        lw=2.1,
        label="four-state model",
    )
    top.set_yscale("log")
    top.set_ylim(8.0e-6, 3.0e-1)
    top.set_ylabel(r"$\phi$  [$h^3\,{\rm Mpc}^{-3}\,{\rm dex}^{-1}$]")
    top.legend(frameon=False, ncol=3, loc="upper right")
    top.tick_params(labelbottom=False)
    resolved = data["fine_smf_resolved"]
    bottom.axhspan(-0.30, 0.30, color=COLORS["reduced"], alpha=0.10)
    bottom.axhline(0.0, color="#444444", lw=1.0)
    bottom.step(
        centres[resolved],
        data["fine_static_smf_fractional_difference"][resolved],
        where="mid",
        color=COLORS["static"],
        lw=1.6,
        ls="--",
    )
    bottom.step(
        centres[resolved],
        data["fine_reduced_autonomous_smf_fractional_difference"][resolved],
        where="mid",
        color=COLORS["reduced"],
        lw=2.0,
    )
    bottom.axhline(0.30, color=COLORS["reduced"], lw=0.8, ls="--")
    bottom.axhline(-0.30, color=COLORS["reduced"], lw=0.8, ls="--")
    worst = np.nanargmax(
        np.where(
            resolved,
            np.abs(data["fine_reduced_autonomous_smf_fractional_difference"]),
            np.nan,
        )
    )
    bottom.annotate(
        f"{100.0 * data['fine_reduced_autonomous_smf_fractional_difference'][worst]:+.1f}%",
        (
            centres[worst],
            data["fine_reduced_autonomous_smf_fractional_difference"][worst],
        ),
        xytext=(18, 24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["reduced"]},
        color=COLORS["reduced"],
        weight="bold",
    )
    bottom.set_ylim(-0.8, 2.3)
    bottom.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    bottom.set_ylabel("fractional\ndifference")
    top.set_title("Four states recover the broad SMF but miss one fine mass bin")
    save_figure(figure, path)


def stellar_mass_scatter_figure(data, path):
    selected = data["resolved_galaxy"]
    reference = data["sage_stellar_mass"][selected] * 1.0e10 / HUBBLE_H
    predictions = (
        ("Static halo mapping", data["static_stellar_mass"][selected], COLORS["static"]),
        ("Four-state baryon cycle", data["reduced_stellar_mass"][selected], COLORS["reduced"]),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharex=True, sharey=True)
    grid = np.linspace(8.0, 12.2, 200)
    for axis, (title, prediction, color) in zip(axes, predictions):
        predicted = np.maximum(prediction * 1.0e10 / HUBBLE_H, 1.0e5)
        axis.hexbin(
            np.log10(reference),
            np.log10(predicted),
            gridsize=55,
            mincnt=1,
            bins="log",
            cmap="Blues" if color == COLORS["reduced"] else "Greys",
        )
        axis.plot(grid, grid, color="#222222", lw=1.2)
        axis.fill_between(
            grid,
            grid + np.log10(0.7),
            grid + np.log10(1.3),
            color=color,
            alpha=0.14,
        )
        axis.set_title(title)
        axis.set_xlabel(r"SAGE16  $\log_{10}(M_\star/M_\odot)$")
        axis.set_xlim(8.0, 12.1)
        axis.set_ylim(8.0, 12.1)
    axes[0].set_ylabel(r"Reduced prediction  $\log_{10}(M_\star/M_\odot)$")
    figure.suptitle("Following the baryon cycle carries information a static halo map loses")
    save_figure(figure, path)


def scope_figure(summary, data, path):
    metrics = summary["test_metrics"]
    figure = plt.figure(figsize=(10.4, 6.1))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.25), wspace=0.34)
    accuracy = figure.add_subplot(grid[0, 0])
    labels = ["stellar mass", "cold gas", "SFR"]
    values = 100.0 * np.asarray(
        [
            metrics["reduced_stellar_mass"]["fraction_within_30_percent"],
            metrics["reduced_cold_gas"]["fraction_within_30_percent"],
            metrics["reduced_star_formation_rate"]["fraction_within_30_percent"],
        ]
    )
    colors = [COLORS["reduced"], COLORS["cold"], COLORS["failed"]]
    bars = accuracy.barh(labels[::-1], values[::-1], color=colors[::-1], alpha=0.86)
    accuracy.axvline(70.0, color="#333333", ls="--", lw=1.0, label="stellar gate")
    accuracy.set_xlim(0.0, 100.0)
    accuracy.set_xlabel("objects within 30%  [%]")
    accuracy.set_title("What the reduction predicts well")
    for bar, value in zip(bars, values[::-1]):
        accuracy.text(
            value + 1.5, bar.get_y() + bar.get_height() / 2.0, f"{value:.1f}%", va="center"
        )

    quenching = figure.add_subplot(grid[0, 1])
    quench = metrics["reduced_quenched_classification"]
    bars = quenching.bar(
        ["SAGE16", "four-state model"],
        100.0 * np.asarray([quench["sage_quenched_fraction"], quench["reduced_quenched_fraction"]]),
        color=[COLORS["sage"], COLORS["reduced"]],
        width=0.62,
    )
    quenching.set_ylim(0.0, 55.0)
    quenching.set_ylabel("quenched central fraction  [%]")
    quenching.set_title("Quenching physics has not survived")
    for bar in bars:
        quenching.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.2,
            f"{bar.get_height():.1f}%",
            ha="center",
        )
    figure.suptitle("Passing one prediction is not the same as reproducing SAGE")
    save_figure(figure, path)


def build_report(
    summary,
    validation,
    rejected_candidate,
    data,
    figures,
    products,
    input_json,
    input_arrays,
    validation_json,
    rejected_candidate_json,
):
    acceptance = summary["acceptance"]
    metrics = summary["test_metrics"]
    stellar = metrics["reduced_stellar_mass"]
    fine = metrics["reduced_fine_autonomous_smf"]
    coarse = metrics["reduced_coarse_autonomous_smf"]
    quench = metrics["reduced_quenched_classification"]
    validation_stellar = validation["test_metrics"]["reduced_stellar_mass"]
    validation_coarse = validation["test_metrics"]["reduced_coarse_autonomous_smf"]
    candidate_stellar = rejected_candidate["stellar_mass"]
    primary_status = (
        DiagnosticStatus.PASSED if acceptance["overall_pass"] else DiagnosticStatus.FAILED
    )
    health = (
        Diagnostic(
            key="stellar_contract",
            title="Predeclared stellar-mass contract",
            status=primary_status,
            summary=(
                f"{100.0 * stellar['fraction_within_30_percent']:.1f}% of resolved galaxies "
                f"are within 30%, only {100.0 * (stellar['fraction_within_30_percent'] - 0.70):.1f} percentage points above the gate; "
                f"the worst populated 0.4-dex SMF bin differs by "
                f"{100.0 * coarse['maximum_resolved_fractional_difference']:.1f}%."
            ),
            tolerance="at least 70% of individual masses within 30%, and every 0.4-dex bin with at least 20 SAGE galaxies within 30%",
        ),
        Diagnostic(
            key="fine_smf",
            title="Fine-bin stellar mass function",
            status=DiagnosticStatus.WARNING,
            summary=(
                f"The worst populated 0.2-dex bin differs by "
                f"{100.0 * fine['maximum_resolved_fractional_difference']:.1f}%, outside 30%."
            ),
            tolerance="diagnostic only; not part of the locked acceptance gate",
        ),
        Diagnostic(
            key="secondary_predictions",
            title="Cold gas, SFR, and quenching",
            status=DiagnosticStatus.FAILED,
            summary=(
                f"Only {100.0 * metrics['reduced_cold_gas']['fraction_within_30_percent']:.1f}% "
                f"of cold-gas masses and {100.0 * metrics['reduced_star_formation_rate']['fraction_within_30_percent']:.1f}% "
                f"of SFRs are within 30%; the quenched fraction is "
                f"{100.0 * quench['reduced_quenched_fraction']:.1f}% versus "
                f"{100.0 * quench['sage_quenched_fraction']:.1f}% in SAGE."
            ),
            notes=(
                "These were declared secondary diagnostics before fitting, not silently promoted after seeing the result.",
            ),
        ),
        Diagnostic(
            key="conservation",
            title="Reduced baryon conservation",
            status=DiagnosticStatus.PASSED,
            summary=(
                "Every local cooling/star-formation/feedback update is an explicit transfer; "
                f"the maximum held-out residual is {metrics['maximum_local_conservation_residual']:.3e} in SAGE mass units."
            ),
        ),
    )

    findings = markdown(
        "The answer is **yes for a deliberately narrow prediction, and no for SAGE as a whole**.",
        f"- The four-state model passes the locked z=0 stellar-mass test on the untouched replication partition, but narrowly: **{100.0 * stellar['fraction_within_30_percent']:.1f}%** of resolved galaxies are within 30%, only **{100.0 * (stellar['fraction_within_30_percent'] - 0.70):.1f} percentage points** above the gate. The worst populated 0.4-dex mass-function bin is **{100.0 * coarse['maximum_resolved_fractional_difference']:.1f}%** from SAGE.",
        f"- A static halo-to-stellar-mass mapping reaches only **{100.0 * metrics['static_stellar_mass']['fraction_within_30_percent']:.1f}%** within 30%; explicit gas history and merger topology carry useful information.",
        f"- Fine 0.2-dex structure is not fully reproduced: the worst populated bin misses by **{100.0 * fine['maximum_resolved_fractional_difference']:.1f}%**.",
        f"- The same model does **not** reproduce gas or ongoing activity: its quenched-central fraction is **{100.0 * quench['reduced_quenched_fraction']:.1f}%**, versus **{100.0 * quench['sage_quenched_fraction']:.1f}%** for SAGE.",
        f"- Partition 4 was used for candidate selection: the four-state model reached **{100.0 * validation_stellar['fraction_within_30_percent']:.2f}%** within 30%, versus **{100.0 * candidate_stellar['fraction_within_30_percent']:.2f}%** for a five-state model with explicit ejected gas; both had a **{100.0 * validation_coarse['maximum_resolved_fractional_difference']:.2f}%** worst coarse-bin error. The five-state fit also drove its reincorporation time to **{rejected_candidate['parameter_values'][9]:.2f} Gyr**, the lower bound, so that state was removed before partition 5 was opened.",
    )

    sections = (
        ReportSection(
            key="answer",
            title="How much of SAGE can four variables reproduce?",
            summary="The minimal model preserves much of the integrated z=0 stellar-mass prediction, but not the present-day baryon cycle or quenching state.",
            body=findings,
            artifacts=(figures["structure"],),
        ),
        ReportSection(
            key="contract",
            title="What did we require before fitting?",
            summary="The 30% statement was made testable before the final replication catalogue was opened.",
            body=markdown(
                "Development used Mini-Millennium partitions 1--3. Partition 4 compared locked four- and five-state candidates and selected the simpler model because the extra ejected reservoir did not improve the target. The rejected candidate sent a mass-dependent fraction of feedback to a separate ejected reservoir and returned it exponentially on one fitted reincorporation timescale. The four-state form, coefficients, fitting data, mass resolution, binning, and thresholds were then frozen before partition 5 was opened.",
                "The primary gate requires at least 70% of resolved individual z=0 stellar masses to lie within 30% of SAGE16, plus every 0.4-dex stellar-mass-function bin containing at least 20 SAGE galaxies to lie within 30%. The familiar 0.2-dex SMF is retained as a stricter diagnostic. Cold gas, SFR, and quenching were declared secondary tests rather than optimized acceptance quantities.",
                "This is therefore a claim about **z=0 stellar mass**, not a claim that the reduced model is interchangeable with SAGE16 for arbitrary observables or histories.",
            ),
            diagnostics=(health[0], health[1]),
        ),
        ReportSection(
            key="information",
            title="What information does the merger-tree baryon cycle add?",
            summary="Following four evolving states is substantially more faithful than mapping peak halo mass directly to stellar mass on the replication partition.",
            body="The static comparison has four fitted coefficients but no memory. The dynamical model retains the same raw halo merger trees and carries gas supply, star formation, feedback, and a minimal regulation memory forward through time.",
            artifacts=(figures["scatter"],),
        ),
        ReportSection(
            key="smf",
            title="Does the minimal model preserve the stellar mass function?",
            summary=(
                f"Yes at the predeclared 0.4-dex resolution; the 0.2-dex diagnostic still exposes a {100.0 * fine['maximum_resolved_fractional_difference']:.1f}% local discrepancy."
            ),
            body="The upper panel is the conventional z=0 stellar mass function. The lower panel makes the 30% requirement visible. Only bins with at least 20 SAGE galaxies enter the residual test, so a single sparsely occupied high-mass bin cannot masquerade as a robust percentage statement.",
            artifacts=(figures["smf"],),
        ),
        ReportSection(
            key="scope",
            title="Which SAGE predictions did not survive the reduction?",
            summary="Cold gas, current SFR, and the quenched population fail decisively; integrated stellar mass alone is an insufficient model-selection target.",
            body=markdown(
                "The four-state fit only minimized robust log stellar-mass residuals. Its poor secondary predictions are therefore an honest out-of-objective test, not a surprise hidden by retuning.",
                "The failure suggests that a future broader reduction must retain more of the regulation history or change its fitting target. It does **not** justify adding states until an untouched test shows that they improve a predeclared observable set.",
            ),
            artifacts=(figures["scope"],),
            diagnostics=(health[2],),
        ),
        ReportSection(
            key="model",
            title="What is the actual reduced model?",
            summary="A conservative forced reservoir model with three mass reservoirs, one regulation-memory variable, nine fitted coefficients, and explicit merger events.",
            body=markdown(
                r"The state is $x=(M_{\rm CGM},M_{\rm cold},M_\star,M_{\rm BH,proxy})$. Halo mass, spin, redshift, infall budgets, and the raw merger topology are external forcing/events.",
                r"Cooling transfers CGM gas to cold gas. Above a spin-dependent threshold, cold gas is processed into long-lived stars and feedback return. The mass-loading factor is a halo-mass power law. Cooling is reduced by a halo-mass term and by the accumulated black-hole proxy. All finite transfers are capped analytically through exponential depletion factors, so local reservoirs remain non-negative and baryon conservation is structural.",
                "This is a teacher--student reduction calibrated to SAGE16. It is not a replacement physical model, and its coefficients should not be interpreted as newly measured SAGE parameters.",
            ),
            links=(
                ReportLink(
                    "Reduced-model source",
                    "https://github.com/yipihey/mimic-jax/blob/main/mimic_jax/sage16/reduced.py",
                ),
            ),
        ),
        ReportSection(
            key="next",
            title="What is the next scientifically defensible test?",
            summary="Broaden the acceptance contract before adding complexity, then demand that every extra state improve an untouched partition.",
            notes=(
                "Define population-level cold-gas, SFR, and quenched-fraction tolerances, not only individual ratios near zero.",
                "Fit the same four-state structure to that multi-observable objective and reserve a new Mini-Millennium partition for the final test.",
                "Only then retest explicit ejected-gas/reincorporation or stored AGN-heating memory; keep a state only if it improves held-out observables.",
                "Test redshift evolution after z=0 targets pass. A z=0 emulator can hide the wrong history.",
            ),
        ),
        ReportSection(
            key="validation",
            title="Why trust this conclusion?",
            summary="The positive and negative claims are tied to explicit checks, and failed diagnostics remain visible.",
            diagnostics=health,
            artifacts=(
                products["summary"],
                products["validation"],
                products["rejected_candidate"],
                products["arrays"],
            ),
        ),
    )

    provenance = capture_provenance(
        repository=REPOSITORY,
        command=(
            sys.executable,
            "examples/build_sage16_minimal_model_report.py",
            "--input-json",
            str(input_json),
            "--input-arrays",
            str(input_arrays),
            "--validation-json",
            str(validation_json),
            "--rejected-candidate-json",
            str(rejected_candidate_json),
        ),
        input_paths=(input_json, input_arrays, validation_json, rejected_candidate_json),
        random_seeds={},
    )
    parameter_metadata = {
        "StarFormationTimescaleGyr": (
            "Gyr",
            "Effective cold-gas processing timescale.",
        ),
        "CoolingTimescaleGyr": ("Gyr", "Effective circumgalactic cooling timescale."),
        "FeedbackMassLoadingAtPivot": (
            "dimensionless",
            "Cold-to-circumgalactic mass loading at the fixed halo-mass pivot.",
        ),
        "FeedbackHaloMassSlope": (
            "dimensionless",
            "Halo-mass slope of the effective feedback loading.",
        ),
        "QuenchingHaloMass": (
            "1e10 Msun/h",
            "Effective halo-mass scale that suppresses cooling.",
        ),
        "QuenchingSlope": (
            "dimensionless",
            "Sharpness of halo-mass cooling suppression.",
        ),
        "ColdGasThresholdPerSpin": (
            "1e10 Msun/h",
            "Cold-gas threshold per unit halo-spin magnitude.",
        ),
        "CoolingRedshiftExponent": (
            "dimensionless",
            "Power-law redshift dependence of effective cooling.",
        ),
        "BlackHoleQuenchingMass": (
            "1e10 Msun/h proxy",
            "Regulation-memory scale; not a faithful SAGE black-hole mass.",
        ),
    }
    parameters = tuple(
        ParameterValue(
            name=name,
            value=float(value),
            unit=parameter_metadata[name][0],
            description=parameter_metadata[name][1],
        )
        for name, value in summary["parameters"].items()
    )
    return RunReport(
        identity=RunIdentity(
            run_id="sage16-minimal-model",
            title="How much of SAGE can we remove?",
            model="SAGE16 teacher / four-state reduced baryon cycle",
            dataset="Mini-Millennium partitions 1--3 development; partition 4 model selection; partition 5 untouched replication",
            parameter_set="nine coefficients fitted to SAGE16 z=0 stellar mass",
            integration_method="two conservative analytic transfer substeps per tree interval",
            summary="A held-out test of how much SAGE16 z=0 stellar-mass information survives in a much smaller reservoir model.",
        ),
        provenance=provenance,
        health=health,
        sections=sections,
        overview_metrics=(
            ScalarMetric("states", "Reduced state fields", 4),
            ScalarMetric("coefficients", "Fitted coefficients", 9),
            ScalarMetric(
                "stellar_within_30",
                "Resolved galaxies within 30%",
                100.0 * stellar["fraction_within_30_percent"],
                unit="%",
            ),
            ScalarMetric(
                "coarse_smf_error",
                "Worst populated 0.4-dex SMF error",
                100.0 * coarse["maximum_resolved_fractional_difference"],
                unit="%",
            ),
        ),
        headline_artifacts=(
            figures["structure"],
            figures["smf"],
            figures["scope"],
        ),
        parameters=parameters,
        links=(
            ReportLink(
                "Mini-Millennium science report",
                "../mini-millennium-sage16-science-program/index.md",
            ),
            ReportLink(
                "Faithful SAGE16 implementation", "../mini-millennium-sage16-initial/index.md"
            ),
            ReportLink("Machine-readable arrays", "assets/mini-millennium-sage16-minimal.npz"),
        ),
    )


def main():
    arguments = parse_arguments()
    if (
        not arguments.input_json.is_file()
        or not arguments.input_arrays.is_file()
        or not arguments.validation_json.is_file()
        or not arguments.rejected_candidate_json.is_file()
    ):
        raise SystemExit("Run scripts/analyze_sage16_reduction.py before building the report")
    summary = json.loads(arguments.input_json.read_text(encoding="utf-8"))
    validation = json.loads(arguments.validation_json.read_text(encoding="utf-8"))
    rejected_candidate = json.loads(arguments.rejected_candidate_json.read_text(encoding="utf-8"))
    data = dict(np.load(arguments.input_arrays, allow_pickle=False))
    output = arguments.output_dir
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    shutil.copy2(arguments.input_json, assets / "mini-millennium-sage16-minimal.json")
    shutil.copy2(arguments.input_arrays, assets / "mini-millennium-sage16-minimal.npz")
    shutil.copy2(
        arguments.validation_json,
        assets / "mini-millennium-sage16-minimal-validation-p4.json",
    )
    shutil.copy2(
        arguments.rejected_candidate_json,
        assets / "mini-millennium-sage16-minimal-rejected-ejected-p4.json",
    )
    products = {
        "summary": Artifact(
            key="minimal_summary",
            title="Minimal-model analysis summary",
            path="assets/mini-millennium-sage16-minimal.json",
            media_type="application/json",
            role="data",
            description="Acceptance contract, fitted coefficients, scalar tests, and limitations.",
        ),
        "arrays": Artifact(
            key="minimal_arrays",
            title="Minimal-model scientific arrays",
            path="assets/mini-millennium-sage16-minimal.npz",
            media_type="application/x-npz",
            role="data",
            description="Replication-partition galaxy predictions and stellar-mass-function arrays.",
        ),
        "validation": Artifact(
            key="minimal_validation_summary",
            title="Partition-4 model-selection summary",
            path="assets/mini-millennium-sage16-minimal-validation-p4.json",
            media_type="application/json",
            role="data",
            description="The held-out candidate-selection result preceding the untouched partition-5 replication.",
        ),
        "rejected_candidate": Artifact(
            key="minimal_rejected_candidate",
            title="Rejected explicit-ejected-reservoir candidate",
            path="assets/mini-millennium-sage16-minimal-rejected-ejected-p4.json",
            media_type="application/json",
            role="data",
            description="State, coefficients, boundary-saturating reincorporation time, and partition-4 metrics for the rejected five-state trial.",
        ),
    }
    figure_definitions = {
        "structure": (
            "The selected four-state model",
            "The retained reservoirs, forcing, transfers, and explicitly discarded SAGE16 detail.",
        ),
        "smf": (
            "Held-out stellar mass function",
            "SAGE16, a static halo mapping, and the four-state model on untouched partition 5.",
        ),
        "scatter": (
            "Held-out individual stellar masses",
            "A static mapping compared with the history-dependent four-state baryon cycle.",
        ),
        "scope": (
            "What the reduction does and does not preserve",
            "Within-30% rates and the failed quenched-fraction prediction.",
        ),
    }
    figures = {
        key: figure_artifact(key, title, description)
        for key, (title, description) in figure_definitions.items()
    }
    structure_figure(assets / "structure.svg")
    stellar_mass_function_figure(data, assets / "smf.svg")
    stellar_mass_scatter_figure(data, assets / "scatter.svg")
    scope_figure(summary, data, assets / "scope.svg")
    written = write_report(
        build_report(
            summary,
            validation,
            rejected_candidate,
            data,
            figures,
            products,
            arguments.input_json,
            arguments.input_arrays,
            arguments.validation_json,
            arguments.rejected_candidate_json,
        ),
        output,
    )
    print(written.markdown_path)
    print(written.manifest_path)


if __name__ == "__main__":
    main()
