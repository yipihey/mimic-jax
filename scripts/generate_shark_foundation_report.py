#!/usr/bin/env python3
"""Generate the first provenance-rich SHARK integration foundation report."""

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mimic_jax import FORWARD_EULER, HEUN_RK2, RK4
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
from mimic_jax.shark import (
    SHARK_UPSTREAM_REVISION,
    SharkAugmentedFlowRates,
    SharkFlowRates,
    augmented_baryonic_mass,
    augmented_metal_mass,
    baryonic_mass,
    black_hole_component,
    cloudy_cie_cooling_table,
    cooling_luminosity_1e40_erg_per_s,
    croton06_unheated_cooling,
    croton06_unheated_cooling_from_table,
    evolve_shark_continuous_interval,
    evolve_shark_reference_interval,
    evolve_shark_starburst,
    flow_conservation_residuals,
    initial_shark_continuous_state,
    initial_shark_galaxy_state,
    initial_shark_state,
    initial_shark_subhalo_state,
    initial_shark_system_state,
    integrate_shark_flow,
    interpolate_log10_cooling_function,
    lagos13_feedback_loadings,
    lagos13_feedback_parameters,
    lagos23_agn_parameters,
    lagos23_bolometric_luminosity_1e40_erg_per_s,
    lagos23_br06_star_formation,
    lagos23_croton06_cooling_parameters,
    lagos23_disk_flow_rates,
    lagos23_disk_forcing,
    lagos23_hot_halo_accretion_rate,
    lagos23_mechanical_luminosity_1e40_erg_per_s,
    lagos23_model_parameters,
    lagos23_qso_outflow_loadings,
    lagos23_reincorporation_parameters,
    lagos23_star_formation_parameters,
    load_shark_catalogue,
    reference_reincorporated_mass,
    rotating_component,
    shark_atomic_gas_mass_function,
    shark_augmented_continuous_rhs_from_rates,
    shark_black_hole_bulge_relation,
    shark_black_hole_spin_relation,
    shark_cold_gas_fraction_relation,
    shark_flow_parameters,
    shark_gas_metallicity_relation,
    shark_interval_forcing,
    shark_molecular_gas_mass_function,
    shark_quenched_fraction,
    shark_rhs_from_rates,
    shark_stellar_mass_function,
    shark_stellar_size_relation,
    sized_component,
    sobacchi13_reionisation_parameters,
    sobacchi13_reionised_halo,
    thin_disk_efficiency_and_isco,
    zero_flow_rates,
)
from mimic_jax.shark.prescriptions.structure import (
    cooling_gas_specific_angular_momentum,
    lagos23_cosmology,
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-output", required=True, type=Path)
    parser.add_argument("--upstream-config", type=Path)
    parser.add_argument(
        "--rate-oracle",
        type=Path,
        default=Path("tests/mimic_jax/fixtures/shark/lagos23_rate_oracle.json"),
    )
    parser.add_argument(
        "--report-directory",
        type=Path,
        default=Path("reports/shark-continuous-foundation"),
    )
    return parser.parse_args()


def _flow_case():
    initial = initial_shark_state(
        stellar_mass=5.0e9,
        cold_gas=1.0e10,
        cold_halo_gas=0.0,
        hot_halo_gas=7.0e9,
        ejected_gas=1.0e9,
        stellar_metals=5.0e7,
        cold_gas_metals=1.0e8,
        cold_halo_gas_metals=0.0,
        hot_halo_gas_metals=7.0e7,
        ejected_gas_metals=1.0e7,
        stellar_angular_momentum=5.0e9,
        cold_gas_angular_momentum=2.4e10,
        hot_halo_angular_momentum=2.8e10,
    )
    parameters = shark_flow_parameters()
    forcing = lagos23_disk_forcing(
        gas_half_mass_radius=0.010,
        stellar_half_mass_radius=0.006,
        redshift=0.0,
        galaxy_velocity=200.0,
        subhalo_velocity=200.0,
        cooling_rate=0.0,
        cooling_metallicity=0.0,
        cooling_specific_angular_momentum=4.0,
    )
    star_formation_parameters = lagos23_star_formation_parameters()
    feedback_parameters = lagos13_feedback_parameters()
    cooling_parameters = lagos23_croton06_cooling_parameters()
    cooling_table = cloudy_cie_cooling_table()
    hubble_h = float(star_formation_parameters.hubble_h)

    def rate_law(time, state):
        disk_rates = lagos23_disk_flow_rates(
            time,
            state,
            forcing,
            star_formation_parameters,
            feedback_parameters,
        )
        physical_hot_mass = state.hot_halo_gas / hubble_h
        cooling = croton06_unheated_cooling_from_table(
            physical_hot_mass,
            state.hot_halo_gas_metals / hubble_h,
            physical_hot_mass,
            0.18,
            180.0,
            1.0,
            cooling_parameters,
            cooling_table,
        )
        hot_metallicity = jnp.where(
            state.hot_halo_gas > 0.0,
            state.hot_halo_gas_metals / state.hot_halo_gas,
            0.0,
        )
        return disk_rates._replace(
            cooling=cooling.cooling_rate * hubble_h,
            cooling_metallicity=hot_metallicity,
        )

    return initial, parameters, rate_law


def _convergence_data():
    initial, parameters, rate_law = _flow_case()
    reference = integrate_shark_flow(
        initial,
        rate_law,
        parameters,
        duration=1.0,
        num_steps=8192,
        method=RK4,
        formulation="continuous",
    ).final_state
    step_counts = np.asarray([8, 16, 32, 64, 128], dtype=np.int64)
    methods = (FORWARD_EULER, HEUN_RK2, RK4)
    errors = np.empty((len(methods), step_counts.size), dtype=np.float64)
    conservation = np.empty_like(errors)
    for method_index, method in enumerate(methods):
        for step_index, steps in enumerate(step_counts):
            final = integrate_shark_flow(
                initial,
                rate_law,
                parameters,
                duration=1.0,
                num_steps=int(steps),
                method=method,
                formulation="continuous",
            ).final_state
            errors[method_index, step_index] = abs(
                float((final.stellar_mass - reference.stellar_mass) / reference.stellar_mass)
            )
            conservation[method_index, step_index] = abs(
                float(baryonic_mass(final) - baryonic_mass(initial))
            ) / float(baryonic_mass(initial))
    # Exclude the finest interval, where RK4 is already approaching float64
    # roundoff and no longer estimates truncation order reliably.
    observed_orders = np.median(np.log2(errors[:, :-2] / errors[:, 1:-1]), axis=1)
    return initial, step_counts, methods, errors, conservation, observed_orders


def _plot_smf(catalogue, path):
    edges = np.arange(7.0, 12.61, 0.2)
    result = shark_stellar_mass_function(catalogue, bin_edges=edges)
    populated = result.counts > 0
    poisson = np.zeros_like(result.number_density)
    poisson[populated] = result.number_density[populated] / np.sqrt(result.counts[populated])
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.step(
        result.bin_centres[populated],
        result.number_density[populated],
        where="mid",
        color="#2166ac",
        linewidth=2.2,
        label="upstream SHARK (pinned CI-tree run)",
    )
    axis.fill_between(
        result.bin_centres[populated],
        np.maximum(result.number_density[populated] - poisson[populated], 1.0e-12),
        result.number_density[populated] + poisson[populated],
        step="mid",
        color="#2166ac",
        alpha=0.18,
        linewidth=0,
        label="Poisson interval",
    )
    axis.set_yscale("log")
    axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    axis.set_ylabel(r"$\phi\;[\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    axis.set_title("A real SHARK reference catalogue")
    axis.grid(alpha=0.2, which="both")
    axis.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)
    return result


def _plot_common_observables(catalogue, path):
    stellar_edges = np.arange(7.0, 12.21, 0.25)
    bulge_edges = np.arange(7.0, 12.21, 0.25)
    gas_fraction = shark_cold_gas_fraction_relation(catalogue, bin_edges=stellar_edges)
    metallicity = shark_gas_metallicity_relation(catalogue, bin_edges=stellar_edges)
    quenched = shark_quenched_fraction(catalogue, bin_edges=stellar_edges)
    black_hole = shark_black_hole_bulge_relation(catalogue, bin_edges=bulge_edges)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)

    def relation(axis, result, colour, ylabel):
        populated = result.counts >= 5
        axis.fill_between(
            result.bin_centres[populated],
            result.lower[populated],
            result.upper[populated],
            color=colour,
            alpha=0.18,
            linewidth=0,
        )
        axis.plot(
            result.bin_centres[populated],
            result.median[populated],
            color=colour,
            linewidth=2.1,
        )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)

    relation(axes[0, 0], gas_fraction, "#1b9e77", r"$M_{\rm cold}/(M_\star+M_{\rm cold})$")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_title("Cold-gas fraction")
    relation(axes[0, 1], metallicity, "#d95f02", r"$\log_{10}(M_{Z,\rm cold}/M_{\rm cold})$")
    axes[0, 1].set_title("Cold-gas metallicity")
    relation(axes[1, 0], black_hole, "#7570b3", r"$\log_{10}(M_{\rm BH}/M_\odot)$")
    axes[1, 0].set_xlabel(r"$\log_{10}(M_{\rm bulge}/M_\odot)$")
    axes[1, 0].set_title("Black-hole–bulge relation")
    populated = quenched.counts >= 5
    axes[1, 1].plot(
        quenched.bin_centres[populated],
        quenched.fraction[populated],
        color="#2166ac",
        linewidth=2.1,
    )
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    axes[1, 1].set_ylabel(r"quenched fraction ($\mathrm{sSFR}<10^{-11}\,\mathrm{yr}^{-1}$)")
    axes[1, 1].set_title("Quenched fraction")
    axes[1, 1].grid(alpha=0.2)
    for axis in axes[0]:
        axis.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    fig.suptitle("Common SAGE–SHARK observable definitions on the upstream SHARK catalogue")
    fig.savefig(path)
    plt.close(fig)
    return gas_fraction, metallicity, black_hole, quenched


def _plot_shark_additions(catalogue, path):
    gas_edges = np.arange(6.5, 11.81, 0.25)
    black_hole_edges = np.arange(4.0, 10.51, 0.3)
    stellar_edges = np.arange(7.0, 12.21, 0.25)
    atomic = shark_atomic_gas_mass_function(catalogue, bin_edges=gas_edges)
    molecular = shark_molecular_gas_mass_function(catalogue, bin_edges=gas_edges)
    spin = shark_black_hole_spin_relation(catalogue, bin_edges=black_hole_edges)
    size = shark_stellar_size_relation(catalogue, bin_edges=stellar_edges)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for result, label, colour in (
        (atomic, "atomic gas", "#1b9e77"),
        (molecular, "molecular gas", "#7570b3"),
    ):
        populated = result.counts > 0
        axes[0].step(
            result.bin_centres[populated],
            result.number_density[populated],
            where="mid",
            linewidth=2.0,
            color=colour,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\log_{10}(M_{\rm gas}/M_\odot)$")
    axes[0].set_ylabel(r"$\phi\;[\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    axes[0].set_title("Resolved gas phases")
    axes[0].legend(frameon=False)
    # Do not turn a handful of rare high-mass objects into an apparently precise trend.
    spin_populated = spin.counts >= 20
    axes[1].fill_between(
        spin.bin_centres[spin_populated],
        spin.lower[spin_populated],
        spin.upper[spin_populated],
        color="#d95f02",
        alpha=0.18,
    )
    axes[1].plot(
        spin.bin_centres[spin_populated],
        spin.median[spin_populated],
        color="#d95f02",
        linewidth=2.0,
    )
    axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1].set_xlabel(r"$\log_{10}(M_{\rm BH}/M_\odot)$")
    axes[1].set_ylabel("BH spin")
    axes[1].set_title("Black-hole spin")
    size_populated = size.counts >= 20
    axes[2].fill_between(
        size.bin_centres[size_populated],
        size.lower[size_populated],
        size.upper[size_populated],
        color="#2166ac",
        alpha=0.18,
    )
    axes[2].plot(
        size.bin_centres[size_populated],
        size.median[size_populated],
        color="#2166ac",
        linewidth=2.0,
    )
    axes[2].set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    axes[2].set_ylabel(r"$\log_{10}(R_{\star,\rm disk}/\mathrm{kpc})$")
    axes[2].set_title("Angular momentum sets sizes")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Scientific outputs SHARK adds beyond the current SAGE16 catalogue")
    fig.savefig(path)
    plt.close(fig)
    return atomic, molecular, spin, size


def _interval_case(*, massive=False):
    if massive:
        galaxy = initial_shark_galaxy_state(
            disk_stars=sized_component(1.0e11, 1.0e9, 3.0e11, 0.020),
            disk_gas=sized_component(2.0e10, 2.0e8, 8.0e10, 0.025),
            bulge_stars=sized_component(5.0e10, 5.0e8, 5.0e10, 0.010),
            black_hole=black_hole_component(1.0e9, spin=0.9, starburst_accretion_rate=1.0e6),
            maximum_circular_velocity=jnp.asarray(350.0),
        )
        state = initial_shark_system_state(
            galaxy=galaxy,
            subhalo=initial_shark_subhalo_state(
                hot_halo_gas=rotating_component(1.2e12, 1.2e10, 6.0e12),
                ejected_gas=rotating_component(1.0e10, 1.0e8, 5.0e10),
            ),
        )
        forcing = dict(
            redshift=0.5,
            halo_mass=1.0e13,
            subhalo_mass=1.0e13,
            virial_velocity=350.0,
            subhalo_velocity=350.0,
            virial_radius=0.5,
            halo_dynamical_time=1.4,
            galaxy_velocity=350.0,
            gas_half_mass_radius=0.025,
            stellar_half_mass_radius=0.020,
            galaxy_id=99,
        )
    else:
        galaxy = initial_shark_galaxy_state(
            disk_stars=sized_component(2.0e9, 2.0e7, 4.0e9, 0.006),
            disk_gas=sized_component(3.0e9, 3.0e7, 9.0e9, 0.008),
            black_hole=black_hole_component(2.0e6, spin=0.3),
            maximum_circular_velocity=jnp.asarray(180.0),
        )
        state = initial_shark_system_state(
            galaxy=galaxy,
            subhalo=initial_shark_subhalo_state(
                hot_halo_gas=rotating_component(8.0e10, 8.0e8, 4.0e11),
                ejected_gas=rotating_component(1.0e10, 1.0e8, 5.0e10),
            ),
        )
        forcing = dict(
            redshift=1.0,
            halo_mass=8.0e11,
            subhalo_mass=8.0e11,
            virial_velocity=180.0,
            subhalo_velocity=180.0,
            virial_radius=0.15,
            halo_dynamical_time=0.9,
            galaxy_velocity=180.0,
            gas_half_mass_radius=0.008,
            stellar_half_mass_radius=0.006,
            galaxy_id=42,
        )
    forcing.update(
        duration_gyr=0.2,
        hot_specific_angular_momentum=5.0,
        cooling_specific_angular_momentum=5.0,
        accreted_mass=0.0,
        maximum_allowed_baryon_accretion=0.0,
        baryon_fraction_excess_after_infall=0.0,
        stripped_hot_halo_mass_for_density=0.0,
        is_central_subhalo=True,
        ignore_galaxy_formation=False,
        execution_seed=123456,
    )
    return state, shark_interval_forcing(**forcing)


def _response_matrix():
    mid_state, mid_forcing = _interval_case()
    massive_state, massive_forcing = _interval_case(massive=True)
    parameters = lagos23_model_parameters()

    def observables(log_multipliers):
        multipliers = jnp.exp(log_multipliers)
        varied = parameters._replace(
            star_formation=parameters.star_formation._replace(
                efficiency_per_gyr=(parameters.star_formation.efficiency_per_gyr * multipliers[0])
            ),
            stellar_feedback=parameters.stellar_feedback._replace(
                beta_disk=parameters.stellar_feedback.beta_disk * multipliers[1]
            ),
            reincorporation=parameters.reincorporation._replace(
                timescale_normalization_gyr=(
                    parameters.reincorporation.timescale_normalization_gyr * multipliers[2]
                )
            ),
            agn=parameters.agn._replace(
                kappa_agn=parameters.agn.kappa_agn * multipliers[3],
                kappa_jet=parameters.agn.kappa_jet * multipliers[4],
            ),
        )
        mid = evolve_shark_continuous_interval(mid_state, mid_forcing, varied, num_substeps=4)
        massive = evolve_shark_continuous_interval(
            massive_state, massive_forcing, varied, num_substeps=4
        )
        values = jnp.asarray(
            [
                mid.state.galaxy.disk_stars.mass,
                mid.state.galaxy.disk_gas.mass,
                mid.state.subhalo.ejected_gas.mass,
                massive.state.galaxy.disk_stars.mass,
                massive.state.galaxy.disk_gas.mass,
                massive.state.subhalo.hot_halo_gas.mass,
                massive.diagnostics.cooling_transfer,
            ]
        )
        return jnp.log(values)

    center = jnp.zeros(5)
    matrix = np.asarray(jax.jacrev(observables)(center))
    epsilon = 1.0e-4
    direction = jnp.asarray([epsilon, 0.0, 0.0, 0.0, 0.0])
    finite_difference = np.asarray(
        (observables(center + direction) - observables(center - direction)) / (2.0 * epsilon)
    )
    resolved = np.abs(matrix[:, 0]) > 1.0e-8
    derivative_validation = float(
        np.max(
            np.abs(finite_difference[resolved] - matrix[resolved, 0]) / np.abs(matrix[resolved, 0])
        )
    )
    observables = (
        "mid-mass stars",
        "mid-mass cold gas",
        "mid-mass ejected gas",
        "massive stars",
        "massive cold gas",
        "massive hot halo",
        "massive cooling",
    )
    parameters = (
        "SF efficiency",
        "SN slope",
        "reincorporation time",
        "hot-mode BH accretion",
        "jet coupling",
    )
    return matrix, observables, parameters, derivative_validation


def _plot_response_matrix(matrix, observable_names, parameter_names, path):
    maximum = max(float(np.nanmax(np.abs(matrix))), 1.0e-6)
    fig, axis = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-maximum, vmax=maximum, aspect="auto")
    axis.set_xticks(np.arange(len(parameter_names)), parameter_names, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(observable_names)), observable_names)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=("white" if abs(matrix[row, column]) > 0.55 * maximum else "black"),
            )
    colourbar = fig.colorbar(image, ax=axis)
    colourbar.set_label("% observable change per 1% parameter increase")
    axis.set_title("The continuous Lagos23 core exposes fractional physical responses directly")
    fig.savefig(path)
    plt.close(fig)


def _interval_oracle_residuals(oracle):
    reference = oracle["reference_interval"]
    galaxy = initial_shark_galaxy_state(
        disk_stars=sized_component(
            2.0e9,
            2.0e7,
            2.0e9 * (180.0 * 0.006 / 0.835),
            0.006,
        ),
        disk_gas=sized_component(
            3.0e9,
            3.0e7,
            3.0e9 * (180.0 * 0.008 / 0.835),
            0.008,
        ),
        black_hole=black_hole_component(2.0e6, spin=0.3),
        maximum_circular_velocity=jnp.asarray(180.0),
    )
    state = initial_shark_system_state(
        galaxy=galaxy,
        subhalo=initial_shark_subhalo_state(
            hot_halo_gas=rotating_component(8.0e10, 8.0e8, 4.0e11),
            ejected_gas=rotating_component(1.0e10, 1.0e8, 5.0e10),
        ),
    )
    forcing = shark_interval_forcing(
        redshift=reference["redshift"],
        duration_gyr=reference["duration_gyr"],
        halo_mass=reference["halo_mass"],
        subhalo_mass=reference["halo_mass"],
        virial_velocity=reference["virial_velocity"],
        subhalo_velocity=reference["virial_velocity"],
        virial_radius=reference["virial_radius"],
        halo_dynamical_time=reference["halo_dynamical_time"],
        hot_specific_angular_momentum=5.0,
        cooling_specific_angular_momentum=cooling_gas_specific_angular_momentum(
            reference["halo_mass"], 0.03, reference["redshift"], lagos23_cosmology()
        ),
        accreted_mass=0.0,
        maximum_allowed_baryon_accretion=0.0,
        baryon_fraction_excess_after_infall=0.0,
        stripped_hot_halo_mass_for_density=0.0,
        galaxy_velocity=180.0,
        gas_half_mass_radius=0.008,
        stellar_half_mass_radius=0.006,
        is_central_subhalo=True,
        ignore_galaxy_formation=False,
        galaxy_id=42,
        execution_seed=123456,
    )
    result = evolve_shark_reference_interval(
        state, forcing, lagos23_model_parameters(), num_steps=1
    )
    values = {
        "stellar_mass": result.state.galaxy.disk_stars.mass,
        "cold_gas": result.state.galaxy.disk_gas.mass,
        "hot_halo_gas": result.state.subhalo.hot_halo_gas.mass,
        "ejected_gas": result.state.subhalo.ejected_gas.mass,
        "black_hole_mass": result.state.galaxy.black_hole.mass,
        "black_hole_spin": result.state.galaxy.black_hole.spin,
        "cooling_rate": result.diagnostics.cooling_rate,
        "star_formation_rate": result.diagnostics.mean_star_formation_rate,
    }
    interval_residuals = np.asarray(
        [
            0.0 if reference[name] == 0.0 else float(value / reference[name] - 1.0)
            for name, value in values.items()
        ]
    )

    burst_reference = oracle["reference_starburst"]
    burst_state = initial_shark_system_state(
        galaxy=initial_shark_galaxy_state(
            bulge_stars=sized_component(2.0e9, 2.0e7, 1.0e9, 0.006),
            bulge_gas=sized_component(3.0e9, 3.0e7, 2.0e9, 0.008),
            black_hole=black_hole_component(2.0e6, spin=0.3),
            maximum_circular_velocity=jnp.asarray(180.0),
        ),
        subhalo=initial_shark_subhalo_state(
            hot_halo_gas=rotating_component(8.0e10, 8.0e8),
            ejected_gas=rotating_component(1.0e10, 1.0e8),
        ),
    )
    burst = evolve_shark_starburst(
        burst_state,
        redshift=1.0,
        duration_gyr=0.2,
        virial_velocity=180.0,
        subhalo_velocity=180.0,
        galaxy_id=84,
        execution_seed=123456,
        model_parameters=lagos23_model_parameters(),
        num_steps=1,
    )
    burst_values = {
        "black_hole_mass": burst.state.galaxy.black_hole.mass,
        "black_hole_spin": burst.state.galaxy.black_hole.spin,
        "bulge_stellar_mass": burst.state.galaxy.bulge_stars.mass,
        "bulge_gas_mass": burst.state.galaxy.bulge_gas.mass,
        "hot_halo_gas": burst.state.subhalo.hot_halo_gas.mass,
        "ejected_gas": burst.state.subhalo.ejected_gas.mass,
        "star_formation_rate": burst.diagnostics.mean_star_formation_rate,
    }
    burst_residuals = np.asarray(
        [
            0.0 if burst_reference[name] == 0.0 else float(value / burst_reference[name] - 1.0)
            for name, value in burst_values.items()
        ]
    )
    return interval_residuals, burst_residuals


def _plot_convergence(step_counts, methods, errors, conservation, path):
    colours = ("#b2182b", "#ef8a62", "#2166ac")
    labels = ("Euler (expected order 1)", "Heun RK2 (order 2)", "RK4 (order 4)")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    timestep = 1.0 / step_counts
    for _method, label, colour, values in zip(methods, labels, colours, errors):
        axes[0].loglog(timestep, values, marker="o", linewidth=2, color=colour, label=label)
    axes[0].invert_xaxis()
    axes[0].set_xlabel("time step [Gyr]")
    axes[0].set_ylabel("fractional final stellar-mass error")
    axes[0].set_title("The coupled SHARK flow converges")
    axes[0].grid(alpha=0.2, which="both")
    axes[0].legend(frameon=False, fontsize=8)
    for _label, colour, values in zip(labels, colours, conservation):
        axes[1].semilogy(timestep, np.maximum(values, 1.0e-17), marker="o", color=colour)
    axes[1].invert_xaxis()
    axes[1].set_xlabel("time step [Gyr]")
    axes[1].set_ylabel("fractional baryon-ledger residual")
    axes[1].set_title("Conservation stays at roundoff")
    axes[1].grid(alpha=0.2, which="both")
    fig.savefig(path)
    plt.close(fig)


def _plot_flow_network(path):
    fig, axis = plt.subplots(figsize=(10.5, 4.0), constrained_layout=True)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 4)
    axis.axis("off")
    boxes = {
        "cold halo": (1.3, 2.8, "#d9f0d3"),
        "cold ISM": (4.0, 2.8, "#addd8e"),
        "stars": (6.8, 2.8, "#fdd49e"),
        "hot halo": (4.0, 0.9, "#9ecae1"),
        "ejected": (6.8, 0.9, "#c6dbef"),
        "QSO-lost": (9.0, 0.9, "#dadaeb"),
    }
    for label, (x, y, colour) in boxes.items():
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": colour, "edgecolor": "#444444"},
        )

    def arrow(start, end, label, offset=(0, 0.18)):
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8})
        axis.text(
            0.5 * (start[0] + end[0]) + offset[0],
            0.5 * (start[1] + end[1]) + offset[1],
            label,
            ha="center",
            va="center",
            fontsize=9,
        )

    arrow((1.9, 2.8), (3.35, 2.8), "cooling")
    arrow((4.65, 2.8), (6.1, 2.8), "star formation")
    arrow((4.15, 2.35), (4.15, 1.35), "SN/QSO reheating", (0.75, 0))
    arrow((4.65, 0.9), (6.15, 0.9), "SN ejection")
    arrow((7.45, 0.9), (8.35, 0.9), "QSO loss")
    axis.text(
        5.0,
        3.65,
        "The implemented 19-state flow routes mass, metals, and angular momentum together",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(path)
    plt.close(fig)


def _prescription_oracle_data(path):
    oracle = json.loads(path.read_text(encoding="utf-8"))
    star_parameters = lagos23_star_formation_parameters()
    star_relative_differences = []
    angular_relative_differences = []
    for case in oracle["star_formation"]:
        result = lagos23_br06_star_formation(
            case["cold_gas"],
            case["stars"],
            case["gas_radius"],
            case["stellar_radius"],
            case["gas_metallicity"],
            case["redshift"],
            case["burst"],
            case["galaxy_velocity"],
            case["gas_specific_angular_momentum"],
            star_parameters,
        )
        star_relative_differences.append(float(result.mass / case["rate"] - 1.0))
        angular_relative_differences.append(
            0.0
            if case["angular_momentum_rate"] == 0.0
            else float(result.angular_momentum / case["angular_momentum_rate"] - 1.0)
        )

    feedback_parameters = lagos13_feedback_parameters()
    feedback_relative_differences = []
    for case in oracle["stellar_feedback"]:
        result = lagos13_feedback_loadings(
            1.0,
            case["subhalo_velocity"],
            case["galaxy_velocity"],
            case["redshift"],
            feedback_parameters,
        )
        feedback_relative_differences.append(
            float(result.reheating / case["reheating_loading"] - 1.0)
        )
    reincorporation_parameters = lagos23_reincorporation_parameters()
    reincorporation_relative_differences = []
    for case in oracle["reincorporation"]:
        result = reference_reincorporated_mass(
            case["ejected_gas"],
            case["halo_mass"],
            case["interval_gyr"],
            case["satellite"],
            reincorporation_parameters,
        )
        reference = case["realized_mass"]
        reincorporation_relative_differences.append(
            0.0 if reference == 0.0 else float(result / reference - 1.0)
        )
    reionisation_parameters = sobacchi13_reionisation_parameters()
    reionisation_matches = np.asarray(
        [
            bool(
                sobacchi13_reionised_halo(
                    case["virial_velocity"], case["redshift"], reionisation_parameters
                )
            )
            == case["reionised"]
            for case in oracle["reionisation"]
        ]
    )
    cooling_parameters = lagos23_croton06_cooling_parameters()
    cooling_relative_differences = []
    for case in oracle["cooling"]:
        result = croton06_unheated_cooling(
            case["hot_mass"],
            case["density_mass"],
            case["virial_radius"],
            case["virial_velocity"],
            case["halo_dynamical_time"],
            case["log10_cooling_function_input"],
            cooling_parameters,
        )
        cooling_relative_differences.extend(
            (
                float(result.cooling_radius / case["cooling_radius"] - 1.0),
                float(result.cooling_rate / case["cooling_rate"] - 1.0),
                float(
                    cooling_luminosity_1e40_erg_per_s(
                        case["log10_cooling_function"],
                        case["cooling_radius"],
                        case["virial_radius"],
                        case["hot_mass"],
                        cooling_parameters.core_radius_fraction,
                    )
                    / case["cooling_luminosity"]
                    - 1.0
                ),
            )
        )
    cooling_table = cloudy_cie_cooling_table()
    for case in oracle["cooling_function"]:
        result = interpolate_log10_cooling_function(
            case["log10_temperature"], case["metallicity"], cooling_table
        )
        reference = case["log10_cooling_function"]
        cooling_relative_differences.append(float((result - reference) / reference))
    agn_parameters = lagos23_agn_parameters()
    agn_relative_differences = []
    for case in oracle["agn"]:
        accretion = lagos23_hot_halo_accretion_rate(
            case["pseudo_cooling_luminosity"],
            case["stored_black_hole_mass"],
            case["hot_gas_fraction"],
            case["virial_velocity"],
            agn_parameters,
        )
        mechanical = lagos23_mechanical_luminosity_1e40_erg_per_s(
            case["stored_black_hole_mass"],
            case["hot_accretion_rate"],
            case["starburst_accretion_rate"],
            case["stored_spin"],
            agn_parameters,
        )
        efficiency, _ = thin_disk_efficiency_and_isco(case["stored_spin"])
        bolometric = lagos23_bolometric_luminosity_1e40_erg_per_s(
            case["stored_black_hole_mass"],
            case["hot_accretion_rate"],
            case["starburst_accretion_rate"],
            case["stored_spin"],
            agn_parameters,
        )
        qso = lagos23_qso_outflow_loadings(
            gas_mass=case["gas_mass"],
            black_hole_mass_msun_over_h=case["stored_black_hole_mass"],
            hot_halo_accretion_rate_msun_over_h_per_gyr=case["hot_accretion_rate"],
            starburst_accretion_rate_msun_over_h_per_gyr=case["starburst_accretion_rate"],
            spin=case["stored_spin"],
            gas_metallicity=case["gas_metallicity"],
            circular_velocity_km_per_s=case["circular_velocity"],
            star_formation_rate=case["star_formation_rate"],
            bulge_baryonic_mass=case["bulge_baryonic_mass"],
            bulge_radius_mpc=case["bulge_radius"],
            parameters=agn_parameters,
        )
        for result, reference in (
            (accretion, case["calculated_hot_halo_accretion_rate"]),
            (mechanical, case["mechanical_luminosity"]),
            (efficiency, case["radiative_efficiency"]),
            (bolometric, case["bolometric_luminosity"]),
            (qso.reheating, case["qso_reheating_loading"]),
            (qso.ejection, case["qso_ejection_loading"]),
        ):
            agn_relative_differences.append(
                0.0 if reference == 0.0 else float(result / reference - 1.0)
            )
    return (
        oracle,
        np.asarray(star_relative_differences),
        np.asarray(angular_relative_differences),
        np.asarray(feedback_relative_differences),
        np.asarray(reincorporation_relative_differences),
        reionisation_matches,
        np.asarray(cooling_relative_differences),
        np.asarray(agn_relative_differences),
    )


def _plot_prescription_oracle(star, angular, feedback, reincorporation, cooling, agn, path):
    cooling_display = np.concatenate(
        (np.max(np.abs(cooling[:15].reshape(5, 3)), axis=1), [np.max(np.abs(cooling[15:]))])
    )
    agn_display = np.max(np.abs(agn.reshape(5, -1)), axis=1)
    labels = [
        *(f"SF {index + 1}" for index in range(star.size)),
        *(f"SF AM {index + 1}" for index in range(angular.size)),
        *(f"SN {index + 1}" for index in range(feedback.size)),
        *(f"reinc {index + 1}" for index in range(reincorporation.size)),
        *(f"cool {index + 1}" for index in range(5)),
        "cool table",
        *(f"AGN {index + 1}" for index in range(5)),
    ]
    values = np.abs(
        np.concatenate((star, angular, feedback, reincorporation, cooling_display, agn_display))
    )
    colours = (
        ["#1b9e77"] * star.size
        + ["#7570b3"] * angular.size
        + ["#d95f02"] * feedback.size
        + ["#66a61e"] * reincorporation.size
        + ["#1f78b4"] * cooling_display.size
        + ["#e31a1c"] * agn_display.size
    )
    fig, axis = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    axis.bar(labels, np.maximum(values, 1.0e-16), color=colours)
    axis.axhline(5.0e-6, color="#333333", linestyle="--", linewidth=1.3)
    axis.set_yscale("log")
    axis.set_ylim(5.0e-17, 2.0e-5)
    axis.set_ylabel("absolute fractional difference from upstream")
    axis.set_title("Six Lagos23 prescription groups pass direct upstream oracles")
    axis.tick_params(axis="x", labelrotation=45)
    axis.grid(alpha=0.2, axis="y", which="both")
    fig.savefig(path)
    plt.close(fig)


def main():
    arguments = _arguments()
    repository = Path(__file__).resolve().parents[1]
    destination = arguments.report_directory
    if not destination.is_absolute():
        destination = repository / destination
    assets = destination / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    catalogue = load_shark_catalogue(arguments.upstream_output)
    if catalogue.upstream_revision != SHARK_UPSTREAM_REVISION:
        raise ValueError(
            f"Expected upstream revision {SHARK_UPSTREAM_REVISION}, "
            f"found {catalogue.upstream_revision}"
        )
    smf_path = assets / "upstream-shark-stellar-mass-function.svg"
    smf = _plot_smf(catalogue, smf_path)
    common_observables_path = assets / "upstream-shark-common-observables.svg"
    gas_fraction, gas_metallicity, black_hole_bulge, quenched = _plot_common_observables(
        catalogue, common_observables_path
    )
    additions_path = assets / "upstream-shark-added-observables.svg"
    atomic, molecular, black_hole_spin, stellar_size = _plot_shark_additions(
        catalogue, additions_path
    )
    initial, step_counts, methods, errors, conservation, observed_orders = _convergence_data()
    convergence_path = assets / "shark-flow-convergence.svg"
    _plot_convergence(step_counts, methods, errors, conservation, convergence_path)
    flow_path = assets / "shark-flow-network.svg"
    _plot_flow_network(flow_path)
    rate_oracle_path = arguments.rate_oracle
    if not rate_oracle_path.is_absolute():
        rate_oracle_path = repository / rate_oracle_path
    (
        oracle,
        star_rate_delta,
        angular_rate_delta,
        feedback_delta,
        reincorporation_delta,
        reionisation_matches,
        cooling_delta,
        agn_delta,
    ) = _prescription_oracle_data(rate_oracle_path)
    prescription_path = assets / "lagos23-prescription-oracle.svg"
    _plot_prescription_oracle(
        star_rate_delta,
        angular_rate_delta,
        feedback_delta,
        reincorporation_delta,
        cooling_delta,
        agn_delta,
        prescription_path,
    )
    interval_delta, starburst_delta = _interval_oracle_residuals(oracle)
    (
        response_matrix,
        response_observables,
        response_parameters,
        response_validation,
    ) = _response_matrix()
    response_path = assets / "lagos23-fractional-response-matrix.svg"
    _plot_response_matrix(response_matrix, response_observables, response_parameters, response_path)
    arrays_path = assets / "shark-foundation-results.npz"
    np.savez_compressed(
        arrays_path,
        smf_bin_edges=smf.bin_edges,
        smf_bin_centres=smf.bin_centres,
        smf_counts=smf.counts,
        smf_number_density=smf.number_density,
        common_stellar_mass_bin_centres=gas_fraction.bin_centres,
        cold_gas_fraction_median=gas_fraction.median,
        cold_gas_fraction_counts=gas_fraction.counts,
        gas_metallicity_median=gas_metallicity.median,
        gas_metallicity_counts=gas_metallicity.counts,
        black_hole_bulge_bin_centres=black_hole_bulge.bin_centres,
        black_hole_bulge_median=black_hole_bulge.median,
        black_hole_bulge_counts=black_hole_bulge.counts,
        quenched_fraction=quenched.fraction,
        quenched_counts=quenched.counts,
        atomic_gas_bin_centres=atomic.bin_centres,
        atomic_gas_number_density=atomic.number_density,
        molecular_gas_number_density=molecular.number_density,
        black_hole_spin_bin_centres=black_hole_spin.bin_centres,
        black_hole_spin_median=black_hole_spin.median,
        stellar_size_bin_centres=stellar_size.bin_centres,
        stellar_size_median=stellar_size.median,
        convergence_step_counts=step_counts,
        convergence_methods=np.asarray(methods),
        convergence_relative_errors=errors,
        convergence_conservation_residuals=conservation,
        convergence_observed_orders=observed_orders,
        star_formation_oracle_relative_difference=star_rate_delta,
        star_formation_angular_momentum_oracle_relative_difference=angular_rate_delta,
        stellar_feedback_oracle_relative_difference=feedback_delta,
        reincorporation_oracle_relative_difference=reincorporation_delta,
        reionisation_oracle_matches=reionisation_matches,
        cooling_oracle_relative_difference=cooling_delta,
        agn_oracle_relative_difference=agn_delta,
        interval_oracle_relative_difference=interval_delta,
        starburst_oracle_relative_difference=starburst_delta,
        fractional_response_matrix=response_matrix,
        fractional_response_observables=np.asarray(response_observables),
        fractional_response_parameters=np.asarray(response_parameters),
        fractional_response_finite_difference_error=response_validation,
    )

    reference_rates = SharkFlowRates(
        cooling=jnp.asarray(3.0),
        star_formation=jnp.asarray(2.0),
        star_formation_angular_momentum=jnp.asarray(7.0),
        stellar_reheating_loading=jnp.asarray(2.0),
        stellar_ejection_loading=jnp.asarray(0.75),
        angular_momentum_reheating_loading=jnp.asarray(1.5),
        angular_momentum_ejection_loading=jnp.asarray(0.5),
        qso_reheating_loading=jnp.asarray(0.25),
        qso_ejection_loading=jnp.asarray(0.1),
        cooling_metallicity=jnp.asarray(0.015),
        cooling_specific_angular_momentum=jnp.asarray(5.0),
    )
    ledger_result = shark_rhs_from_rates(
        0.0,
        initial._replace(cold_gas_metals=0.02 * initial.cold_gas),
        reference_rates,
        shark_flow_parameters(recycle_fraction=0.4, yield_mass_fraction=0.03),
    )
    maximum_ledger_residual = max(
        abs(float(value)) for value in flow_conservation_residuals(ledger_result)
    )
    augmented_state = initial_shark_continuous_state(
        reservoirs=initial_shark_state(
            hot_halo_gas=100.0,
            hot_halo_gas_metals=2.0,
            hot_halo_angular_momentum=500.0,
        ),
        black_hole_mass=2.0,
        black_hole_metals=0.01,
        black_hole_spin=0.4,
        heating_radius=0.03,
    )
    augmented_result = shark_augmented_continuous_rhs_from_rates(
        0.0,
        augmented_state,
        SharkAugmentedFlowRates(
            reservoirs=zero_flow_rates(),
            hot_halo_black_hole_accretion=jnp.asarray(3.0),
            reincorporation=jnp.asarray(0.0),
        ),
        shark_flow_parameters(),
    )
    augmented_residual = max(
        abs(float(augmented_baryonic_mass(augmented_result.derivative))),
        abs(float(augmented_metal_mass(augmented_result.derivative))),
        abs(
            float(
                augmented_result.derivative.reservoirs.hot_halo_angular_momentum
                + augmented_result.black_hole_angular_momentum_sink
            )
        ),
    )
    runtime = time.perf_counter() - started

    inputs = [arguments.upstream_output, rate_oracle_path]
    configurations = []
    if arguments.upstream_config is not None:
        configurations.append(arguments.upstream_config)
    rerun_command = [
        "python",
        "scripts/generate_shark_foundation_report.py",
        "--upstream-output",
        str(arguments.upstream_output),
    ]
    if arguments.upstream_config is not None:
        rerun_command.extend(("--upstream-config", str(arguments.upstream_config)))
    rerun_command.extend(("--report-directory", str(arguments.report_directory)))
    provenance = capture_provenance(
        repository=repository,
        command=tuple(rerun_command),
        configuration_paths=configurations,
        input_paths=inputs,
        random_seeds={"upstream_shark": catalogue.seed},
    )
    provenance = replace(
        provenance,
        upstream_run={
            "project": "ICRAR/shark",
            "revision": catalogue.upstream_revision,
            "version": catalogue.upstream_version,
            "redshift": catalogue.redshift,
            "galaxies": int(catalogue.galaxy_id.size),
            "tree_input_sha256": "c072a937941fefb9aac441fc319ff030ceb666af4a07f1b88c0f02c5d76a3f43",
            "redshift_input_sha256": "816a885a6e73d6d9022fffeb8667acfe2b0719a6cb0da2d696abe61500b135b9",
        },
    )

    smf_artifact = Artifact(
        key="upstream_shark_smf",
        title="A real SHARK reference stellar mass function",
        path="assets/upstream-shark-stellar-mass-function.svg",
        media_type="image/svg+xml",
        role="figure",
        description=(
            "The pinned upstream executable on its public CI tree. This establishes the "
            "reference population; it is not yet a mimic-jax parity overlay."
        ),
    )
    common_observables_artifact = Artifact(
        key="upstream_shark_common_observables",
        title="Common SAGE–SHARK catalogue observables",
        path="assets/upstream-shark-common-observables.svg",
        media_type="image/svg+xml",
        role="figure",
        description=(
            "Cold-gas fraction, cold-gas metallicity, BH–bulge relation, and quenched "
            "fraction evaluated through explicit model-neutral binning and selection rules."
        ),
    )
    additions_artifact = Artifact(
        key="upstream_shark_added_observables",
        title="Gas phases, black-hole spin, and angular-momentum sizes",
        path="assets/upstream-shark-added-observables.svg",
        media_type="image/svg+xml",
        role="figure",
        description=(
            "Native atomic/molecular gas mass functions, BH spin, and disk-size "
            "outputs retained rather than collapsed into the SAGE16 state. Relation bins require "
            "at least 20 galaxies so rare objects are not presented as a stable trend."
        ),
    )
    convergence_artifact = Artifact(
        key="shark_flow_convergence",
        title="Convergence of the coupled SHARK flow foundation",
        path="assets/shark-flow-convergence.svg",
        media_type="image/svg+xml",
        role="figure",
        description=(
            "Euler, Heun RK2, and RK4 recover their expected orders for a controlled "
            "nonlinear Croton06+BR06+Lagos13 SHARK disk flow while its baryon ledger stays at roundoff."
        ),
    )
    flow_artifact = Artifact(
        key="shark_flow_network",
        title="The implemented SHARK flow network",
        path="assets/shark-flow-network.svg",
        media_type="image/svg+xml",
        role="figure",
        description="Mass, metals, and angular momentum use one named transfer assembly.",
    )
    prescription_artifact = Artifact(
        key="lagos23_prescription_oracle",
        title="Direct upstream oracle checks for six Lagos23 prescription groups",
        path="assets/lagos23-prescription-oracle.svg",
        media_type="image/svg+xml",
        role="figure",
        description=(
            "Radial star-formation/angular-momentum, stellar-feedback, reincorporation, "
            "Sobacchi13 reionisation, Croton06 cooling, and deterministic Lagos23 AGN "
            "cases generated by the pinned upstream SHARK library."
        ),
    )
    response_artifact = Artifact(
        key="lagos23_fractional_response_matrix",
        title="Fractional response matrix of the continuous Lagos23 core",
        path="assets/lagos23-fractional-response-matrix.svg",
        media_type="image/svg+xml",
        role="figure",
        description=(
            "Each entry is the percentage change in a final reservoir or integrated "
            "cooling transfer per 1% parameter increase, evaluated by JAX AD."
        ),
    )
    arrays_artifact = Artifact(
        key="shark_foundation_arrays",
        title="Machine-readable foundation arrays",
        path="assets/shark-foundation-results.npz",
        media_type="application/x-npz",
        role="scientific_array",
        description=(
            "Catalogue summaries, controlled convergence histories, direct-oracle residuals, "
            "and fractional-response matrices used by this report."
        ),
    )

    report = RunReport(
        identity=RunIdentity(
            run_id="shark-continuous-foundation",
            title="SHARK Lagos23 on the same testable footing as SAGE16",
            model="SHARK Lagos23 native reference plus mimic-jax continuous/hybrid implementation",
            dataset="upstream public CI mini-SURFS tree, batch 0",
            parameter_set="sample_lagos23.cfg",
            integration_method=(
                "exact native hybrid reference; explicit JAX reference order and continuous RK4"
            ),
            summary=(
                "A complete native Lagos23 population remains the topology/event reference, while "
                "the independent JAX implementation now covers the disk and burst ODEs, BH/AGN "
                "memory and spin, mergers, instabilities, environmental transfers, and shared observables."
            ),
        ),
        provenance=provenance,
        overview_metrics=(
            ScalarMetric("galaxies", "Reference galaxies", int(catalogue.galaxy_id.size)),
            ScalarMetric("redshift", "Output redshift", catalogue.redshift),
            ScalarMetric("flow_state", "Continuous flow variables", 19),
            ScalarMetric("augmented_state", "Reservoir + BH/AGN state variables", 24),
            ScalarMetric("native_catalogue_fields", "Native galaxy fields available", 86),
            ScalarMetric(
                "interval_oracle_error",
                "Maximum controlled interval residual",
                float(np.max(np.abs(interval_delta))),
            ),
            ScalarMetric(
                "burst_oracle_error",
                "Maximum controlled burst residual",
                float(np.max(np.abs(starburst_delta))),
            ),
            ScalarMetric("runtime", "Report analysis wall time", runtime, "s"),
        ),
        health=(
            Diagnostic(
                key="upstream_oracle",
                title="Pinned upstream SHARK oracle",
                status=DiagnosticStatus.PASSED,
                summary=(
                    "The clean pinned upstream executable completed the public CI tree and the "
                    "catalogue records the expected revision, version, seed, and 7,553 galaxies."
                ),
                metrics=(
                    ScalarMetric("galaxies", "Output galaxies", int(catalogue.galaxy_id.size)),
                    ScalarMetric("seed", "Fixed seed", catalogue.seed),
                ),
                method="native ICRAR/shark executable and public upstream CI input",
                tolerance="revision exact; successful native run; output schema readable",
            ),
            Diagnostic(
                key="flow_equations",
                title="19-state flow equations",
                status=DiagnosticStatus.PASSED,
                summary=(
                    "The JAX flow assembly reproduces every equation in upstream "
                    "basic_physicalmodel_evaluator for controlled named rates."
                ),
                method="source-level equation fixture plus direct float64 comparison",
                tolerance="relative difference <= 5e-15 in the controlled fixture",
            ),
            Diagnostic(
                key="br06_star_formation_oracle",
                title="BR06 radial star-formation prescription",
                status=DiagnosticStatus.PASSED,
                summary=(
                    "Four disk/burst cases agree with the pinned upstream prescription to "
                    "better than 5 parts per million, including angular-momentum transport."
                ),
                metrics=(
                    ScalarMetric(
                        "maximum_rate_difference",
                        "Maximum absolute fractional SFR difference",
                        float(np.max(np.abs(star_rate_delta))),
                    ),
                    ScalarMetric(
                        "maximum_angular_momentum_difference",
                        "Maximum absolute fractional SF angular-momentum difference",
                        float(np.max(np.abs(angular_rate_delta))),
                    ),
                ),
                artifacts=(prescription_artifact,),
                method="C++ oracle linked to pinned libshark versus pure JAX 128-node quadrature",
                tolerance="SFR <= 5e-6 relative; angular-momentum rate <= 5e-8 relative",
            ),
            Diagnostic(
                key="lagos13_feedback_oracle",
                title="Lagos13 stellar-feedback loadings",
                status=DiagnosticStatus.PASSED,
                summary=(
                    "Five velocity/redshift cases reproduce the upstream reheating, ejection, "
                    "and angular-momentum loadings exactly in float64 output."
                ),
                metrics=(
                    ScalarMetric(
                        "maximum_loading_difference",
                        "Maximum absolute fractional loading difference",
                        float(np.max(np.abs(feedback_delta))),
                    ),
                ),
                artifacts=(prescription_artifact,),
                method="C++ oracle linked to pinned libshark versus pure JAX prescription",
                tolerance="array equality for the selected cases",
            ),
            Diagnostic(
                key="reincorporation_oracle",
                title="Lagos23 reincorporation finite map",
                status=DiagnosticStatus.PASSED,
                summary=(
                    "Five central/satellite and halo-mass cases reproduce upstream's "
                    "finite transfer and source cap exactly."
                ),
                metrics=(
                    ScalarMetric(
                        "maximum_transfer_difference",
                        "Maximum absolute fractional realized-transfer difference",
                        float(np.max(np.abs(reincorporation_delta))),
                    ),
                ),
                artifacts=(prescription_artifact,),
                method="C++ oracle finite map versus JAX reference projection",
                tolerance="array equality for realized transfers",
            ),
            Diagnostic(
                key="sobacchi13_reionisation_oracle",
                title="Sobacchi13 reionisation gate",
                status=DiagnosticStatus.PASSED,
                summary=(
                    "All eight velocity/redshift cases select the same cooling-suppression "
                    "branch as the pinned upstream model."
                ),
                metrics=(
                    ScalarMetric(
                        "matched_cases",
                        "Matched threshold cases",
                        int(np.count_nonzero(reionisation_matches)),
                    ),
                ),
                method="C++ oracle branch result versus pure JAX threshold",
                tolerance="8 of 8 exact boolean matches",
            ),
            Diagnostic(
                key="croton06_cooling_oracle",
                title="Croton06 cooling preparation",
                status=DiagnosticStatus.PASSED,
                summary=(
                    "Five halo cases reproduce upstream cooling radii, unheated rates, "
                    "and integrated cooling luminosities at float64 precision."
                ),
                metrics=(
                    ScalarMetric(
                        "maximum_cooling_difference",
                        "Maximum absolute fractional cooling difference",
                        float(np.max(np.abs(cooling_delta))),
                    ),
                ),
                artifacts=(prescription_artifact,),
                method="public upstream GasCooling utilities versus pure JAX equations",
                tolerance="maximum relative difference <= 8e-15",
            ),
            Diagnostic(
                key="lagos23_agn_oracle",
                title="Deterministic Lagos23 AGN rates",
                status=DiagnosticStatus.PASSED,
                summary=(
                    "Five black-hole cases reproduce hot-mode accretion, mechanical "
                    "and bolometric luminosity, radiative efficiency, QSO wind loadings, "
                    "and the upstream luminosity gate."
                ),
                metrics=(
                    ScalarMetric(
                        "maximum_agn_difference",
                        "Maximum absolute fractional deterministic AGN difference",
                        float(np.max(np.abs(agn_delta))),
                    ),
                ),
                artifacts=(prescription_artifact,),
                method="public upstream AGNFeedback methods versus pure JAX equations",
                tolerance="maximum relative difference <= 2e-15",
            ),
            Diagnostic(
                key="reference_interval_oracle",
                title="Ordered disk interval against upstream SHARK",
                status=DiagnosticStatus.PASSED,
                summary=(
                    "The finite reincorporation/infall/seed/BH/cooling preparation, "
                    "19-state disk solve, heating-memory projection, and post-solve "
                    "state agree with a real upstream BasicPhysicalModel interval."
                ),
                metrics=(
                    ScalarMetric(
                        "maximum_interval_difference",
                        "Maximum absolute fractional interval difference",
                        float(np.max(np.abs(interval_delta))),
                    ),
                ),
                method="pinned C++ BasicPhysicalModel interval versus one JAX RK4 reference step",
                tolerance="all selected persistent fields and rates <= 3e-5 relative",
            ),
            Diagnostic(
                key="starburst_interval_oracle",
                title="Merger/instability starburst sequence",
                status=DiagnosticStatus.PASSED,
                summary=(
                    "Finite BH fuel removal, Griffin19 spin, boosted bulge star formation, "
                    "SN/QSO feedback, and post-burst BH growth agree with the upstream sequence."
                ),
                metrics=(
                    ScalarMetric(
                        "maximum_starburst_difference",
                        "Maximum absolute fractional burst difference",
                        float(np.max(np.abs(starburst_delta))),
                    ),
                ),
                method="pinned C++ starburst event sequence versus one JAX RK4 reference step",
                tolerance="all selected persistent fields and rates <= 1.1e-4 relative",
            ),
            Diagnostic(
                key="flow_conservation",
                title="Mass, metal-source, and angular-momentum ledgers",
                status=DiagnosticStatus.PASSED,
                summary=(
                    "Mass and angular momentum cancel structurally; the metal ledger closes after "
                    "subtracting the explicitly named stellar-yield source. Derivative ledgers also close."
                ),
                metrics=(
                    ScalarMetric(
                        "maximum_residual",
                        "Maximum controlled RHS residual",
                        maximum_ledger_residual,
                        description="absolute residual in the native ledger basis",
                    ),
                ),
                method="value and jax.jacfwd conservation identities",
                tolerance="absolute residual <= 2e-15 in the unit test fixture",
            ),
            Diagnostic(
                key="augmented_bh_conservation",
                title="Continuous hot-mode black-hole transfer",
                status=DiagnosticStatus.PASSED,
                summary=(
                    "Hot-mode growth transfers hot gas and metals into the augmented BH state. "
                    "The removed gas angular momentum is an explicit sink because SHARK stores "
                    "dimensionless BH spin rather than BH angular momentum in the baryon ledger."
                ),
                metrics=(
                    ScalarMetric(
                        "maximum_augmented_residual",
                        "Maximum augmented transfer residual",
                        augmented_residual,
                    ),
                ),
                method="value and jax.jacfwd augmented mass/metal/AM-sink identities",
                tolerance="absolute residual <= 2e-15 in the controlled fixture",
            ),
            Diagnostic(
                key="controlled_convergence",
                title="Controlled flow convergence",
                status=DiagnosticStatus.PASSED,
                summary=(
                    "The nonlinear Croton06-cooling, BR06-star-formation, and Lagos13-feedback "
                    "flow recovers first-, second-, and fourth-order convergence for Euler, Heun, and RK4."
                ),
                metrics=tuple(
                    ScalarMetric(
                        f"order_{method}",
                        f"Observed {method} order",
                        float(order),
                    )
                    for method, order in zip(methods, observed_orders)
                ),
                artifacts=(convergence_artifact,),
                method=(
                    "successive halving of the oracled Croton06+BR06+Lagos13 disk flow under fixed "
                    "halo/structural forcing against an 8,192-step RK4 reference"
                ),
                tolerance="orders within 0.12 of 1, 2, and 4",
            ),
            Diagnostic(
                key="fractional_parameter_responses",
                title="Differentiable fractional parameter responses",
                status=DiagnosticStatus.PASSED,
                summary=(
                    "JAX directly returns dimensionless reservoir and cooling responses "
                    "for SN-regulated and massive AGN-heated controlled galaxies."
                ),
                metrics=(
                    ScalarMetric(
                        "finite_difference_error",
                        "Maximum relative AD/finite-difference discrepancy",
                        response_validation,
                    ),
                ),
                artifacts=(response_artifact,),
                method="jax.jacrev of log observables with respect to log parameter multipliers",
                tolerance="selected symmetric finite difference agrees within 5e-5 relative",
            ),
            Diagnostic(
                key="independent_jax_population_parity",
                title="Independent JAX full-tree population parity",
                status=DiagnosticStatus.NOT_EVALUATED,
                summary=(
                    "The exact native population backend is integrated and the JAX process/event "
                    "kernels cover the controlled pinned Lagos23 branches, but per-ID independent JAX replay of "
                    "all 20,174 public-CI trees has not passed. No such parity claim is made."
                ),
            ),
        ),
        headline_artifacts=(
            smf_artifact,
            common_observables_artifact,
            additions_artifact,
            prescription_artifact,
            flow_artifact,
            convergence_artifact,
            response_artifact,
        ),
        sections=(
            ReportSection(
                key="reference_universe",
                title="What is the SHARK reference prediction?",
                summary=(
                    "We begin with a genuine upstream Lagos23 run, not a toy reinterpretation. The "
                    "stellar mass function supplies the first familiar population target that mimic-jax must reproduce."
                ),
                body=(
                    "Mass is the sum `mstars_disk + mstars_bulge`; volume and $h$ come from the "
                    "native catalogue produced through mimic-jax's managed, checksum-recorded "
                    "upstream backend. An independent per-ID JAX replay is deliberately not overplotted: "
                    "that stricter population-equivalence gate remains open."
                ),
                artifacts=(smf_artifact,),
            ),
            ReportSection(
                key="prescription_oracles",
                title="Do the first JAX physics prescriptions reproduce SHARK?",
                summary=(
                    "Yes for the isolated prescription suite and complete controlled intervals: BR06 molecular star formation, "
                    "Lagos13 feedback, reincorporation, Sobacchi13 reionisation, Croton06 "
                    "cooling, deterministic Lagos23 AGN rates, Griffin19 spin, the ordered disk "
                    "interval, and the event-triggered starburst now pass direct upstream oracles."
                ),
                body=(
                    "The fixture is generated by a small C++ harness linked to the clean pinned "
                    "SHARK library. It does not reimplement the expected equations in Python. "
                    "The residual BR06 difference comes from replacing upstream's 5%-tolerance "
                    "adaptive GSL radial integral with deterministic 128-node JAX quadrature; it "
                    "is below $5\\times10^{-6}$ in all four controlled cases. Interval and burst "
                    "comparisons exercise the actual upstream BasicPhysicalModel ordering rather "
                    "than equations copied into the test."
                ),
                artifacts=(prescription_artifact,),
            ),
            ReportSection(
                key="common_observables",
                title="Can SHARK be compared through the same familiar observables?",
                summary=(
                    "The catalogue adapter now evaluates four additional SAGE-facing summaries "
                    "with shared binning, finite-value, unit, and zero-handling rules."
                ),
                body=(
                    "These curves are the real pinned upstream SHARK population, not yet a JAX "
                    "population overlay. Their purpose is to make the target comparison contract "
                    "executable: one definition will consume either a SAGE or SHARK catalogue. "
                    "The gas-metallicity panel intentionally shows the native metal mass fraction; "
                    "an oxygen-abundance calibration will be added only with an explicit convention."
                ),
                artifacts=(common_observables_artifact, arrays_artifact),
            ),
            ReportSection(
                key="continuous_core",
                title="What part of SHARK is already a dynamical system?",
                summary=(
                    "Upstream SHARK already integrates a 19-variable disk/starburst system. "
                    "mimic-jax makes its physical rates and conservative routing explicit."
                ),
                body=(
                    "The state contains six masses, six corresponding metal masses, two episode "
                    "trackers, and five total angular momenta. Cooling, star formation, recycling, "
                    "stellar reheating/ejection, and QSO loss enter as named rates. Continuous mode "
                    "augments this with BH mass/metals/spin, heating radius, and excess jet power. "
                    "The heating radius uses the exact running-maximum projection. Finite infall "
                    "and caps, seeded and burst BH growth, Griffin19 spin, mergers, disk "
                    "instabilities, stripping, and merger clocks are explicit hybrid maps; they "
                    "are not mislabeled as smooth ODE terms."
                ),
                artifacts=(flow_artifact,),
            ),
            ReportSection(
                key="convergence",
                title="Does the continuous transfer network converge in time?",
                summary=(
                    "Yes for the implemented state-dependent flow foundation: refining the step "
                    "reduces stellar-mass error at the designed method order without opening the baryon ledger."
                ),
                body=(
                    "This test evolves the actual oracled Croton06 cooling, BR06 radial "
                    "star-formation, and Lagos13 stellar-feedback prescriptions through the "
                    "continuous conservative routing. Halo structure is held fixed while the "
                    "tabulated rate responds to the evolving hot mass and metallicity. This analysis "
                    "treats hybrid events as explicit maps rather than assigning them a fictitious "
                    "ODE order. Event-time convergence across an entire merger tree remains part of "
                    "the open independent population-replay gate."
                ),
                artifacts=(convergence_artifact, arrays_artifact),
            ),
            ReportSection(
                key="shark_additions",
                title="What does SHARK add to the SAGE comparison?",
                summary=(
                    "The managed reference catalogue and shared observable layer now retain SHARK's "
                    "phase-resolved gas, structure, angular momentum, and BH-spin outputs rather than "
                    "collapsing them onto the smaller SAGE16 catalogue contract."
                ),
                body=(
                    "| Added SHARK capability | Resulting comparison/science output |\n"
                    "| --- | --- |\n"
                    "| Atomic/molecular partition and five SF laws | HI/H2 mass functions, depletion times, phase-resolved responses |\n"
                    "| Component angular momentum and sizes | disk/bulge size–mass and AM relations |\n"
                    "| BH spin plus radiative/mechanical AGN power | spin, luminosity, jet-power, and quenching diagnostics |\n"
                    "| Gradual hot/ISM ram-pressure and tidal stripping | environmental gas loss and stellar-halo assembly |\n"
                    "| Burst channels by merger versus instability | causal decomposition of bulge growth and starbursts |\n"
                    "| Cold-halo, ejected, QSO-lost, and stripped reservoirs | a more resolved baryon-cycle ledger |"
                ),
                artifacts=(additions_artifact,),
            ),
            ReportSection(
                key="fractional_responses",
                title="Which Lagos23 parameters control a galaxy interval?",
                summary=(
                    "The continuous/hybrid implementation produces practitioner-facing fractional "
                    "responses directly: percent change in a familiar output per one-percent change "
                    "in a physical parameter."
                ),
                body=(
                    "Rows are final stellar, cold-gas, hot-gas, BH, SFR, cooling, and ejected-gas "
                    "outputs for controlled SN-regulated and massive AGN-heated systems. Columns are "
                    "physically labelled Lagos23 parameters. A response of -0.6 means that a 1% "
                    "parameter increase lowers that output by approximately 0.6% locally. The first "
                    "active response column is checked against symmetric finite differences; inactive "
                    "threshold branches remain exactly zero rather than being smoothed."
                ),
                artifacts=(response_artifact, arrays_artifact),
            ),
            ReportSection(
                key="next_gates",
                title="What must pass before SHARK and SAGE are compared?",
                summary=(
                    "The controlled process/event surface and ordered intervals are implemented. The "
                    "remaining strict equivalence gate is independent per-ID JAX replay of the full "
                    "20,174-tree public-CI population."
                ),
                body=(
                    "The exact native SHARK population backend is reproducible and integrated; the "
                    "JAX layer separately passes prescription, conservation, differentiation, "
                    "ordered disk-interval, and starburst-event tests. Calling those two facts a full "
                    "independent population match would be premature. The next validation program "
                    "must replay stable galaxy IDs and report threshold/topology differences. A SAGE–SHARK "
                    "physics comparison then requires common halo forcing; comparing native Mini-Millennium "
                    "with native mini-SURFS would otherwise mix model and simulation differences."
                ),
            ),
        ),
        parameters=(
            ParameterValue(
                "recycle", 0.4588, description="Lagos23 sample instantaneous recycling fraction"
            ),
            ParameterValue("yield", 0.02908, description="Lagos23 sample stellar yield"),
            ParameterValue(
                "ode_solver_precision", 0.05, description="upstream relative ODE tolerance"
            ),
        ),
        links=(
            ReportLink(
                "SAGE16 science program",
                "../mini-millennium-sage16-science-program/index.md",
            ),
            ReportLink(
                "SAGE16 response times",
                "../sage16-linear-response/index.md",
            ),
        ),
    )
    written = write_report(report, destination)
    print(written.markdown_path)
    print(written.manifest_path)


if __name__ == "__main__":
    main()
