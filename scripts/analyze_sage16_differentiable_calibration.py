#!/usr/bin/env python3
"""Fit two SAGE16 parameters to the observed low-redshift stellar mass function."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from mimic_jax.inference import (  # noqa: E402
    LocalLogResponseEmulator,
    LogGaussianObservation,
    fit_local_log_response,
    independent_log_covariance,
    local_log_posterior,
    random_walk_metropolis,
    validate_log_posterior_gradient,
)
from mimic_jax.io import open_lhalo_partition  # noqa: E402
from mimic_jax.sage16 import (  # noqa: E402
    evolve_lhalo_partition,
    fiducial_parameters,
    linearize_lhalo_partition,
    load_baldry2008_stellar_mass_function,
    load_scale_factors,
    snapshot_timing,
    soft_stellar_mass_function,
    state_field_array,
    state_tangent_matrix,
    stellar_mass_function,
)

HUBBLE_H = 0.73
BOX_SIZE_MPC_OVER_H = 62.5
TREE_FILE_COUNT = 8
SNAPSHOT = 63
PARAMETER_NAMES = ("FeedbackReheatingEpsilon", "ReIncorporationFactor")
FIT_MASS_MIN = 8.5
FIT_MASS_MAX = 11.15
MINIMUM_MODEL_COUNT = 10
SURROGATE_MAXIMUM_ERROR_DEX = 0.05
NONLINEAR_STEP_TOLERANCE = 5.0e-3
TRAINING_LEVELS = ((-0.18, -0.09, 0.0), (-0.24, -0.12, 0.0))
HELD_OUT_POINTS = ((-0.135, -0.18), (-0.135, -0.06), (-0.045, -0.18), (-0.045, -0.06))


def parse_arguments():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trees",
        type=Path,
        default=repository / "simulations/mini-millennium/snapshots/trees_063.1",
    )
    parser.add_argument(
        "--scale-factors",
        type=Path,
        default=repository / "simulations/mini-millennium/mini-millennium.a_list",
    )
    parser.add_argument(
        "--observation",
        type=Path,
        default=repository / "data/observations/baldry2008_stellar_mass_function.csv",
    )
    parser.add_argument(
        "--baseline-arrays",
        type=Path,
        default=repository / "archive/mini-millennium-sage16-parameter-responses.npz",
    )
    parser.add_argument("--global-tree-offset", type=int, default=3432)
    parser.add_argument("--num-substeps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-members", type=int, default=512)
    parser.add_argument("--bandwidth-dex", type=float, default=0.05)
    parser.add_argument("--mcmc-steps", type=int, default=50_000)
    parser.add_argument("--mcmc-burn-in", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=481516)
    parser.add_argument("--skip-nonlinear", action="store_true")
    parser.add_argument("--skip-held-out", action="store_true")
    parser.add_argument("--tree-count", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-arrays", type=Path, required=True)
    return parser.parse_args()


def maximum_resident_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def progress(label):
    def update(event):
        if event["event"] == "snapshot":
            print(
                f"[{label}] snapshot={event['snapshot']:02d} groups={event['groups']}",
                flush=True,
            )

    return update


def snapshot_records(result):
    return tuple(result.records_by_snapshot.get(SNAPSHOT, ()))


def selected_parameter_values(parameters):
    return np.asarray([float(getattr(parameters, name)) for name in PARAMETER_NAMES])


def parameters_from_log_ratios(fiducial, log_ratios):
    values = selected_parameter_values(fiducial) * np.exp(np.asarray(log_ratios))
    return fiducial._replace(
        **{
            name: jnp.asarray(value, dtype=jnp.float64)
            for name, value in zip(PARAMETER_NAMES, values)
        }
    )


def population_summaries(records, edges, bandwidth_dex):
    volume = BOX_SIZE_MPC_OVER_H**3 / TREE_FILE_COUNT
    masses = state_field_array(records, "StellarMass").astype(np.float64)
    hard = stellar_mass_function(
        masses,
        volume_mpc_over_h_cubed=volume,
        hubble_h=HUBBLE_H,
        bin_edges=edges,
    )
    soft = np.asarray(
        soft_stellar_mass_function(
            jnp.asarray(masses),
            volume_mpc_over_h_cubed=volume,
            hubble_h=HUBBLE_H,
            bin_edges=edges,
            bandwidth_dex=bandwidth_dex,
        )
    )
    return masses, hard, soft


def linearized_population(
    partition,
    timing,
    parameters,
    arguments,
    tree_indices,
    edges,
    label,
):
    started = time.perf_counter()
    result = linearize_lhalo_partition(
        partition,
        timing,
        tree_indices=tree_indices,
        global_tree_offset=arguments.global_tree_offset,
        num_substeps=arguments.num_substeps,
        output_snapshots=(SNAPSHOT,),
        batch_size=arguments.batch_size,
        max_batch_members=arguments.max_batch_members,
        member_binning="power_of_two",
        parameters=parameters,
        parameter_names=PARAMETER_NAMES,
        progress_callback=progress(label),
    )
    if not result.success:
        raise RuntimeError(f"{label} linearized partition evolution failed")
    records = snapshot_records(result)
    masses, hard, soft = population_summaries(records, edges, arguments.bandwidth_dex)
    tangents = state_tangent_matrix(records, "StellarMass").astype(np.float64)

    def observable(values):
        return soft_stellar_mass_function(
            values,
            volume_mpc_over_h_cubed=BOX_SIZE_MPC_OVER_H**3 / TREE_FILE_COUNT,
            hubble_h=HUBBLE_H,
            bin_edges=edges,
            bandwidth_dex=arguments.bandwidth_dex,
        )

    raw = np.asarray(
        jax.vmap(lambda direction: jax.jvp(observable, (jnp.asarray(masses),), (direction,))[1])(
            jnp.asarray(tangents.T)
        )
    ).T
    parameter_values = selected_parameter_values(parameters)
    response = np.full_like(raw, np.nan)
    positive = soft > 0.0
    response[positive] = raw[positive] * parameter_values[None, :] / soft[positive, None]
    return {
        "hard": hard.number_density,
        "counts": hard.counts,
        "soft": soft,
        "response": response,
        "records": len(records),
        "seconds": time.perf_counter() - started,
    }


def primal_population(
    partition,
    timing,
    parameters,
    arguments,
    tree_indices,
    edges,
    label,
):
    started = time.perf_counter()
    result = evolve_lhalo_partition(
        partition,
        timing,
        tree_indices=tree_indices,
        global_tree_offset=arguments.global_tree_offset,
        num_substeps=arguments.num_substeps,
        output_snapshots=(SNAPSHOT,),
        batch_size=arguments.batch_size,
        max_batch_members=arguments.max_batch_members,
        member_binning="power_of_two",
        parameters=parameters,
        progress_callback=progress(label),
    )
    if not result.success:
        raise RuntimeError(f"{label} partition evolution failed")
    records = snapshot_records(result)
    _, hard, soft = population_summaries(records, edges, arguments.bandwidth_dex)
    return {
        "hard": hard.number_density,
        "counts": hard.counts,
        "soft": soft,
        "records": len(records),
        "seconds": time.perf_counter() - started,
    }


def interpolate_observation(observation, centres):
    values = np.interp(centres, observation.coordinate, observation.values)
    lower = np.interp(centres, observation.coordinate, observation.lower_errors)
    upper = np.interp(centres, observation.coordinate, observation.upper_errors)
    return values, lower, upper


def make_fit_problem(centres, hard, counts, response, observation_values, lower, upper, mask):
    covariance = independent_log_covariance(
        observation_values[mask],
        lower[mask],
        upper[mask],
        model_counts=counts[mask],
    )
    names = tuple(f"log10(M*)={value:.2f}" for value in centres[mask])
    observed = LogGaussianObservation(
        observation_values[mask],
        covariance,
        names,
        ("Mpc^-3 dex^-1",) * len(names),
    )
    emulator = LocalLogResponseEmulator(
        hard[mask],
        response[mask],
        np.ones(len(PARAMETER_NAMES)),
        names,
        PARAMETER_NAMES,
    )
    return emulator, observed


def log_prediction_error(actual, predicted, mask):
    valid = mask & (actual > 0.0) & (predicted > 0.0)
    values = np.log10(actual[valid] / predicted[valid])
    return values, float(np.max(np.abs(values))), float(np.median(np.abs(values)))


def quadratic_features(log_parameter_ratios):
    """Second-order features with conventional one-half diagonal factors."""

    values = jnp.asarray(log_parameter_ratios, dtype=jnp.float64)
    return jnp.asarray((0.5 * values[0] ** 2, values[0] * values[1], 0.5 * values[1] ** 2))


def fit_gradient_enhanced_quadratic(training_points, training_log_changes, elasticities):
    """Fit curvature while holding the fiducial value and JAX gradient fixed."""

    points = np.asarray(training_points, dtype=np.float64)
    changes = np.asarray(training_log_changes, dtype=np.float64)
    response = np.asarray(elasticities, dtype=np.float64)
    features = np.asarray([quadratic_features(point) for point in points])
    remainder = changes - points @ response.T
    coefficients, _, _, _ = np.linalg.lstsq(features, remainder, rcond=None)
    return coefficients


def quadratic_log_change(log_parameter_ratios, elasticities, coefficients):
    values = jnp.asarray(log_parameter_ratios, dtype=jnp.float64)
    return jnp.asarray(elasticities) @ values + quadratic_features(values) @ jnp.asarray(
        coefficients
    )


def optimize_quadratic_likelihood(
    baseline_values,
    elasticities,
    coefficients,
    observation,
    initial,
):
    precision = observation.precision
    observed = observation.log_values
    baseline_log = jnp.log(jnp.asarray(baseline_values))

    def negative_log_likelihood(values):
        residual = (
            baseline_log + quadratic_log_change(values, elasticities, coefficients) - observed
        )
        return 0.5 * residual @ precision @ residual

    value_and_gradient = jax.jit(jax.value_and_grad(negative_log_likelihood))

    def scipy_objective(values):
        value, gradient = value_and_gradient(jnp.asarray(values))
        return float(value), np.asarray(gradient, dtype=np.float64)

    bounds = tuple((min(levels), max(levels)) for levels in TRAINING_LEVELS)
    result = minimize(
        scipy_objective,
        np.asarray(initial, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"ftol": 1.0e-12, "gtol": 1.0e-9, "maxiter": 200},
    )
    if not result.success:
        raise RuntimeError(f"Quadratic response optimization failed: {result.message}")
    optimum = jnp.asarray(result.x, dtype=jnp.float64)
    hessian = jax.hessian(negative_log_likelihood)(optimum)
    covariance = jnp.linalg.inv(hessian)
    return result, negative_log_likelihood, hessian, covariance


def _unconstrained_newton_calibration():
    """Retained failure diagnostic: hard-bin Newton steps can leave the local regime."""
    arguments = parse_arguments()
    started = time.perf_counter()
    baseline = dict(np.load(arguments.baseline_arrays, allow_pickle=False))
    centres = baseline["stellar_mass_bin_centres"].astype(np.float64)
    edges = baseline["stellar_mass_bin_edges"].astype(np.float64)
    baseline_names = tuple(str(value) for value in baseline["parameter_names"])
    parameter_indices = [baseline_names.index(name) for name in PARAMETER_NAMES]
    baseline_response = baseline["parameter_response"][:, parameter_indices].astype(np.float64)
    baseline_hard = baseline["hard_smf"].astype(np.float64)
    baseline_soft = baseline["soft_smf"].astype(np.float64)
    baseline_counts = baseline["hard_smf_counts"].astype(np.int64)

    observation = load_baldry2008_stellar_mass_function(
        arguments.observation,
        hubble_h=HUBBLE_H,
    )
    observation_values, observation_lower, observation_upper = interpolate_observation(
        observation, centres
    )
    fit_mask = (
        (centres >= FIT_MASS_MIN)
        & (centres <= FIT_MASS_MAX)
        & (baseline_counts >= MINIMUM_MODEL_COUNT)
        & (observation_values > observation_lower)
        & (baseline_hard > 0.0)
        & np.all(np.isfinite(baseline_response), axis=1)
    )
    initial_emulator, fit_observation = make_fit_problem(
        centres,
        baseline_hard,
        baseline_counts,
        baseline_response,
        observation_values,
        observation_lower,
        observation_upper,
        fit_mask,
    )
    initial_fit = fit_local_log_response(initial_emulator, fit_observation)
    print(
        "[initial] ratios="
        f"{np.asarray(initial_fit.parameter_ratios)} "
        f"chi2={initial_fit.chi_square_fiducial:.3f}->{initial_fit.chi_square_best:.3f}",
        flush=True,
    )

    arrays = {
        "stellar_mass_bin_centres": centres,
        "stellar_mass_bin_edges": edges,
        "fit_mask": fit_mask,
        "observation_mass": observation.coordinate,
        "observation_smf": observation.values,
        "observation_error_lower": observation.lower_errors,
        "observation_error_upper": observation.upper_errors,
        "observation_interpolated": observation_values,
        "observation_log_covariance": np.asarray(fit_observation.log_covariance),
        "baseline_hard_smf": baseline_hard,
        "baseline_soft_smf": baseline_soft,
        "baseline_counts": baseline_counts,
        "baseline_response": baseline_response,
        "parameter_names": np.asarray(PARAMETER_NAMES),
        "initial_log_parameter_step": np.asarray(initial_fit.log_parameter_ratios),
        "initial_covariance": np.asarray(initial_fit.covariance),
    }
    summary = {
        "schema_version": "mimic-jax-differentiable-calibration/v1",
        "observation": {
            "source": observation.source,
            "doi": observation.doi,
            "working_likelihood": (
                "diagonal Gaussian in ln(phi); quoted Baldry envelope plus fixed fiducial "
                "Mini-Millennium Poisson variance"
            ),
            "known_omissions": [
                "observational bin-to-bin covariance",
                "Mini-Millennium sample/cosmic variance beyond Poisson counting noise",
                "stellar-mass systematic covariance",
                "model discrepancy",
            ],
        },
        "fit_definition": {
            "parameter_names": list(PARAMETER_NAMES),
            "log10_stellar_mass_range": [FIT_MASS_MIN, FIT_MASS_MAX],
            "minimum_model_count": MINIMUM_MODEL_COUNT,
            "fitted_bins": int(np.count_nonzero(fit_mask)),
            "implicit_prior": False,
            "surrogate_gate_maximum_error_dex": SURROGATE_MAXIMUM_ERROR_DEX,
        },
        "initial_local_fit": {
            "parameter_ratios": np.asarray(initial_fit.parameter_ratios).tolist(),
            "log_parameter_step": np.asarray(initial_fit.log_parameter_ratios).tolist(),
            "chi_square_fiducial": initial_fit.chi_square_fiducial,
            "chi_square_predicted": initial_fit.chi_square_best,
            "condition_number": initial_fit.condition_number,
        },
        "nonlinear_evaluated": False,
    }

    if not arguments.skip_nonlinear:
        partition = open_lhalo_partition(arguments.trees)
        timing = snapshot_timing(load_scale_factors(arguments.scale_factors))
        if arguments.tree_count is None:
            tree_indices = tuple(range(partition.tree_count))
        else:
            tree_indices = tuple(range(min(arguments.tree_count, partition.tree_count)))
        if len(tree_indices) != partition.tree_count:
            raise ValueError(
                "Nonlinear calibration currently requires the complete archived partition so "
                "the baseline response, counts, and volume remain consistent"
            )
        fiducial = fiducial_parameters()
        centre_log_ratios = np.asarray(initial_fit.log_parameter_ratios)
        initial_predicted_hard = baseline_hard * np.exp(baseline_response @ centre_log_ratios)
        iteration_records = []
        final_linearized = None
        final_fit = None
        first_candidate = None
        for iteration in range(arguments.maximum_iterations):
            centre_parameters = parameters_from_log_ratios(fiducial, centre_log_ratios)
            linearized = linearized_population(
                partition,
                timing,
                centre_parameters,
                arguments,
                tree_indices,
                edges,
                f"iteration-{iteration + 1}",
            )
            emulator, current_observation = make_fit_problem(
                centres,
                linearized["hard"],
                baseline_counts,
                linearized["response"],
                observation_values,
                observation_lower,
                observation_upper,
                fit_mask,
            )
            correction = fit_local_log_response(emulator, current_observation)
            iteration_records.append(
                {
                    "iteration": iteration + 1,
                    "centre_log_parameter_ratios": centre_log_ratios.tolist(),
                    "centre_parameter_ratios": np.exp(centre_log_ratios).tolist(),
                    "correction": np.asarray(correction.log_parameter_ratios).tolist(),
                    "chi_square_centre": correction.chi_square_fiducial,
                    "chi_square_predicted": correction.chi_square_best,
                    "linearized_seconds": linearized["seconds"],
                }
            )
            final_linearized = linearized
            final_fit = correction
            if first_candidate is None:
                first_candidate = linearized
            print(
                f"[iteration {iteration + 1}] correction="
                f"{np.asarray(correction.log_parameter_ratios)} "
                f"chi2={correction.chi_square_fiducial:.3f}->{correction.chi_square_best:.3f}",
                flush=True,
            )
            if np.max(np.abs(np.asarray(correction.log_parameter_ratios))) <= (
                NONLINEAR_STEP_TOLERANCE
            ):
                break
            centre_log_ratios = centre_log_ratios + np.asarray(correction.log_parameter_ratios)
        converged = bool(
            np.max(np.abs(np.asarray(final_fit.log_parameter_ratios))) <= NONLINEAR_STEP_TOLERANCE
        )
        if not converged:
            raise RuntimeError(
                "Nonlinear Gauss-Newton iterations did not meet the predeclared step tolerance"
            )

        # The first iteration is the exact evaluation of the fiducial response proposal.
        initial_error, initial_maximum, initial_median = log_prediction_error(
            first_candidate["hard"], initial_predicted_hard, fit_mask
        )

        final_emulator, final_observation = make_fit_problem(
            centres,
            final_linearized["hard"],
            baseline_counts,
            final_linearized["response"],
            observation_values,
            observation_lower,
            observation_upper,
            fit_mask,
        )
        automatic_gradient, finite_gradient, gradient_error = validate_log_posterior_gradient(
            final_fit.log_parameter_ratios,
            final_emulator,
            final_observation,
            relative_steps=(1.0e-2, 3.0e-3, 1.0e-3),
        )
        chain = random_walk_metropolis(
            lambda values: local_log_posterior(values, final_emulator, final_observation),
            final_fit.log_parameter_ratios,
            2.4**2 / len(PARAMETER_NAMES) * np.asarray(final_fit.covariance),
            num_steps=arguments.mcmc_steps,
            burn_in=arguments.mcmc_burn_in,
            seed=arguments.seed,
        )

        held_out_log_offsets = []
        held_out_actual_hard = []
        held_out_actual_soft = []
        held_out_seconds = []
        finite_difference_soft = []
        if not arguments.skip_held_out:
            eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(final_fit.covariance))
            mode_offset = np.asarray(final_fit.log_parameter_ratios)
            for mode in range(len(PARAMETER_NAMES)):
                direction = np.sqrt(eigenvalues[mode]) * eigenvectors[:, mode]
                for sign in (-1.0, 1.0):
                    offset = mode_offset + sign * direction
                    tested = parameters_from_log_ratios(fiducial, centre_log_ratios + offset)
                    result = primal_population(
                        partition,
                        timing,
                        tested,
                        arguments,
                        tree_indices,
                        edges,
                        f"posterior-mode-{mode + 1}-{sign:+.0f}",
                    )
                    held_out_log_offsets.append(offset)
                    held_out_actual_hard.append(result["hard"])
                    held_out_actual_soft.append(result["soft"])
                    held_out_seconds.append(result["seconds"])

            finite_step = 1.0e-2
            for parameter_index, name in enumerate(PARAMETER_NAMES):
                sides = []
                for sign in (-1.0, 1.0):
                    offset = np.zeros(len(PARAMETER_NAMES))
                    offset[parameter_index] = sign * finite_step
                    tested = parameters_from_log_ratios(fiducial, centre_log_ratios + offset)
                    result = primal_population(
                        partition,
                        timing,
                        tested,
                        arguments,
                        tree_indices,
                        edges,
                        f"finite-{name}-{sign:+.0f}",
                    )
                    sides.append(result["soft"])
                    held_out_seconds.append(result["seconds"])
                finite_difference_soft.append(
                    (np.log(sides[1]) - np.log(sides[0])) / (2.0 * finite_step)
                )

        held_out_log_offsets = np.asarray(held_out_log_offsets, dtype=np.float64)
        held_out_actual_hard = np.asarray(held_out_actual_hard, dtype=np.float64)
        held_out_actual_soft = np.asarray(held_out_actual_soft, dtype=np.float64)
        if held_out_log_offsets.size:
            held_out_predicted_hard = np.asarray(
                [
                    final_linearized["hard"] * np.exp(final_linearized["response"] @ offset)
                    for offset in held_out_log_offsets
                ]
            )
            held_out_errors = []
            held_out_maximum = []
            held_out_median = []
            for actual, predicted in zip(held_out_actual_hard, held_out_predicted_hard):
                error, maximum, median = log_prediction_error(actual, predicted, fit_mask)
                held_out_errors.append(error)
                held_out_maximum.append(maximum)
                held_out_median.append(median)
            posterior_maximum_error = float(np.max(held_out_maximum))
            posterior_median_error = float(np.median(held_out_median))
        else:
            held_out_predicted_hard = np.empty_like(held_out_actual_hard)
            held_out_errors = []
            posterior_maximum_error = float("nan")
            posterior_median_error = float("nan")

        finite_difference_soft = np.asarray(finite_difference_soft, dtype=np.float64)
        if finite_difference_soft.size:
            derivative_absolute_error = np.abs(
                finite_difference_soft[:, fit_mask] - final_linearized["response"][fit_mask].T
            )
            derivative_maximum_error = float(np.max(derivative_absolute_error))
        else:
            derivative_absolute_error = np.empty((0, np.count_nonzero(fit_mask)))
            derivative_maximum_error = float("nan")

        chain_mean = np.mean(chain.samples, axis=0)
        chain_covariance = np.cov(chain.samples, rowvar=False)
        covariance_relative_error = float(
            np.linalg.norm(chain_covariance - np.asarray(final_fit.covariance))
            / np.linalg.norm(np.asarray(final_fit.covariance))
        )
        final_parameter_ratios = np.exp(
            centre_log_ratios + np.asarray(final_fit.log_parameter_ratios)
        )
        arrays.update(
            {
                "first_candidate_hard_smf": first_candidate["hard"],
                "initial_surrogate_predicted_hard_smf": initial_predicted_hard,
                "initial_surrogate_error_dex": initial_error,
                "final_centre_log_parameter_ratios": centre_log_ratios,
                "final_parameter_ratios": final_parameter_ratios,
                "final_hard_smf": final_linearized["hard"],
                "final_soft_smf": final_linearized["soft"],
                "final_counts": final_linearized["counts"],
                "final_response": final_linearized["response"],
                "final_correction": np.asarray(final_fit.log_parameter_ratios),
                "final_covariance": np.asarray(final_fit.covariance),
                "final_hessian": np.asarray(final_fit.hessian),
                "posterior_chain": chain.samples[::5],
                "posterior_chain_log_probability": chain.log_probabilities[::5],
                "held_out_log_offsets": held_out_log_offsets,
                "held_out_actual_hard_smf": held_out_actual_hard,
                "held_out_predicted_hard_smf": held_out_predicted_hard,
                "held_out_actual_soft_smf": held_out_actual_soft,
                "finite_difference_soft_response": finite_difference_soft,
                "finite_difference_response_absolute_error": derivative_absolute_error,
                "likelihood_gradient_automatic": np.asarray(automatic_gradient),
                "likelihood_gradient_finite_difference": np.asarray(finite_gradient),
                "likelihood_gradient_absolute_error": np.asarray(gradient_error),
            }
        )
        summary.update(
            {
                "nonlinear_evaluated": True,
                "tree_file": str(arguments.trees),
                "tree_count": len(tree_indices),
                "records": int(final_linearized["records"]),
                "iterations": iteration_records,
                "converged": converged,
                "initial_surrogate_validation": {
                    "passed": initial_maximum <= SURROGATE_MAXIMUM_ERROR_DEX,
                    "maximum_absolute_error_dex": initial_maximum,
                    "median_absolute_error_dex": initial_median,
                },
                "final_fit": {
                    "parameter_ratios": final_parameter_ratios.tolist(),
                    "parameter_values": (
                        selected_parameter_values(fiducial) * final_parameter_ratios
                    ).tolist(),
                    "one_sigma_log": np.asarray(final_fit.one_sigma_log).tolist(),
                    "one_sigma_ratio_lower": (
                        np.exp(centre_log_ratios) * np.asarray(final_fit.one_sigma_ratio_lower)
                    ).tolist(),
                    "one_sigma_ratio_upper": (
                        np.exp(centre_log_ratios) * np.asarray(final_fit.one_sigma_ratio_upper)
                    ).tolist(),
                    "chi_square": final_fit.chi_square_fiducial,
                    "degrees_of_freedom": int(np.count_nonzero(fit_mask) - len(PARAMETER_NAMES)),
                    "condition_number": final_fit.condition_number,
                    "uncertainty_interpretation": (
                        "local Laplace/linear-Gaussian working-likelihood interval; not a "
                        "complete observational posterior"
                    ),
                },
                "posterior_reference": {
                    "method": "random-walk Metropolis on the final local response model",
                    "steps": arguments.mcmc_steps,
                    "burn_in": arguments.mcmc_burn_in,
                    "acceptance_fraction": chain.acceptance_fraction,
                    "mean_log_parameter_ratios": chain_mean.tolist(),
                    "laplace_covariance_relative_error": covariance_relative_error,
                },
                "held_out_surrogate_validation": {
                    "evaluated": not arguments.skip_held_out,
                    "maximum_absolute_error_dex": (
                        posterior_maximum_error if np.isfinite(posterior_maximum_error) else None
                    ),
                    "median_absolute_error_dex": (
                        posterior_median_error if np.isfinite(posterior_median_error) else None
                    ),
                    "passed": bool(
                        np.isfinite(posterior_maximum_error)
                        and posterior_maximum_error <= SURROGATE_MAXIMUM_ERROR_DEX
                    ),
                },
                "derivative_validation": {
                    "evaluated": not arguments.skip_held_out,
                    "finite_log_parameter_step": 1.0e-2,
                    "maximum_absolute_elasticity_error": (
                        derivative_maximum_error if np.isfinite(derivative_maximum_error) else None
                    ),
                },
                "runtime": {
                    "linearized_seconds": float(
                        np.sum([item["linearized_seconds"] for item in iteration_records])
                    ),
                    "held_out_primal_seconds": float(np.sum(held_out_seconds)),
                },
            }
        )

    summary["backend"] = jax.default_backend()
    summary["jax_version"] = jax.__version__
    summary["peak_resident_bytes"] = maximum_resident_bytes()
    summary["total_seconds"] = time.perf_counter() - started
    arguments.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output_arrays, **arrays)
    summary["arrays"] = arguments.output_arrays.name
    arguments.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(arguments.output_json)
    print(arguments.output_arrays)


def main():
    """Run the gradient-enhanced quadratic-emulator calibration."""

    arguments = parse_arguments()
    started = time.perf_counter()
    baseline = dict(np.load(arguments.baseline_arrays, allow_pickle=False))
    centres = baseline["stellar_mass_bin_centres"].astype(np.float64)
    edges = baseline["stellar_mass_bin_edges"].astype(np.float64)
    baseline_names = tuple(str(value) for value in baseline["parameter_names"])
    parameter_indices = [baseline_names.index(name) for name in PARAMETER_NAMES]
    baseline_response = baseline["parameter_response"][:, parameter_indices].astype(np.float64)
    baseline_hard = baseline["hard_smf"].astype(np.float64)
    baseline_soft = baseline["soft_smf"].astype(np.float64)
    baseline_counts = baseline["hard_smf_counts"].astype(np.int64)
    observation = load_baldry2008_stellar_mass_function(
        arguments.observation,
        hubble_h=HUBBLE_H,
    )
    observation_values, observation_lower, observation_upper = interpolate_observation(
        observation, centres
    )
    fit_mask = (
        (centres >= FIT_MASS_MIN)
        & (centres <= FIT_MASS_MAX)
        & (baseline_counts >= MINIMUM_MODEL_COUNT)
        & (observation_values > observation_lower)
        & (baseline_hard > 0.0)
        & (baseline_soft > 0.0)
        & np.all(np.isfinite(baseline_response), axis=1)
    )
    initial_emulator, fit_observation = make_fit_problem(
        centres,
        baseline_hard,
        baseline_counts,
        baseline_response,
        observation_values,
        observation_lower,
        observation_upper,
        fit_mask,
    )
    initial_fit = fit_local_log_response(initial_emulator, fit_observation)
    initial_q = np.asarray(initial_fit.log_parameter_ratios)
    print(
        "[initial] ratios="
        f"{np.asarray(initial_fit.parameter_ratios)} "
        f"chi2={initial_fit.chi_square_fiducial:.3f}->{initial_fit.chi_square_best:.3f}",
        flush=True,
    )

    arrays = {
        "stellar_mass_bin_centres": centres,
        "stellar_mass_bin_edges": edges,
        "fit_mask": fit_mask,
        "observation_mass": observation.coordinate,
        "observation_smf": observation.values,
        "observation_error_lower": observation.lower_errors,
        "observation_error_upper": observation.upper_errors,
        "observation_interpolated": observation_values,
        "observation_log_covariance": np.asarray(fit_observation.log_covariance),
        "baseline_hard_smf": baseline_hard,
        "baseline_soft_smf": baseline_soft,
        "baseline_counts": baseline_counts,
        "baseline_response": baseline_response,
        "parameter_names": np.asarray(PARAMETER_NAMES),
        "initial_log_parameter_step": initial_q,
        "initial_covariance": np.asarray(initial_fit.covariance),
    }
    summary = {
        "schema_version": "mimic-jax-differentiable-calibration/v2",
        "observation": {
            "source": observation.source,
            "doi": observation.doi,
            "working_likelihood": (
                "diagonal Gaussian in ln(phi); quoted Baldry envelope plus fixed fiducial "
                "Mini-Millennium Poisson variance"
            ),
            "known_omissions": [
                "observational bin-to-bin covariance",
                "Mini-Millennium sample/cosmic variance beyond Poisson counting noise",
                "stellar-mass systematic covariance",
                "model discrepancy",
            ],
        },
        "fit_definition": {
            "parameter_names": list(PARAMETER_NAMES),
            "log10_stellar_mass_range": [FIT_MASS_MIN, FIT_MASS_MAX],
            "minimum_model_count": MINIMUM_MODEL_COUNT,
            "fitted_bins": int(np.count_nonzero(fit_mask)),
            "prior": "uniform within the stated logarithmic emulator design box",
            "training_levels": [list(values) for values in TRAINING_LEVELS],
            "held_out_points": [list(values) for values in HELD_OUT_POINTS],
            "surrogate_gate_maximum_error_dex": SURROGATE_MAXIMUM_ERROR_DEX,
        },
        "initial_local_fit": {
            "parameter_ratios": np.asarray(initial_fit.parameter_ratios).tolist(),
            "log_parameter_step": initial_q.tolist(),
            "chi_square_fiducial": initial_fit.chi_square_fiducial,
            "chi_square_predicted": initial_fit.chi_square_best,
            "condition_number": initial_fit.condition_number,
        },
        "nonlinear_evaluated": False,
    }

    if not arguments.skip_nonlinear:
        partition = open_lhalo_partition(arguments.trees)
        timing = snapshot_timing(load_scale_factors(arguments.scale_factors))
        if arguments.tree_count is None:
            tree_indices = tuple(range(partition.tree_count))
        else:
            tree_indices = tuple(range(min(arguments.tree_count, partition.tree_count)))
        if len(tree_indices) != partition.tree_count:
            raise ValueError(
                "The exact emulator design requires the complete archived partition so the "
                "baseline response, counts, and volume remain consistent"
            )
        fiducial = fiducial_parameters()
        training_points = np.asarray(
            [(left, right) for left in TRAINING_LEVELS[0] for right in TRAINING_LEVELS[1]],
            dtype=np.float64,
        )
        training_hard = []
        training_soft = []
        training_seconds = []
        for index, point in enumerate(training_points):
            if np.allclose(point, 0.0):
                training_hard.append(baseline_hard)
                training_soft.append(baseline_soft)
                training_seconds.append(0.0)
                continue
            result = primal_population(
                partition,
                timing,
                parameters_from_log_ratios(fiducial, point),
                arguments,
                tree_indices,
                edges,
                f"training-{index + 1}-of-{len(training_points) - 1}",
            )
            training_hard.append(result["hard"])
            training_soft.append(result["soft"])
            training_seconds.append(result["seconds"])
        training_hard = np.asarray(training_hard)
        training_soft = np.asarray(training_soft)
        training_log_changes = np.log(training_soft[:, fit_mask] / baseline_soft[fit_mask][None, :])
        coefficients = fit_gradient_enhanced_quadratic(
            training_points,
            training_log_changes,
            baseline_response[fit_mask],
        )
        training_predicted_log_changes = np.asarray(
            [
                quadratic_log_change(point, baseline_response[fit_mask], coefficients)
                for point in training_points
            ]
        )
        training_errors_dex = (training_log_changes - training_predicted_log_changes) / np.log(10.0)

        optimization, negative_log_likelihood, hessian, covariance = optimize_quadratic_likelihood(
            baseline_hard[fit_mask],
            baseline_response[fit_mask],
            coefficients,
            fit_observation,
            np.clip(
                initial_q,
                [min(values) for values in TRAINING_LEVELS],
                [max(values) for values in TRAINING_LEVELS],
            ),
        )
        optimum = np.asarray(optimization.x)
        covariance = np.asarray(covariance)
        hessian = np.asarray(hessian)
        condition_number = float(np.linalg.cond(hessian))
        one_sigma = np.sqrt(np.diag(covariance))
        bounds = np.asarray(
            [(min(levels), max(levels)) for levels in TRAINING_LEVELS], dtype=np.float64
        )

        def bounded_log_probability(values):
            values = jnp.asarray(values)
            in_bounds = jnp.all(
                (values >= jnp.asarray(bounds[:, 0])) & (values <= jnp.asarray(bounds[:, 1]))
            )
            return jnp.where(in_bounds, -negative_log_likelihood(values), -jnp.inf)

        chain = random_walk_metropolis(
            bounded_log_probability,
            optimum,
            2.4**2 / len(PARAMETER_NAMES) * covariance,
            num_steps=arguments.mcmc_steps,
            burn_in=arguments.mcmc_burn_in,
            seed=arguments.seed,
        )
        chain_quantiles = np.quantile(chain.samples, [0.16, 0.5, 0.84], axis=0)

        initial_candidate = primal_population(
            partition,
            timing,
            parameters_from_log_ratios(fiducial, initial_q),
            arguments,
            tree_indices,
            edges,
            "linear-response-candidate",
        )
        initial_predicted_hard = baseline_hard * np.exp(baseline_response @ initial_q)
        initial_error, initial_maximum, initial_median = log_prediction_error(
            initial_candidate["hard"], initial_predicted_hard, fit_mask
        )

        validation_points = np.asarray(HELD_OUT_POINTS, dtype=np.float64)
        validation_hard = []
        validation_soft = []
        validation_seconds = []
        if not arguments.skip_held_out:
            for index, point in enumerate(validation_points):
                result = primal_population(
                    partition,
                    timing,
                    parameters_from_log_ratios(fiducial, point),
                    arguments,
                    tree_indices,
                    edges,
                    f"held-out-{index + 1}-of-{len(validation_points)}",
                )
                validation_hard.append(result["hard"])
                validation_soft.append(result["soft"])
                validation_seconds.append(result["seconds"])
        else:
            validation_points = np.empty((0, len(PARAMETER_NAMES)))

        optimum_result = primal_population(
            partition,
            timing,
            parameters_from_log_ratios(fiducial, optimum),
            arguments,
            tree_indices,
            edges,
            "emulator-optimum",
        )
        exact_seconds = (
            training_seconds
            + validation_seconds
            + [
                initial_candidate["seconds"],
                optimum_result["seconds"],
            ]
        )
        validation_hard = np.asarray(validation_hard, dtype=np.float64)
        validation_soft = np.asarray(validation_soft, dtype=np.float64)
        validation_predictions = np.asarray(
            [
                baseline_hard[fit_mask]
                * np.exp(
                    np.asarray(
                        quadratic_log_change(point, baseline_response[fit_mask], coefficients)
                    )
                )
                for point in validation_points
            ]
        )
        linear_validation_predictions = np.asarray(
            [
                baseline_hard[fit_mask] * np.exp(baseline_response[fit_mask] @ point)
                for point in validation_points
            ]
        )
        if validation_points.size:
            validation_errors_dex = np.log10(validation_hard[:, fit_mask] / validation_predictions)
            linear_validation_errors_dex = np.log10(
                validation_hard[:, fit_mask] / linear_validation_predictions
            )
        else:
            validation_errors_dex = np.empty((0, np.count_nonzero(fit_mask)))
            linear_validation_errors_dex = np.empty_like(validation_errors_dex)
        optimum_prediction = baseline_hard[fit_mask] * np.exp(
            np.asarray(quadratic_log_change(optimum, baseline_response[fit_mask], coefficients))
        )
        optimum_error_dex = np.log10(optimum_result["hard"][fit_mask] / optimum_prediction)
        all_validation_errors = np.concatenate(
            (validation_errors_dex.reshape(-1), optimum_error_dex.reshape(-1))
        )
        validation_maximum = float(np.max(np.abs(all_validation_errors)))
        validation_median = float(np.median(np.abs(all_validation_errors)))
        emulator_passed = validation_maximum <= SURROGATE_MAXIMUM_ERROR_DEX

        exact_residual = np.log(optimum_result["hard"][fit_mask]) - np.asarray(
            fit_observation.log_values
        )
        exact_chi_square = float(
            exact_residual @ np.asarray(fit_observation.precision) @ exact_residual
        )
        chain_covariance = np.cov(chain.samples, rowvar=False)
        covariance_relative_error = float(
            np.linalg.norm(chain_covariance - covariance) / np.linalg.norm(covariance)
        )
        boundary_distance_sigma = np.minimum(
            (optimum - bounds[:, 0]) / one_sigma,
            (bounds[:, 1] - optimum) / one_sigma,
        )
        final_parameter_values = selected_parameter_values(fiducial) * np.exp(optimum)
        arrays.update(
            {
                "training_points": training_points,
                "training_hard_smf": training_hard,
                "training_soft_smf": training_soft,
                "training_log_changes": training_log_changes,
                "training_predicted_log_changes": training_predicted_log_changes,
                "training_errors_dex": training_errors_dex,
                "quadratic_coefficients": coefficients,
                "optimum_log_parameter_ratios": optimum,
                "optimum_parameter_values": final_parameter_values,
                "optimum_hessian": hessian,
                "optimum_covariance": covariance,
                "optimum_exact_hard_smf": optimum_result["hard"],
                "optimum_exact_soft_smf": optimum_result["soft"],
                "optimum_predicted_hard_smf_fit_bins": optimum_prediction,
                "optimum_error_dex": optimum_error_dex,
                "initial_candidate_hard_smf": initial_candidate["hard"],
                "initial_predicted_hard_smf": initial_predicted_hard,
                "initial_error_dex": initial_error,
                "held_out_points": validation_points,
                "held_out_hard_smf": validation_hard,
                "held_out_soft_smf": validation_soft,
                "held_out_quadratic_predictions_fit_bins": validation_predictions,
                "held_out_linear_predictions_fit_bins": linear_validation_predictions,
                "held_out_quadratic_errors_dex": validation_errors_dex,
                "held_out_linear_errors_dex": linear_validation_errors_dex,
                "posterior_chain": chain.samples[::5],
                "posterior_chain_log_probability": chain.log_probabilities[::5],
                "posterior_quantiles_log_parameter_ratios": chain_quantiles,
            }
        )
        summary.update(
            {
                "nonlinear_evaluated": True,
                "tree_file": str(arguments.trees),
                "tree_count": len(tree_indices),
                "records": int(optimum_result["records"]),
                "initial_surrogate_validation": {
                    "passed": initial_maximum <= SURROGATE_MAXIMUM_ERROR_DEX,
                    "maximum_absolute_error_dex": initial_maximum,
                    "median_absolute_error_dex": initial_median,
                },
                "emulator": {
                    "kind": "fiducial-value/JAX-gradient-constrained quadratic log response",
                    "exact_training_runs": int(
                        np.count_nonzero(np.any(training_points != 0, axis=1))
                    ),
                    "training_maximum_absolute_error_dex": float(
                        np.max(np.abs(training_errors_dex))
                    ),
                    "held_out_runs": int(validation_points.shape[0]) + 1,
                    "held_out_maximum_absolute_error_dex": validation_maximum,
                    "held_out_median_absolute_error_dex": validation_median,
                    "linear_held_out_maximum_absolute_error_dex": (
                        float(np.max(np.abs(linear_validation_errors_dex)))
                        if linear_validation_errors_dex.size
                        else None
                    ),
                    "passed": emulator_passed,
                    "scientific_use": (
                        "accepted for the reported local intervals"
                        if emulator_passed
                        else "rejected for parameter intervals"
                    ),
                },
                "fit": {
                    "optimizer": "L-BFGS-B with exact JAX objective gradient on the emulator",
                    "success": bool(optimization.success),
                    "iterations": int(optimization.nit),
                    "function_evaluations": int(optimization.nfev),
                    "log_parameter_ratios": optimum.tolist(),
                    "parameter_ratios": np.exp(optimum).tolist(),
                    "parameter_values": final_parameter_values.tolist(),
                    "laplace_one_sigma_log": one_sigma.tolist(),
                    "mcmc_16_50_84_log": chain_quantiles.tolist(),
                    "emulator_chi_square": float(2.0 * optimization.fun),
                    "exact_sage_chi_square": exact_chi_square,
                    "fiducial_chi_square": initial_fit.chi_square_fiducial,
                    "degrees_of_freedom": int(np.count_nonzero(fit_mask) - len(PARAMETER_NAMES)),
                    "hessian_condition_number": condition_number,
                    "distance_to_design_boundary_sigma": boundary_distance_sigma.tolist(),
                    "interval_status": "available" if emulator_passed else "unavailable",
                    "uncertainty_interpretation": (
                        (
                            "working-likelihood intervals conditional on the stated diagonal "
                            "errors, uniform design-box prior, one Mini-Millennium partition, "
                            "and accepted quadratic-emulator validation"
                        )
                        if emulator_passed
                        else (
                            "diagnostic only; final parameter intervals are unavailable because "
                            "the quadratic emulator failed its held-out validation gate"
                        )
                    ),
                },
                "mcmc_reference": {
                    "method": "random-walk Metropolis on the same bounded emulator likelihood",
                    "steps": arguments.mcmc_steps,
                    "burn_in": arguments.mcmc_burn_in,
                    "acceptance_fraction": chain.acceptance_fraction,
                    "laplace_covariance_relative_error": covariance_relative_error,
                },
                "runtime": {
                    "exact_sage_seconds": float(np.sum(exact_seconds)),
                    "exact_sage_runs": int(len(exact_seconds) - 1),
                    "emulator_optimizer_seconds": None,
                    "baseline_linearized_seconds": 433.333007833,
                    "baseline_primal_seconds": 147.038510417,
                },
            }
        )

    summary["backend"] = jax.default_backend()
    summary["jax_version"] = jax.__version__
    summary["peak_resident_bytes"] = maximum_resident_bytes()
    summary["total_seconds"] = time.perf_counter() - started
    arguments.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output_arrays, **arrays)
    summary["arrays"] = arguments.output_arrays.name
    arguments.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(arguments.output_json)
    print(arguments.output_arrays)


if __name__ == "__main__":
    main()
