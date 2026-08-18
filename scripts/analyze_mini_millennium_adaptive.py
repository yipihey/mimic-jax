#!/usr/bin/env python3
"""Test adaptive integration on continuous SAGE16 intervals from Mini-Millennium."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from analyze_mini_millennium_science_program import json_ready  # noqa: E402

from mimic_jax import ADAPTIVE_SUCCESS, RK4  # noqa: E402
from mimic_jax.io import open_lhalo_partition  # noqa: E402
from mimic_jax.sage16 import (  # noqa: E402
    ODE_STATE_NAMES,
    evolve_lhalo_partition,
    fiducial_parameters,
    integrate_sage16_ode,
    integrate_sage16_ode_adaptive,
    load_cooling_tables,
    load_scale_factors,
    ode_state_from_galaxy,
    sage16_units,
    snapshot_timing,
)
from mimic_jax.sage16.ode import calculate_continuous_cooling_rate  # noqa: E402


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trees",
        type=Path,
        default=Path("simulations/mini-millennium/snapshots/trees_063.1"),
    )
    parser.add_argument(
        "--scale-factors",
        type=Path,
        default=Path("simulations/mini-millennium/mini-millennium.a_list"),
    )
    parser.add_argument("--global-tree-offset", type=int, default=3432)
    parser.add_argument("--tree-count", type=int, default=64)
    parser.add_argument("--case-count", type=int, default=64)
    parser.add_argument("--snapshot", type=int, default=63)
    parser.add_argument("--reference-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-members", type=int, default=512)
    parser.add_argument(
        "--compilation-cache-dir",
        type=Path,
        default=Path("archive/jax-cache"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-arrays", type=Path, required=True)
    return parser.parse_args()


def maximum_resident_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def stack_pytrees(values):
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *values)


def select_cases(result, snapshot, maximum):
    records = [
        record
        for by_snapshot in result.records_by_tree
        for record in by_snapshot.get(snapshot, ())
        if int(record.halo.Type) == 0
        and float(record.state.StellarMass) > 0.0
        and float(record.state.DiskScaleRadius) > 0.0
        and float(record.halo.dT) > 0.0
        and float(record.halo.Rvir) > 0.0
        and float(record.halo.Vvir) > 0.0
    ]
    records.sort(key=lambda record: float(record.state.StellarMass))
    if len(records) > maximum:
        indices = np.linspace(0, len(records) - 1, maximum, dtype=np.int32)
        records = [records[int(index)] for index in indices]
    if not records:
        raise RuntimeError("No valid central galaxies were available for adaptive analysis")
    return records


def relative_reservoir_errors(candidate, reference):
    denominator = np.maximum(np.abs(reference[..., :4]), 1.0e-10)
    return np.abs(candidate[..., :4] - reference[..., :4]) / denominator


def state_matrix(state):
    return np.stack([np.asarray(getattr(state, name)) for name in ODE_STATE_NAMES], axis=-1)


def main():
    arguments = parse_arguments()
    arguments.compilation_cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update(
        "jax_compilation_cache_dir",
        str(arguments.compilation_cache_dir.resolve()),
    )
    partition = open_lhalo_partition(arguments.trees)
    if not 0 < arguments.tree_count <= partition.tree_count:
        raise SystemExit("tree_count must be within the input partition")
    tree_indices = tuple(
        int(value)
        for value in np.linspace(
            0,
            partition.tree_count - 1,
            num=arguments.tree_count,
            dtype=np.int32,
        )
    )
    timing = snapshot_timing(load_scale_factors(arguments.scale_factors))
    print("[adaptive] obtaining representative z=0 SAGE16 states", flush=True)
    sample_started = time.perf_counter()
    evolved = evolve_lhalo_partition(
        partition,
        timing,
        tree_indices=tree_indices,
        global_tree_offset=arguments.global_tree_offset,
        num_substeps=10,
        output_snapshots=(arguments.snapshot,),
        batch_size=arguments.batch_size,
        max_batch_members=arguments.max_batch_members,
        member_binning="power_of_two",
    )
    sample_seconds = time.perf_counter() - sample_started
    if not evolved.success:
        raise SystemExit("representative-state evolution reported failure")
    records = select_cases(evolved, arguments.snapshot, arguments.case_count)
    candidate_case_count = len(records)
    states = stack_pytrees([ode_state_from_galaxy(record.state) for record in records])
    halos = stack_pytrees([record.halo for record in records])
    disk_scale_radius = jnp.asarray(
        [record.state.DiskScaleRadius for record in records], dtype=jnp.float64
    )
    galaxy_ids = np.asarray([record.halo.UniqueGalaxyID for record in records], dtype=np.int64)
    parameters = fiducial_parameters()
    units = sage16_units()
    tables = load_cooling_tables()

    def fixed_one(state, halo, radius, num_steps):
        return integrate_sage16_ode(
            state,
            halo,
            radius,
            parameters,
            units,
            tables,
            num_steps=num_steps,
            method=RK4,
        ).final_state

    def reference_one(state, halo, radius):
        return integrate_sage16_ode(
            state,
            halo,
            radius,
            parameters,
            units,
            tables,
            num_steps=arguments.reference_steps,
            method=RK4,
        ).states

    print(f"[adaptive] RK4 reference steps={arguments.reference_steps}", flush=True)
    reference_started = time.perf_counter()
    reference_history = jax.jit(jax.vmap(reference_one))(states, halos, disk_scale_radius)
    jax.block_until_ready(reference_history.StellarMass)
    reference_history_values = state_matrix(reference_history)
    reference_values = reference_history_values[:, -1]
    cooling_radius = jax.jit(
        jax.vmap(
            lambda history, halo: jax.vmap(
                lambda state: calculate_continuous_cooling_rate(state, halo, units, tables)[1]
            )(history)
        )
    )(reference_history, halos)
    jax.block_until_ready(cooling_radius)
    reference_seconds = time.perf_counter() - reference_started
    cold_critical = (
        0.19
        * np.asarray(halos.Vvir)
        * float(parameters.StarFormingDiskFactor)
        * np.asarray(disk_scale_radius)
    )
    star_formation_margin = reference_history_values[..., ODE_STATE_NAMES.index("ColdGas")]
    star_formation_margin = star_formation_margin - cold_critical[:, None]
    one_star_formation_branch = np.all(star_formation_margin > 0.0, axis=1) | np.all(
        star_formation_margin <= 0.0, axis=1
    )
    cooling_margin = np.asarray(cooling_radius) - np.asarray(halos.Rvir)[:, None]
    one_cooling_branch = np.all(cooling_margin > 0.0, axis=1) | np.all(
        cooling_margin <= 0.0, axis=1
    )
    smooth_segment = (
        np.all(reference_history_values[..., :4] > 1.0e-8, axis=(1, 2))
        & np.all(reference_history_values[..., 4:] >= 0.0, axis=(1, 2))
        & np.all(np.isfinite(reference_history_values), axis=(1, 2))
        & one_star_formation_branch
        & one_cooling_branch
    )
    if np.count_nonzero(smooth_segment) < 4:
        raise RuntimeError("Too few sampled intervals remain on a smooth positive RHS branch")
    states = jax.tree_util.tree_map(lambda value: value[smooth_segment], states)
    halos = jax.tree_util.tree_map(lambda value: value[smooth_segment], halos)
    disk_scale_radius = disk_scale_radius[smooth_segment]
    galaxy_ids = galaxy_ids[smooth_segment]
    reference_values = reference_values[smooth_segment]
    case_count = int(np.count_nonzero(smooth_segment))

    fixed_step_counts = (1, 2, 4, 8, 16, 32, 64)
    fixed_values = []
    fixed_seconds = []
    for count in fixed_step_counts:
        print(f"[adaptive] fixed RK4 steps={count}", flush=True)
        started = time.perf_counter()
        runner = jax.jit(
            jax.vmap(lambda state, halo, radius: fixed_one(state, halo, radius, count))
        )
        result = runner(states, halos, disk_scale_radius)
        jax.block_until_ready(result.StellarMass)
        fixed_seconds.append(time.perf_counter() - started)
        fixed_values.append(state_matrix(result))
    fixed_values = np.asarray(fixed_values)
    fixed_errors = relative_reservoir_errors(fixed_values, reference_values)

    tolerances = (1.0e-3, 1.0e-5, 1.0e-7, 1.0e-9)
    adaptive_values = []
    adaptive_status = []
    accepted_steps = []
    rejected_steps = []
    rhs_evaluations = []
    maximum_stability_products = []
    adaptive_seconds = []

    for tolerance in tolerances:
        print(f"[adaptive] Dormand-Prince tolerance={tolerance:.0e}", flush=True)

        def adaptive_one(state, halo, radius):
            return integrate_sage16_ode_adaptive(
                state,
                halo,
                radius,
                parameters,
                units,
                tables,
                relative_tolerance=tolerance,
                absolute_tolerance=tolerance * 1.0e-3,
                initial_step=halo.dT,
                jacobian_stability_factor=1.0,
                max_steps=256,
                max_attempts=1024,
            )

        started = time.perf_counter()
        solution = jax.jit(jax.vmap(adaptive_one))(states, halos, disk_scale_radius)
        jax.block_until_ready(solution.final_state.StellarMass)
        adaptive_seconds.append(time.perf_counter() - started)
        adaptive_values.append(state_matrix(solution.final_state))
        adaptive_status.append(np.asarray(solution.status))
        accepted_steps.append(np.asarray(solution.accepted_steps))
        rejected_steps.append(np.asarray(solution.rejected_steps))
        rhs_evaluations.append(np.asarray(solution.rhs_evaluations))
        step_product = np.asarray(solution.accepted_step_sizes * solution.accepted_jacobian_norms)
        maximum_stability_products.append(np.max(np.nan_to_num(step_product, nan=0.0), axis=1))

    adaptive_values = np.asarray(adaptive_values)
    adaptive_status = np.asarray(adaptive_status)
    accepted_steps = np.asarray(accepted_steps)
    rejected_steps = np.asarray(rejected_steps)
    rhs_evaluations = np.asarray(rhs_evaluations)
    maximum_stability_products = np.asarray(maximum_stability_products)
    adaptive_errors = relative_reservoir_errors(adaptive_values, reference_values)
    adaptive_baryon_residual = np.sum(adaptive_values[..., :4], axis=-1) - np.sum(
        state_matrix(states)[..., :4], axis=-1
    )
    stellar_log_shift = np.log10(
        adaptive_values[..., ODE_STATE_NAMES.index("StellarMass")]
        / reference_values[..., ODE_STATE_NAMES.index("StellarMass")]
    )

    cold_critical = (
        0.19
        * np.asarray(halos.Vvir)
        * float(parameters.StarFormingDiskFactor)
        * np.asarray(disk_scale_radius)
    )
    star_formation_margin = np.asarray(states.ColdGas) - cold_critical
    if not np.any(star_formation_margin > 0.0):
        raise RuntimeError("No sampled continuous interval has active quiescent star formation")
    representative = int(np.argmax(star_formation_margin))
    representative_state = jax.tree_util.tree_map(lambda value: value[representative], states)
    representative_halo = jax.tree_util.tree_map(lambda value: value[representative], halos)
    representative_radius = disk_scale_radius[representative]
    derivative_tolerance = 1.0e-8

    def final_stellar_mass(efficiency):
        varied = parameters._replace(SfrEfficiency=efficiency)
        return integrate_sage16_ode_adaptive(
            representative_state,
            representative_halo,
            representative_radius,
            varied,
            units,
            tables,
            relative_tolerance=derivative_tolerance,
            absolute_tolerance=derivative_tolerance * 1.0e-3,
            initial_step=representative_halo.dT,
            jacobian_stability_factor=1.0,
            max_steps=256,
            max_attempts=1024,
        ).final_state.StellarMass

    automatic_derivative = jax.grad(final_stellar_mass)(parameters.SfrEfficiency)
    finite_difference_steps = np.asarray((1.0e-2, 3.0e-3, 1.0e-3), dtype=np.float64)
    finite_difference_derivatives = []
    for epsilon in finite_difference_steps:
        finite_difference_derivatives.append(
            (
                final_stellar_mass(parameters.SfrEfficiency * (1.0 + epsilon))
                - final_stellar_mass(parameters.SfrEfficiency * (1.0 - epsilon))
            )
            / (2.0 * epsilon * parameters.SfrEfficiency)
        )
    finite_difference_derivatives = np.asarray(finite_difference_derivatives)
    derivative_relative_errors = np.abs(
        finite_difference_derivatives - float(automatic_derivative)
    ) / abs(float(automatic_derivative))

    arrays = {
        "galaxy_ids": galaxy_ids,
        "state_names": np.asarray(ODE_STATE_NAMES),
        "initial_values": state_matrix(states),
        "halo_virial_mass": np.asarray(halos.Mvir),
        "halo_virial_velocity": np.asarray(halos.Vvir),
        "disk_scale_radius": np.asarray(disk_scale_radius),
        "interval_duration": np.asarray(halos.dT),
        "reference_values": reference_values,
        "fixed_step_counts": np.asarray(fixed_step_counts, dtype=np.int32),
        "fixed_values": fixed_values,
        "fixed_relative_reservoir_errors": fixed_errors,
        "fixed_rhs_evaluations": 4 * np.asarray(fixed_step_counts, dtype=np.int32),
        "fixed_elapsed_seconds": np.asarray(fixed_seconds),
        "adaptive_tolerances": np.asarray(tolerances),
        "adaptive_values": adaptive_values,
        "adaptive_relative_reservoir_errors": adaptive_errors,
        "adaptive_status": adaptive_status,
        "adaptive_accepted_steps": accepted_steps,
        "adaptive_rejected_steps": rejected_steps,
        "adaptive_rhs_evaluations": rhs_evaluations,
        "adaptive_maximum_stability_product": maximum_stability_products,
        "adaptive_baryon_residual": adaptive_baryon_residual,
        "adaptive_stellar_log_shift": stellar_log_shift,
        "adaptive_elapsed_seconds": np.asarray(adaptive_seconds),
        "derivative_parameter_value": np.asarray(parameters.SfrEfficiency),
        "derivative_galaxy_id": galaxy_ids[representative],
        "derivative_automatic": np.asarray(automatic_derivative),
        "derivative_finite_difference_steps": finite_difference_steps,
        "derivative_finite_difference": finite_difference_derivatives,
        "derivative_finite_difference_relative_error": derivative_relative_errors,
    }
    arguments.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output_arrays, **arrays)

    maximum_errors = np.max(adaptive_errors, axis=(1, 2))
    median_errors = np.median(np.max(adaptive_errors, axis=2), axis=1)
    payload = {
        "schema_version": "mimic-jax-mini-millennium-adaptive-continuous/v1",
        "scope": "continuous quiescent SAGE16 RHS under fixed halo forcing; no finite maps or events inside adaptive steps",
        "tree_count": arguments.tree_count,
        "input_halos": int(np.sum(partition.tree_halo_counts[np.asarray(tree_indices)])),
        "candidate_case_count": candidate_case_count,
        "case_count": case_count,
        "excluded_boundary_or_threshold_cases": candidate_case_count - case_count,
        "selection_contract": "all four mass reservoirs remain >1e-8 and the reference trajectory crosses neither the quiescent-star-formation nor cooling-regime threshold",
        "snapshot": arguments.snapshot,
        "reference_method": RK4,
        "reference_steps": arguments.reference_steps,
        "jacobian_control": "tolerance-scaled infinity norm of D^-1 (d f/d x) D",
        "jacobian_stability_factor": 1.0,
        "adaptive_tolerances": tolerances,
        "adaptive_success_counts": [
            int(np.count_nonzero(values == ADAPTIVE_SUCCESS)) for values in adaptive_status
        ],
        "adaptive_maximum_relative_reservoir_error": maximum_errors,
        "adaptive_median_maximum_relative_reservoir_error": median_errors,
        "adaptive_maximum_absolute_baryon_residual": np.max(
            np.abs(adaptive_baryon_residual), axis=1
        ),
        "adaptive_maximum_absolute_stellar_log_shift_dex": np.max(
            np.abs(stellar_log_shift), axis=1
        ),
        "adaptive_median_accepted_steps": np.median(accepted_steps, axis=1),
        "adaptive_maximum_accepted_steps": np.max(accepted_steps, axis=1),
        "adaptive_median_rhs_evaluations": np.median(rhs_evaluations, axis=1),
        "adaptive_maximum_stability_product": np.max(maximum_stability_products, axis=1),
        "derivative_validation": {
            "parameter": "SfrEfficiency",
            "galaxy_id": int(galaxy_ids[representative]),
            "automatic": float(automatic_derivative),
            "finite_difference_steps": finite_difference_steps,
            "finite_difference": finite_difference_derivatives,
            "maximum_relative_error": float(np.max(derivative_relative_errors)),
        },
        "elapsed_seconds": {
            "state_sample": sample_seconds,
            "reference_compile_and_run": reference_seconds,
            "fixed_compile_and_run": fixed_seconds,
            "adaptive_compile_and_run": adaptive_seconds,
        },
        "peak_resident_bytes": maximum_resident_bytes(),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "arrays": arguments.output_arrays.name,
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
