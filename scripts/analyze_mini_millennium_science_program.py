#!/usr/bin/env python3
"""Run the first differentiable SAGE16 Mini-Millennium science program."""

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

from mimic_jax.io import open_lhalo_partition  # noqa: E402
from mimic_jax.sage16 import (  # noqa: E402
    evolve_lhalo_partition,
    fiducial_parameters,
    group_baryon_inventory,
    linearize_lhalo_partition,
    load_scale_factors,
    process_perturbations,
    snapshot_timing,
    soft_stellar_mass_function,
    state_field_array,
    state_tangent_matrix,
    stellar_mass_function,
)

HUBBLE_H = 0.73
BOX_SIZE_MPC_OVER_H = 62.5
TOTAL_TREE_FILES = 8
GLOBAL_BARYON_FRACTION = 0.17
DEFAULT_PARAMETERS = (
    "SfrEfficiency",
    "FeedbackReheatingEpsilon",
    "FeedbackEjectionEfficiency",
    "ReIncorporationFactor",
    "RadioModeEfficiency",
    "BlackHoleGrowthRate",
    "QuasarModeEfficiency",
)
DEFAULT_PROCESSES = (
    "cooling",
    "sn_reheating",
    "sn_ejection",
    "reincorporation",
    "agn_heating",
)
RESERVOIRS = (
    "ColdGas",
    "HotGas",
    "EjectedGas",
    "StellarMass",
    "ICS",
    "BlackHoleMass",
)


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
    parser.add_argument("--tree-start", type=int, default=0)
    parser.add_argument("--tree-count", type=int)
    parser.add_argument("--snapshot", type=int, default=63)
    parser.add_argument("--num-substeps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-members", type=int, default=512)
    parser.add_argument("--bandwidth-dex", type=float, default=0.05)
    parser.add_argument("--history-epochs", type=int, default=8)
    parser.add_argument("--history-sample-per-bin", type=int, default=12)
    parser.add_argument("--convergence-tree-count", type=int, default=500)
    parser.add_argument(
        "--compilation-cache-dir",
        type=Path,
        default=Path("archive/jax-cache"),
    )
    parser.add_argument(
        "--skip-finite-difference",
        action="store_true",
        help="omit the expensive full-tree derivative validation reruns",
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="omit selected-tree historical process responses",
    )
    parser.add_argument(
        "--skip-convergence",
        action="store_true",
        help="omit population timestep refinement",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-arrays", type=Path, required=True)
    return parser.parse_args()


def maximum_resident_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def progress(label):
    state = {"batches": 0, "seconds": 0.0}

    def update(event):
        if event["event"] == "snapshot":
            print(
                f"[{label}] snapshot={event['snapshot']:02d} groups={event['groups']}",
                flush=True,
            )
        elif event["event"] == "batch":
            state["batches"] += 1
            state["seconds"] += float(event["elapsed_seconds"])

    update.state = state
    return update


def records_for_snapshot(result, snapshot):
    return tuple(result.records_by_snapshot.get(snapshot, ()))


def record_identifiers(records):
    return np.asarray([int(record.halo.UniqueGalaxyID) for record in records], dtype=np.int64)


def aligned_masses(result, snapshot, identifiers):
    records = records_for_snapshot(result, snapshot)
    by_id = {int(record.halo.UniqueGalaxyID): float(record.state.StellarMass) for record in records}
    if set(by_id) != set(int(value) for value in identifiers):
        missing = len(set(int(value) for value in identifiers) - set(by_id))
        extra = len(set(by_id) - set(int(value) for value in identifiers))
        raise ValueError(f"perturbed z=0 identities changed: missing={missing}, extra={extra}")
    return np.asarray([by_id[int(identifier)] for identifier in identifiers], dtype=np.float64)


def soft_mass_function_and_tangents(
    masses,
    mass_tangents,
    *,
    bin_edges,
    bandwidth_dex,
    volume,
):
    masses = jnp.asarray(masses, dtype=jnp.float64)
    mass_tangents = jnp.asarray(mass_tangents, dtype=jnp.float64)

    def observable(values):
        return soft_stellar_mass_function(
            values,
            volume_mpc_over_h_cubed=volume,
            hubble_h=HUBBLE_H,
            bin_edges=bin_edges,
            bandwidth_dex=bandwidth_dex,
        )

    density = observable(masses)
    tangents = jax.vmap(lambda direction: jax.jvp(observable, (masses,), (direction,))[1])(
        mass_tangents.T
    )
    return np.asarray(density), np.asarray(tangents).T


def parameter_summary_matrix(records, parameter_names, parameter_values, soft_density, soft_raw):
    rows = []
    names = []
    valid_bins = soft_density > 0.0
    mass_centres = np.arange(8.05, 12.0, 0.1)
    for label, lower, upper in (
        ("low-mass abundance (8.5–9.5)", 8.5, 9.5),
        ("knee abundance (9.5–10.5)", 9.5, 10.5),
        ("massive abundance (10.5–11.5)", 10.5, 11.5),
    ):
        selected = valid_bins & (mass_centres >= lower) & (mass_centres < upper)
        value = np.sum(soft_density[selected])
        derivative = np.sum(soft_raw[selected], axis=0)
        rows.append(
            derivative * parameter_values / value
            if value > 0.0
            else np.full(len(parameter_names), np.nan)
        )
        names.append(label)

    for field, label in (
        ("StellarMass", "stellar-mass density"),
        ("StarFormationRate", "star-formation-rate density"),
        ("ColdGas", "cold-gas density"),
        ("EjectedGas", "ejected-gas density"),
        ("BlackHoleMass", "black-hole-mass density"),
    ):
        values = state_field_array(records, field).astype(np.float64)
        tangents = state_tangent_matrix(records, field).astype(np.float64)
        total = np.sum(values)
        if total <= 0.0:
            rows.append(np.full(len(parameter_names), np.nan))
        else:
            rows.append(np.sum(tangents, axis=0) * parameter_values / total)
        names.append(label)
    return tuple(names), np.asarray(rows, dtype=np.float64)


def response_similarity(values):
    parameter_count = values.shape[1]
    result = np.full((parameter_count, parameter_count), np.nan, dtype=np.float64)
    for left in range(parameter_count):
        for right in range(parameter_count):
            valid = np.isfinite(values[:, left]) & np.isfinite(values[:, right])
            if not np.any(valid):
                continue
            left_values = values[valid, left]
            right_values = values[valid, right]
            denominator = np.linalg.norm(left_values) * np.linalg.norm(right_values)
            if denominator > 0.0:
                result[left, right] = np.dot(left_values, right_values) / denominator
    return result


def catalogue_inventory(records):
    return group_baryon_inventory(
        unique_galaxy_id=np.asarray([record.halo.UniqueGalaxyID for record in records]),
        unique_central_galaxy_id=np.asarray(
            [record.halo.UniqueCentralGalaxyID for record in records]
        ),
        galaxy_type=np.asarray([record.halo.Type for record in records]),
        central_halo_mass=np.asarray([record.halo.Mvir for record in records]),
        reservoirs={
            name: np.asarray([getattr(record.state, name) for record in records])
            for name in RESERVOIRS
        },
        hubble_h=HUBBLE_H,
        global_baryon_fraction=GLOBAL_BARYON_FRACTION,
        halo_mass_bin_edges=np.arange(9.5, 14.1, 0.25),
    )


def parameter_finite_differences(
    partition,
    timing,
    tree_indices,
    baseline_identifiers,
    baseline_response,
    parameter_names,
    parameter_values,
    arguments,
    mass_edges,
    volume,
):
    cases = (
        ("SfrEfficiency", 1.0e-2),
        ("SfrEfficiency", 3.0e-3),
        ("FeedbackReheatingEpsilon", 1.0e-2),
        ("RadioModeEfficiency", 1.0e-2),
    )
    responses = []
    errors = []
    base_parameters = fiducial_parameters()
    for name, relative_step in cases:
        print(f"[finite difference] {name} step={relative_step:g}", flush=True)
        value = float(getattr(base_parameters, name))
        perturbed_masses = []
        for sign in (-1.0, 1.0):
            parameters = base_parameters._replace(
                **{name: jnp.asarray(value * (1.0 + sign * relative_step))}
            )
            result = evolve_lhalo_partition(
                partition,
                timing,
                tree_indices=tree_indices,
                global_tree_offset=arguments.global_tree_offset,
                num_substeps=arguments.num_substeps,
                output_snapshots=(arguments.snapshot,),
                batch_size=arguments.batch_size,
                max_batch_members=arguments.max_batch_members,
                member_binning="power_of_two",
                parameters=parameters,
            )
            perturbed_masses.append(
                aligned_masses(result, arguments.snapshot, baseline_identifiers)
            )
        lower = np.asarray(
            soft_stellar_mass_function(
                jnp.asarray(perturbed_masses[0]),
                volume_mpc_over_h_cubed=volume,
                hubble_h=HUBBLE_H,
                bin_edges=mass_edges,
                bandwidth_dex=arguments.bandwidth_dex,
            )
        )
        upper = np.asarray(
            soft_stellar_mass_function(
                jnp.asarray(perturbed_masses[1]),
                volume_mpc_over_h_cubed=volume,
                hubble_h=HUBBLE_H,
                bin_edges=mass_edges,
                bandwidth_dex=arguments.bandwidth_dex,
            )
        )
        response = np.full(upper.shape, np.nan, dtype=np.float64)
        positive = (upper > 0.0) & (lower > 0.0)
        response[positive] = (np.log(upper[positive]) - np.log(lower[positive])) / (
            np.log1p(relative_step) - np.log1p(-relative_step)
        )
        automatic = baseline_response[:, parameter_names.index(name)]
        responses.append(response)
        errors.append(np.abs(response - automatic))
    return (
        tuple(name for name, _ in cases),
        np.asarray([step for _, step in cases]),
        np.asarray(responses),
        np.asarray(errors),
    )


def select_history_trees(linearized, snapshot, mass_bin_edges, per_bin):
    selected_trees = []
    selected_bins = []
    selected_ids = []
    counts = []
    for bin_index, (lower, upper) in enumerate(zip(mass_bin_edges[:-1], mass_bin_edges[1:])):
        candidates = []
        midpoint = 0.5 * (lower + upper)
        for tree_index, records_by_snapshot in zip(
            linearized.tree_indices, linearized.records_by_tree
        ):
            for record in records_by_snapshot.get(snapshot, ()):
                if int(record.halo.Type) != 0 or float(record.state.StellarMass) <= 0.0:
                    continue
                logarithmic_mass = np.log10(float(record.state.StellarMass) * 1.0e10 / HUBBLE_H)
                if lower <= logarithmic_mass < upper:
                    candidates.append(
                        (
                            abs(logarithmic_mass - midpoint),
                            int(record.halo.UniqueGalaxyID),
                            int(tree_index),
                        )
                    )
        candidates.sort()
        chosen = candidates[:per_bin]
        counts.append(len(chosen))
        for _, identifier, tree_index in chosen:
            selected_trees.append(tree_index)
            selected_bins.append(bin_index)
            selected_ids.append(identifier)
    return (
        tuple(sorted(set(selected_trees))),
        np.asarray(selected_trees, dtype=np.int32),
        np.asarray(selected_bins, dtype=np.int32),
        np.asarray(selected_ids, dtype=np.int64),
        np.asarray(counts, dtype=np.int32),
    )


def aggregate_selected_stellar_mass(
    result,
    snapshot,
    target_tree_indices,
    selected_ids,
    selected_bins,
    bin_count,
):
    sums = np.zeros(bin_count, dtype=np.float64)
    counts = np.zeros(bin_count, dtype=np.int32)
    record_lookup = dict(zip(result.tree_indices, result.records_by_tree))
    for tree_index, identifier, bin_index in zip(target_tree_indices, selected_ids, selected_bins):
        central = [
            record
            for record in record_lookup[int(tree_index)].get(snapshot, ())
            if int(record.halo.Type) == 0 and int(record.halo.UniqueGalaxyID) == int(identifier)
        ]
        if len(central) != 1:
            raise ValueError("each selected target must identify exactly one z=0 central")
        sums[bin_index] += float(central[0].state.StellarMass)
        counts[bin_index] += 1
    return sums, counts


def historical_process_analysis(
    partition,
    timing,
    linearized,
    tree_indices,
    arguments,
):
    mass_bin_edges = np.asarray([8.0, 9.0, 9.75, 10.5, 11.25, 12.0])
    (
        selected_trees,
        target_tree_indices,
        selected_bins,
        selected_ids,
        sample_counts,
    ) = select_history_trees(
        linearized,
        arguments.snapshot,
        mass_bin_edges,
        arguments.history_sample_per_bin,
    )
    ln_edges = np.linspace(
        np.log(float(np.min(timing.scale_factor))),
        np.log(float(np.max(timing.scale_factor))),
        arguments.history_epochs + 1,
    )
    callback = progress("history AD")
    started = time.perf_counter()
    response_run = linearize_lhalo_partition(
        partition,
        timing,
        tree_indices=selected_trees,
        global_tree_offset=arguments.global_tree_offset,
        num_substeps=arguments.num_substeps,
        output_snapshots=(arguments.snapshot,),
        batch_size=arguments.batch_size,
        max_batch_members=arguments.max_batch_members,
        member_binning="power_of_two",
        process_names=DEFAULT_PROCESSES,
        ln_scale_factor_edges=ln_edges,
        progress_callback=callback,
    )
    elapsed = time.perf_counter() - started
    bin_count = mass_bin_edges.size - 1
    process_count = len(DEFAULT_PROCESSES)
    epoch_count = arguments.history_epochs
    mass_sum = np.zeros(bin_count, dtype=np.float64)
    tangent_sum = np.zeros((bin_count, process_count * epoch_count), dtype=np.float64)
    counts = np.zeros(bin_count, dtype=np.int32)
    response_lookup = dict(zip(response_run.tree_indices, response_run.records_by_tree))
    for tree_index, identifier, bin_index in zip(target_tree_indices, selected_ids, selected_bins):
        central = [
            record
            for record in response_lookup[int(tree_index)].get(arguments.snapshot, ())
            if int(record.halo.Type) == 0 and int(record.halo.UniqueGalaxyID) == int(identifier)
        ]
        if len(central) != 1:
            raise ValueError("each selected history target must have one final central")
        record = central[0]
        mass_sum[bin_index] += float(record.state.StellarMass)
        tangent_sum[bin_index] += np.asarray(record.state_tangent.StellarMass)
        counts[bin_index] += 1
    response = np.full_like(tangent_sum, np.nan)
    valid = mass_sum > 0.0
    response[valid] = tangent_sum[valid] / mass_sum[valid, None]
    response = response.reshape(bin_count, process_count, epoch_count)

    baseline_histories = evolve_lhalo_partition(
        partition,
        timing,
        tree_indices=selected_trees,
        global_tree_offset=arguments.global_tree_offset,
        num_substeps=arguments.num_substeps,
        output_snapshots=None,
        batch_size=arguments.batch_size,
        max_batch_members=arguments.max_batch_members,
        member_binning="power_of_two",
    )
    history_inventory = np.full(
        (bin_count, len(timing.scale_factor), len(RESERVOIRS)), np.nan, dtype=np.float64
    )
    halo_growth = np.full((bin_count, len(timing.scale_factor)), np.nan, dtype=np.float64)
    stellar_growth = np.full_like(halo_growth, np.nan)
    inventory_numerator = np.zeros_like(history_inventory)
    inventory_denominator = np.zeros((bin_count, len(timing.scale_factor)))
    halo_growth_values = [[[] for _ in timing.scale_factor] for _ in range(bin_count)]
    stellar_growth_values = [[[] for _ in timing.scale_factor] for _ in range(bin_count)]
    baseline_lookup = dict(zip(baseline_histories.tree_indices, baseline_histories.records_by_tree))
    for tree_index, bin_index, identifier in zip(target_tree_indices, selected_bins, selected_ids):
        records_by_snapshot = baseline_lookup[int(tree_index)]
        main_records = {}
        for snapshot, records in records_by_snapshot.items():
            match = [record for record in records if int(record.halo.UniqueGalaxyID) == identifier]
            if len(match) == 1:
                main_records[snapshot] = match[0]
        for snapshot, record in main_records.items():
            halo_mass = float(record.halo.Mvir)
            if halo_mass > 0.0:
                inventory_denominator[bin_index, snapshot] += GLOBAL_BARYON_FRACTION * halo_mass
                for reservoir_index, reservoir in enumerate(RESERVOIRS):
                    inventory_numerator[bin_index, snapshot, reservoir_index] += float(
                        getattr(record.state, reservoir)
                    )
        ordered = sorted(main_records)
        for previous, current in zip(ordered[:-1], ordered[1:]):
            previous_record = main_records[previous]
            current_record = main_records[current]
            delta_time = float(timing.lookback_time[previous] - timing.lookback_time[current])
            previous_halo = float(previous_record.halo.Mvir)
            current_halo = float(current_record.halo.Mvir)
            previous_stars = float(previous_record.state.StellarMass)
            current_stars = float(current_record.state.StellarMass)
            if delta_time > 0.0 and previous_halo > 0.0 and current_halo > 0.0:
                halo_growth_values[bin_index][current].append(
                    (np.log(current_halo) - np.log(previous_halo)) / delta_time
                )
            if delta_time > 0.0 and previous_stars > 0.0 and current_stars > 0.0:
                stellar_growth_values[bin_index][current].append(
                    (np.log(current_stars) - np.log(previous_stars)) / delta_time
                )
    populated = inventory_denominator > 0.0
    history_inventory[populated] = (
        inventory_numerator[populated] / inventory_denominator[populated, None]
    )
    for bin_index in range(bin_count):
        for snapshot in range(len(timing.scale_factor)):
            if halo_growth_values[bin_index][snapshot]:
                halo_growth[bin_index, snapshot] = np.median(
                    halo_growth_values[bin_index][snapshot]
                )
            if stellar_growth_values[bin_index][snapshot]:
                stellar_growth[bin_index, snapshot] = np.median(
                    stellar_growth_values[bin_index][snapshot]
                )

    validation = {}
    if not arguments.skip_finite_difference:
        candidate_directions = []
        for process in ("cooling", "agn_heating"):
            process_index = DEFAULT_PROCESSES.index(process)
            collapsed = np.nanmax(np.abs(response[:, process_index]), axis=0)
            candidate_directions.append((process, int(np.nanargmax(collapsed))))
        for process, epoch in candidate_directions:
            print(f"[history finite difference] {process} epoch={epoch}", flush=True)
            perturbed_observables = []
            snapshot_epoch = np.asarray(
                [
                    int(
                        np.clip(
                            np.searchsorted(
                                ln_edges,
                                np.log(float(scale_factor)),
                                side="right",
                            )
                            - 1,
                            0,
                            epoch_count - 1,
                        )
                    )
                    for scale_factor in timing.scale_factor
                ]
            )
            for sign in (-1.0, 1.0):
                schedule = np.zeros(len(timing.scale_factor), dtype=np.float64)
                schedule[snapshot_epoch == epoch] = sign * 1.0e-2
                result = evolve_lhalo_partition(
                    partition,
                    timing,
                    tree_indices=selected_trees,
                    global_tree_offset=arguments.global_tree_offset,
                    num_substeps=arguments.num_substeps,
                    output_snapshots=(arguments.snapshot,),
                    batch_size=arguments.batch_size,
                    max_batch_members=arguments.max_batch_members,
                    member_binning="power_of_two",
                    perturbations=process_perturbations(**{process: schedule}),
                )
                values, value_counts = aggregate_selected_stellar_mass(
                    result,
                    arguments.snapshot,
                    target_tree_indices,
                    selected_ids,
                    selected_bins,
                    bin_count,
                )
                if not np.array_equal(value_counts, counts):
                    raise ValueError("historical finite difference changed sample membership")
                perturbed_observables.append(values)
            finite_difference = (
                np.log(perturbed_observables[1]) - np.log(perturbed_observables[0])
            ) / 2.0e-2
            automatic = response[:, DEFAULT_PROCESSES.index(process), epoch]
            validation[f"{process}:epoch_{epoch}"] = {
                "automatic": automatic,
                "finite_difference": finite_difference,
                "absolute_error": np.abs(automatic - finite_difference),
            }

    return {
        "mass_bin_edges": mass_bin_edges,
        "sample_counts": sample_counts,
        "selected_tree_indices": np.asarray(selected_trees, dtype=np.int32),
        "selected_target_tree_indices": target_tree_indices,
        "selected_galaxy_ids": selected_ids,
        "ln_scale_factor_edges": ln_edges,
        "redshift_edges": np.exp(-ln_edges) - 1.0,
        "response": response,
        "history_inventory": history_inventory,
        "halo_growth": halo_growth,
        "stellar_growth": stellar_growth,
        "validation": validation,
        "elapsed_seconds": elapsed,
        "batch_seconds": callback.state["seconds"],
    }


def convergence_analysis(
    partition,
    timing,
    tree_indices,
    baseline,
    arguments,
    mass_edges,
    volume,
):
    sample_indices = tuple(tree_indices[: arguments.convergence_tree_count])
    baseline_records = []
    baseline_lookup = dict(zip(baseline.tree_indices, baseline.records_by_tree))
    for tree_index in sample_indices:
        baseline_records.extend(baseline_lookup[tree_index].get(arguments.snapshot, ()))
    baseline_masses = state_field_array(baseline_records, "StellarMass")
    baseline_hard_counts = stellar_mass_function(
        baseline_masses,
        volume_mpc_over_h_cubed=volume,
        hubble_h=HUBBLE_H,
        bin_edges=mass_edges,
    ).counts
    baseline_soft = np.asarray(
        soft_stellar_mass_function(
            jnp.asarray(baseline_masses),
            volume_mpc_over_h_cubed=volume,
            hubble_h=HUBBLE_H,
            bin_edges=mass_edges,
            bandwidth_dex=arguments.bandwidth_dex,
        )
    )
    step_counts = (5, 10, 20, 40, 80)
    soft = []
    scalar_names = ("StellarMass", "ColdGas", "EjectedGas", "BlackHoleMass")
    scalars = []
    elapsed = []
    for substeps in step_counts:
        if substeps == arguments.num_substeps:
            records = baseline_records
            elapsed.append(0.0)
        else:
            print(f"[convergence] substeps={substeps}", flush=True)
            started = time.perf_counter()
            result = evolve_lhalo_partition(
                partition,
                timing,
                tree_indices=sample_indices,
                global_tree_offset=arguments.global_tree_offset,
                num_substeps=substeps,
                output_snapshots=(arguments.snapshot,),
                batch_size=arguments.batch_size,
                max_batch_members=arguments.max_batch_members,
                member_binning="power_of_two",
            )
            elapsed.append(time.perf_counter() - started)
            records = records_for_snapshot(result, arguments.snapshot)
        masses = state_field_array(records, "StellarMass")
        soft.append(
            np.asarray(
                soft_stellar_mass_function(
                    jnp.asarray(masses),
                    volume_mpc_over_h_cubed=volume,
                    hubble_h=HUBBLE_H,
                    bin_edges=mass_edges,
                    bandwidth_dex=arguments.bandwidth_dex,
                )
            )
        )
        scalars.append([float(np.sum(state_field_array(records, name))) for name in scalar_names])
    soft = np.asarray(soft)
    relative_to_fine = np.full_like(soft, np.nan)
    positive = soft[-1] > 0.0
    relative_to_fine[:, positive] = (soft[:, positive] - soft[-1, positive]) / soft[-1, positive]
    scalars = np.asarray(scalars)
    scalar_relative_to_fine = (scalars - scalars[-1]) / scalars[-1]
    successive_soft_difference = soft[1:] - soft[:-1]
    successive_scalar_difference = scalars[1:] - scalars[:-1]
    return {
        "tree_count": len(sample_indices),
        "step_counts": np.asarray(step_counts),
        "soft_smf": soft,
        "soft_smf_relative_to_fine": relative_to_fine,
        "scalar_names": np.asarray(scalar_names),
        "scalar_values": scalars,
        "scalar_relative_to_fine": scalar_relative_to_fine,
        "successive_soft_smf_difference": successive_soft_difference,
        "successive_scalar_difference": successive_scalar_difference,
        "elapsed_seconds": np.asarray(elapsed),
        "baseline_soft_smf": baseline_soft,
        "baseline_hard_counts": baseline_hard_counts,
    }


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def main():
    arguments = parse_arguments()
    if arguments.compilation_cache_dir is not None:
        arguments.compilation_cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update(
            "jax_compilation_cache_dir",
            str(arguments.compilation_cache_dir.resolve()),
        )
    partition = open_lhalo_partition(arguments.trees)
    tree_end = (
        partition.tree_count
        if arguments.tree_count is None
        else arguments.tree_start + arguments.tree_count
    )
    if not 0 <= arguments.tree_start < tree_end <= partition.tree_count:
        raise SystemExit("requested tree interval is outside the input partition")
    tree_indices = tuple(range(arguments.tree_start, tree_end))
    timing = snapshot_timing(load_scale_factors(arguments.scale_factors))
    parameters = fiducial_parameters()
    parameter_names = DEFAULT_PARAMETERS
    parameter_values = np.asarray(
        [float(getattr(parameters, name)) for name in parameter_names], dtype=np.float64
    )
    volume = BOX_SIZE_MPC_OVER_H**3 / TOTAL_TREE_FILES
    mass_edges = np.arange(8.0, 12.1, 0.1, dtype=np.float64)

    callback = progress("parameter AD")
    started = time.perf_counter()
    linearized = linearize_lhalo_partition(
        partition,
        timing,
        tree_indices=tree_indices,
        global_tree_offset=arguments.global_tree_offset,
        num_substeps=arguments.num_substeps,
        output_snapshots=(arguments.snapshot,),
        batch_size=arguments.batch_size,
        max_batch_members=arguments.max_batch_members,
        member_binning="power_of_two",
        parameter_names=parameter_names,
        progress_callback=callback,
    )
    parameter_elapsed = time.perf_counter() - started
    if not linearized.success:
        raise SystemExit("linearized partition evolution reported failure")
    records = records_for_snapshot(linearized, arguments.snapshot)
    identifiers = record_identifiers(records)
    masses = state_field_array(records, "StellarMass").astype(np.float64)
    mass_tangents = state_tangent_matrix(records, "StellarMass").astype(np.float64)
    hard = stellar_mass_function(
        masses,
        volume_mpc_over_h_cubed=volume,
        hubble_h=HUBBLE_H,
        bin_edges=mass_edges,
    )
    soft_density, soft_raw = soft_mass_function_and_tangents(
        masses,
        mass_tangents,
        bin_edges=mass_edges,
        bandwidth_dex=arguments.bandwidth_dex,
        volume=volume,
    )
    parameter_response = np.full_like(soft_raw, np.nan)
    positive_soft_density = soft_density > 0.0
    parameter_response[positive_soft_density] = (
        soft_raw[positive_soft_density]
        * parameter_values[None, :]
        / soft_density[positive_soft_density, None]
    )
    summary_names, summary_response = parameter_summary_matrix(
        records,
        parameter_names,
        parameter_values,
        soft_density,
        soft_raw,
    )
    similarity = response_similarity(
        np.concatenate((parameter_response[hard.counts >= 5], summary_response), axis=0)
    )
    dominant_parameter = np.full(hard.counts.shape, -1, dtype=np.int32)
    resolved = hard.counts >= 5
    dominant_parameter[resolved] = np.nanargmax(np.abs(parameter_response[resolved]), axis=1)
    inventory = catalogue_inventory(records)

    bandwidth_values = np.asarray([0.025, 0.05, 0.1], dtype=np.float64)
    bandwidth_density = np.stack(
        [
            np.asarray(
                soft_stellar_mass_function(
                    jnp.asarray(masses),
                    volume_mpc_over_h_cubed=volume,
                    hubble_h=HUBBLE_H,
                    bin_edges=mass_edges,
                    bandwidth_dex=float(bandwidth),
                )
            )
            for bandwidth in bandwidth_values
        ]
    )

    finite_difference = None
    if not arguments.skip_finite_difference:
        finite_difference = parameter_finite_differences(
            partition,
            timing,
            tree_indices,
            identifiers,
            parameter_response,
            parameter_names,
            parameter_values,
            arguments,
            mass_edges,
            volume,
        )

    history = None
    if not arguments.skip_history:
        history = historical_process_analysis(
            partition,
            timing,
            linearized,
            tree_indices,
            arguments,
        )

    convergence = None
    if not arguments.skip_convergence:
        convergence = convergence_analysis(
            partition,
            timing,
            tree_indices,
            linearized,
            arguments,
            mass_edges,
            volume,
        )

    arrays = {
        "stellar_mass_bin_edges": mass_edges,
        "stellar_mass_bin_centres": hard.bin_centres,
        "hard_smf_counts": hard.counts,
        "hard_smf": hard.number_density,
        "soft_smf": soft_density,
        "soft_smf_raw_derivatives": soft_raw,
        "parameter_response": parameter_response,
        "parameter_names": np.asarray(parameter_names),
        "parameter_values": parameter_values,
        "dominant_parameter": dominant_parameter,
        "summary_observable_names": np.asarray(summary_names),
        "summary_response": summary_response,
        "parameter_similarity": similarity,
        "bandwidth_values": bandwidth_values,
        "bandwidth_density": bandwidth_density,
        "halo_mass_bin_edges": inventory.halo_mass_bin_edges,
        "halo_mass_bin_centres": inventory.halo_mass_bin_centres,
        "group_counts": inventory.group_counts,
        "reservoir_names": np.asarray(inventory.reservoir_names),
        "baryon_allotment_fractions": inventory.allotment_fractions,
    }
    if finite_difference is not None:
        arrays.update(
            {
                "fd_parameter_names": np.asarray(finite_difference[0]),
                "fd_relative_steps": finite_difference[1],
                "fd_parameter_response": finite_difference[2],
                "fd_absolute_error": finite_difference[3],
            }
        )
    if history is not None:
        arrays.update(
            {
                "history_mass_bin_edges": history["mass_bin_edges"],
                "history_sample_counts": history["sample_counts"],
                "history_selected_tree_indices": history["selected_tree_indices"],
                "history_selected_target_tree_indices": history["selected_target_tree_indices"],
                "history_selected_galaxy_ids": history["selected_galaxy_ids"],
                "history_ln_scale_factor_edges": history["ln_scale_factor_edges"],
                "history_redshift_edges": history["redshift_edges"],
                "history_process_names": np.asarray(DEFAULT_PROCESSES),
                "historical_process_response": history["response"],
                "history_reservoir_names": np.asarray(RESERVOIRS),
                "history_baryon_inventory": history["history_inventory"],
                "history_scale_factor": timing.scale_factor,
                "history_redshift": timing.redshift,
                "halo_log_growth_rate": history["halo_growth"],
                "stellar_log_growth_rate": history["stellar_growth"],
            }
        )
        for key, validation in history["validation"].items():
            safe_key = key.replace(":", "_")
            arrays[f"history_validation_{safe_key}_automatic"] = validation["automatic"]
            arrays[f"history_validation_{safe_key}_finite_difference"] = validation[
                "finite_difference"
            ]
            arrays[f"history_validation_{safe_key}_absolute_error"] = validation["absolute_error"]
    if convergence is not None:
        arrays.update(
            {
                "convergence_step_counts": convergence["step_counts"],
                "convergence_soft_smf": convergence["soft_smf"],
                "convergence_soft_smf_relative_to_fine": convergence["soft_smf_relative_to_fine"],
                "convergence_scalar_names": convergence["scalar_names"],
                "convergence_scalar_values": convergence["scalar_values"],
                "convergence_scalar_relative_to_fine": convergence["scalar_relative_to_fine"],
                "convergence_elapsed_seconds": convergence["elapsed_seconds"],
            }
        )
    arguments.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output_arrays, **arrays)

    resolved_soft_difference = (
        np.abs(bandwidth_density[1, resolved] - hard.number_density[resolved])
        / hard.number_density[resolved]
    )
    findings = {
        "dominant_parameter_by_resolved_bin": {
            parameter_names[index]: int(np.count_nonzero(dominant_parameter == index))
            for index in range(len(parameter_names))
        },
        "largest_absolute_parameter_response": float(
            np.nanmax(np.abs(parameter_response[resolved]))
        ),
        "maximum_soft_vs_hard_fractional_difference": float(np.nanmax(resolved_soft_difference)),
        "median_soft_vs_hard_fractional_difference": float(np.nanmedian(resolved_soft_difference)),
    }
    if finite_difference is not None:
        resolved_errors = finite_difference[3][:, resolved]
        findings["maximum_resolved_parameter_fd_absolute_error"] = float(np.nanmax(resolved_errors))
        findings["median_resolved_parameter_fd_absolute_error"] = float(
            np.nanmedian(resolved_errors)
        )
    if history is not None:
        response = history["response"]
        findings["largest_absolute_historical_response"] = float(np.nanmax(np.abs(response)))
        findings["history_sample_size"] = int(np.sum(history["sample_counts"]))
    if convergence is not None:
        resolved_convergence = hard.counts >= 5
        findings["maximum_default_vs_20_substep_smf_fractional_difference"] = float(
            np.nanmax(np.abs(convergence["soft_smf_relative_to_fine"][1, resolved_convergence]))
        )

    payload = {
        "schema_version": "mimic-jax-mini-millennium-science-program/v1",
        "tree_file": str(arguments.trees),
        "tree_start": arguments.tree_start,
        "tree_end": tree_end,
        "tree_count": len(tree_indices),
        "input_halos": int(np.sum(partition.tree_halo_counts[arguments.tree_start : tree_end])),
        "snapshot": arguments.snapshot,
        "records": len(records),
        "parameter_names": parameter_names,
        "process_names": DEFAULT_PROCESSES if history is not None else (),
        "parameter_derivative_method": linearized.derivative_method,
        "process_derivative_method": (
            "jax.linearize forward chain rule" if history is not None else "not evaluated"
        ),
        "smf_estimator": {
            "kind": "Gaussian-CDF finite-volume estimator",
            "bandwidth_dex": arguments.bandwidth_dex,
            "hard_bin_width_dex": float(np.diff(mass_edges)[0]),
        },
        "num_substeps": arguments.num_substeps,
        "parameter_evolution_seconds": parameter_elapsed,
        "parameter_batch_seconds": callback.state["seconds"],
        "history_evolution_seconds": (None if history is None else history["elapsed_seconds"]),
        "convergence_tree_count": (0 if convergence is None else convergence["tree_count"]),
        "peak_resident_bytes": maximum_resident_bytes(),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "findings": findings,
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
