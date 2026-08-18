#!/usr/bin/env python3
"""Diagnose oscillatory stellar-mass-function differences under substep changes."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from analyze_mini_millennium_science_program import (  # noqa: E402
    BOX_SIZE_MPC_OVER_H,
    HUBBLE_H,
    TOTAL_TREE_FILES,
    json_ready,
)

from mimic_jax.io import open_lhalo_partition  # noqa: E402
from mimic_jax.sage16 import (  # noqa: E402
    evolve_lhalo_partition,
    load_scale_factors,
    snapshot_timing,
    soft_stellar_mass_function,
    stellar_mass_function,
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
    parser.add_argument("--tree-count", type=int, default=500)
    parser.add_argument("--snapshot", type=int, default=63)
    parser.add_argument("--coarse-substeps", type=int, default=10)
    parser.add_argument("--fine-substeps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-members", type=int, default=512)
    parser.add_argument(
        "--bandwidths-dex",
        type=float,
        nargs="+",
        default=(0.03, 0.05, 0.1, 0.2),
    )
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


def records_with_tree_index(result, snapshot):
    rows = []
    for tree_index, records_by_snapshot in zip(result.tree_indices, result.records_by_tree):
        for record in records_by_snapshot.get(snapshot, ()):
            rows.append((tree_index, record))
    return rows


def catalogue_arrays(rows):
    fields = (
        "StellarMass",
        "ColdGas",
        "HotGas",
        "EjectedGas",
        "BlackHoleMass",
        "BulgeMass",
        "ICS",
    )
    arrays = {
        "tree_index": np.asarray([tree_index for tree_index, _ in rows], dtype=np.int32),
        "galaxy_id": np.asarray([record.halo.UniqueGalaxyID for _, record in rows], dtype=np.int64),
        "galaxy_type": np.asarray([record.halo.Type for _, record in rows], dtype=np.int32),
    }
    for field in fields:
        arrays[field] = np.asarray(
            [getattr(record.state, field) for _, record in rows], dtype=np.float64
        )
    return arrays


def aligned_catalogues(coarse, fine):
    coarse_lookup = {int(identifier): index for index, identifier in enumerate(coarse["galaxy_id"])}
    fine_lookup = {int(identifier): index for index, identifier in enumerate(fine["galaxy_id"])}
    common_ids = np.asarray(sorted(set(coarse_lookup) & set(fine_lookup)), dtype=np.int64)
    coarse_indices = np.asarray([coarse_lookup[int(value)] for value in common_ids], dtype=np.int32)
    fine_indices = np.asarray([fine_lookup[int(value)] for value in common_ids], dtype=np.int32)
    return common_ids, coarse_indices, fine_indices


def finite_relative(candidate, reference):
    result = np.full_like(reference, np.nan, dtype=np.float64)
    valid = reference != 0.0
    result[valid] = (candidate[valid] - reference[valid]) / reference[valid]
    return result


def population_masks(catalogue):
    """Return the complete, central, and satellite catalogue selections."""
    galaxy_type = catalogue["galaxy_type"]
    return (
        np.ones(galaxy_type.shape, dtype=bool),
        galaxy_type == 0,
        galaxy_type != 0,
    )


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
    catalogues = {}
    elapsed = {}
    for label, substeps in (
        ("coarse", arguments.coarse_substeps),
        ("fine", arguments.fine_substeps),
    ):
        print(f"[ringing] {label} substeps={substeps}", flush=True)
        started = time.perf_counter()
        result = evolve_lhalo_partition(
            partition,
            timing,
            tree_indices=tree_indices,
            global_tree_offset=arguments.global_tree_offset,
            num_substeps=substeps,
            output_snapshots=(arguments.snapshot,),
            batch_size=arguments.batch_size,
            max_batch_members=arguments.max_batch_members,
            member_binning="power_of_two",
        )
        elapsed[label] = time.perf_counter() - started
        if not result.success:
            raise SystemExit(f"{label} evolution reported failure")
        catalogues[label] = catalogue_arrays(records_with_tree_index(result, arguments.snapshot))

    coarse = catalogues["coarse"]
    fine = catalogues["fine"]
    common_ids, coarse_indices, fine_indices = aligned_catalogues(coarse, fine)
    coarse_mass = coarse["StellarMass"][coarse_indices]
    fine_mass = fine["StellarMass"][fine_indices]
    positive = (coarse_mass > 0.0) & (fine_mass > 0.0)
    logarithmic_fine_mass = np.log10(fine_mass[positive] * 1.0e10 / HUBBLE_H)
    logarithmic_shift = np.log10(coarse_mass[positive] / fine_mass[positive])
    mass_edges = np.arange(8.0, 12.1, 0.1, dtype=np.float64)
    volume = BOX_SIZE_MPC_OVER_H**3 / TOTAL_TREE_FILES
    population_names = ("all", "centrals", "satellites")
    hard = []
    hard_counts = []
    soft = []
    for current in (coarse, fine):
        current_hard = []
        current_counts = []
        current_soft = []
        for mask in population_masks(current):
            selected_mass = current["StellarMass"][mask]
            hard_summary = stellar_mass_function(
                selected_mass,
                volume_mpc_over_h_cubed=volume,
                hubble_h=HUBBLE_H,
                bin_edges=mass_edges,
            )
            current_hard.append(hard_summary.number_density)
            current_counts.append(hard_summary.counts)
            current_soft.append(
                [
                    np.asarray(
                        soft_stellar_mass_function(
                            selected_mass,
                            volume_mpc_over_h_cubed=volume,
                            hubble_h=HUBBLE_H,
                            bin_edges=mass_edges,
                            bandwidth_dex=bandwidth,
                        )
                    )
                    for bandwidth in arguments.bandwidths_dex
                ]
            )
        hard.append(current_hard)
        hard_counts.append(current_counts)
        soft.append(current_soft)
    hard = np.asarray(hard)
    hard_counts = np.asarray(hard_counts)
    soft = np.asarray(soft)
    hard_relative = np.stack(
        [finite_relative(hard[0, index], hard[1, index]) for index in range(hard.shape[1])]
    )
    soft_relative = np.stack(
        [
            [
                finite_relative(soft[0, population, bandwidth], soft[1, population, bandwidth])
                for bandwidth in range(soft.shape[2])
            ]
            for population in range(soft.shape[1])
        ]
    )

    shift_edges = np.arange(8.0, 11.6, 0.2)
    shift_bin = np.digitize(logarithmic_fine_mass, shift_edges) - 1
    shift_count = np.zeros(shift_edges.size - 1, dtype=np.int32)
    shift_median = np.full(shift_edges.size - 1, np.nan)
    shift_q16 = np.full_like(shift_median, np.nan)
    shift_q84 = np.full_like(shift_median, np.nan)
    for index in range(shift_count.size):
        selected = shift_bin == index
        shift_count[index] = np.count_nonzero(selected)
        if shift_count[index]:
            shift_q16[index], shift_median[index], shift_q84[index] = np.quantile(
                logarithmic_shift[selected], (0.16, 0.5, 0.84)
            )

    common_difference = coarse_mass - fine_mass
    order = np.argsort(np.abs(common_difference))[::-1]
    top = order[: min(20, order.size)]
    coarse_only = np.asarray(sorted(set(coarse["galaxy_id"]) - set(common_ids)), dtype=np.int64)
    fine_only = np.asarray(sorted(set(fine["galaxy_id"]) - set(common_ids)), dtype=np.int64)
    total_difference = float(np.sum(coarse["StellarMass"]) - np.sum(fine["StellarMass"]))
    common_total_difference = float(np.sum(common_difference))
    arrays = {
        "stellar_mass_bin_edges": mass_edges,
        "stellar_mass_bin_centres": mass_edges[:-1] + 0.05,
        "bandwidths_dex": np.asarray(arguments.bandwidths_dex),
        "population_names": np.asarray(population_names),
        "hard_smf": hard,
        "hard_smf_counts": hard_counts,
        "hard_smf_fractional_difference": hard_relative,
        "soft_smf": soft,
        "soft_smf_fractional_difference": soft_relative,
        "common_galaxy_ids": common_ids,
        "common_tree_indices": coarse["tree_index"][coarse_indices],
        "common_coarse_galaxy_type": coarse["galaxy_type"][coarse_indices],
        "common_fine_galaxy_type": fine["galaxy_type"][fine_indices],
        "coarse_stellar_mass": coarse_mass,
        "fine_stellar_mass": fine_mass,
        "fine_log_stellar_mass": logarithmic_fine_mass,
        "coarse_minus_fine_log_stellar_mass": logarithmic_shift,
        "shift_bin_edges": shift_edges,
        "shift_bin_centres": shift_edges[:-1] + 0.1,
        "shift_bin_counts": shift_count,
        "shift_median": shift_median,
        "shift_q16": shift_q16,
        "shift_q84": shift_q84,
        "coarse_only_galaxy_ids": coarse_only,
        "fine_only_galaxy_ids": fine_only,
        "largest_difference_galaxy_ids": common_ids[top],
        "largest_difference_tree_indices": coarse["tree_index"][coarse_indices[top]],
        "largest_stellar_mass_differences": common_difference[top],
        "largest_coarse_stellar_masses": coarse_mass[top],
        "largest_fine_stellar_masses": fine_mass[top],
    }
    arguments.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output_arrays, **arrays)

    bandwidth_maximum = []
    bandwidth_median = []
    for population, values_by_bandwidth in enumerate(soft_relative):
        resolved = hard_counts[1, population] >= 5
        current_maximum = []
        current_median = []
        for values in values_by_bandwidth:
            finite = resolved & np.isfinite(values)
            current_maximum.append(float(np.max(np.abs(values[finite]))))
            current_median.append(float(np.median(np.abs(values[finite]))))
        bandwidth_maximum.append(current_maximum)
        bandwidth_median.append(current_median)
    payload = {
        "schema_version": "mimic-jax-mini-millennium-timestep-ringing/v1",
        "tree_selection": "spread",
        "tree_count": arguments.tree_count,
        "input_halos": int(np.sum(partition.tree_halo_counts[np.asarray(tree_indices)])),
        "coarse_substeps": arguments.coarse_substeps,
        "fine_substeps": arguments.fine_substeps,
        "coarse_galaxy_count": int(coarse["galaxy_id"].size),
        "fine_galaxy_count": int(fine["galaxy_id"].size),
        "common_galaxy_count": int(common_ids.size),
        "common_galaxy_type_change_count": int(
            np.count_nonzero(
                coarse["galaxy_type"][coarse_indices] != fine["galaxy_type"][fine_indices]
            )
        ),
        "coarse_only_galaxy_count": int(coarse_only.size),
        "fine_only_galaxy_count": int(fine_only.size),
        "positive_common_galaxy_count": int(np.count_nonzero(positive)),
        "median_coarse_minus_fine_log_stellar_mass": float(np.median(logarithmic_shift)),
        "fraction_with_larger_coarse_stellar_mass": float(np.mean(logarithmic_shift > 0.0)),
        "total_stellar_mass_difference": total_difference,
        "common_identity_stellar_mass_difference": common_total_difference,
        "common_identity_fraction_of_total_difference": (
            common_total_difference / total_difference if total_difference != 0.0 else None
        ),
        "bandwidths_dex": arguments.bandwidths_dex,
        "population_names": population_names,
        "soft_smf_median_absolute_fractional_difference": bandwidth_median,
        "soft_smf_maximum_absolute_fractional_difference": bandwidth_maximum,
        "elapsed_seconds": elapsed,
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
