#!/usr/bin/env python3
"""Produce durable science summaries for one matched Mini-Millennium partition."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import h5py
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from mimic_jax.io import open_lhalo_partition  # noqa: E402
from mimic_jax.sage16 import (  # noqa: E402
    binned_fraction,
    binned_quantiles,
    evolve_lhalo_partition,
    group_baryon_inventory,
    load_scale_factors,
    record_to_catalogue,
    safe_fractional_difference,
    snapshot_timing,
    stellar_mass_function,
)

HUBBLE_H = 0.73
BOX_SIZE_MPC_OVER_H = 62.5
GLOBAL_BARYON_FRACTION = 0.17
TOTAL_TREE_FILES = 8
RESERVOIR_FIELDS = (
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
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path("output/sage16-mini-millennium/model_001.hdf5"),
    )
    parser.add_argument("--tree-start", type=int, default=0)
    parser.add_argument("--tree-count", type=int)
    parser.add_argument("--global-tree-offset", type=int, default=3432)
    parser.add_argument("--partition-index", type=int, default=1)
    parser.add_argument("--snapshot", type=int, default=63)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-members", type=int, default=512)
    parser.add_argument(
        "--member-binning", choices=("exact", "power_of_two"), default="power_of_two"
    )
    parser.add_argument(
        "--compilation-cache-dir",
        type=Path,
        help="optional persistent JAX compilation cache",
    )
    parser.add_argument(
        "--equivalence-json",
        type=Path,
        help="optional all-snapshot equivalence result to cross-reference",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-arrays", type=Path, required=True)
    return parser.parse_args()


def maximum_resident_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def read_upstream_rows(path, snapshot, tree_start, tree_end):
    with h5py.File(path, "r") as upstream:
        group = f"Snap{snapshot:03d}"
        counts = upstream[f"{group}/TreeHalosPerSnap"][:]
        if tree_end > counts.size:
            raise ValueError("requested tree interval exceeds the upstream partition")
        offsets = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
        start = int(offsets[tree_start])
        stop = int(offsets[tree_end - 1] + counts[tree_end - 1])
        return upstream[f"{group}/Galaxies"][start:stop]


def rows_from_result(result, partition, timing, snapshot):
    rows = []
    for tree_index, records_by_snapshot in zip(result.tree_indices, result.records_by_tree):
        tree = partition.read_tree(tree_index)
        rows.extend(
            record_to_catalogue(record, tree, timing)
            for record in records_by_snapshot.get(snapshot, ())
        )
    return rows


def matched_columns(rows, reference):
    actual_by_id = {int(row["UniqueGalaxyID"]): row for row in rows}
    reference_by_id = {int(row["UniqueGalaxyID"]): row for row in reference}
    if actual_by_id.keys() != reference_by_id.keys():
        missing = len(reference_by_id.keys() - actual_by_id.keys())
        extra = len(actual_by_id.keys() - reference_by_id.keys())
        raise ValueError(f"z=0 identities differ: missing={missing}, extra={extra}")
    identifiers = np.asarray(sorted(reference_by_id), dtype=np.int64)
    fields = (
        "UniqueGalaxyID",
        "UniqueCentralGalaxyID",
        "Type",
        "Mvir",
        "CentralMvir",
        *RESERVOIR_FIELDS,
        "StarFormationRate",
        "Cooling",
        "Heating",
    )
    actual = {
        field: np.asarray([actual_by_id[int(identifier)][field] for identifier in identifiers])
        for field in fields
    }
    expected = {
        field: np.asarray([reference_by_id[int(identifier)][field] for identifier in identifiers])
        for field in fields
    }
    return identifiers, actual, expected


def inventory(columns, halo_mass_edges):
    return group_baryon_inventory(
        unique_galaxy_id=columns["UniqueGalaxyID"],
        unique_central_galaxy_id=columns["UniqueCentralGalaxyID"],
        galaxy_type=columns["Type"],
        central_halo_mass=columns["Mvir"],
        reservoirs={name: columns[name] for name in RESERVOIR_FIELDS},
        hubble_h=HUBBLE_H,
        global_baryon_fraction=GLOBAL_BARYON_FRACTION,
        halo_mass_bin_edges=halo_mass_edges,
    )


def central_history_summaries(columns, stellar_mass_edges):
    central = (columns["Type"] == 0) & (columns["StellarMass"] > 0.0)
    logarithmic_stellar_mass = np.log10(columns["StellarMass"][central] * 1.0e10 / HUBBLE_H)
    stellar_mass_msun = columns["StellarMass"][central] * 1.0e10 / HUBBLE_H
    specific_sfr = columns["StarFormationRate"][central] / stellar_mass_msun
    quenched = specific_sfr < 1.0e-11
    counts, quenched_counts, quenched_fraction = binned_fraction(
        logarithmic_stellar_mass, quenched, bin_edges=stellar_mass_edges
    )
    cooling = np.where(columns["Cooling"][central] > 0.0, columns["Cooling"][central], np.nan)
    heating = np.where(columns["Heating"][central] > 0.0, columns["Heating"][central], np.nan)
    cooling_counts, cooling_quantiles = binned_quantiles(
        logarithmic_stellar_mass, cooling, bin_edges=stellar_mass_edges
    )
    heating_counts, heating_quantiles = binned_quantiles(
        logarithmic_stellar_mass, heating, bin_edges=stellar_mass_edges
    )
    return {
        "central_counts": counts,
        "quenched_counts": quenched_counts,
        "quenched_fraction": quenched_fraction,
        "cooling_counts": cooling_counts,
        "cooling_quantiles": cooling_quantiles,
        "heating_counts": heating_counts,
        "heating_quantiles": heating_quantiles,
    }


def dominant_regimes(inventory_result, minimum_groups=10):
    fractions = inventory_result.allotment_fractions
    valid = (inventory_result.group_counts >= minimum_groups) & np.any(
        np.isfinite(fractions), axis=1
    )
    regimes = []
    current = None
    for index in np.flatnonzero(valid):
        dominant = int(np.nanargmax(fractions[index]))
        if (
            current is None
            or dominant != current["reservoir_index"]
            or index != current["last"] + 1
        ):
            if current is not None:
                regimes.append(current)
            current = {
                "reservoir_index": dominant,
                "first": int(index),
                "last": int(index),
            }
        else:
            current["last"] = int(index)
    if current is not None:
        regimes.append(current)
    return [
        {
            "reservoir": inventory_result.reservoir_names[regime["reservoir_index"]],
            "log10_halo_mass_min": float(inventory_result.halo_mass_bin_edges[regime["first"]]),
            "log10_halo_mass_max": float(inventory_result.halo_mass_bin_edges[regime["last"] + 1]),
        }
        for regime in regimes
    ]


def main():
    arguments = parse_arguments()
    if arguments.compilation_cache_dir is not None:
        arguments.compilation_cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update(
            "jax_compilation_cache_dir", str(arguments.compilation_cache_dir.resolve())
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

    started = time.perf_counter()
    result = evolve_lhalo_partition(
        partition,
        timing,
        tree_indices=tree_indices,
        global_tree_offset=arguments.global_tree_offset,
        output_snapshots=(arguments.snapshot,),
        batch_size=arguments.batch_size,
        max_batch_members=arguments.max_batch_members,
        member_binning=arguments.member_binning,
    )
    evolution_seconds = time.perf_counter() - started
    if not result.success:
        raise SystemExit("partition evolution reported failure")

    reference_rows = read_upstream_rows(
        arguments.upstream, arguments.snapshot, arguments.tree_start, tree_end
    )
    actual_rows = rows_from_result(result, partition, timing, arguments.snapshot)
    identifiers, actual, expected = matched_columns(actual_rows, reference_rows)

    volume = BOX_SIZE_MPC_OVER_H**3 / TOTAL_TREE_FILES
    stellar_mass_edges = np.arange(8.0, 12.1, 0.1, dtype=np.float64)
    halo_mass_edges = np.arange(9.5, 14.1, 0.25, dtype=np.float64)
    actual_smf = stellar_mass_function(
        actual["StellarMass"],
        volume_mpc_over_h_cubed=volume,
        hubble_h=HUBBLE_H,
        bin_edges=stellar_mass_edges,
    )
    expected_smf = stellar_mass_function(
        expected["StellarMass"],
        volume_mpc_over_h_cubed=volume,
        hubble_h=HUBBLE_H,
        bin_edges=stellar_mass_edges,
    )
    smf_fractional_difference, smf_fractional_valid = safe_fractional_difference(
        actual_smf.number_density, expected_smf.number_density
    )
    actual_inventory = inventory(actual, halo_mass_edges)
    expected_inventory = inventory(expected, halo_mass_edges)
    actual_central = central_history_summaries(actual, stellar_mass_edges)
    expected_central = central_history_summaries(expected, stellar_mass_edges)

    resolved_smf = expected_smf.counts >= 5
    smf_bin_mismatches = int(
        np.count_nonzero(actual_smf.counts[resolved_smf] != expected_smf.counts[resolved_smf])
    )
    maximum_smf_difference = (
        float(np.nanmax(np.abs(smf_fractional_difference[resolved_smf])))
        if np.any(resolved_smf)
        else None
    )
    resolved_inventory = expected_inventory.group_counts >= 20
    actual_total_fraction = np.nansum(actual_inventory.allotment_fractions, axis=1)
    expected_total_fraction = np.nansum(expected_inventory.allotment_fractions, axis=1)
    inventory_difference, inventory_valid = safe_fractional_difference(
        actual_total_fraction, expected_total_fraction
    )
    resolved_inventory &= inventory_valid
    maximum_inventory_difference = (
        float(np.nanmax(np.abs(inventory_difference[resolved_inventory])))
        if np.any(resolved_inventory)
        else None
    )
    stellar_mass_difference = np.abs(
        actual["StellarMass"].astype(np.float64) - expected["StellarMass"].astype(np.float64)
    )
    resolved_stellar_mass = np.abs(expected["StellarMass"]) > 2.0e-6
    stellar_mass_relative_difference = np.zeros_like(stellar_mass_difference)
    stellar_mass_relative_difference[resolved_stellar_mass] = stellar_mass_difference[
        resolved_stellar_mass
    ] / np.abs(expected["StellarMass"][resolved_stellar_mass])
    resolved_quenched = expected_central["central_counts"] >= 5
    quenched_difference = np.abs(
        actual_central["quenched_fraction"] - expected_central["quenched_fraction"]
    )
    maximum_quenched_difference = (
        float(np.nanmax(quenched_difference[resolved_quenched]))
        if np.any(resolved_quenched)
        else None
    )

    arrays = {
        "stellar_mass_bin_edges": stellar_mass_edges,
        "stellar_mass_bin_centres": actual_smf.bin_centres,
        "upstream_smf_counts": expected_smf.counts,
        "mimic_jax_smf_counts": actual_smf.counts,
        "upstream_smf": expected_smf.number_density,
        "mimic_jax_smf": actual_smf.number_density,
        "smf_fractional_difference": smf_fractional_difference,
        "smf_fractional_valid": smf_fractional_valid,
        "halo_mass_bin_edges": halo_mass_edges,
        "halo_mass_bin_centres": actual_inventory.halo_mass_bin_centres,
        "reservoir_names": np.asarray(RESERVOIR_FIELDS),
        "group_counts": actual_inventory.group_counts,
        "upstream_baryon_allotment_fractions": expected_inventory.allotment_fractions,
        "mimic_jax_baryon_allotment_fractions": actual_inventory.allotment_fractions,
        "upstream_total_baryon_fraction": expected_total_fraction,
        "mimic_jax_total_baryon_fraction": actual_total_fraction,
        "stellar_summary_bin_centres": actual_smf.bin_centres,
    }
    for prefix, values in (("upstream", expected_central), ("mimic_jax", actual_central)):
        for name, value in values.items():
            arrays[f"{prefix}_{name}"] = value

    arguments.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output_arrays, **arrays)
    equivalence = None
    if arguments.equivalence_json is not None:
        equivalence = json.loads(arguments.equivalence_json.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "mimic-jax-mini-millennium-science/v1",
        "partition_index": arguments.partition_index,
        "tree_start": arguments.tree_start,
        "tree_end": tree_end,
        "tree_count": len(tree_indices),
        "global_tree_offset": arguments.global_tree_offset,
        "input_halos": int(partition.tree_halo_counts[arguments.tree_start : tree_end].sum()),
        "snapshot": arguments.snapshot,
        "records_matched": int(identifiers.size),
        "groups_evolved": int(result.groups_evolved),
        "volume_mpc_over_h_cubed": volume,
        "hubble_h": HUBBLE_H,
        "evolution_seconds": evolution_seconds,
        "peak_resident_bytes": maximum_resident_bytes(),
        "backend": jax.default_backend(),
        "arrays": arguments.output_arrays.name,
        "metrics": {
            "stellar_mass_values_exact": bool(
                np.array_equal(actual["StellarMass"], expected["StellarMass"])
            ),
            "stellar_mass_values_different": int(np.count_nonzero(stellar_mass_difference != 0.0)),
            "maximum_resolved_stellar_mass_relative_difference": float(
                np.max(stellar_mass_relative_difference)
            ),
            "all_smf_bin_mismatches": int(
                np.count_nonzero(actual_smf.counts != expected_smf.counts)
            ),
            "resolved_smf_bins": int(np.count_nonzero(resolved_smf)),
            "resolved_smf_bin_mismatches": smf_bin_mismatches,
            "maximum_resolved_smf_fractional_difference": maximum_smf_difference,
            "resolved_baryon_inventory_bins": int(np.count_nonzero(resolved_inventory)),
            "maximum_resolved_total_baryon_fractional_difference": (maximum_inventory_difference),
            "resolved_quenched_fraction_bins": int(np.count_nonzero(resolved_quenched)),
            "maximum_resolved_quenched_fraction_absolute_difference": (maximum_quenched_difference),
        },
        "dominant_baryon_reservoir_regimes": dominant_regimes(actual_inventory),
        "all_snapshot_equivalence": equivalence,
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
