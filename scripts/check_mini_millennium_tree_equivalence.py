#!/usr/bin/env python3
"""Compare one mimic-jax tree against an upstream MIMIC HDF5 catalogue."""

import argparse
import math
from pathlib import Path

import h5py
import numpy as np

from mimic_jax.io import open_lhalo_partition
from mimic_jax.sage16 import (
    evolve_lhalo_tree,
    load_scale_factors,
    record_to_catalogue,
    snapshot_timing,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=int, default=1575)
    parser.add_argument("--snapshot", type=int, default=63)
    parser.add_argument(
        "--all-output-snapshots",
        action="store_true",
        help="compare every snapshot table present in the upstream catalogue",
    )
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument(
        "--trees",
        type=Path,
        default=Path("simulations/mini-millennium/snapshots/trees_063.0"),
    )
    parser.add_argument(
        "--scale-factors",
        type=Path,
        default=Path("simulations/mini-millennium/mini-millennium.a_list"),
    )
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path("output/sage16-mini-millennium/model_000.hdf5"),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.partition != 0:
        raise SystemExit("this initial checker currently requires the explicit global tree offset")
    partition = open_lhalo_partition(arguments.trees)
    tree = partition.read_tree(arguments.tree)
    timing = snapshot_timing(load_scale_factors(arguments.scale_factors))
    result = evolve_lhalo_tree(tree, timing, tree_index=arguments.tree)
    with h5py.File(arguments.upstream, "r") as upstream:
        snapshots = (
            sorted(int(name[4:]) for name in upstream if name.startswith("Snap"))
            if arguments.all_output_snapshots
            else [arguments.snapshot]
        )
        expected_by_snapshot = {}
        for snapshot in snapshots:
            group = f"Snap{snapshot:03d}"
            counts = upstream[f"{group}/TreeHalosPerSnap"][:]
            start = int(np.sum(counts[: arguments.tree], dtype=np.int64))
            expected_by_snapshot[snapshot] = upstream[f"{group}/Galaxies"][
                start : start + int(counts[arguments.tree])
            ]

    comparisons = 0
    record_count = 0
    mismatches = []
    maximum_relative_error = 0.0
    for snapshot, expected in expected_by_snapshot.items():
        actual = [
            record_to_catalogue(record, tree, timing)
            for record in result.records_by_snapshot.get(snapshot, ())
        ]
        if len(actual) != len(expected):
            mismatches.append((snapshot, "record_count", len(actual), len(expected), math.nan))
            continue
        record_count += len(actual)
        actual_by_id = {int(row["UniqueGalaxyID"]): row for row in actual}
        expected_by_id = {int(row["UniqueGalaxyID"]): row for row in expected}
        if actual_by_id.keys() != expected_by_id.keys():
            mismatches.append(
                (
                    snapshot,
                    "UniqueGalaxyID_set",
                    actual_by_id.keys(),
                    expected_by_id.keys(),
                    math.nan,
                )
            )
            continue
        for identifier in expected_by_id:
            actual_row = actual_by_id[identifier]
            expected_row = expected_by_id[identifier]
            for field in expected.dtype.names:
                comparisons += 1
                observed = np.asarray(actual_row[field])
                reference = np.asarray(expected_row[field])
                if observed.dtype.kind in "iu":
                    equal = np.array_equal(observed, reference)
                    relative_error = 0.0
                else:
                    if reference.dtype.itemsize == 4:
                        relative_tolerance, absolute_tolerance = 2.0e-6, 2.0e-6
                    elif field in ("Cooling", "Heating"):
                        # These log10 diagnostics accumulate mixed-precision transfer
                        # powers; use the same explicit tolerance as float reservoirs.
                        relative_tolerance, absolute_tolerance = 2.0e-6, 2.0e-6
                    else:
                        relative_tolerance, absolute_tolerance = 2.0e-12, 2.0e-12
                    equal = np.allclose(
                        observed,
                        reference,
                        rtol=relative_tolerance,
                        atol=absolute_tolerance,
                        equal_nan=True,
                    )
                    difference = np.abs(observed - reference)
                    resolved = np.abs(reference) > absolute_tolerance
                    relative_error = (
                        float(np.max(difference[resolved] / np.abs(reference[resolved])))
                        if np.any(resolved)
                        else 0.0
                    )
                    maximum_relative_error = max(maximum_relative_error, relative_error)
                if not equal:
                    mismatches.append(
                        (snapshot, identifier, field, observed, reference, relative_error)
                    )

    print(
        f"tree={arguments.tree} snapshots={snapshots} records={record_count} "
        f"groups={result.groups_evolved} success={result.success} "
        f"max_relative_error={maximum_relative_error:.3e}"
    )
    if mismatches:
        for mismatch in mismatches[:20]:
            print("mismatch", mismatch)
        raise SystemExit(f"{len(mismatches)} field comparisons exceeded their tolerance")
    print(f"all {comparisons} field comparisons passed")


if __name__ == "__main__":
    main()
