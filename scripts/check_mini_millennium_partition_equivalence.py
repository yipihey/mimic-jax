#!/usr/bin/env python3
"""Batch Mini-Millennium trees and compare their catalogue fields with upstream MIMIC."""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from mimic_jax.io import open_lhalo_partition
from mimic_jax.sage16 import (
    evolve_lhalo_partition,
    load_scale_factors,
    record_to_catalogue,
    snapshot_timing,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-start", type=int, default=0)
    parser.add_argument("--tree-count", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-members", type=int, default=512)
    parser.add_argument(
        "--member-binning",
        choices=("exact", "power_of_two"),
        default="exact",
        help="FoF member shape policy (default: exact reference workspaces)",
    )
    parser.add_argument("--global-tree-offset", type=int, default=0)
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
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the machine-readable equivalence summary",
    )
    return parser.parse_args()


def tolerances(field, dtype):
    if dtype.itemsize == 4 or field in ("Cooling", "Heating"):
        return 2.0e-6, 2.0e-6
    return 2.0e-12, 2.0e-12


def main():
    arguments = parse_arguments()
    partition = open_lhalo_partition(arguments.trees)
    end = (
        partition.tree_count
        if arguments.tree_count is None
        else arguments.tree_start + arguments.tree_count
    )
    if not 0 <= arguments.tree_start < end <= partition.tree_count:
        raise SystemExit(
            f"requested tree interval [{arguments.tree_start}, {end}) is outside "
            f"[0, {partition.tree_count})"
        )
    tree_indices = tuple(range(arguments.tree_start, end))
    timing = snapshot_timing(load_scale_factors(arguments.scale_factors))
    with h5py.File(arguments.upstream, "r") as upstream:
        snapshots = sorted(int(name[4:]) for name in upstream if name.startswith("Snap"))

    started = time.perf_counter()

    def report_progress(progress):
        if progress["event"] == "snapshot":
            print(
                f"snapshot={progress['snapshot']} groups={progress['groups']} "
                f"shapes={progress['shape_count']} "
                f"maximum_members={progress['maximum_members']}",
                flush=True,
            )
        elif progress["event"] == "shape" and progress["member_count"] >= 32:
            print(
                f"  large_shape members={progress['member_count']} "
                f"central_indices={progress['central_indices']} groups={progress['groups']} "
                f"batch={progress['batch_size']}",
                flush=True,
            )

    result = evolve_lhalo_partition(
        partition,
        timing,
        tree_indices=tree_indices,
        global_tree_offset=arguments.global_tree_offset,
        output_snapshots=snapshots,
        batch_size=arguments.batch_size,
        max_batch_members=arguments.max_batch_members,
        member_binning=arguments.member_binning,
        progress_callback=report_progress,
    )
    elapsed = time.perf_counter() - started

    mismatches = []
    comparisons = 0
    records_compared = 0
    maximum_relative_error = {}
    with h5py.File(arguments.upstream, "r") as upstream:
        counts_by_snapshot = {
            snapshot: upstream[f"Snap{snapshot:03d}/TreeHalosPerSnap"][:] for snapshot in snapshots
        }
        offsets_by_snapshot = {
            snapshot: np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
            for snapshot, counts in counts_by_snapshot.items()
        }
        for tree_index, records_by_snapshot in zip(result.tree_indices, result.records_by_tree):
            tree = partition.read_tree(tree_index)
            for snapshot in snapshots:
                group = f"Snap{snapshot:03d}"
                count = int(counts_by_snapshot[snapshot][tree_index])
                start = int(offsets_by_snapshot[snapshot][tree_index])
                expected = upstream[f"{group}/Galaxies"][start : start + count]
                actual = [
                    record_to_catalogue(record, tree, timing)
                    for record in records_by_snapshot.get(snapshot, ())
                ]
                if len(actual) != len(expected):
                    mismatches.append(
                        (tree_index, snapshot, "record_count", len(actual), len(expected))
                    )
                    continue
                records_compared += len(actual)
                actual_by_id = {int(row["UniqueGalaxyID"]): row for row in actual}
                expected_by_id = {int(row["UniqueGalaxyID"]): row for row in expected}
                if actual_by_id.keys() != expected_by_id.keys():
                    mismatches.append((tree_index, snapshot, "UniqueGalaxyID_set"))
                    continue
                for identifier, expected_row in expected_by_id.items():
                    actual_row = actual_by_id[identifier]
                    for field in expected.dtype.names:
                        comparisons += 1
                        observed = np.asarray(actual_row[field])
                        reference = np.asarray(expected_row[field])
                        if observed.dtype.kind in "iu":
                            equal = np.array_equal(observed, reference)
                        else:
                            relative_tolerance, absolute_tolerance = tolerances(
                                field, reference.dtype
                            )
                            equal = np.allclose(
                                observed,
                                reference,
                                rtol=relative_tolerance,
                                atol=absolute_tolerance,
                                equal_nan=True,
                            )
                            resolved = np.abs(reference) > absolute_tolerance
                            relative_error = (
                                float(
                                    np.max(
                                        np.abs(observed[resolved] - reference[resolved])
                                        / np.abs(reference[resolved])
                                    )
                                )
                                if np.any(resolved)
                                else 0.0
                            )
                            maximum_relative_error[field] = max(
                                maximum_relative_error.get(field, 0.0), relative_error
                            )
                        if not equal:
                            mismatches.append(
                                (
                                    tree_index,
                                    snapshot,
                                    identifier,
                                    field,
                                    observed,
                                    reference,
                                )
                            )

    largest_fields = sorted(maximum_relative_error.items(), key=lambda item: item[1], reverse=True)[
        :10
    ]
    print(
        f"trees=[{arguments.tree_start},{end}) input_halos="
        f"{int(partition.tree_halo_counts[arguments.tree_start:end].sum())} "
        f"groups={result.groups_evolved} records={records_compared} fields={comparisons} "
        f"elapsed_seconds={elapsed:.6f} success={result.success}"
    )
    print("largest_relative_errors", largest_fields)
    payload = {
        "diagnostic": "mini_millennium_partition_equivalence",
        "tree_start": arguments.tree_start,
        "tree_end": end,
        "tree_count": len(tree_indices),
        "batch_size": arguments.batch_size,
        "max_batch_members": arguments.max_batch_members,
        "member_binning": arguments.member_binning,
        "global_tree_offset": arguments.global_tree_offset,
        "input_halos": int(partition.tree_halo_counts[arguments.tree_start : end].sum()),
        "groups_evolved": result.groups_evolved,
        "records_compared": records_compared,
        "field_comparisons": comparisons,
        "mismatches": len(mismatches),
        "elapsed_seconds": elapsed,
        "success": result.success,
        "largest_relative_errors": {field: error for field, error in largest_fields},
        "tolerances": {
            "float32_and_cooling_heating": {"rtol": 2.0e-6, "atol": 2.0e-6},
            "other_float64": {"rtol": 2.0e-12, "atol": 2.0e-12},
            "integer_fields": "exact",
        },
    }
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if mismatches:
        for mismatch in mismatches[:20]:
            print("mismatch", mismatch)
        raise SystemExit(f"{len(mismatches)} comparisons exceeded their tolerance")
    print("all requested Mini-Millennium catalogue comparisons passed")


if __name__ == "__main__":
    main()
