#!/usr/bin/env python3
"""Benchmark cold and warm mimic-jax Mini-Millennium partition evolution."""

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from mimic_jax.io import open_lhalo_partition  # noqa: E402
from mimic_jax.sage16 import (  # noqa: E402
    evolve_lhalo_partition,
    load_scale_factors,
    record_to_catalogue,
    snapshot_timing,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-start", type=int, default=0)
    parser.add_argument("--tree-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-members", type=int, default=512)
    parser.add_argument(
        "--compilation-cache-dir",
        type=Path,
        help="optional persistent JAX compilation cache used across processes",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--member-binning",
        choices=("exact", "power_of_two"),
        default="power_of_two",
    )
    parser.add_argument(
        "--output-snapshots",
        default="63",
        help="comma-separated snapshots retained for catalogue conversion",
    )
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
    return parser.parse_args()


def maximum_resident_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class ProgressTimings:
    def __init__(self):
        self.batch_seconds = 0.0
        self.batch_calls = 0
        self.shape_calls = set()

    def __call__(self, event):
        if event["event"] == "shape":
            self.shape_calls.add((event["member_count"], event["batch_size"]))
        elif event["event"] == "batch":
            self.batch_seconds += event["elapsed_seconds"]
            self.batch_calls += 1


def catalogue_digest(result, partition, timing):
    digest = hashlib.sha256()
    count = 0
    started = time.perf_counter()
    for tree_index, records_by_snapshot in zip(result.tree_indices, result.records_by_tree):
        tree = partition.read_tree(tree_index)
        for snapshot in sorted(records_by_snapshot):
            for record in records_by_snapshot[snapshot]:
                count += 1
                row = record_to_catalogue(record, tree, timing)
                for name, value in row.items():
                    array = np.asarray(value)
                    digest.update(name.encode("utf-8"))
                    digest.update(array.dtype.str.encode("ascii"))
                    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
                    digest.update(array.tobytes())
    return count, digest.hexdigest(), time.perf_counter() - started


def main():
    arguments = parse_arguments()
    if arguments.tree_count <= 0 or arguments.repeats <= 0:
        raise SystemExit("tree-count and repeats must be positive")
    if arguments.compilation_cache_dir is not None:
        arguments.compilation_cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update(
            "jax_compilation_cache_dir",
            str(arguments.compilation_cache_dir.resolve()),
        )
    output_snapshots = tuple(
        int(value) for value in arguments.output_snapshots.split(",") if value.strip()
    )
    setup_started = time.perf_counter()
    partition = open_lhalo_partition(arguments.trees)
    end = arguments.tree_start + arguments.tree_count
    if not 0 <= arguments.tree_start < end <= partition.tree_count:
        raise SystemExit(
            f"requested tree interval [{arguments.tree_start}, {end}) is outside "
            f"[0, {partition.tree_count})"
        )
    tree_indices = tuple(range(arguments.tree_start, end))
    timing = snapshot_timing(load_scale_factors(arguments.scale_factors))
    setup_seconds = time.perf_counter() - setup_started

    runs = []
    for repeat in range(arguments.repeats):
        progress = ProgressTimings()
        started = time.perf_counter()
        result = evolve_lhalo_partition(
            partition,
            timing,
            tree_indices=tree_indices,
            output_snapshots=output_snapshots,
            batch_size=arguments.batch_size,
            max_batch_members=arguments.max_batch_members,
            member_binning=arguments.member_binning,
            progress_callback=progress,
        )
        evolution_seconds = time.perf_counter() - started
        record_count, digest, catalogue_seconds = catalogue_digest(result, partition, timing)
        runs.append(
            {
                "repeat": repeat + 1,
                "evolution_seconds": evolution_seconds,
                "batch_seconds": progress.batch_seconds,
                "host_seconds": max(evolution_seconds - progress.batch_seconds, 0.0),
                "catalogue_seconds": catalogue_seconds,
                "batch_calls": progress.batch_calls,
                "executable_shapes": len(progress.shape_calls),
                "groups": result.groups_evolved,
                "records": record_count,
                "success": result.success,
                "catalogue_sha256": digest,
                "peak_resident_bytes": maximum_resident_bytes(),
            }
        )

    payload = {
        "benchmark": "mini_millennium_partition",
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "jax_compilation_cache_dir": jax.config.jax_compilation_cache_dir,
        "tree_start": arguments.tree_start,
        "tree_count": arguments.tree_count,
        "input_halos": int(partition.tree_halo_counts[arguments.tree_start : end].sum()),
        "batch_size": arguments.batch_size,
        "max_batch_members": arguments.max_batch_members,
        "member_binning": arguments.member_binning,
        "output_snapshots": output_snapshots,
        "setup_seconds": setup_seconds,
        "runs": runs,
        "repeat_catalogues_identical": len({run["catalogue_sha256"] for run in runs}) == 1,
    }
    if len(runs) > 1:
        payload["first_to_best_warm_ratio"] = runs[0]["evolution_seconds"] / min(
            run["evolution_seconds"] for run in runs[1:]
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not all(run["success"] for run in runs):
        raise SystemExit("one or more evolution runs reported failure")
    if not payload["repeat_catalogues_identical"]:
        raise SystemExit("repeated runs produced different catalogue digests")


if __name__ == "__main__":
    main()
