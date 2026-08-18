#!/usr/bin/env python3
"""Validate selected finite-epoch process responses over several step sizes."""

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
    aggregate_selected_stellar_mass,
    json_ready,
)

from mimic_jax.io import open_lhalo_partition  # noqa: E402
from mimic_jax.sage16 import (  # noqa: E402
    evolve_lhalo_partition,
    load_scale_factors,
    process_perturbations,
    snapshot_timing,
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
        "--history-arrays",
        type=Path,
        default=Path("archive/mini-millennium-sage16-history-responses.npz"),
    )
    parser.add_argument("--global-tree-offset", type=int, default=3432)
    parser.add_argument("--snapshot", type=int, default=63)
    parser.add_argument("--num-substeps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-members", type=int, default=512)
    parser.add_argument(
        "--relative-steps",
        type=float,
        nargs="+",
        default=(1.0e-2, 3.0e-3, 1.0e-3),
    )
    parser.add_argument(
        "--processes",
        nargs="+",
        default=("cooling", "agn_heating"),
    )
    parser.add_argument("--epoch", type=int, default=6)
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


def main():
    arguments = parse_arguments()
    arguments.compilation_cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update(
        "jax_compilation_cache_dir",
        str(arguments.compilation_cache_dir.resolve()),
    )
    with np.load(arguments.history_arrays, allow_pickle=False) as stored:
        history = dict(stored)
    process_names = [str(value) for value in history["history_process_names"]]
    unknown = sorted(set(arguments.processes) - set(process_names))
    if unknown:
        raise SystemExit(f"unknown historical processes: {unknown}")
    response = history["historical_process_response"]
    sample_counts = history["history_sample_counts"]
    selected_bins = np.repeat(np.arange(sample_counts.size, dtype=np.int32), sample_counts)
    target_trees = history["history_selected_target_tree_indices"]
    identifiers = history["history_selected_galaxy_ids"]
    selected_trees = tuple(int(value) for value in history["history_selected_tree_indices"])
    ln_edges = history["history_ln_scale_factor_edges"]

    partition = open_lhalo_partition(arguments.trees)
    timing = snapshot_timing(load_scale_factors(arguments.scale_factors))
    snapshot_epoch = np.asarray(
        [
            int(
                np.clip(
                    np.searchsorted(ln_edges, np.log(float(scale_factor)), side="right") - 1,
                    0,
                    len(ln_edges) - 2,
                )
            )
            for scale_factor in timing.scale_factor
        ]
    )
    finite_difference = np.full(
        (len(arguments.processes), len(arguments.relative_steps), sample_counts.size),
        np.nan,
        dtype=np.float64,
    )
    elapsed = np.zeros((len(arguments.processes), len(arguments.relative_steps), 2))
    for process_index, process in enumerate(arguments.processes):
        for step_index, step in enumerate(arguments.relative_steps):
            perturbed_sums = []
            for sign_index, sign in enumerate((-1.0, 1.0)):
                print(
                    f"[history finite difference] process={process} epoch={arguments.epoch} "
                    f"epsilon={sign * step:+.4g}",
                    flush=True,
                )
                schedule = np.zeros(len(timing.scale_factor), dtype=np.float64)
                schedule[snapshot_epoch == arguments.epoch] = sign * step
                started = time.perf_counter()
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
                elapsed[process_index, step_index, sign_index] = time.perf_counter() - started
                sums, counts = aggregate_selected_stellar_mass(
                    result,
                    arguments.snapshot,
                    target_trees,
                    identifiers,
                    selected_bins,
                    sample_counts.size,
                )
                if not np.array_equal(counts, sample_counts):
                    raise ValueError("finite-epoch rerun changed the selected sample")
                perturbed_sums.append(sums)
            finite_difference[process_index, step_index] = (
                np.log(perturbed_sums[1]) - np.log(perturbed_sums[0])
            ) / (2.0 * step)

    automatic = np.stack(
        [
            response[:, process_names.index(process), arguments.epoch]
            for process in arguments.processes
        ]
    )
    absolute_error = np.abs(finite_difference - automatic[:, None, :])
    arrays = {
        "process_names": np.asarray(arguments.processes),
        "epoch": np.asarray(arguments.epoch),
        "relative_steps": np.asarray(arguments.relative_steps),
        "automatic_response": automatic,
        "finite_difference_response": finite_difference,
        "absolute_error": absolute_error,
        "sample_counts": sample_counts,
        "mass_bin_edges": history["history_mass_bin_edges"],
        "elapsed_seconds": elapsed,
    }
    arguments.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output_arrays, **arrays)

    finite = np.isfinite(absolute_error)
    maximum = np.full((len(arguments.processes), len(arguments.relative_steps)), np.nan)
    median = np.full_like(maximum, np.nan)
    for process_index in range(len(arguments.processes)):
        for step_index in range(len(arguments.relative_steps)):
            valid = finite[process_index, step_index]
            if np.any(valid):
                maximum[process_index, step_index] = np.max(
                    absolute_error[process_index, step_index, valid]
                )
                median[process_index, step_index] = np.median(
                    absolute_error[process_index, step_index, valid]
                )
    payload = {
        "schema_version": "mimic-jax-mini-millennium-history-validation/v1",
        "history_arrays": str(arguments.history_arrays),
        "process_names": arguments.processes,
        "epoch": arguments.epoch,
        "relative_steps": arguments.relative_steps,
        "maximum_absolute_error": maximum,
        "median_absolute_error": median,
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
