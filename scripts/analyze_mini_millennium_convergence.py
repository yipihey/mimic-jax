#!/usr/bin/env python3
"""Measure population-level SAGE16 timestep effects on familiar observables."""

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
    TOTAL_TREE_FILES,
    convergence_analysis,
    json_ready,
)

from mimic_jax.io import open_lhalo_partition  # noqa: E402
from mimic_jax.sage16 import (  # noqa: E402
    evolve_lhalo_partition,
    load_scale_factors,
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
    parser.add_argument("--global-tree-offset", type=int, default=3432)
    parser.add_argument("--tree-start", type=int, default=1500)
    parser.add_argument("--tree-count", type=int, default=500)
    parser.add_argument(
        "--tree-selection",
        choices=("spread", "contiguous"),
        default="spread",
        help="spread samples the requested count across the complete partition",
    )
    parser.add_argument("--snapshot", type=int, default=63)
    parser.add_argument("--num-substeps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-members", type=int, default=512)
    parser.add_argument("--bandwidth-dex", type=float, default=0.05)
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
    partition = open_lhalo_partition(arguments.trees)
    if not 0 < arguments.tree_count <= partition.tree_count:
        raise SystemExit("tree_count must be within the input partition")
    if arguments.tree_selection == "spread":
        tree_indices = tuple(
            int(value)
            for value in np.linspace(
                0,
                partition.tree_count - 1,
                num=arguments.tree_count,
                dtype=np.int32,
            )
        )
    else:
        tree_end = arguments.tree_start + arguments.tree_count
        if not 0 <= arguments.tree_start < tree_end <= partition.tree_count:
            raise SystemExit("requested tree interval is outside the input partition")
        tree_indices = tuple(range(arguments.tree_start, tree_end))
    if len(set(tree_indices)) != arguments.tree_count:
        raise SystemExit("tree selection did not produce the requested number of unique trees")
    timing = snapshot_timing(load_scale_factors(arguments.scale_factors))
    print("[convergence] baseline substeps=10", flush=True)
    baseline_started = time.perf_counter()
    baseline = evolve_lhalo_partition(
        partition,
        timing,
        tree_indices=tree_indices,
        global_tree_offset=arguments.global_tree_offset,
        num_substeps=arguments.num_substeps,
        output_snapshots=(arguments.snapshot,),
        batch_size=arguments.batch_size,
        max_batch_members=arguments.max_batch_members,
        member_binning="power_of_two",
    )
    baseline_seconds = time.perf_counter() - baseline_started
    if not baseline.success:
        raise SystemExit("baseline convergence sample reported failure")
    arguments.convergence_tree_count = arguments.tree_count
    mass_edges = np.arange(8.0, 12.1, 0.1, dtype=np.float64)
    convergence = convergence_analysis(
        partition,
        timing,
        tree_indices,
        baseline,
        arguments,
        mass_edges,
        BOX_SIZE_MPC_OVER_H**3 / TOTAL_TREE_FILES,
    )

    arrays = {
        "stellar_mass_bin_edges": mass_edges,
        "stellar_mass_bin_centres": mass_edges[:-1] + 0.5 * np.diff(mass_edges),
        "baseline_hard_smf_counts": convergence["baseline_hard_counts"],
        "convergence_step_counts": convergence["step_counts"],
        "convergence_soft_smf": convergence["soft_smf"],
        "convergence_soft_smf_relative_to_fine": convergence["soft_smf_relative_to_fine"],
        "convergence_scalar_names": convergence["scalar_names"],
        "convergence_scalar_values": convergence["scalar_values"],
        "convergence_scalar_relative_to_fine": convergence["scalar_relative_to_fine"],
        "convergence_successive_soft_smf_difference": convergence["successive_soft_smf_difference"],
        "convergence_successive_scalar_difference": convergence["successive_scalar_difference"],
        "convergence_elapsed_seconds": convergence["elapsed_seconds"],
    }
    arguments.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output_arrays, **arrays)

    resolved = convergence["baseline_hard_counts"] >= 5
    default_difference = np.abs(convergence["soft_smf_relative_to_fine"][1])
    coarse_difference = np.abs(convergence["soft_smf_relative_to_fine"][0])
    scalar_names = tuple(str(value) for value in convergence["scalar_names"])
    scalar_default = {
        name: float(value)
        for name, value in zip(
            scalar_names,
            convergence["scalar_relative_to_fine"][1],
        )
    }
    payload = {
        "schema_version": "mimic-jax-mini-millennium-convergence/v1",
        "tree_file": str(arguments.trees),
        "tree_selection": arguments.tree_selection,
        "tree_start": int(min(tree_indices)),
        "tree_end": int(max(tree_indices)) + 1,
        "tree_count": arguments.tree_count,
        "input_halos": int(np.sum(partition.tree_halo_counts[np.asarray(tree_indices)])),
        "step_counts": convergence["step_counts"],
        "fine_reference_substeps": int(convergence["step_counts"][-1]),
        "baseline_evolution_seconds": baseline_seconds,
        "refinement_evolution_seconds": convergence["elapsed_seconds"],
        "resolved_smf_bins": int(np.count_nonzero(resolved)),
        "maximum_default_vs_fine_smf_fractional_difference": (
            float(np.nanmax(default_difference[resolved])) if np.any(resolved) else None
        ),
        "median_default_vs_fine_smf_fractional_difference": (
            float(np.nanmedian(default_difference[resolved])) if np.any(resolved) else None
        ),
        "maximum_coarse_vs_fine_smf_fractional_difference": (
            float(np.nanmax(coarse_difference[resolved])) if np.any(resolved) else None
        ),
        "default_vs_fine_scalar_fractional_difference": scalar_default,
        "smf_estimator": {
            "kind": "Gaussian-CDF finite-volume estimator",
            "bandwidth_dex": arguments.bandwidth_dex,
        },
        "forcing": "piecewise constant within each merger-tree interval",
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
