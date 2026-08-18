#!/usr/bin/env python3
"""Calculate mass-stratified historical SAGE16 process responses."""

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
    DEFAULT_PROCESSES,
    RESERVOIRS,
    historical_process_analysis,
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
    parser.add_argument("--snapshot", type=int, default=63)
    parser.add_argument("--num-substeps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-members", type=int, default=512)
    parser.add_argument("--history-epochs", type=int, default=8)
    parser.add_argument("--history-sample-per-bin", type=int, default=12)
    parser.add_argument("--skip-finite-difference", action="store_true")
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
    timing = snapshot_timing(load_scale_factors(arguments.scale_factors))
    tree_indices = tuple(range(partition.tree_count))

    print("[history] evolving complete fiducial partition for deterministic selection", flush=True)
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
        raise SystemExit("baseline partition evolution reported failure")
    history = historical_process_analysis(
        partition,
        timing,
        baseline,
        tree_indices,
        arguments,
    )

    arrays = {
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
    validation_summary = {}
    for key, validation in history["validation"].items():
        safe_key = key.replace(":", "_")
        arrays[f"history_validation_{safe_key}_automatic"] = validation["automatic"]
        arrays[f"history_validation_{safe_key}_finite_difference"] = validation["finite_difference"]
        arrays[f"history_validation_{safe_key}_absolute_error"] = validation["absolute_error"]
        finite = np.isfinite(validation["absolute_error"])
        validation_summary[key] = {
            "maximum_absolute_error": (
                float(np.max(validation["absolute_error"][finite])) if np.any(finite) else None
            ),
            "median_absolute_error": (
                float(np.median(validation["absolute_error"][finite])) if np.any(finite) else None
            ),
        }
    arguments.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output_arrays, **arrays)

    response = history["response"]
    process_peaks = {}
    for process_index, process in enumerate(DEFAULT_PROCESSES):
        absolute = np.abs(response[:, process_index])
        if np.any(np.isfinite(absolute)):
            mass_index, epoch_index = np.unravel_index(np.nanargmax(absolute), absolute.shape)
            process_peaks[process] = {
                "response": float(response[mass_index, process_index, epoch_index]),
                "log10_stellar_mass_min": float(history["mass_bin_edges"][mass_index]),
                "log10_stellar_mass_max": float(history["mass_bin_edges"][mass_index + 1]),
                "redshift_edge_high": float(history["redshift_edges"][epoch_index]),
                "redshift_edge_low": float(history["redshift_edges"][epoch_index + 1]),
            }
    payload = {
        "schema_version": "mimic-jax-mini-millennium-history/v1",
        "tree_file": str(arguments.trees),
        "tree_count_used_for_selection": partition.tree_count,
        "input_halos": partition.total_halos,
        "selected_tree_count": int(history["selected_tree_indices"].size),
        "selected_galaxy_count": int(np.sum(history["sample_counts"])),
        "sample_counts": history["sample_counts"],
        "process_names": DEFAULT_PROCESSES,
        "epoch_count": arguments.history_epochs,
        "derivative_method": "jax.linearize forward chain rule",
        "normalization": "d ln(mean final stellar mass) / d finite-epoch epsilon",
        "baseline_evolution_seconds": baseline_seconds,
        "response_evolution_seconds": history["elapsed_seconds"],
        "response_batch_seconds": history["batch_seconds"],
        "peak_resident_bytes": maximum_resident_bytes(),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "process_peaks": process_peaks,
        "finite_difference_validation": validation_summary,
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
