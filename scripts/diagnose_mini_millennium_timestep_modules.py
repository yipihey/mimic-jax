#!/usr/bin/env python3
"""Ablate hybrid SAGE16 modules in a strong timestep-sensitive tree."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from analyze_mini_millennium_science_program import json_ready  # noqa: E402

from mimic_jax.io import open_lhalo_partition  # noqa: E402
from mimic_jax.sage16 import (  # noqa: E402
    evolve_lhalo_tree,
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
    parser.add_argument("--tree-index", type=int, default=91)
    parser.add_argument("--global-tree-offset", type=int, default=3432)
    parser.add_argument("--coarse-substeps", type=int, default=10)
    parser.add_argument("--fine-substeps", type=int, default=80)
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


def aggregate_snapshots(result, snapshot_count, fields):
    values = np.full((snapshot_count, len(fields)), np.nan, dtype=np.float64)
    counts = np.zeros(snapshot_count, dtype=np.int32)
    for snapshot, records in result.records_by_snapshot.items():
        counts[snapshot] = len(records)
        for field_index, field in enumerate(fields):
            values[snapshot, field_index] = sum(
                float(getattr(record.state, field)) for record in records
            )
    return values, counts


def main():
    arguments = parse_arguments()
    arguments.compilation_cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update(
        "jax_compilation_cache_dir",
        str(arguments.compilation_cache_dir.resolve()),
    )
    partition = open_lhalo_partition(arguments.trees)
    if not 0 <= arguments.tree_index < partition.tree_count:
        raise SystemExit("tree_index is outside the input partition")
    timing = snapshot_timing(load_scale_factors(arguments.scale_factors))
    tree = partition.read_tree(arguments.tree_index)
    disabled = -np.inf
    conditions = {
        "fiducial": process_perturbations(),
        "no_disk_instability": process_perturbations(disk_instability=disabled),
        "no_satellite_stripping": process_perturbations(satellite_stripping=disabled),
        "no_agn_heating": process_perturbations(agn_heating=disabled),
        "no_quasar_or_starburst": process_perturbations(
            quasar_mode=disabled,
            starburst=disabled,
        ),
    }
    fields = (
        "StellarMass",
        "BlackHoleMass",
        "ColdGas",
        "HotGas",
        "EjectedGas",
        "BulgeMass",
        "Rheat",
        "StarFormationRate",
    )
    substeps = (arguments.coarse_substeps, arguments.fine_substeps)
    snapshot_values = np.full(
        (len(conditions), len(substeps), timing.scale_factor.size, len(fields)),
        np.nan,
        dtype=np.float64,
    )
    record_counts = np.zeros(snapshot_values.shape[:-1], dtype=np.int32)
    elapsed = np.zeros((len(conditions), len(substeps)), dtype=np.float64)
    for condition_index, (condition, perturbations) in enumerate(conditions.items()):
        for substep_index, num_substeps in enumerate(substeps):
            print(
                f"[module-ablation] {condition} substeps={num_substeps}",
                flush=True,
            )
            started = time.perf_counter()
            result = evolve_lhalo_tree(
                tree,
                timing,
                tree_index=arguments.tree_index,
                global_tree_offset=arguments.global_tree_offset,
                num_substeps=num_substeps,
                perturbations=perturbations,
            )
            elapsed[condition_index, substep_index] = time.perf_counter() - started
            if not result.success:
                raise SystemExit(f"{condition}/{num_substeps} evolution reported failure")
            values, counts = aggregate_snapshots(result, timing.scale_factor.size, fields)
            snapshot_values[condition_index, substep_index] = values
            record_counts[condition_index, substep_index] = counts

    final_snapshot = int(np.flatnonzero(np.isfinite(snapshot_values[0, 0, :, 0]))[-1])
    final_values = snapshot_values[:, :, final_snapshot, :]
    final_coarse_to_fine_ratio = np.divide(
        final_values[:, 0],
        final_values[:, 1],
        out=np.full_like(final_values[:, 0], np.nan),
        where=final_values[:, 1] != 0.0,
    )
    arrays = {
        "condition_names": np.asarray(tuple(conditions)),
        "substeps": np.asarray(substeps, dtype=np.int32),
        "snapshot": np.arange(timing.scale_factor.size, dtype=np.int32),
        "scale_factor": np.asarray(timing.scale_factor),
        "redshift": 1.0 / np.asarray(timing.scale_factor) - 1.0,
        "field_names": np.asarray(fields),
        "snapshot_values": snapshot_values,
        "record_counts": record_counts,
        "final_values": final_values,
        "final_coarse_to_fine_ratio": final_coarse_to_fine_ratio,
        "elapsed_seconds": elapsed,
    }
    arguments.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output_arrays, **arrays)

    stellar_index = fields.index("StellarMass")
    black_hole_index = fields.index("BlackHoleMass")

    def finite_or_none(value):
        value = float(value)
        return value if np.isfinite(value) else None

    ratios = {
        condition: {
            "stellar_mass": finite_or_none(final_coarse_to_fine_ratio[index, stellar_index]),
            "black_hole_mass": finite_or_none(final_coarse_to_fine_ratio[index, black_hole_index]),
        }
        for index, condition in enumerate(conditions)
    }
    payload = {
        "schema_version": "mimic-jax-mini-millennium-timestep-module-ablation/v1",
        "tree_index": arguments.tree_index,
        "input_halos": int(partition.tree_halo_counts[arguments.tree_index]),
        "selection_reason": "one of the largest matched-galaxy stellar-mass shifts in the 500-tree timestep diagnostic",
        "coarse_substeps": arguments.coarse_substeps,
        "fine_substeps": arguments.fine_substeps,
        "conditions": tuple(conditions),
        "disabled_process_multiplier": 0.0,
        "final_snapshot": final_snapshot,
        "final_coarse_to_fine_ratio": ratios,
        "elapsed_seconds": elapsed,
        "peak_resident_bytes": maximum_resident_bytes(),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "arrays": arguments.output_arrays.name,
        "interpretation_scope": "single deliberately timestep-sensitive case study; not a population-average attribution",
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
