#!/usr/bin/env python3
"""Run the pinned native SHARK population oracle with durable provenance."""

import argparse
from pathlib import Path

from mimic_jax.shark.reference import (
    prepare_reference_config,
    run_reference_shark,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shark-source", required=True, type=Path)
    parser.add_argument("--shark-executable", required=True, type=Path)
    parser.add_argument("--config-template", required=True, type=Path)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--redshifts", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--model-name", default="lagos23-reference")
    parser.add_argument("--simulation-name", default="mini-SURFS")
    parser.add_argument("--snapshot", type=int, default=199)
    parser.add_argument("--subvolume", type=int, default=0)
    arguments = parser.parse_args()

    output = arguments.output_directory.resolve()
    effective_config = prepare_reference_config(
        arguments.config_template,
        output / "effective-shark.cfg",
        tree_file=arguments.tree,
        redshift_file=arguments.redshifts,
        output_directory=output,
        seed=arguments.seed,
        model_name=arguments.model_name,
        simulation_batch=arguments.subvolume,
    )
    result = run_reference_shark(
        executable=arguments.shark_executable,
        config=effective_config,
        output_directory=output,
        tree_file=arguments.tree,
        redshift_file=arguments.redshifts,
        snapshot=arguments.snapshot,
        subvolume=arguments.subvolume,
        model_name=arguments.model_name,
        simulation_name=arguments.simulation_name,
        seed=arguments.seed,
        source_directory=arguments.shark_source,
    )
    print(result.catalogue)


if __name__ == "__main__":
    main()
