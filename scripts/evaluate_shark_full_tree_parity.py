#!/usr/bin/env python3
"""Evaluate every native SHARK RHS call with the compiled JAX population kernel."""

import argparse

import jax

jax.config.update("jax_enable_x64", True)

from mimic_jax.shark.population_parity import evaluate_shark_population_parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="binary trace produced by the opt-in upstream patch")
    parser.add_argument("tree", help="public SHARK tree_199.<subvolume>.hdf5 input")
    parser.add_argument("output", help="destination JSON summary")
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--rtol", type=float, default=1.1e-4)
    parser.add_argument("--warn-rtol", type=float, default=1.5e-4)
    parser.add_argument("--atol", type=float, default=1.0e-8)
    parser.add_argument(
        "--instrumented-output-root",
        help="traced native model root containing <snapshot>/0/galaxies.hdf5",
    )
    parser.add_argument(
        "--reference-output-root",
        help="clean native model root containing <snapshot>/0/galaxies.hdf5",
    )
    arguments = parser.parse_args()
    result = evaluate_shark_population_parity(
        arguments.trace,
        arguments.tree,
        batch_size=arguments.batch_size,
        relative_tolerance=arguments.rtol,
        warning_relative_tolerance=arguments.warn_rtol,
        absolute_tolerance=arguments.atol,
        instrumented_output_root=arguments.instrumented_output_root,
        reference_output_root=arguments.reference_output_root,
    )
    result.write_json(arguments.output)
    print(
        f"{'PASS' if result.strict_passed else 'WARN' if result.passed else 'FAIL'}: "
        f"{result.rhs_evaluations:,} RHS calls, "
        f"maximum rate relative difference {result.rates.maximum_relative_error:.6g}, "
        f"{result.rates.failing_values} strict exceptions, "
        f"{result.elapsed_seconds:.3f} s steady-state JAX replay"
    )
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
