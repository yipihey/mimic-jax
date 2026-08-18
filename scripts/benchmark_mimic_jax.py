#!/usr/bin/env python3
"""Benchmark the implemented quiescent SAGE16 subset without hiding JIT compilation."""

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mimic_jax.sage16 import (  # noqa: E402
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    quiescent_disk_step,
    sage16_units,
    step_context,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    return parser.parse_args()


def synchronize(value):
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), value)
    return value


def timed(callable_, repeats):
    start = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = synchronize(callable_())
    return (time.perf_counter() - start) / repeats, result


def benchmark(batch_size, warmups, repeats):
    state = initial_galaxy_state(
        ColdGas=10.0,
        HotGas=5.0,
        StellarMass=2.0,
        DiskScaleRadius=0.01,
    )
    halo = initial_halo_forcing(Vvir=150.0, dT=1.0e-4)
    context = step_context(time_interval=1.0e-4)
    parameters = fiducial_parameters()
    units = sage16_units()

    def scalar_step():
        return quiescent_disk_step(
            state,
            state,
            halo,
            halo,
            context,
            parameters,
            units,
        )

    eager_seconds, _ = timed(scalar_step, repeats)
    compiled_scalar = jax.jit(scalar_step)
    scalar_first_seconds, _ = timed(compiled_scalar, 1)
    for _ in range(warmups):
        synchronize(compiled_scalar())
    scalar_warmed_seconds, _ = timed(compiled_scalar, repeats)

    cold_values = jnp.linspace(8.0, 12.0, batch_size, dtype=jnp.float32)
    states = jax.vmap(lambda cold: state._replace(ColdGas=cold))(cold_values)
    halos = jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (batch_size,) + value.shape),
        halo,
    )

    @jax.jit
    def batched_step():
        return jax.vmap(
            quiescent_disk_step,
            in_axes=(0, 0, 0, 0, None, None, None),
        )(states, states, halos, halos, context, parameters, units)

    batch_first_seconds, _ = timed(batched_step, 1)
    for _ in range(warmups):
        synchronize(batched_step())
    batch_warmed_seconds, _ = timed(batched_step, repeats)

    return {
        "scope": "implemented quiescent SAGE16 subset; not an end-to-end MIMIC comparison",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "batch_size": batch_size,
        "warmups": warmups,
        "repeats": repeats,
        "eager_scalar_seconds": eager_seconds,
        "jit_scalar_first_call_seconds": scalar_first_seconds,
        "jit_scalar_warmed_seconds": scalar_warmed_seconds,
        "jit_vmap_first_call_seconds": batch_first_seconds,
        "jit_vmap_warmed_batch_seconds": batch_warmed_seconds,
        "jit_vmap_warmed_seconds_per_galaxy": batch_warmed_seconds / batch_size,
    }


def main() -> int:
    arguments = parse_arguments()
    if arguments.batch_size < 1 or arguments.warmups < 0 or arguments.repeats < 1:
        raise ValueError("batch-size and repeats must be positive; warmups cannot be negative")
    result = benchmark(arguments.batch_size, arguments.warmups, arguments.repeats)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
