#!/usr/bin/env python3
"""Run a controlled timestep-refinement study of the upstream SAGE16 update."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mimic_jax import timestep_refinement_study  # noqa: E402
from mimic_jax.sage16 import (  # noqa: E402
    baryonic_mass,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    load_cooling_tables,
    sage16_units,
    step_context,
    subcycle_upstream_sequential_central,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional NPZ output path")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    initial = initial_galaxy_state(
        ColdGas=2.0,
        HotGas=10.0,
        EjectedGas=1.0,
        StellarMass=1.0,
        MetalsColdGas=0.04,
        MetalsHotGas=0.2,
        MetalsEjectedGas=0.02,
        MetalsStellarMass=0.02,
        BlackHoleMass=0.01,
        DiskScaleRadius=0.01,
    )
    halo = initial_halo_forcing(Mvir=100.0, Rvir=0.2, Vvir=200.0, dT=0.01)
    context = step_context(time_interval=0.01)
    parameters = fiducial_parameters()
    units = sage16_units()
    tables = load_cooling_tables()

    def run(num_substeps):
        return subcycle_upstream_sequential_central(
            initial,
            halo,
            context,
            parameters,
            units,
            tables,
            num_substeps=num_substeps,
        )

    study = timestep_refinement_study(
        run,
        lambda result: jnp.asarray(
            [
                result.final_state.StellarMass,
                result.final_state.ColdGas,
                result.final_state.HotGas,
                result.final_state.EjectedGas,
                result.final_state.BlackHoleMass,
            ]
        ),
        substeps=(1, 2, 4, 8),
        observable_names=(
            "stellar_mass",
            "cold_gas",
            "hot_gas",
            "ejected_gas",
            "black_hole_mass",
        ),
        observable_units=("1e10 Msun/h",) * 5,
    )

    print("method=upstream_sequential forcing=piecewise_constant")
    print("substeps " + " ".join(f"{name:>17s}" for name in study.observable_names))
    for num_substeps, values in zip(study.substeps, study.observable_values):
        print(f"{int(num_substeps):8d} " + " ".join(f"{float(value):17.9g}" for value in values))
    print("\nabsolute difference from the 8-substep provisional reference")
    for num_substeps, errors in zip(study.substeps, study.absolute_errors):
        print(f"{int(num_substeps):8d} " + " ".join(f"{float(error):17.6e}" for error in errors))
    for num_substeps in (1, 2, 4, 8):
        result = run(num_substeps)
        residual = float(baryonic_mass(result.final_state) - baryonic_mass(initial))
        print(f"baryon residual at {num_substeps:2d} substeps: {residual:+.6e}")

    if arguments.output is not None:
        study.save(arguments.output)
        print(f"saved {arguments.output}")
    orders = np.asarray(study.observed_orders)
    finite_orders = orders[np.isfinite(orders)]
    if finite_orders.size:
        print(f"observed orders span {finite_orders.min():.3f} to {finite_orders.max():.3f}")
    print("scope: controlled central slice; no Mini-Millennium convergence claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
