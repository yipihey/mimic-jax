#!/usr/bin/env python3
"""Demonstrate physically normalized responses for the controlled SAGE16 subset."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mimic_jax import (  # noqa: E402
    parameter_response_matrix,
    process_response_tensor,
    uniform_ln_scale_factor_edges,
    validate_parameter_response,
    validate_process_response,
)
from mimic_jax.sage16 import (  # noqa: E402
    PROCESS_NAMES,
    evolve_upstream_sequential_central_history,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    load_cooling_tables,
    perturbations_from_matrix,
    quiescent_disk_step,
    sage16_units,
    step_context,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for response archives with scientific metadata",
    )
    return parser.parse_args()


def stack_record(record, count):
    return jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (count,) + value.shape),
        record,
    )


def parameter_example():
    state = initial_galaxy_state(
        ColdGas=10.0,
        HotGas=5.0,
        StellarMass=2.0,
        DiskScaleRadius=0.01,
    )
    halo = initial_halo_forcing(Vvir=150.0, dT=1.0e-4)
    context = step_context(time_interval=1.0e-4)
    units = sage16_units()
    parameters = fiducial_parameters()

    def observables(current_parameters):
        result = quiescent_disk_step(
            state,
            state,
            halo,
            halo,
            context,
            current_parameters,
            units,
        )
        return jnp.asarray([result.galaxy.StellarMass, result.galaxy.ColdGas])

    response = parameter_response_matrix(
        observables,
        parameters,
        parameter_names=("SfrEfficiency", "FeedbackReheatingEpsilon"),
        observable_names=("stellar_mass", "cold_gas"),
        observable_units=("1e10 Msun/h", "1e10 Msun/h"),
        parameter_units=("dimensionless", "dimensionless"),
    )
    validation = validate_parameter_response(response, observables, parameters)
    return response, validation


def history_example():
    num_epochs = 4
    state = initial_galaxy_state(
        ColdGas=2.0,
        HotGas=10.0,
        EjectedGas=1.0,
        StellarMass=1.0,
        MetalsColdGas=0.04,
        MetalsHotGas=0.2,
        MetalsEjectedGas=0.02,
        MetalsStellarMass=0.02,
        BlackHoleMass=1.0e-5,
        DiskScaleRadius=0.01,
    )
    halos = stack_record(
        initial_halo_forcing(Mvir=100.0, Vvir=200.0, Rvir=0.2, dT=0.01),
        num_epochs,
    )
    contexts = stack_record(step_context(time_interval=0.01), num_epochs)
    parameters = fiducial_parameters()
    units = sage16_units()
    cooling_tables = load_cooling_tables()

    def observables(epsilon):
        history = evolve_upstream_sequential_central_history(
            state,
            halos,
            contexts,
            parameters,
            units,
            cooling_tables,
            perturbations_from_matrix(epsilon),
        )
        return jnp.asarray([history.final_state.StellarMass])

    baseline = evolve_upstream_sequential_central_history(
        state,
        halos,
        contexts,
        parameters,
        units,
        cooling_tables,
    )
    process_reference_values = jnp.stack(
        [
            baseline.diagnostics.cooling_budget.gas,
            baseline.diagnostics.star_formation.formed_stars,
            baseline.diagnostics.star_formation.cold_to_hot,
            baseline.diagnostics.star_formation.hot_to_ejected,
            baseline.diagnostics.reincorporation.gas,
            baseline.diagnostics.radio_mode.heating_mass,
            baseline.diagnostics.infall.requested,
            jnp.zeros_like(baseline.diagnostics.infall.requested),
            baseline.diagnostics.disk_instability.unstable_gas,
            baseline.diagnostics.quasar_mode.black_hole_accreted,
            baseline.diagnostics.starburst.formed_stars,
        ]
    )
    response = process_response_tensor(
        observables,
        process_names=PROCESS_NAMES,
        ln_scale_factor_edges=uniform_ln_scale_factor_edges(4.0, 0.0, num_epochs),
        observable_names=("final_stellar_mass",),
        observable_units=("1e10 Msun/h",),
        process_reference_values=process_reference_values,
    )
    validation = validate_process_response(response, observables, log_rate_steps=(1.0e-2,))
    return response, validation


def print_parameter_response(response, validation):
    print("\nFractional parameter responses")
    for observable_index, observable in enumerate(response.observable_names):
        for parameter_index, parameter in enumerate(response.parameter_names):
            value = float(response.values[observable_index, parameter_index])
            print(
                f"  {parameter}: a 1% increase produces approximately a "
                f"{value:+.5f}% change in {observable}"
            )
    for step, errors in zip(validation.relative_steps, validation.absolute_error):
        print(f"  finite-difference step={float(step):.1e}, max abs error={np.max(errors):.3e}")


def print_history_response(response, validation):
    print("\nFinite-epoch process responses for final stellar mass")
    for process_index, process in enumerate(response.process_names):
        values = " ".join(f"{value:+.5f}" for value in response.values[0, process_index])
        print(f"  {process:18s} {values}")
    redshift = " -> ".join(f"{value:.2f}" for value in response.redshift_edges)
    print(f"  redshift edges: {redshift}")
    print(
        "  each value is the approximate % change in final stellar mass per 1% "
        "process increase in that finite epoch"
    )
    automatic = np.asarray(response.values)
    errors = np.asarray(validation.absolute_error)
    resolved = np.abs(automatic) > 1.0e-2
    if np.any(resolved):
        print(
            "  finite-difference max abs error for |response| > 1e-2: "
            f"{np.max(errors[:, resolved]):.3e}"
        )
    print(
        "  smaller final-reservoir responses can fall below float32 finite-difference "
        "resolution; the AGN transfer derivative has a separate focused validation test"
    )


def main() -> int:
    arguments = parse_arguments()
    parameter_response, parameter_validation = parameter_example()
    history_response, history_validation = history_example()
    print_parameter_response(parameter_response, parameter_validation)
    print_history_response(history_response, history_validation)

    if arguments.output_dir is not None:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        parameter_response.save(arguments.output_dir / "parameter_response.npz")
        history_response.save(arguments.output_dir / "historical_process_response.npz")
        print(f"\nSaved response archives to {arguments.output_dir}")
    print(
        "\nScope: controlled implemented central chain; shared-central satellite "
        "evolution and merger events are excluded, and this is not a "
        "Mini-Millennium science result."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
