#!/usr/bin/env python3
"""Run pinned native Sapphire in its own environment and export a common artifact.

This script deliberately does not import mimic-jax: the projects currently
require incompatible JAX versions.  It is integration glue only; all physical
rates and derivatives are evaluated by Sapphire's own Pandya23 closure.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "mimic-jax-sapphire-native/v1"
EXPECTED_REVISION = "ee50e858e3427de50368c32205001248849b8be0"
STATE_NAMES = ("M_star", "M_ism", "M_cgm", "Eth_cgm", "MZ_star", "MZ_ism", "MZ_cgm")
FORCING_NAMES = ("Mdot_in_dm", "Mvir", "Rvir", "Vvir", "NFW_c")
PARAMETER_NAMES = (
    "A_M",
    "alpha0_M",
    "alphaz_M",
    "beta_M",
    "A_E",
    "alpha0_E",
    "alphaz_E",
    "beta_E",
    "A_SF",
    "alpha0_SF",
    "alphaz_SF",
    "beta_SF",
    "A_Z",
    "alpha0_Z",
    "alphaz_Z",
    "beta_Z",
)
AMPLITUDE_PARAMETER_INDICES = (0, 4, 8, 12)
OBSERVABLE_NAMES = (
    "stellar_mass",
    "ism_mass",
    "cgm_mass",
    "star_formation_rate",
    "stellar_metallicity",
    "ism_metallicity",
)
RATE_NAMES = (
    "Mdot_in_halo",
    "Edot_in_halo",
    "Edot_cool",
    "Mdot_cool",
    "Mdot_sfr",
    "Mdot_wind",
    "Edot_wind",
    "Edot_out_halo",
    "Mdot_out_halo",
    "MZdot_sfr",
    "MZdot_cool",
    "MZdot_yield",
    "MZdot_wind",
    "MZdot_in_halo",
    "MZdot_out_halo",
)


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(source: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def coordinate(name, unit, description, label=None):
    return {
        "name": name,
        "label": name.replace("_", " ") if label is None else label,
        "unit": unit,
        "description": description,
    }


def write_artifact(directory: Path, manifest, arrays) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    normalized = {name: np.asarray(array) for name, array in sorted(arrays.items())}
    array_path = directory / "arrays.npz"
    np.savez_compressed(array_path, **normalized)
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["arrays"] = {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in normalized.items()
    }
    manifest["array_file"] = {
        "path": "arrays.npz",
        "sha256": file_hash(array_path),
        "size_bytes": array_path.stat().st_size,
    }
    (directory / "artifact.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    source = arguments.source.resolve()
    data = arguments.data.resolve()
    case_path = arguments.case.resolve()
    output = arguments.output.resolve()
    revision = git_output(source, "rev-parse", "HEAD")
    if revision != EXPECTED_REVISION:
        raise RuntimeError(
            f"Sapphire bridge expected revision {EXPECTED_REVISION}, received {revision}"
        )
    case = json.loads(case_path.read_text(encoding="utf-8"))
    if case.get("schema_version") != "mimic-jax-sapphire-case/v1":
        raise ValueError(f"Unsupported Sapphire case schema {case.get('schema_version')!r}")

    sys.path.insert(0, str(source))
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import diffrax
    import jax
    import jax.numpy as jnp
    from astropy import units as u
    from sapphire.models import pandya23

    jax.config.update("jax_enable_x64", True)

    fixed = {
        "alpha_n": -1.5,
        "alpha_T": 0.0,
        "f_recycle": 0.4,
        "yZ": 0.02,
        "A_Zin_halo": 0.25,
    }
    cosmology = {
        "Om0": 0.3075,
        "Ode0": 0.6910,
        "Ob0": 0.0486,
        "h0": 0.6774,
        "sigma8": 0.8159,
        "Tcmb0": 2.7255,
        "Neff": 3.0460,
        "ns": 0.9667,
    }
    config = {
        "data_path": str(data),
        "coolfunc": "sd93",
        "clip_etaE_max": 1.0,
        "params_fixed_astro": fixed,
        "params_cosmo": cosmology,
    }
    integrator, saveat_fn, _ = pandya23.setup(config)

    initial_state = jnp.asarray([case["initial_state"][name] for name in STATE_NAMES])
    forcing = jnp.asarray([case["forcing"][name] for name in FORCING_NAMES])
    raw_parameters = jnp.asarray([case["parameters"][name] for name in PARAMETER_NAMES])
    start_time = float(case["start_time_gyr"])
    end_time = float(case["end_time_gyr"])
    sample_count = int(case["sample_count"])
    solver_config = case["solver"]
    seconds_per_year = float(u.yr.to("s"))
    seconds_per_gyr = seconds_per_year * 1.0e9

    forcing_times = jnp.asarray([start_time, end_time])

    def native_parameters(values):
        transformed = values
        for index in AMPLITUDE_PARAMETER_INDICES:
            transformed = transformed.at[index].set(10.0 ** transformed[index])
        return jnp.concatenate((transformed, jnp.zeros((1,), dtype=transformed.dtype)))

    def interpolators(values, log_input_controls):
        adjusted = values * jnp.exp(log_input_controls)
        result = []
        for value in adjusted:
            series = jnp.log10(jnp.repeat(value, 2))
            coefficients = diffrax.backward_hermite_coefficients(forcing_times, series)
            result.append(diffrax.CubicInterpolation(forcing_times, coefficients))
        return tuple(result)

    def physical_derivative(time_gyr, state, input_controls, parameters):
        log_time_seconds = jnp.log10(time_gyr * seconds_per_gyr)
        log_state = jnp.log10(state)
        arguments_native = (native_parameters(parameters), interpolators(forcing, input_controls))
        logarithmic_derivative = integrator(log_time_seconds, log_state, arguments_native)
        return state / time_gyr * logarithmic_derivative

    def auxiliary(time_gyr, state, input_controls, parameters):
        log_time_seconds = jnp.log10(time_gyr * seconds_per_gyr)
        log_state = jnp.log10(state)
        arguments_native = (native_parameters(parameters), interpolators(forcing, input_controls))
        _, information = saveat_fn(log_time_seconds, log_state, arguments_native)
        return information

    def observables(time_gyr, state, input_controls, parameters):
        information = auxiliary(time_gyr, state, input_controls, parameters)
        return jnp.asarray(
            [
                state[0],
                state[1],
                state[2],
                information["Mdot_sfr"],
                state[4] / state[0],
                state[5] / state[1],
            ]
        )

    zero_controls = jnp.zeros((len(FORCING_NAMES),), dtype=jnp.float64)
    saved_times_gyr = jnp.linspace(start_time, end_time, sample_count)
    saved_log_times = jnp.log10(saved_times_gyr * seconds_per_gyr)

    def solve(parameters, input_controls, relative_tolerance, absolute_tolerance, save_times):
        integration_arguments = (
            native_parameters(parameters),
            interpolators(forcing, input_controls),
        )
        return diffrax.diffeqsolve(
            diffrax.ODETerm(integrator),
            diffrax.Tsit5(scan_kind="bounded"),
            jnp.log10(start_time * seconds_per_gyr),
            jnp.log10(end_time * seconds_per_gyr),
            1.0e-10,
            jnp.log10(initial_state),
            integration_arguments,
            saveat=diffrax.SaveAt(ts=save_times),
            stepsize_controller=diffrax.PIDController(
                rtol=relative_tolerance, atol=absolute_tolerance
            ),
            adjoint=diffrax.DirectAdjoint(),
            max_steps=int(solver_config["max_steps"]),
            throw=True,
        )

    solution = solve(
        raw_parameters,
        zero_controls,
        float(solver_config["rtol"]),
        float(solver_config["atol"]),
        saved_log_times,
    )
    trajectory_state = 10.0**solution.ys
    final_state = trajectory_state[-1]
    final_time = jnp.asarray(end_time)

    state_jacobian, input_jacobian = jax.jacfwd(
        lambda state, controls: physical_derivative(final_time, state, controls, raw_parameters),
        argnums=(0, 1),
    )(final_state, zero_controls)
    output_jacobian, direct_input_jacobian = jax.jacfwd(
        lambda state, controls: observables(final_time, state, controls, raw_parameters),
        argnums=(0, 1),
    )(final_state, zero_controls)
    parameter_state_jacobian = jax.jacfwd(
        lambda parameters: physical_derivative(final_time, final_state, zero_controls, parameters)
    )(raw_parameters)
    parameter_output_jacobian = jax.jacfwd(
        lambda parameters: observables(final_time, final_state, zero_controls, parameters)
    )(raw_parameters)

    def trajectory_endpoint_observables(parameters):
        endpoint = solve(
            parameters,
            zero_controls,
            float(solver_config["rtol"]),
            float(solver_config["atol"]),
            saved_log_times[-1:],
        )
        endpoint_state = 10.0 ** endpoint.ys[-1]
        return observables(final_time, endpoint_state, zero_controls, parameters)

    trajectory_parameter_output_jacobian = jax.jacfwd(trajectory_endpoint_observables)(
        raw_parameters
    )
    state_derivative = physical_derivative(final_time, final_state, zero_controls, raw_parameters)
    observable_values = observables(final_time, final_state, zero_controls, raw_parameters)
    auxiliary_values = auxiliary(final_time, final_state, zero_controls, raw_parameters)
    rate_values = jnp.asarray([auxiliary_values[name] for name in RATE_NAMES])

    local_finite_difference_step = 1.0e-5
    state_jacobian_finite_difference = []
    for index in range(len(STATE_NAMES)):
        offset = (
            jnp.zeros_like(final_state)
            .at[index]
            .set(local_finite_difference_step * final_state[index])
        )
        plus = physical_derivative(final_time, final_state + offset, zero_controls, raw_parameters)
        minus = physical_derivative(final_time, final_state - offset, zero_controls, raw_parameters)
        state_jacobian_finite_difference.append(
            (plus - minus) / (2.0 * local_finite_difference_step * final_state[index])
        )
    state_jacobian_finite_difference = jnp.stack(state_jacobian_finite_difference, axis=1)

    input_jacobian_finite_difference = []
    for index in range(len(FORCING_NAMES)):
        offset = jnp.zeros_like(zero_controls).at[index].set(local_finite_difference_step)
        plus = physical_derivative(final_time, final_state, zero_controls + offset, raw_parameters)
        minus = physical_derivative(final_time, final_state, zero_controls - offset, raw_parameters)
        input_jacobian_finite_difference.append(
            (plus - minus) / (2.0 * local_finite_difference_step)
        )
    input_jacobian_finite_difference = jnp.stack(input_jacobian_finite_difference, axis=1)

    parameter_output_jacobian_finite_difference = []
    for index in range(len(PARAMETER_NAMES)):
        offset = jnp.zeros_like(raw_parameters).at[index].set(local_finite_difference_step)
        plus = observables(final_time, final_state, zero_controls, raw_parameters + offset)
        minus = observables(final_time, final_state, zero_controls, raw_parameters - offset)
        parameter_output_jacobian_finite_difference.append(
            (plus - minus) / (2.0 * local_finite_difference_step)
        )
    parameter_output_jacobian_finite_difference = jnp.stack(
        parameter_output_jacobian_finite_difference, axis=1
    )

    trajectory_finite_difference_steps = jnp.asarray(
        [1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5], dtype=jnp.float64
    )

    def trajectory_finite_difference(step):
        columns = []
        for index in range(len(PARAMETER_NAMES)):
            offset = jnp.zeros_like(raw_parameters).at[index].set(step)
            plus = trajectory_endpoint_observables(raw_parameters + offset)
            minus = trajectory_endpoint_observables(raw_parameters - offset)
            columns.append((plus - minus) / (2.0 * step))
        return jnp.stack(columns, axis=1)

    trajectory_parameter_output_jacobian_finite_difference_scan = jnp.stack(
        tuple(trajectory_finite_difference(step) for step in trajectory_finite_difference_steps)
    )
    trajectory_reference_index = 2
    trajectory_parameter_output_jacobian_finite_difference = (
        trajectory_parameter_output_jacobian_finite_difference_scan[trajectory_reference_index]
    )

    strict_tolerance = min(float(solver_config["rtol"]), 1.0e-10)
    strict_solution = solve(
        raw_parameters,
        zero_controls,
        strict_tolerance,
        strict_tolerance,
        saved_log_times[-1:],
    )
    strict_final_state = 10.0 ** strict_solution.ys[-1]
    convergence_fraction = (final_state - strict_final_state) / strict_final_state

    mass_rate_scale = 1.0e9
    baryon_conservation_residual = (
        jnp.sum(state_derivative[:3])
        - auxiliary_values["Mdot_in_halo"] * mass_rate_scale
        + auxiliary_values["Mdot_out_halo"] * mass_rate_scale
    )
    metal_conservation_residual = (
        jnp.sum(state_derivative[4:])
        - auxiliary_values["MZdot_yield"] * mass_rate_scale
        - auxiliary_values["MZdot_in_halo"] * mass_rate_scale
        + auxiliary_values["MZdot_out_halo"] * mass_rate_scale
    )

    def relative_matrix_error(automatic, finite_difference):
        return jnp.linalg.norm(automatic - finite_difference) / jnp.maximum(
            jnp.linalg.norm(finite_difference), jnp.finfo(jnp.float64).tiny
        )

    trajectory_finite_difference_errors = jnp.asarray(
        [
            relative_matrix_error(trajectory_parameter_output_jacobian, finite_difference)
            for finite_difference in trajectory_parameter_output_jacobian_finite_difference_scan
        ]
    )

    derivative_validation = {
        "state_jacobian_relative_l2_error": float(
            relative_matrix_error(state_jacobian, state_jacobian_finite_difference)
        ),
        "input_jacobian_relative_l2_error": float(
            relative_matrix_error(input_jacobian, input_jacobian_finite_difference)
        ),
        "parameter_output_jacobian_relative_l2_error": float(
            relative_matrix_error(
                parameter_output_jacobian, parameter_output_jacobian_finite_difference
            )
        ),
        "trajectory_parameter_output_jacobian_relative_l2_error": float(
            relative_matrix_error(
                trajectory_parameter_output_jacobian,
                trajectory_parameter_output_jacobian_finite_difference,
            )
        ),
    }

    trajectory_observables = jnp.stack(
        tuple(
            observables(time, state, zero_controls, raw_parameters)
            for time, state in zip(saved_times_gyr, trajectory_state)
        )
    )
    rate_units = {name: ("erg/s" if name.startswith("Edot") else "Msun/yr") for name in RATE_NAMES}
    state_units = ("Msun", "Msun", "Msun", "erg", "Msun", "Msun", "Msun")
    observable_units = ("Msun", "Msun", "Msun", "Msun/yr", "mass fraction", "mass fraction")
    final_redshift = float(auxiliary_values["redshift"])

    arrays = {
        "direct_input_jacobian": np.asarray(direct_input_jacobian),
        "conservation_residuals": np.asarray(
            [baryon_conservation_residual, metal_conservation_residual]
        ),
        "convergence_fraction": np.asarray(convergence_fraction),
        "convergence_strict_state": np.asarray(strict_final_state),
        "forcing_values": np.asarray(forcing),
        "input_jacobian": np.asarray(input_jacobian),
        "input_jacobian_finite_difference": np.asarray(input_jacobian_finite_difference),
        "linearization_state": np.asarray(final_state),
        "observable_values": np.asarray(observable_values),
        "output_jacobian": np.asarray(output_jacobian),
        "parameter_output_jacobian": np.asarray(parameter_output_jacobian),
        "parameter_output_jacobian_finite_difference": np.asarray(
            parameter_output_jacobian_finite_difference
        ),
        "parameter_state_jacobian": np.asarray(parameter_state_jacobian),
        "parameter_values": np.asarray(raw_parameters),
        "rate_values": np.asarray(rate_values),
        "state_derivative": np.asarray(state_derivative),
        "state_jacobian": np.asarray(state_jacobian),
        "state_jacobian_finite_difference": np.asarray(state_jacobian_finite_difference),
        "trajectory_observables": np.asarray(trajectory_observables),
        "trajectory_parameter_output_jacobian": np.asarray(trajectory_parameter_output_jacobian),
        "trajectory_parameter_output_jacobian_finite_difference": np.asarray(
            trajectory_parameter_output_jacobian_finite_difference
        ),
        "trajectory_parameter_output_jacobian_finite_difference_scan": np.asarray(
            trajectory_parameter_output_jacobian_finite_difference_scan
        ),
        "trajectory_parameter_output_jacobian_finite_difference_steps": np.asarray(
            trajectory_finite_difference_steps
        ),
        "trajectory_parameter_output_jacobian_finite_difference_errors": np.asarray(
            trajectory_finite_difference_errors
        ),
        "trajectory_state": np.asarray(trajectory_state),
        "trajectory_times_gyr": np.asarray(saved_times_gyr),
    }
    manifest = {
        "model": {
            "name": "sapphire",
            "label": "Sapphire Pandya23",
            "version": importlib.metadata.version("sapphire-jax"),
            "revision": revision,
            "repository": "https://github.com/virajpandya/sapphire",
            "formulation": "native seven-state Pandya23 continuous central-galaxy model",
        },
        "qualification": (
            "Native Sapphire under constant smooth halo forcing. This controlled experiment "
            "does not represent its TNG/CDHMAH population or add merger topology."
        ),
        "case": case,
        "coordinates": {
            "state": [
                coordinate(name, unit, description)
                for name, unit, description in zip(
                    STATE_NAMES,
                    state_units,
                    (
                        "Long-lived stellar mass.",
                        "Interstellar gas mass.",
                        "Circumgalactic gas mass.",
                        "Circumgalactic thermal energy.",
                        "Stellar metal mass.",
                        "ISM metal mass.",
                        "CGM metal mass.",
                    ),
                )
            ],
            "input": [
                coordinate(
                    name,
                    "fractional forcing change",
                    f"Log-fractional perturbation of native halo input {name}.",
                )
                for name in FORCING_NAMES
            ],
            "parameter": [
                coordinate(
                    name,
                    "native Sapphire coordinate",
                    "Raw configuration derivative; A_* is log10 normalization and slopes are linear.",
                )
                for name in PARAMETER_NAMES
            ],
            "observable": [
                coordinate(name, unit, "Native local Sapphire observable.")
                for name, unit in zip(OBSERVABLE_NAMES, observable_units)
            ],
            "rate": [
                coordinate(name, rate_units[name], "Native auxiliary Pandya23 rate.")
                for name in RATE_NAMES
            ],
        },
        "linearization_point": {
            "time_gyr": end_time,
            "redshift": final_redshift,
            "halo_mass_msun": float(forcing[1]),
            "forcing_assumption": "constant physical halo coordinates over the integration interval",
        },
        "solver": {
            "engine": "diffrax",
            "method": "Tsit5",
            "step_control": "PIDController",
            "rtol": float(solver_config["rtol"]),
            "atol": float(solver_config["atol"]),
            "max_steps": int(solver_config["max_steps"]),
            "result": str(solution.result),
            "statistics": {name: int(value) for name, value in solution.stats.items()},
        },
        "derivatives": {
            "method": "native Sapphire JAX jacfwd",
            "local_method": "jax.jacfwd through the native Pandya23 RHS and observables",
            "trajectory_method": (
                "jax.jacfwd through the native Diffrax Tsit5 adaptive trajectory and "
                "endpoint observables"
            ),
            "state_coordinates": "physical rather than Sapphire's internal log10 coordinates",
            "input_coordinates": "natural-log fractional halo-forcing perturbations",
            "parameter_coordinates": "raw Sapphire configuration coordinates",
            "local_parameter_output_scope": (
                "instantaneous observable derivative at fixed final state and forcing"
            ),
            "trajectory_parameter_output_scope": (
                "end-to-end derivative of final observables through the native adaptive solve"
            ),
            "local_finite_difference_step": local_finite_difference_step,
            "trajectory_finite_difference_steps": [
                float(step) for step in trajectory_finite_difference_steps
            ],
            "trajectory_finite_difference_reference_step": float(
                trajectory_finite_difference_steps[trajectory_reference_index]
            ),
            "validation": derivative_validation,
        },
        "conservation": {
            "coordinates": [
                {
                    "name": "baryons",
                    "unit": "Msun/Gyr",
                    "boundary": "Mdot_in_halo source and Mdot_out_halo sink",
                },
                {
                    "name": "metals",
                    "unit": "Msun/Gyr",
                    "boundary": "stellar yield plus enriched inflow, minus enriched CGM outflow",
                },
            ],
            "qualification": "Open-system budgets reconstructed from native auxiliary rates.",
        },
        "convergence": {
            "comparison_tolerance": strict_tolerance,
            "quantity": "fractional difference of requested-tolerance final state from stricter solve",
            "maximum_absolute_fractional_difference": float(jnp.max(jnp.abs(convergence_fraction))),
        },
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "source_revision": revision,
            "source_dirty": bool(git_output(source, "status", "--porcelain")),
            "case_sha256": file_hash(case_path),
            "cooling_table": {
                "path": str((data / "coolfunc" / "newcool.dat").resolve()),
                "sha256": file_hash(data / "coolfunc" / "newcool.dat"),
                "release": "v0.130",
                "release_asset": "sapphire-data.tar.gz",
                "source_url": (
                    "https://github.com/virajpandya/sapphire/releases/download/"
                    "v0.130/sapphire-data.tar.gz"
                ),
            },
            "software": {
                "python": platform.python_version(),
                "jax": jax.__version__,
                "diffrax": importlib.metadata.version("diffrax"),
                "sapphire-jax": importlib.metadata.version("sapphire-jax"),
                "numpy": np.__version__,
            },
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in jax.devices()],
            },
        },
    }
    write_artifact(output, manifest, arrays)


if __name__ == "__main__":
    main()
