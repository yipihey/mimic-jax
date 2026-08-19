#!/usr/bin/env python3
"""Measure local Laplace/linear responses along fiducial Mini-Millennium SAGE16 histories."""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from mimic_jax import (  # noqa: E402
    RK4,
    frequency_response,
    integrate_fixed_step,
    linearize_state_space,
    step_response,
)
from mimic_jax.io import open_lhalo_partition  # noqa: E402
from mimic_jax.sage16 import (  # noqa: E402
    ODE_STATE_NAMES,
    evolve_lhalo_partition,
    fiducial_parameters,
    hybrid_state_from_galaxy,
    load_cooling_tables,
    load_scale_factors,
    ode_state_from_galaxy,
    process_perturbations,
    sage16_hybrid_rhs_and_rates,
    sage16_ode_rhs_and_rates,
    sage16_units,
    snapshot_timing,
)
from mimic_jax.sage16.ode import calculate_continuous_cooling_rate  # noqa: E402

SECONDS_PER_GYR = 365.25 * 24.0 * 3600.0 * 1.0e9
HYBRID_DYNAMIC_NAMES = ODE_STATE_NAMES + ("BlackHoleMass",)
RESPONSE_STATE_NAMES = ODE_STATE_NAMES[:4]
DEFAULT_SNAPSHOTS = (20, 28, 36, 44, 52, 58, 63)


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
    parser.add_argument("--tree-count", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-members", type=int, default=512)
    parser.add_argument("--snapshots", type=int, nargs="+", default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--pulse-steps", type=int, default=512)
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


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def physical_mass(code_mass, hubble_h):
    return np.asarray(code_mass) * 1.0e10 / hubble_h


def stack_pytrees(values):
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *values)


def stratified_tree_indices(partition, count):
    """Mix uniform index coverage with the largest trees needed for massive haloes."""

    uniform_count = max(1, count // 2)
    uniform = np.linspace(0, partition.tree_count - 1, uniform_count, dtype=np.int32)
    largest = np.argsort(partition.tree_halo_counts)[::-1]
    selected = list(int(value) for value in uniform)
    for value in largest:
        candidate = int(value)
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) == count:
            break
    return tuple(sorted(selected))


def ode_local_model(
    record,
    parameters,
    units,
    tables,
    *,
    output_names=("ColdGas", "star_formation"),
    background_perturbations=None,
):
    state = ode_state_from_galaxy(record.state)
    control = jnp.zeros((1,), dtype=jnp.float64)
    background_perturbations = dict(background_perturbations or {})

    def perturbation(value):
        return process_perturbations(cooling=value[0], **background_perturbations)

    def rhs(current, value):
        return sage16_ode_rhs_and_rates(
            0.0,
            current,
            record.halo,
            record.state.DiskScaleRadius,
            parameters,
            units,
            tables,
            perturbation(value),
        ).derivative

    def output(current, value):
        result = sage16_ode_rhs_and_rates(
            0.0,
            current,
            record.halo,
            record.state.DiskScaleRadius,
            parameters,
            units,
            tables,
            perturbation(value),
        )
        values = []
        for name in output_names:
            if name in ODE_STATE_NAMES:
                values.append(getattr(current, name))
            else:
                values.append(getattr(result.rates, name))
        return jnp.stack(values)

    model = linearize_state_space(rhs, output, state, control)
    baseline = output(state, control)
    rates = sage16_ode_rhs_and_rates(
        0.0,
        state,
        record.halo,
        record.state.DiskScaleRadius,
        parameters,
        units,
        tables,
    ).rates
    return state, model, baseline, rates


def physical_time_model(model, time_unit_gyr):
    return model._replace(
        state_jacobian=model.state_jacobian / time_unit_gyr,
        input_jacobian=model.input_jacobian / time_unit_gyr,
    )


def modal_diagnostics(model, input_index=0, output_index=0):
    matrix = np.asarray(model.state_jacobian)
    input_vector = np.asarray(model.input_jacobian)[:, input_index]
    output_vector = np.asarray(model.output_jacobian)[output_index]
    eigenvalues, right = np.linalg.eig(matrix)
    try:
        left = np.linalg.inv(right)
    except np.linalg.LinAlgError:
        return eigenvalues, np.full(eigenvalues.shape, np.nan), right
    residues = np.asarray(
        [
            (output_vector @ right[:, index]) * (left[index] @ input_vector)
            for index in range(len(eigenvalues))
        ]
    )
    return eigenvalues, residues, right


def dominant_stable_mode(model, *, input_index=0, output_index=0):
    eigenvalues, residues, right = modal_diagnostics(model, input_index, output_index)
    stable = np.real(eigenvalues) < -1.0e-8
    finite = stable & np.isfinite(residues) & (np.abs(residues) > 0.0)
    if not np.any(finite):
        return None
    candidates = np.flatnonzero(finite)
    selected = candidates[int(np.argmax(np.abs(residues[candidates])))]
    return {
        "index": int(selected),
        "pole": eigenvalues[selected],
        "residue": residues[selected],
        "timescale": -1.0 / np.real(eigenvalues[selected]),
        "eigenvector": right[:, selected],
        "all_poles": eigenvalues,
        "all_residues": residues,
    }


def record_is_candidate(record):
    return (
        int(record.halo.Type) == 0
        and float(record.halo.Mvir) > 0.0
        and float(record.halo.Rvir) > 0.0
        and float(record.halo.Vvir) > 0.0
        and float(record.state.HotGas) > 0.0
        and float(record.state.ColdGas) > 0.0
        and float(record.state.DiskScaleRadius) > 0.0
    )


def select_response_records(evolved, units, maximum_per_mass_bin=24):
    mass_edges = np.asarray([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 14.5])
    selected = []
    for snapshot in sorted(evolved.records_by_snapshot):
        candidates = [
            record
            for record in evolved.records_by_snapshot[snapshot]
            if record_is_candidate(record)
        ]
        masses = np.asarray(
            [np.log10(physical_mass(record.halo.Mvir, units.Hubble_h)) for record in candidates]
        )
        for lower, upper in zip(mass_edges[:-1], mass_edges[1:]):
            indices = np.flatnonzero((masses >= lower) & (masses < upper))
            if len(indices) > maximum_per_mass_bin:
                indices = indices[
                    np.linspace(0, len(indices) - 1, maximum_per_mass_bin, dtype=np.int32)
                ]
            selected.extend((snapshot, candidates[int(index)]) for index in indices)
    return selected


def collect_local_measurements(evolved, timing, parameters, units, tables, time_unit_gyr):
    selected = select_response_records(evolved, units)
    states = stack_pytrees([ode_state_from_galaxy(record.state) for _, record in selected])
    halos = stack_pytrees([record.halo for _, record in selected])
    radii = jnp.asarray([record.state.DiskScaleRadius for _, record in selected], dtype=jnp.float64)
    control = jnp.zeros((1,), dtype=jnp.float64)

    def one_model(state, halo, radius):
        def rhs(current, value):
            return sage16_ode_rhs_and_rates(
                0.0,
                current,
                halo,
                radius,
                parameters,
                units,
                tables,
                process_perturbations(cooling=value[0]),
            ).derivative

        def output(current, value):
            result = sage16_ode_rhs_and_rates(
                0.0,
                current,
                halo,
                radius,
                parameters,
                units,
                tables,
                process_perturbations(cooling=value[0]),
            )
            return jnp.stack((current.ColdGas, result.rates.star_formation))

        model = linearize_state_space(rhs, output, state, control)
        baseline = output(state, control)
        rates = sage16_ode_rhs_and_rates(0.0, state, halo, radius, parameters, units, tables).rates
        return model, baseline, rates

    models, baselines, rates = jax.jit(jax.vmap(one_model))(states, halos, radii)
    jax.block_until_ready(models.state_jacobian)
    measurements = []
    for index, (snapshot, record) in enumerate(selected):
        model = jax.tree_util.tree_map(lambda value: value[index], models)
        physical = physical_time_model(model, time_unit_gyr)
        mode = dominant_stable_mode(physical, output_index=0)
        if mode is None or not np.isfinite(mode["timescale"]):
            continue
        measurements.append(
            {
                "record": record,
                "snapshot": snapshot,
                "redshift": float(timing.redshift[snapshot]),
                "log_halo_mass": float(np.log10(physical_mass(record.halo.Mvir, units.Hubble_h))),
                "log_stellar_mass": (
                    float(np.log10(physical_mass(record.state.StellarMass, units.Hubble_h)))
                    if float(record.state.StellarMass) > 0.0
                    else np.nan
                ),
                "timescale": float(mode["timescale"]),
                "cooling": float(rates.cooling[index]),
                "star_formation": float(rates.star_formation[index]),
                "model": physical,
                "baseline": np.asarray(baselines[index]),
                "mode": mode,
            }
        )
    return measurements


def choose_pulse_case(measurements):
    eligible = [
        item
        for item in measurements
        if item["star_formation"] > 0.0
        and item["cooling"] > 0.0
        and 0.08 <= item["timescale"] <= 3.0
        and 9.5 <= item["log_stellar_mass"] <= 10.8
    ]
    if not eligible:
        raise RuntimeError(
            "No active smooth central was available for the cooling-pulse experiment"
        )
    eligible.sort(
        key=lambda item: (abs(item["redshift"] - 0.5), abs(item["log_stellar_mass"] - 10.2))
    )
    return eligible[0]


def state_series_matrix(states):
    return np.stack([np.asarray(getattr(states, name)) for name in ODE_STATE_NAMES], axis=-1)


def cooling_pulse_experiment(case, parameters, units, tables, time_unit_gyr, pulse_steps):
    record = case["record"]
    initial = ode_state_from_galaxy(record.state)
    model = case["model"]
    response_time = case["timescale"]
    duration_gyr = float(np.clip(4.0 * response_time, 0.35, 3.0))
    pulse_duration_gyr = float(np.clip(0.35 * response_time, 0.04, 0.35))
    duration = duration_gyr / time_unit_gyr
    pulse_duration = pulse_duration_gyr / time_unit_gyr
    amplitudes = np.asarray([0.001, 0.01, 0.05], dtype=np.float64)

    def nonlinear(epsilon):
        def rhs(current_time, state):
            active = jnp.where(current_time < pulse_duration, epsilon, 0.0)
            return sage16_ode_rhs_and_rates(
                current_time,
                state,
                record.halo,
                record.state.DiskScaleRadius,
                parameters,
                units,
                tables,
                process_perturbations(cooling=active),
            ).derivative

        return integrate_fixed_step(
            rhs,
            initial,
            duration=duration,
            num_steps=pulse_steps,
            method=RK4,
        )

    baseline_solution = nonlinear(0.0)
    baseline_states = baseline_solution.states
    baseline_sfr = jax.vmap(
        lambda state: sage16_ode_rhs_and_rates(
            0.0,
            state,
            record.halo,
            record.state.DiskScaleRadius,
            parameters,
            units,
            tables,
        ).rates.star_formation
    )(baseline_states)
    baseline_cold = np.asarray(baseline_states.ColdGas)
    baseline_sfr = np.asarray(baseline_sfr)
    times_gyr = np.asarray(baseline_solution.times) * time_unit_gyr
    unit_step = np.asarray(step_response(model, jnp.asarray(times_gyr)))[:, :, 0]
    delayed_times = np.maximum(times_gyr - pulse_duration_gyr, 0.0)
    delayed_step = np.array(step_response(model, jnp.asarray(delayed_times)))[:, :, 0]
    delayed_step[times_gyr < pulse_duration_gyr] = 0.0

    nonlinear_cold = []
    nonlinear_sfr = []
    linear_cold = []
    linear_sfr = []
    errors = []
    for amplitude in amplitudes:
        epsilon = float(np.log1p(amplitude))
        perturbed = nonlinear(epsilon)
        perturbed_sfr = jax.vmap(
            lambda state: sage16_ode_rhs_and_rates(
                0.0,
                state,
                record.halo,
                record.state.DiskScaleRadius,
                parameters,
                units,
                tables,
            ).rates.star_formation
        )(perturbed.states)

        nonlinear_cold_fraction = (
            np.asarray(perturbed.states.ColdGas) - baseline_cold
        ) / baseline_cold
        nonlinear_sfr_fraction = (np.asarray(perturbed_sfr) - baseline_sfr) / baseline_sfr
        linear_output = epsilon * (unit_step - delayed_step)
        linear_cold_fraction = linear_output[:, 0] / baseline_cold
        linear_sfr_fraction = linear_output[:, 1] / baseline_sfr
        nonlinear_cold.append(nonlinear_cold_fraction)
        nonlinear_sfr.append(nonlinear_sfr_fraction)
        linear_cold.append(linear_cold_fraction)
        linear_sfr.append(linear_sfr_fraction)
        comparisons = []
        for measured, predicted in (
            (nonlinear_cold_fraction, linear_cold_fraction),
            (nonlinear_sfr_fraction, linear_sfr_fraction),
        ):
            scale = max(float(np.max(np.abs(measured))), 1.0e-12)
            comparisons.append(float(np.sqrt(np.mean((measured - predicted) ** 2)) / scale))
        errors.append(comparisons)

    return {
        "times_gyr": times_gyr,
        "amplitudes": amplitudes,
        "nonlinear_cold_fraction": np.asarray(nonlinear_cold),
        "nonlinear_sfr_fraction": np.asarray(nonlinear_sfr),
        "linear_cold_fraction": np.asarray(linear_cold),
        "linear_sfr_fraction": np.asarray(linear_sfr),
        "normalized_rmse": np.asarray(errors),
        "duration_gyr": duration_gyr,
        "pulse_duration_gyr": pulse_duration_gyr,
    }


def frequency_and_modes(case, parameters, units, tables, time_unit_gyr):
    model = case["model"]
    timescales = np.logspace(-2.0, 1.5, 240)
    omega = 1.0 / timescales
    response = np.asarray(frequency_response(model, omega))[:, :, 0]
    baseline = case["baseline"]
    fractional = response / baseline[None, :]

    record = case["record"]
    reff = float(parameters.StarFormingDiskFactor * record.state.DiskScaleRadius)
    tau_star = reff / float(record.halo.Vvir) / float(parameters.SfrEfficiency) * time_unit_gyr
    tau_equilibrium = tau_star / float(
        1.0 - parameters.RecycleFraction + parameters.FeedbackReheatingEpsilon
    )
    cooling_time = float(record.state.HotGas) / case["cooling"] * time_unit_gyr
    reincorporation_rate = sage16_ode_rhs_and_rates(
        0.0,
        ode_state_from_galaxy(record.state),
        record.halo,
        record.state.DiskScaleRadius,
        parameters,
        units,
        tables,
    ).rates.reincorporation
    reincorporation_time = (
        float(record.state.EjectedGas / reincorporation_rate) * time_unit_gyr
        if float(reincorporation_rate) > 0.0
        else np.nan
    )

    eigenvalues, residues, right = modal_diagnostics(model, output_index=1)
    stable = np.flatnonzero((np.real(eigenvalues) < -1.0e-8) & np.isfinite(residues))
    stable = stable[np.argsort(np.abs(residues[stable]))[::-1]]
    selected = []
    maximum_residue = np.max(np.abs(residues[stable])) if len(stable) else 0.0
    for index in stable:
        if np.abs(residues[index]) < 1.0e-6 * maximum_residue:
            continue
        conjugate_already_selected = any(
            np.isclose(eigenvalues[index], np.conj(eigenvalues[other]), rtol=1.0e-7, atol=1.0e-10)
            for other in selected
        )
        if not conjugate_already_selected:
            selected.append(int(index))
        if len(selected) == 4:
            break
    selected = np.asarray(selected, dtype=np.int32)
    mode_times = -1.0 / np.real(eigenvalues[selected])
    compositions = []
    for index in selected:
        vector = np.abs(right[:4, index])
        compositions.append(vector / np.sum(vector) if np.sum(vector) > 0.0 else vector)
    variants = (
        ("fiducial", {}),
        ("SN feedback locally removed", {"sn_reheating": -50.0, "sn_ejection": -50.0}),
        ("reincorporation locally removed", {"reincorporation": -50.0}),
    )
    variant_response = []
    variant_mode_times = []
    for _, background in variants:
        _, variant_model, variant_baseline, _ = ode_local_model(
            record,
            parameters,
            units,
            tables,
            background_perturbations=background,
        )
        variant_model = physical_time_model(variant_model, time_unit_gyr)
        variant_frequency = np.asarray(frequency_response(variant_model, omega))[:, :, 0]
        variant_response.append(variant_frequency[:, 1] / float(variant_baseline[1]))
        variant_mode = dominant_stable_mode(variant_model, output_index=1)
        variant_mode_times.append(variant_mode["timescale"] if variant_mode is not None else np.nan)
    return {
        "inverse_angular_frequency_gyr": timescales,
        "frequency_response": fractional,
        "poles_per_gyr": eigenvalues,
        "residues": residues,
        "selected_mode_indices": selected,
        "selected_mode_poles_per_gyr": eigenvalues[selected],
        "selected_mode_times_gyr": mode_times,
        "selected_mode_composition": np.asarray(compositions),
        "recipe_names": np.asarray(
            ["cooling depletion", "star formation", "cold-gas regulator", "reincorporation"]
        ),
        "recipe_times_gyr": np.asarray(
            [cooling_time, tau_star, tau_equilibrium, reincorporation_time]
        ),
        "feedback_variant_names": np.asarray([name for name, _ in variants]),
        "feedback_variant_response": np.asarray(variant_response),
        "feedback_variant_mode_times_gyr": np.asarray(variant_mode_times),
    }


def binned_map(measurements, value_name, *, minimum_count=3):
    mass_edges = np.asarray([10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0])
    redshifts = np.asarray(sorted({item["redshift"] for item in measurements}), dtype=np.float64)
    values = np.full((len(redshifts), len(mass_edges) - 1), np.nan)
    counts = np.zeros_like(values, dtype=np.int32)
    for redshift_index, redshift in enumerate(redshifts):
        at_redshift = [item for item in measurements if item["redshift"] == redshift]
        masses = np.asarray([item["log_halo_mass"] for item in at_redshift])
        measured = np.asarray([item[value_name] for item in at_redshift])
        for mass_index, (lower, upper) in enumerate(zip(mass_edges[:-1], mass_edges[1:])):
            selected = (masses >= lower) & (masses < upper) & np.isfinite(measured)
            counts[redshift_index, mass_index] = np.count_nonzero(selected)
            if counts[redshift_index, mass_index] >= minimum_count:
                values[redshift_index, mass_index] = np.median(measured[selected])
    return mass_edges, redshifts, values, counts


def hybrid_reduced_model(record, parameters, units, tables):
    template = hybrid_state_from_galaxy(record.state)
    initial = jnp.stack([getattr(template, name) for name in HYBRID_DYNAMIC_NAMES])
    control = jnp.zeros((1,), dtype=jnp.float64)

    def unpack(vector):
        return template._replace(
            **{name: vector[index] for index, name in enumerate(HYBRID_DYNAMIC_NAMES)}
        )

    def evaluate(vector, epsilon):
        return sage16_hybrid_rhs_and_rates(
            0.0,
            unpack(vector),
            record.halo,
            parameters,
            units,
            tables,
            perturbations=process_perturbations(cooling=epsilon[0]),
        )

    def rhs(vector, epsilon):
        derivative = evaluate(vector, epsilon).derivative
        return jnp.stack([getattr(derivative, name) for name in HYBRID_DYNAMIC_NAMES])

    def output(vector, epsilon):
        result = evaluate(vector, epsilon)
        return jnp.stack((vector[0], result.rates.star_formation))

    model = linearize_state_space(rhs, output, initial, control)
    result = evaluate(initial, control)
    baseline = output(initial, control)
    return model, baseline, result.rates


def agn_measurements(measurements, parameters, units, tables, time_unit_gyr):
    records = [item["record"] for item in measurements]
    hybrid_states = stack_pytrees([hybrid_state_from_galaxy(record.state) for record in records])
    ode_states = stack_pytrees([ode_state_from_galaxy(record.state) for record in records])
    halos = stack_pytrees([record.halo for record in records])

    def rates_one(state, halo):
        return sage16_hybrid_rhs_and_rates(0.0, state, halo, parameters, units, tables).rates

    def raw_cooling_one(state, halo):
        return calculate_continuous_cooling_rate(state, halo, units, tables)[0]

    batched_rates = jax.jit(jax.vmap(rates_one))(hybrid_states, halos)
    raw_cooling = jax.jit(jax.vmap(raw_cooling_one))(ode_states, halos)
    jax.block_until_ready(batched_rates.cooling)
    candidates = []
    suppression_values = []
    for index, item in enumerate(measurements):
        record = item["record"]
        rates = jax.tree_util.tree_map(lambda value: value[index], batched_rates)
        raw = float(raw_cooling[index])
        suppression = 1.0 - float(rates.cooling) / raw if raw > 0.0 else np.nan
        suppression_values.append(suppression)
        if (
            float(record.state.Rheat) > 0.0
            and item["star_formation"] > 0.0
            and float(rates.agn_heating) > 0.0
            and np.isfinite(suppression)
        ):
            candidates.append((item, rates, suppression))
    if not candidates:
        raise RuntimeError("No active SAGE16 central with stored radio-mode heating was found")
    strongly_suppressed = [value for value in candidates if value[2] >= 0.25]
    selection = strongly_suppressed if strongly_suppressed else candidates
    selection.sort(
        key=lambda value: (value[0]["log_stellar_mass"], value[0]["log_halo_mass"]),
        reverse=True,
    )
    selected_item, rates, suppression = selection[0]
    fiducial_model, fiducial_baseline, _ = hybrid_reduced_model(
        selected_item["record"], parameters, units, tables
    )
    fiducial_baseline = np.asarray(fiducial_baseline)
    no_agn_parameters = parameters._replace(AGNrecipe=jnp.asarray(0, dtype=jnp.int32))
    no_agn_model, no_agn_baseline, _ = hybrid_reduced_model(
        selected_item["record"], no_agn_parameters, units, tables
    )
    fiducial_model = physical_time_model(fiducial_model, time_unit_gyr)
    no_agn_model = physical_time_model(no_agn_model, time_unit_gyr)
    timescales = np.logspace(-2.0, 1.5, 240)
    omega = 1.0 / timescales
    fiducial_response = np.asarray(frequency_response(fiducial_model, omega))[:, :, 0]
    no_agn_response = np.asarray(frequency_response(no_agn_model, omega))[:, :, 0]
    fiducial_fractional = fiducial_response / fiducial_baseline[None, :]
    no_agn_fractional = no_agn_response / no_agn_baseline[None, :]

    suppression_measurements = []
    for item, value in zip(measurements, suppression_values):
        copy = dict(item)
        copy["agn_suppression"] = value
        suppression_measurements.append(copy)
    mass_edges, redshifts, suppression_map, counts = binned_map(
        suppression_measurements, "agn_suppression"
    )
    return {
        "inverse_angular_frequency_gyr": timescales,
        "fiducial_fractional_response": fiducial_fractional,
        "locally_removed_fractional_response": no_agn_fractional,
        "selected_suppression": suppression,
        "selected_log_halo_mass": selected_item["log_halo_mass"],
        "selected_log_stellar_mass": selected_item["log_stellar_mass"],
        "selected_redshift": selected_item["redshift"],
        "selected_rheat": float(selected_item["record"].state.Rheat),
        "selected_agn_heating": float(rates.agn_heating),
        "mass_edges": mass_edges,
        "redshifts": redshifts,
        "suppression_map": suppression_map,
        "map_counts": counts,
    }


def main():
    arguments = parse_arguments()
    arguments.compilation_cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(arguments.compilation_cache_dir.resolve()))
    partition = open_lhalo_partition(arguments.trees)
    if not 0 < arguments.tree_count <= partition.tree_count:
        raise SystemExit("tree_count must be within the input partition")
    snapshots = tuple(sorted(set(arguments.snapshots)))
    tree_indices = stratified_tree_indices(partition, arguments.tree_count)
    timing = snapshot_timing(load_scale_factors(arguments.scale_factors))
    parameters = fiducial_parameters()
    units = sage16_units()
    tables = load_cooling_tables()
    time_unit_gyr = float(units.UnitTime_in_s) / SECONDS_PER_GYR

    started = time.perf_counter()
    print(
        f"[linear-response] evolving {len(tree_indices)} trees at snapshots {snapshots}", flush=True
    )
    evolved = evolve_lhalo_partition(
        partition,
        timing,
        tree_indices=tree_indices,
        global_tree_offset=arguments.global_tree_offset,
        num_substeps=10,
        output_snapshots=snapshots,
        batch_size=arguments.batch_size,
        max_batch_members=arguments.max_batch_members,
        member_binning="power_of_two",
    )
    if not evolved.success:
        raise SystemExit("Mini-Millennium evolution reported failure")
    evolution_seconds = time.perf_counter() - started

    print("[linear-response] linearizing smooth fixed-forcing states", flush=True)
    measurements = collect_local_measurements(
        evolved, timing, parameters, units, tables, time_unit_gyr
    )
    if len(measurements) < 20:
        raise RuntimeError("Too few valid local response measurements")
    pulse_case = choose_pulse_case(measurements)
    pulse = cooling_pulse_experiment(
        pulse_case,
        parameters,
        units,
        tables,
        time_unit_gyr,
        arguments.pulse_steps,
    )
    local = frequency_and_modes(pulse_case, parameters, units, tables, time_unit_gyr)
    mass_edges, redshifts, memory_map, memory_counts = binned_map(measurements, "timescale")
    print("[linear-response] measuring the hybrid AGN cooling response", flush=True)
    agn = agn_measurements(measurements, parameters, units, tables, time_unit_gyr)
    total_seconds = time.perf_counter() - started

    one_percent_index = int(np.flatnonzero(pulse["amplitudes"] == 0.01)[0])
    one_percent_error = float(np.max(pulse["normalized_rmse"][one_percent_index]))
    stable_poles = np.real(local["poles_per_gyr"]) < -1.0e-8
    neutral_poles = np.abs(np.real(local["poles_per_gyr"])) <= 1.0e-8
    summary = {
        "analysis": "local_frozen_coefficient_linear_response",
        "model": "fiducial SAGE16",
        "dataset": "Mini-Millennium input partition 1",
        "tree_file": str(arguments.trees),
        "tree_count": len(tree_indices),
        "tree_indices": tree_indices,
        "tree_sampling": "half uniform in tree index, half largest remaining trees by halo count",
        "snapshots": snapshots,
        "local_state_count": len(measurements),
        "time_unit_gyr": time_unit_gyr,
        "evolution_seconds": evolution_seconds,
        "total_analysis_seconds": total_seconds,
        "maximum_resident_bytes": maximum_resident_bytes(),
        "linear_validation": {
            "amplitudes": pulse["amplitudes"],
            "normalized_rmse": pulse["normalized_rmse"],
            "acceptance_criterion": "maximum cold-gas/SFR normalized RMSE <= 0.05 at 1%",
            "one_percent_maximum_normalized_rmse": one_percent_error,
            "passed": one_percent_error <= 0.05,
        },
        "pulse_case": {
            "unique_galaxy_id": int(pulse_case["record"].halo.UniqueGalaxyID),
            "snapshot": pulse_case["snapshot"],
            "redshift": pulse_case["redshift"],
            "log_halo_mass_msun": pulse_case["log_halo_mass"],
            "log_stellar_mass_msun": pulse_case["log_stellar_mass"],
            "dominant_cooling_memory_gyr": pulse_case["timescale"],
            "pulse_duration_gyr": pulse["pulse_duration_gyr"],
            "experiment_duration_gyr": pulse["duration_gyr"],
        },
        "local_modes": {
            "stable_pole_count": int(np.count_nonzero(stable_poles)),
            "neutral_pole_count": int(np.count_nonzero(neutral_poles)),
            "selected_response_times_gyr": local["selected_mode_times_gyr"],
            "recipe_times_gyr": dict(
                zip(local["recipe_names"].tolist(), local["recipe_times_gyr"].tolist())
            ),
            "feedback_variant_mode_times_gyr": dict(
                zip(
                    local["feedback_variant_names"].tolist(),
                    local["feedback_variant_mode_times_gyr"].tolist(),
                )
            ),
        },
        "agn_case": {
            "redshift": agn["selected_redshift"],
            "log_halo_mass_msun": agn["selected_log_halo_mass"],
            "log_stellar_mass_msun": agn["selected_log_stellar_mass"],
            "instantaneous_cooling_suppression": agn["selected_suppression"],
            "rheat_code_length": agn["selected_rheat"],
            "agn_heating_code_mass_per_time": agn["selected_agn_heating"],
            "selection": "largest stellar mass among active centrals with at least 25% prior-heating cooling suppression",
        },
        "scope": {
            "continuous_subset": list(ODE_STATE_NAMES),
            "forcing": "halo and disk properties frozen at each sampled SAGE trajectory point",
            "events": "mergers, topology changes, thresholds crossed during an experiment, and Rheat projection are outside each local LTI model",
            "agn_comparison": "local fiducial-background comparison with AGNrecipe set to zero; not a rerun of the nonlinear history",
        },
    }
    arrays = {
        "pulse_times_gyr": pulse["times_gyr"],
        "pulse_amplitudes": pulse["amplitudes"],
        "pulse_nonlinear_cold_fraction": pulse["nonlinear_cold_fraction"],
        "pulse_nonlinear_sfr_fraction": pulse["nonlinear_sfr_fraction"],
        "pulse_linear_cold_fraction": pulse["linear_cold_fraction"],
        "pulse_linear_sfr_fraction": pulse["linear_sfr_fraction"],
        "pulse_normalized_rmse": pulse["normalized_rmse"],
        "inverse_angular_frequency_gyr": local["inverse_angular_frequency_gyr"],
        "fractional_frequency_response": local["frequency_response"],
        "local_poles_per_gyr": local["poles_per_gyr"],
        "local_residues": local["residues"],
        "selected_mode_indices": local["selected_mode_indices"],
        "selected_mode_poles_per_gyr": local["selected_mode_poles_per_gyr"],
        "selected_mode_times_gyr": local["selected_mode_times_gyr"],
        "selected_mode_composition": local["selected_mode_composition"],
        "response_state_names": np.asarray(RESPONSE_STATE_NAMES),
        "recipe_names": local["recipe_names"],
        "recipe_times_gyr": local["recipe_times_gyr"],
        "feedback_variant_names": local["feedback_variant_names"],
        "feedback_variant_response": local["feedback_variant_response"],
        "feedback_variant_mode_times_gyr": local["feedback_variant_mode_times_gyr"],
        "memory_mass_edges": mass_edges,
        "memory_redshifts": redshifts,
        "memory_timescale_gyr": memory_map,
        "memory_counts": memory_counts,
        "agn_inverse_angular_frequency_gyr": agn["inverse_angular_frequency_gyr"],
        "agn_fiducial_fractional_response": agn["fiducial_fractional_response"],
        "agn_removed_fractional_response": agn["locally_removed_fractional_response"],
        "agn_mass_edges": agn["mass_edges"],
        "agn_redshifts": agn["redshifts"],
        "agn_suppression": agn["suppression_map"],
        "agn_counts": agn["map_counts"],
        "sample_log_halo_mass": np.asarray([item["log_halo_mass"] for item in measurements]),
        "sample_redshift": np.asarray([item["redshift"] for item in measurements]),
        "sample_memory_timescale_gyr": np.asarray([item["timescale"] for item in measurements]),
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(arguments.output_arrays, **arrays)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
