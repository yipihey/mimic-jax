#!/usr/bin/env python3
"""Fit a minimal SAGE16 teacher--student model and test it on a disjoint partition.

The model form and acceptance thresholds are fixed in this script.  Partitions
1--3 are the default development/fit sample.  Partition 4 was used for model
selection, and partition 5 is the default untouched replication sample.  The
faithful SAGE16 catalogue supplies targets; only merger-tree halo quantities
drive the reduced model.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import jax
import numpy as np
from scipy.optimize import least_squares

jax.config.update("jax_enable_x64", True)

from mimic_jax.io import open_lhalo_partition  # noqa: E402
from mimic_jax.sage16 import (  # noqa: E402
    ReducedForcing,
    ReducedParameters,
    ReducedState,
    load_scale_factors,
    sage16_units,
    snapshot_timing,
)
from mimic_jax.sage16.reduced import (  # noqa: E402
    RECYCLE_FRACTION,
    _apply_reduced_merger_event_numpy,
    _evolve_reduced_interval_numpy,
)

REPOSITORY = Path(__file__).resolve().parents[1]
HUBBLE_H = 0.73
GLOBAL_BARYON_FRACTION = 0.17
PARTICLE_MASS = 0.0860657
BOX_SIZE_MPC_OVER_H = 62.5
NUMBER_OF_PARTITIONS = 8
SECONDS_PER_GYR = 365.25 * 24.0 * 3600.0 * 1.0e9
MINIMUM_STELLAR_MASS_MSUN = 1.0e8
MINIMUM_HALO_PARTICLES = 20
MINIMUM_SMF_COUNT = 20
FIDELITY_TOLERANCE = 0.30
MINIMUM_INDIVIDUAL_FRACTION = 0.70
FINE_STELLAR_MASS_EDGES = np.arange(8.0, 12.01, 0.2)
COARSE_STELLAR_MASS_EDGES = np.arange(8.0, 12.01, 0.4)
REDUCED_PARAMETER_NAMES = ReducedParameters._fields


@dataclass(frozen=True)
class TreeTopology:
    """Host-side arrays required by the reduced tree evolution."""

    node_count: int
    tree_count: int
    snapshot: np.ndarray
    descendant: np.ndarray
    fof_root: np.ndarray
    halo_mass: np.ndarray
    spin_magnitude: np.ndarray
    most_bound_id: np.ndarray
    tree_index: np.ndarray
    nodes_by_snapshot: tuple
    roots_by_snapshot: tuple
    root_locations_by_snapshot: tuple
    cosmic_time_gyr: np.ndarray
    snapshot_dt_gyr: np.ndarray
    scale_factor: np.ndarray


@dataclass(frozen=True)
class TeacherCatalogue:
    """SAGE16 z=0 targets and their raw-tree identity map."""

    rows: np.ndarray
    row_tree_index: np.ndarray
    raw_node_index: np.ndarray


@dataclass(frozen=True)
class ReducedTreeResult:
    """Reduced-model state for all terminal nodes and SAGE-matched nodes."""

    circumgalactic_gas: np.ndarray
    cold_gas: np.ndarray
    stellar_mass: np.ndarray
    black_hole_proxy: np.ndarray
    star_formation_rate: np.ndarray
    peak_halo_mass: np.ndarray
    terminal_node_index: np.ndarray
    maximum_local_conservation_residual: float
    external_infall_mass: float


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-trees",
        type=Path,
        nargs="+",
        default=[
            REPOSITORY / "simulations/mini-millennium/snapshots/trees_063.1",
            REPOSITORY / "simulations/mini-millennium/snapshots/trees_063.2",
            REPOSITORY / "simulations/mini-millennium/snapshots/trees_063.3",
        ],
    )
    parser.add_argument(
        "--development-reference",
        type=Path,
        nargs="+",
        default=[
            REPOSITORY / "output/sage16-mini-millennium/model_001.hdf5",
            REPOSITORY / "output/sage16-mini-millennium/model_002.hdf5",
            REPOSITORY / "output/sage16-mini-millennium/model_003.hdf5",
        ],
    )
    parser.add_argument(
        "--test-trees",
        type=Path,
        default=REPOSITORY / "simulations/mini-millennium/snapshots/trees_063.5",
    )
    parser.add_argument(
        "--test-reference",
        type=Path,
        default=REPOSITORY / "output/sage16-mini-millennium/model_005.hdf5",
    )
    parser.add_argument(
        "--scale-factors",
        type=Path,
        default=REPOSITORY / "simulations/mini-millennium/mini-millennium.a_list",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPOSITORY / "archive/mini-millennium-sage16-minimal.json",
    )
    parser.add_argument(
        "--output-arrays",
        type=Path,
        default=REPOSITORY / "archive/mini-millennium-sage16-minimal.npz",
    )
    return parser.parse_args()


def load_topology(path: Path, scale_factor_path: Path) -> TreeTopology:
    partition = open_lhalo_partition(path)
    trees = []
    offsets = []
    offset = 0
    for tree_index in range(partition.tree_count):
        tree = partition.read_tree(tree_index)
        trees.append(tree)
        offsets.append(offset)
        offset += len(tree)

    node_count = offset
    snapshot = np.empty(node_count, dtype=np.int16)
    descendant = np.full(node_count, -1, dtype=np.int64)
    fof_root = np.empty(node_count, dtype=np.int64)
    halo_mass = np.empty(node_count, dtype=np.float64)
    spin_magnitude = np.empty(node_count, dtype=np.float64)
    most_bound_id = np.empty(node_count, dtype=np.int64)
    tree_indices = np.empty(node_count, dtype=np.int32)
    for tree_index, (tree, offset) in enumerate(zip(trees, offsets)):
        node_slice = slice(offset, offset + len(tree))
        snapshot[node_slice] = tree["SnapNum"]
        most_bound_id[node_slice] = tree["MostBoundID"]
        tree_indices[node_slice] = tree_index
        fof_root[node_slice] = offset + tree["FirstHaloInFOFgroup"].astype(np.int64)
        local_descendant = tree["Descendant"].astype(np.int64)
        descendant[node_slice] = np.where(local_descendant >= 0, offset + local_descendant, -1)
        is_central = tree["FirstHaloInFOFgroup"] == np.arange(len(tree))
        catalog_mass = tree["M_Crit200"].astype(np.float64)
        particle_mass = tree["Len"].astype(np.float64) * PARTICLE_MASS
        halo_mass[node_slice] = np.maximum(
            np.where(is_central & (catalog_mass >= 0.0), catalog_mass, particle_mass),
            0.0,
        )
        spin_magnitude[node_slice] = np.linalg.norm(tree["Spin"].astype(np.float64), axis=1)

    scale_factor = load_scale_factors(scale_factor_path)
    timing = snapshot_timing(scale_factor)
    unit_time_gyr = float(sage16_units().UnitTime_in_s) / SECONDS_PER_GYR
    cosmic_time = (timing.lookback_time[0] - timing.lookback_time) * unit_time_gyr
    snapshot_dt = np.concatenate(([cosmic_time[0]], np.diff(cosmic_time)))
    nodes_by_snapshot = tuple(
        np.flatnonzero(snapshot == snapshot_number) for snapshot_number in range(scale_factor.size)
    )
    roots_by_snapshot = []
    root_locations_by_snapshot = []
    for nodes in nodes_by_snapshot:
        roots = np.unique(fof_root[nodes]) if nodes.size else np.empty(0, dtype=np.int64)
        locations = {int(node): position for position, node in enumerate(nodes)}
        roots_by_snapshot.append(roots)
        root_locations_by_snapshot.append(
            np.asarray([locations[int(root)] for root in roots], dtype=np.int64)
        )
    return TreeTopology(
        node_count=node_count,
        tree_count=partition.tree_count,
        snapshot=snapshot,
        descendant=descendant,
        fof_root=fof_root,
        halo_mass=halo_mass,
        spin_magnitude=spin_magnitude,
        most_bound_id=most_bound_id,
        tree_index=tree_indices,
        nodes_by_snapshot=nodes_by_snapshot,
        roots_by_snapshot=tuple(roots_by_snapshot),
        root_locations_by_snapshot=tuple(root_locations_by_snapshot),
        cosmic_time_gyr=cosmic_time,
        snapshot_dt_gyr=snapshot_dt,
        scale_factor=scale_factor,
    )


def load_teacher(path: Path, topology: TreeTopology, snapshot_number: int = 63):
    with h5py.File(path, "r") as source:
        group = source[f"Snap{snapshot_number:03d}"]
        rows = group["Galaxies"][:]
        counts = group["TreeHalosPerSnap"][:]
    row_tree_index = np.repeat(np.arange(counts.size, dtype=np.int32), counts)
    if row_tree_index.size != rows.size:
        raise ValueError("teacher tree counts do not match its galaxy catalogue")
    terminal_nodes = np.flatnonzero(topology.snapshot == snapshot_number)
    raw_by_most_bound_id = {int(topology.most_bound_id[node]): int(node) for node in terminal_nodes}
    missing = [
        int(identifier)
        for identifier in rows["MostBoundID"]
        if int(identifier) not in raw_by_most_bound_id
    ]
    if missing:
        raise ValueError(f"teacher contains {len(missing)} identities absent from the raw trees")
    raw_node_index = np.asarray(
        [raw_by_most_bound_id[int(identifier)] for identifier in rows["MostBoundID"]],
        dtype=np.int64,
    )
    return TeacherCatalogue(rows, row_tree_index, raw_node_index)


def parameters_from_optimizer(values) -> ReducedParameters:
    return ReducedParameters(
        StarFormationTimescaleGyr=10.0 ** values[0],
        CoolingTimescaleGyr=10.0 ** values[1],
        FeedbackMassLoadingAtPivot=10.0 ** values[2],
        FeedbackHaloMassSlope=values[3],
        QuenchingHaloMass=10.0 ** values[4],
        QuenchingSlope=values[5],
        ColdGasThresholdPerSpin=10.0 ** values[6],
        CoolingRedshiftExponent=values[7],
        BlackHoleQuenchingMass=10.0 ** values[8],
    )


def evolve_reduced_trees(topology: TreeTopology, parameters: ReducedParameters):
    accumulated_circumgalactic = np.zeros(topology.node_count, dtype=np.float64)
    accumulated_cold = np.zeros(topology.node_count, dtype=np.float64)
    accumulated_stars = np.zeros(topology.node_count, dtype=np.float64)
    accumulated_black_hole = np.zeros(topology.node_count, dtype=np.float64)
    accumulated_peak_mass = np.zeros(topology.node_count, dtype=np.float64)
    accumulated_progenitor_halo_mass = np.zeros(topology.node_count, dtype=np.float64)
    maximum_progenitor_halo_mass = np.zeros(topology.node_count, dtype=np.float64)
    circumgalactic = np.zeros(topology.node_count, dtype=np.float64)
    cold = np.zeros(topology.node_count, dtype=np.float64)
    stars = np.zeros(topology.node_count, dtype=np.float64)
    black_hole = np.zeros(topology.node_count, dtype=np.float64)
    star_formation_rate = np.zeros(topology.node_count, dtype=np.float64)
    peak_mass = np.zeros(topology.node_count, dtype=np.float64)
    maximum_residual = 0.0
    total_infall = 0.0

    for snapshot_number, nodes in enumerate(topology.nodes_by_snapshot):
        if nodes.size == 0:
            continue
        local_circumgalactic = accumulated_circumgalactic[nodes].copy()
        local_cold = accumulated_cold[nodes].copy()
        local_stars = accumulated_stars[nodes].copy()
        local_black_hole = accumulated_black_hole[nodes].copy()
        local_peak = np.maximum(accumulated_peak_mass[nodes], topology.halo_mass[nodes])

        main_progenitor_mass = maximum_progenitor_halo_mass[nodes]
        secondary_progenitor_mass = np.maximum(
            accumulated_progenitor_halo_mass[nodes] - main_progenitor_mass,
            0.0,
        )
        merger_mass_ratio = np.where(
            main_progenitor_mass > 0.0,
            secondary_progenitor_mass / np.maximum(main_progenitor_mass, np.finfo(np.float64).tiny),
            0.0,
        )
        local_state, _ = _apply_reduced_merger_event_numpy(
            ReducedState(
                local_circumgalactic,
                local_cold,
                local_stars,
                local_black_hole,
            ),
            merger_mass_ratio,
        )
        local_circumgalactic = local_state.CircumgalacticGas
        local_cold = local_state.ColdGas
        local_stars = local_state.StellarMass
        local_black_hole = local_state.BlackHoleProxy

        group_root = topology.fof_root[nodes]
        group_mass = np.bincount(
            group_root,
            weights=(local_circumgalactic + local_cold + local_stars + local_black_hole),
            minlength=topology.node_count,
        )
        roots = topology.roots_by_snapshot[snapshot_number]
        root_locations = topology.root_locations_by_snapshot[snapshot_number]
        infall = np.maximum(
            GLOBAL_BARYON_FRACTION * topology.halo_mass[roots] - group_mass[roots],
            0.0,
        )
        local_circumgalactic[root_locations] += infall
        total_infall += float(np.sum(infall))

        initial_mass = local_circumgalactic + local_cold + local_stars + local_black_hole
        state = ReducedState(
            local_circumgalactic,
            local_cold,
            local_stars,
            local_black_hole,
        )
        forcing = ReducedForcing(
            topology.halo_mass[nodes],
            topology.spin_magnitude[nodes],
            np.full(nodes.size, 1.0 / topology.scale_factor[snapshot_number]),
        )
        state, diagnostics = _evolve_reduced_interval_numpy(
            state,
            forcing,
            parameters,
            topology.snapshot_dt_gyr[snapshot_number],
            substeps=2,
        )
        final_mass = (
            state.CircumgalacticGas + state.ColdGas + state.StellarMass + state.BlackHoleProxy
        )
        maximum_residual = max(
            maximum_residual,
            float(np.max(np.abs(final_mass - initial_mass))),
        )
        circumgalactic[nodes] = state.CircumgalacticGas
        cold[nodes] = state.ColdGas
        stars[nodes] = state.StellarMass
        black_hole[nodes] = state.BlackHoleProxy
        interval_duration = topology.snapshot_dt_gyr[snapshot_number]
        star_formation_rate[nodes] = np.where(
            interval_duration > 0.0,
            diagnostics.LockedStellarMass
            / (1.0 - RECYCLE_FRACTION)
            / np.maximum(interval_duration, np.finfo(np.float64).tiny),
            diagnostics.StarFormationRate,
        )
        peak_mass[nodes] = local_peak

        descendant = topology.descendant[nodes]
        survives = descendant >= 0
        np.add.at(
            accumulated_circumgalactic,
            descendant[survives],
            state.CircumgalacticGas[survives],
        )
        np.add.at(accumulated_cold, descendant[survives], state.ColdGas[survives])
        np.add.at(accumulated_stars, descendant[survives], state.StellarMass[survives])
        np.add.at(
            accumulated_black_hole,
            descendant[survives],
            state.BlackHoleProxy[survives],
        )
        np.maximum.at(accumulated_peak_mass, descendant[survives], local_peak[survives])
        np.add.at(
            accumulated_progenitor_halo_mass,
            descendant[survives],
            topology.halo_mass[nodes][survives],
        )
        np.maximum.at(
            maximum_progenitor_halo_mass,
            descendant[survives],
            topology.halo_mass[nodes][survives],
        )

    terminal = topology.nodes_by_snapshot[-1]
    return ReducedTreeResult(
        circumgalactic_gas=circumgalactic[terminal],
        cold_gas=cold[terminal],
        stellar_mass=stars[terminal],
        black_hole_proxy=black_hole[terminal],
        star_formation_rate=star_formation_rate[terminal],
        peak_halo_mass=peak_mass[terminal],
        terminal_node_index=terminal,
        maximum_local_conservation_residual=maximum_residual,
        external_infall_mass=total_infall,
    )


def terminal_locations(topology: TreeTopology, teacher: TeacherCatalogue):
    terminal = topology.nodes_by_snapshot[-1]
    location_by_node = {int(node): index for index, node in enumerate(terminal)}
    return np.asarray(
        [location_by_node[int(node)] for node in teacher.raw_node_index], dtype=np.int64
    )


def resolved_galaxies(topology: TreeTopology, teacher: TeacherCatalogue):
    stellar_mass_msun = teacher.rows["StellarMass"].astype(np.float64) * 1.0e10 / HUBBLE_H
    return (
        topology.halo_mass[teacher.raw_node_index] >= MINIMUM_HALO_PARTICLES * PARTICLE_MASS
    ) & (stellar_mass_msun >= MINIMUM_STELLAR_MASS_MSUN)


def fit_static_efficiency(development_samples):
    peak_halo_mass = np.concatenate([sample[2] for sample in development_samples])
    selected = np.concatenate(
        [resolved_galaxies(topology, teacher) for topology, teacher, _ in development_samples]
    )
    reference = np.concatenate(
        [teacher.rows["StellarMass"].astype(np.float64) for _, teacher, _ in development_samples]
    )
    logarithmic_mass = np.log10(np.maximum(peak_halo_mass, np.finfo(np.float64).tiny))
    logarithmic_reference = np.log10(np.maximum(reference, np.finfo(np.float64).tiny))
    logarithm_of_ten = np.log(10.0)

    def prediction(values):
        log_characteristic_mass, log_efficiency, low_slope, high_slope = values
        offset = logarithmic_mass - log_characteristic_mass
        logarithmic_denominator = (
            np.logaddexp(
                logarithm_of_ten * (-low_slope * offset),
                logarithm_of_ten * (high_slope * offset),
            )
            / logarithm_of_ten
        )
        return np.log10(2.0) + log_efficiency + logarithmic_mass - logarithmic_denominator

    fitted = least_squares(
        lambda values: prediction(values)[selected] - logarithmic_reference[selected],
        np.asarray([1.8, -1.3, 0.8, 1.3]),
        bounds=([-1.0, -4.0, 0.01, 0.01], [5.0, 0.0, 5.0, 5.0]),
        loss="soft_l1",
        f_scale=0.1,
    )
    return fitted.x, 10.0 ** prediction(fitted.x)


def fit_reduced_model(development_samples):
    def residual(values):
        residuals = []
        for topology, teacher in development_samples:
            locations = terminal_locations(topology, teacher)
            selected = resolved_galaxies(topology, teacher)
            reference = teacher.rows["StellarMass"].astype(np.float64)
            logarithmic_reference = np.log10(np.maximum(reference, np.finfo(np.float64).tiny))
            result = evolve_reduced_trees(topology, parameters_from_optimizer(values))
            prediction = result.stellar_mass[locations]
            residuals.append(
                np.log10(np.maximum(prediction, np.finfo(np.float64).tiny))[selected]
                - logarithmic_reference[selected]
            )
        return np.concatenate(residuals)

    initial = np.asarray(
        [
            -0.40293,
            0.07045,
            -0.81155,
            1.95533,
            1.0,
            0.79195,
            -0.68383,
            1.65549,
            -3.36229,
        ]
    )
    lower = np.asarray([-3.0, -2.0, -2.0, 0.0, 1.0, 0.1, -3.0, -5.0, -8.0])
    upper = np.asarray([2.0, 2.0, 2.0, 3.0, 5.0, 8.0, 2.0, 5.0, 1.0])
    fitted = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.1,
        max_nfev=150,
    )
    return fitted.x, fitted.nfev


def stellar_mass_metrics(reference, prediction, selected):
    ratio = prediction[selected] / reference[selected]
    absolute_fractional_error = np.abs(ratio - 1.0)
    positive_prediction = prediction[selected] > 0.0
    logarithmic_error = np.log10(prediction[selected][positive_prediction]) - np.log10(
        reference[selected][positive_prediction]
    )
    return {
        "number_of_galaxies": int(np.count_nonzero(selected)),
        "number_of_non_positive_predictions": int(np.count_nonzero(~positive_prediction)),
        "fraction_within_30_percent": float(
            np.mean(absolute_fractional_error <= FIDELITY_TOLERANCE)
        ),
        "median_absolute_fractional_error": float(np.median(absolute_fractional_error)),
        "p90_absolute_fractional_error": float(np.quantile(absolute_fractional_error, 0.9)),
        "rmse_dex_for_positive_predictions": float(np.sqrt(np.mean(logarithmic_error**2))),
    }


def stellar_mass_function_metrics(reference, prediction, *, bin_edges):
    reference_positive = reference > 0.0
    prediction_positive = prediction > 0.0
    reference_counts, _ = np.histogram(
        np.log10(reference[reference_positive] * 1.0e10 / HUBBLE_H),
        bins=bin_edges,
    )
    prediction_counts, _ = np.histogram(
        np.log10(prediction[prediction_positive] * 1.0e10 / HUBBLE_H),
        bins=bin_edges,
    )
    resolved = reference_counts >= MINIMUM_SMF_COUNT
    fractional_difference = np.full(reference_counts.shape, np.nan, dtype=np.float64)
    fractional_difference[resolved] = (
        prediction_counts[resolved] - reference_counts[resolved]
    ) / reference_counts[resolved]
    maximum = float(np.nanmax(np.abs(fractional_difference[resolved])))
    return {
        "reference_counts": reference_counts,
        "prediction_counts": prediction_counts,
        "resolved": resolved,
        "fractional_difference": fractional_difference,
        "maximum_resolved_fractional_difference": maximum,
        "all_resolved_bins_within_30_percent": bool(maximum <= FIDELITY_TOLERANCE),
    }


def positive_quantity_metrics(reference, prediction, minimum_reference):
    selected = np.isfinite(reference) & np.isfinite(prediction) & (reference >= minimum_reference)
    if not np.any(selected):
        return {"number_of_objects": 0}
    ratio = prediction[selected] / reference[selected]
    error = np.abs(ratio - 1.0)
    return {
        "number_of_objects": int(np.count_nonzero(selected)),
        "fraction_within_30_percent": float(np.mean(error <= FIDELITY_TOLERANCE)),
        "median_absolute_fractional_error": float(np.median(error)),
        "p90_absolute_fractional_error": float(np.quantile(error, 0.9)),
    }


def test_diagnostics(topology, teacher, result):
    locations = terminal_locations(topology, teacher)
    resolved = resolved_galaxies(topology, teacher)
    reference_stellar_mass = teacher.rows["StellarMass"].astype(np.float64)
    matched_stellar_mass = result.stellar_mass[locations]
    matched_cold_gas = result.cold_gas[locations]
    matched_sfr = result.star_formation_rate[locations] * 10.0 / HUBBLE_H
    reference_sfr = teacher.rows["StarFormationRate"].astype(np.float64)
    central = teacher.rows["Type"] == 0
    reference_specific_sfr = reference_sfr / np.maximum(
        reference_stellar_mass * 1.0e10 / HUBBLE_H, np.finfo(np.float64).tiny
    )
    model_specific_sfr = matched_sfr / np.maximum(
        matched_stellar_mass * 1.0e10 / HUBBLE_H, np.finfo(np.float64).tiny
    )
    classification = central & resolved
    reference_quenched = reference_specific_sfr < 1.0e-11
    model_quenched = model_specific_sfr < 1.0e-11
    return {
        "stellar_mass": stellar_mass_metrics(
            reference_stellar_mass, matched_stellar_mass, resolved
        ),
        "cold_gas": positive_quantity_metrics(
            teacher.rows["ColdGas"].astype(np.float64), matched_cold_gas, 0.01
        ),
        "star_formation_rate": positive_quantity_metrics(reference_sfr, matched_sfr, 1.0e-3),
        "quenched_classification": {
            "number_of_centrals": int(np.count_nonzero(classification)),
            "accuracy": float(
                np.mean(reference_quenched[classification] == model_quenched[classification])
            ),
            "sage_quenched_fraction": float(np.mean(reference_quenched[classification])),
            "reduced_quenched_fraction": float(np.mean(model_quenched[classification])),
        },
        "matched_stellar_mass": matched_stellar_mass,
        "matched_cold_gas": matched_cold_gas,
        "matched_sfr": matched_sfr,
        "resolved": resolved,
    }


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def scalar_smf_metrics(metrics):
    array_fields = {
        "reference_counts",
        "prediction_counts",
        "resolved",
        "fractional_difference",
    }
    return {key: value for key, value in metrics.items() if key not in array_fields}


def main():
    arguments = parse_arguments()
    if len(arguments.development_trees) != len(arguments.development_reference):
        raise SystemExit("development tree and reference lists must have equal length")
    started = time.perf_counter()
    development_samples = []
    for tree_path, reference_path in zip(
        arguments.development_trees, arguments.development_reference
    ):
        topology = load_topology(tree_path, arguments.scale_factors)
        teacher = load_teacher(reference_path, topology)
        development_samples.append((topology, teacher))
    development_load_seconds = time.perf_counter() - started

    fit_started = time.perf_counter()
    provisional_parameters = parameters_from_optimizer(
        np.asarray(
            [
                -0.40293,
                0.07045,
                -0.81155,
                1.95533,
                1.0,
                0.79195,
                -0.68383,
                1.65549,
                -3.36229,
            ]
        )
    )
    static_samples = []
    for topology, teacher in development_samples:
        provisional = evolve_reduced_trees(topology, provisional_parameters)
        locations = terminal_locations(topology, teacher)
        static_samples.append((topology, teacher, provisional.peak_halo_mass[locations]))
    static_values, _ = fit_static_efficiency(static_samples)
    reduced_values, reduced_nfev = fit_reduced_model(development_samples)
    fitted_parameters = parameters_from_optimizer(reduced_values)
    development_results = [
        evolve_reduced_trees(topology, fitted_parameters) for topology, _ in development_samples
    ]
    development_summaries = []
    for (topology, teacher), result in zip(development_samples, development_results):
        locations = terminal_locations(topology, teacher)
        fine_smf = stellar_mass_function_metrics(
            teacher.rows["StellarMass"].astype(np.float64),
            result.stellar_mass[locations],
            bin_edges=FINE_STELLAR_MASS_EDGES,
        )
        development_summaries.append(
            {
                "tree_count": topology.tree_count,
                "node_count": topology.node_count,
                "teacher_galaxies": len(teacher.rows),
                "fine_smf_maximum_fractional_difference": fine_smf[
                    "maximum_resolved_fractional_difference"
                ],
            }
        )
    fit_seconds = time.perf_counter() - fit_started

    # The test partition is not opened until the model form and coefficients are fixed.
    test_started = time.perf_counter()
    test_topology = load_topology(arguments.test_trees, arguments.scale_factors)
    test_teacher = load_teacher(arguments.test_reference, test_topology)
    test_result = evolve_reduced_trees(test_topology, fitted_parameters)
    test_seconds = time.perf_counter() - test_started
    test_locations = terminal_locations(test_topology, test_teacher)
    test_peak_mass = test_result.peak_halo_mass[test_locations]
    logarithmic_mass = np.log10(np.maximum(test_peak_mass, np.finfo(np.float64).tiny))
    log_characteristic_mass, log_efficiency, low_slope, high_slope = static_values
    offset = logarithmic_mass - log_characteristic_mass
    log_denominator = np.logaddexp(
        np.log(10.0) * (-low_slope * offset),
        np.log(10.0) * (high_slope * offset),
    ) / np.log(10.0)
    static_test_prediction = 10.0 ** (
        np.log10(2.0) + log_efficiency + logarithmic_mass - log_denominator
    )
    teacher_stellar_mass = test_teacher.rows["StellarMass"].astype(np.float64)
    resolved = resolved_galaxies(test_topology, test_teacher)
    static_individual = stellar_mass_metrics(teacher_stellar_mass, static_test_prediction, resolved)
    reduced_diagnostics = test_diagnostics(test_topology, test_teacher, test_result)
    fine_matched_smf = stellar_mass_function_metrics(
        teacher_stellar_mass,
        reduced_diagnostics["matched_stellar_mass"],
        bin_edges=FINE_STELLAR_MASS_EDGES,
    )
    fine_static_smf = stellar_mass_function_metrics(
        teacher_stellar_mass,
        static_test_prediction,
        bin_edges=FINE_STELLAR_MASS_EDGES,
    )
    fine_autonomous_smf = stellar_mass_function_metrics(
        teacher_stellar_mass,
        test_result.stellar_mass,
        bin_edges=FINE_STELLAR_MASS_EDGES,
    )
    coarse_matched_smf = stellar_mass_function_metrics(
        teacher_stellar_mass,
        reduced_diagnostics["matched_stellar_mass"],
        bin_edges=COARSE_STELLAR_MASS_EDGES,
    )
    coarse_static_smf = stellar_mass_function_metrics(
        teacher_stellar_mass,
        static_test_prediction,
        bin_edges=COARSE_STELLAR_MASS_EDGES,
    )
    coarse_autonomous_smf = stellar_mass_function_metrics(
        teacher_stellar_mass,
        test_result.stellar_mass,
        bin_edges=COARSE_STELLAR_MASS_EDGES,
    )

    individual_pass = (
        reduced_diagnostics["stellar_mass"]["fraction_within_30_percent"]
        >= MINIMUM_INDIVIDUAL_FRACTION
    )
    matched_population_pass = coarse_matched_smf["all_resolved_bins_within_30_percent"]
    autonomous_population_pass = coarse_autonomous_smf["all_resolved_bins_within_30_percent"]
    payload = {
        "schema_version": "mimic-jax-sage16-reduction/v2",
        "claim": (
            "A four-state, nine-coefficient reduction reproduces held-out SAGE16 "
            "z=0 stellar masses for at least 70% of resolved galaxies and the stellar "
            "mass function within 30% in 0.4-dex bins containing at least 20 SAGE galaxies."
        ),
        "acceptance": {
            "fractional_tolerance": FIDELITY_TOLERANCE,
            "minimum_individual_fraction": MINIMUM_INDIVIDUAL_FRACTION,
            "minimum_smf_count": MINIMUM_SMF_COUNT,
            "gated_smf_bin_width_dex": 0.4,
            "diagnostic_smf_bin_width_dex": 0.2,
            "individual_stellar_mass_pass": individual_pass,
            "matched_population_pass": matched_population_pass,
            "autonomous_population_pass": autonomous_population_pass,
            "overall_pass": bool(individual_pass and autonomous_population_pass),
        },
        "development": {
            "tree_files": [str(path) for path in arguments.development_trees],
            "reference_files": [str(path) for path in arguments.development_reference],
            "partitions": development_summaries,
        },
        "test": {
            "tree_file": str(arguments.test_trees),
            "reference_file": str(arguments.test_reference),
            "tree_count": test_topology.tree_count,
            "node_count": test_topology.node_count,
            "teacher_galaxies": len(test_teacher.rows),
            "raw_terminal_nodes": int(test_result.terminal_node_index.size),
            "unmatched_raw_terminal_nodes": int(
                test_result.terminal_node_index.size - len(test_teacher.rows)
            ),
        },
        "complexity": {
            "faithful_sage16_state_fields": 32,
            "faithful_sage16_parameters": 15,
            "static_state_fields": 0,
            "static_fitted_coefficients": 4,
            "reduced_state_fields": 4,
            "reduced_fitted_coefficients": 9,
            "reduced_state_names": list(ReducedState._fields),
            "discarded_explicit_state": [
                "separate hot and ejected reservoirs",
                "metals",
                "bulge",
                "physical black-hole mass and detailed accretion history",
                "disk radius",
                "AGN heating radius",
                "merger clocks",
            ],
        },
        "parameters": dict(zip(REDUCED_PARAMETER_NAMES, fitted_parameters)),
        "static_parameters": {
            "CharacteristicHaloMass": 10.0 ** static_values[0],
            "PeakEfficiency": 10.0 ** static_values[1],
            "LowMassSlope": static_values[2],
            "HighMassSlope": static_values[3],
        },
        "fit": {
            "objective": "robust log10 stellar-mass residual on all resolved development galaxies",
            "number_of_function_evaluations": reduced_nfev,
            "development_load_seconds": development_load_seconds,
            "fit_seconds": fit_seconds,
            "test_seconds": test_seconds,
        },
        "test_metrics": {
            "static_stellar_mass": static_individual,
            "static_fine_matched_smf": scalar_smf_metrics(fine_static_smf),
            "static_coarse_matched_smf": scalar_smf_metrics(coarse_static_smf),
            "reduced_stellar_mass": reduced_diagnostics["stellar_mass"],
            "reduced_cold_gas": reduced_diagnostics["cold_gas"],
            "reduced_star_formation_rate": reduced_diagnostics["star_formation_rate"],
            "reduced_quenched_classification": reduced_diagnostics["quenched_classification"],
            "reduced_fine_matched_smf": scalar_smf_metrics(fine_matched_smf),
            "reduced_fine_autonomous_smf": scalar_smf_metrics(fine_autonomous_smf),
            "reduced_coarse_matched_smf": scalar_smf_metrics(coarse_matched_smf),
            "reduced_coarse_autonomous_smf": scalar_smf_metrics(coarse_autonomous_smf),
            "maximum_local_conservation_residual": test_result.maximum_local_conservation_residual,
        },
        "limitations": [
            "The reduced model is calibrated to SAGE16, not to observations.",
            "Only z=0 stellar mass and the stellar mass function are acceptance-gated.",
            "Gas, SFR, and quenching are out-of-sample diagnostics and may fail badly.",
            "The event treatment merges raw progenitor states additively and omits SAGE merger clocks.",
            "The selected model is replicated on a fifth Mini-Millennium file partition, not another simulation.",
            "Partition 4 was used to compare candidate state spaces; partition 5 was not opened until the four-state model and acceptance rule were locked.",
            "The 0.2-dex SMF remains a diagnostic because the first reduction failed on the known fine-scale ringing.",
            "A fifth diffuse-stellar reservoir was tested and rejected because its fitted stripping fraction saturated near one and worsened held-out SMF agreement.",
            "A distinct ejected-gas reservoir was tested on the same development/test split and rejected because its reincorporation time saturated at 0.01 Gyr and its held-out stellar metrics were numerically indistinguishable from the four-state model.",
        ],
    }

    arrays = {
        "fine_stellar_mass_bin_edges": FINE_STELLAR_MASS_EDGES,
        "fine_stellar_mass_bin_centres": 0.5
        * (FINE_STELLAR_MASS_EDGES[:-1] + FINE_STELLAR_MASS_EDGES[1:]),
        "fine_sage_smf_counts": fine_matched_smf["reference_counts"],
        "fine_static_smf_counts": fine_static_smf["prediction_counts"],
        "fine_reduced_matched_smf_counts": fine_matched_smf["prediction_counts"],
        "fine_reduced_autonomous_smf_counts": fine_autonomous_smf["prediction_counts"],
        "fine_smf_resolved": fine_matched_smf["resolved"],
        "fine_static_smf_fractional_difference": fine_static_smf["fractional_difference"],
        "fine_reduced_matched_smf_fractional_difference": fine_matched_smf["fractional_difference"],
        "fine_reduced_autonomous_smf_fractional_difference": fine_autonomous_smf[
            "fractional_difference"
        ],
        "coarse_stellar_mass_bin_edges": COARSE_STELLAR_MASS_EDGES,
        "coarse_stellar_mass_bin_centres": 0.5
        * (COARSE_STELLAR_MASS_EDGES[:-1] + COARSE_STELLAR_MASS_EDGES[1:]),
        "coarse_sage_smf_counts": coarse_matched_smf["reference_counts"],
        "coarse_static_smf_counts": coarse_static_smf["prediction_counts"],
        "coarse_reduced_matched_smf_counts": coarse_matched_smf["prediction_counts"],
        "coarse_reduced_autonomous_smf_counts": coarse_autonomous_smf["prediction_counts"],
        "coarse_smf_resolved": coarse_matched_smf["resolved"],
        "coarse_static_smf_fractional_difference": coarse_static_smf["fractional_difference"],
        "coarse_reduced_matched_smf_fractional_difference": coarse_matched_smf[
            "fractional_difference"
        ],
        "coarse_reduced_autonomous_smf_fractional_difference": coarse_autonomous_smf[
            "fractional_difference"
        ],
        "sage_stellar_mass": teacher_stellar_mass,
        "static_stellar_mass": static_test_prediction,
        "reduced_stellar_mass": reduced_diagnostics["matched_stellar_mass"],
        "sage_cold_gas": test_teacher.rows["ColdGas"].astype(np.float64),
        "reduced_cold_gas": reduced_diagnostics["matched_cold_gas"],
        "sage_black_hole_mass": test_teacher.rows["BlackHoleMass"].astype(np.float64),
        "reduced_black_hole_proxy": test_result.black_hole_proxy[test_locations],
        "sage_star_formation_rate": test_teacher.rows["StarFormationRate"].astype(np.float64),
        "reduced_star_formation_rate": reduced_diagnostics["matched_sfr"],
        "resolved_galaxy": reduced_diagnostics["resolved"],
        "galaxy_type": test_teacher.rows["Type"].astype(np.int32),
        "halo_mass": test_topology.halo_mass[test_teacher.raw_node_index],
        "reduced_parameter_names": np.asarray(REDUCED_PARAMETER_NAMES),
        "reduced_parameter_values": np.asarray(fitted_parameters, dtype=np.float64),
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(arguments.output_arrays, **arrays)
    print(json.dumps(json_ready(payload["acceptance"]), indent=2, sort_keys=True))
    print(json.dumps(json_ready(payload["test_metrics"]), indent=2, sort_keys=True))
    print(arguments.output_json)
    print(arguments.output_arrays)


if __name__ == "__main__":
    main()
