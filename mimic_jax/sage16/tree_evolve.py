"""Host-side L-Halo traversal around the pure JAX SAGE16 group evolution kernel."""

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import jax
import numpy as np

from mimic_jax.sage16.cooling_tables import CoolingTables, load_cooling_tables
from mimic_jax.sage16.group_evolve import (
    evolve_upstream_sequential_group_final,
    evolve_upstream_sequential_group_interval,
)
from mimic_jax.sage16.perturbations import process_perturbations
from mimic_jax.sage16.transfers import UpstreamGroupFinalResult
from mimic_jax.sage16.types import (
    GalaxyState,
    HaloForcing,
    InheritanceDescendant,
    Sage16Parameters,
    Sage16Units,
    StepContext,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    sage16_units,
)


class GalaxyRecord(NamedTuple):
    """One surviving galaxy and the descendant subhalo that owns its output segment."""

    state: GalaxyState
    halo: HaloForcing
    source_halo: int
    state_tangent: Optional[GalaxyState] = None


@dataclass(frozen=True)
class SnapshotTiming:
    """Scale factor, redshift, and upstream-compatible lookback time per snapshot."""

    scale_factor: np.ndarray
    redshift: np.ndarray
    lookback_time: np.ndarray


@dataclass(frozen=True)
class TreeEvolutionResult:
    """Surviving per-halo records and catalogue-ready records from one merger tree."""

    processed_by_halo: Tuple[Tuple[GalaxyRecord, ...], ...]
    records_by_snapshot: Mapping[int, Tuple[GalaxyRecord, ...]]
    groups_evolved: int
    success: bool


@dataclass(frozen=True)
class PartitionEvolutionResult:
    """Catalogue records produced by batched evolution of several input trees."""

    tree_indices: Tuple[int, ...]
    records_by_tree: Tuple[Mapping[int, Tuple[GalaxyRecord, ...]], ...]
    records_by_snapshot: Mapping[int, Tuple[GalaxyRecord, ...]]
    groups_evolved: int
    success: bool


@dataclass
class _TreeWorkspace:
    tree: np.ndarray
    tree_index: int
    global_tree_offset: int
    processed: list
    records_by_snapshot: Dict[int, list]
    roots_by_snapshot: Mapping[int, Tuple[int, ...]]
    groups_evolved: int = 0
    success: bool = True


@dataclass(frozen=True)
class _PreparedGroup:
    workspace: _TreeWorkspace
    snapshot: int
    states: GalaxyState
    halos: HaloForcing
    context: StepContext
    central_index: int
    segments: Tuple[Tuple[int, int, int], ...]
    state_tangents: Optional[GalaxyState] = None


def load_scale_factors(path) -> np.ndarray:
    """Load MIMIC's strictly increasing scale-factor list."""

    values = np.loadtxt(Path(path), dtype=np.float64, comments="#", ndmin=1)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("scale factors must be a non-empty finite positive sequence")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("scale factors must be strictly increasing")
    return values


def _adaptive_simpson(function, lower, upper, tolerance, depth=0, max_depth=20):
    midpoint = (lower + upper) / 2.0
    lower_value = function(lower)
    upper_value = function(upper)
    midpoint_value = function(midpoint)
    whole = (upper - lower) * (lower_value + 4.0 * midpoint_value + upper_value) / 6.0
    left_midpoint = (lower + midpoint) / 2.0
    right_midpoint = (midpoint + upper) / 2.0
    left = (midpoint - lower) * (lower_value + 4.0 * function(left_midpoint) + midpoint_value) / 6.0
    right = (
        (upper - midpoint) * (midpoint_value + 4.0 * function(right_midpoint) + upper_value) / 6.0
    )
    if abs(left + right - whole) <= tolerance or depth >= max_depth:
        return left + right
    return _adaptive_simpson(
        function, lower, midpoint, tolerance / 2.0, depth + 1, max_depth
    ) + _adaptive_simpson(function, midpoint, upper, tolerance / 2.0, depth + 1, max_depth)


def snapshot_timing(scale_factors: Sequence[float], units: Sage16Units = None) -> SnapshotTiming:
    """Reproduce MIMIC's adaptive-Simpson ``Age`` and ``ZZ`` tables."""

    if units is None:
        units = sage16_units()
    scale_factor = np.asarray(scale_factors, dtype=np.float64)
    omega = float(units.Omega)
    omega_lambda = float(units.OmegaLambda)
    hubble = float(units.Hubble)

    def integrand(value):
        denominator = math.sqrt(
            omega / value + (1.0 - omega - omega_lambda) + omega_lambda * value * value
        )
        return 1.0 / denominator if denominator != 0.0 else 0.0

    ages = np.asarray(
        [_adaptive_simpson(integrand, value, 1.0, 1.0e-10) / hubble for value in scale_factor],
        dtype=np.float64,
    )
    return SnapshotTiming(scale_factor, 1.0 / scale_factor - 1.0, ages)


def virial_mass(tree: np.ndarray, halo_index: int, particle_mass: float = 0.0860657) -> float:
    """Apply upstream ``get_virial_mass`` to one raw L-Halo record."""

    halo = tree[halo_index]
    catalog_mass = float(halo["M_Crit200"])
    if int(halo["FirstHaloInFOFgroup"]) == halo_index and catalog_mass >= 0.0:
        return catalog_mass
    return int(halo["Len"]) * particle_mass


def virial_radius(
    mass: float,
    redshift: float,
    units: Sage16Units,
) -> float:
    """Apply upstream's common 200-critical virial-radius definition."""

    one_plus_redshift = 1.0 + redshift
    hubble_squared = float(units.Hubble) ** 2 * (
        float(units.Omega) * one_plus_redshift**3
        + (1.0 - float(units.Omega) - float(units.OmegaLambda)) * one_plus_redshift**2
        + float(units.OmegaLambda)
    )
    critical_density = 3.0 * hubble_squared / (8.0 * math.pi * float(units.G))
    factor = 1.0 / (200.0 * 4.0 * math.pi / 3.0 * critical_density)
    return float(np.cbrt(mass * factor))


def virial_velocity(mass: float, radius: float, units: Sage16Units) -> float:
    """Apply upstream ``sqrt(G Mvir / Rvir)`` with its zero-radius guard."""

    return math.sqrt(float(units.G) * mass / radius) if radius > 0.0 else 0.0


def _linked_indices(tree, first_name, next_name, halo_index):
    linked = []
    current = int(tree[first_name][halo_index])
    seen = set()
    while current >= 0:
        if current >= len(tree) or current in seen:
            raise ValueError(f"invalid or cyclic {next_name} chain from halo {halo_index}")
        seen.add(current)
        linked.append(current)
        current = int(tree[next_name][current])
    return linked


def _fof_members(tree, root):
    members = [root]
    current = int(tree["NextHaloInFOFgroup"][root])
    seen = {root}
    while current >= 0:
        if current >= len(tree) or current in seen:
            raise ValueError(f"invalid or cyclic FoF chain from halo {root}")
        seen.add(current)
        members.append(current)
        current = int(tree["NextHaloInFOFgroup"][current])
    return members


def _stack(records):
    return jax.tree_util.tree_map(lambda *values: np.stack(values), *records)


def _record_at(records, index):
    return jax.tree_util.tree_map(lambda value: value[index], records)


@lru_cache(maxsize=1)
def _host_initial_state():
    return jax.tree_util.tree_map(np.asarray, initial_galaxy_state())


@lru_cache(maxsize=1)
def _host_initial_halo():
    return jax.tree_util.tree_map(np.asarray, initial_halo_forcing())


def _host_reset_snapshot_accumulators(state):
    return state._replace(
        InfallingGas=np.float64(0.0),
        CoolingGas=np.float64(0.0),
        NewStellarMass=np.float64(0.0),
        StarFormationRate=np.float32(0.0),
        QuasarModeBHaccretionMass=np.float32(0.0),
        SupernovaReheatedMass=np.float64(0.0),
        SupernovaEjectedMass=np.float64(0.0),
        Cooling=np.float64(0.0),
        Heating=np.float64(0.0),
        Rcool=np.float64(0.0),
        CoolingLambda=np.float64(0.0),
        SupernovaOutflowRate=np.float32(0.0),
        UnstableDiskGasFraction=np.float64(0.0),
    )


def _host_zero_state_tangent(state, tangent_dimension):
    """Construct one parameter/process tangent record beside a host state."""

    return jax.tree_util.tree_map(
        lambda value: np.zeros(
            (tangent_dimension,) + np.shape(value), dtype=np.asarray(value).dtype
        ),
        state,
    )


def _host_reset_snapshot_accumulator_tangents(tangent):
    """Apply the snapshot-reset map to a tangent record."""

    return tangent._replace(
        **{
            name: np.zeros_like(getattr(tangent, name))
            for name in (
                "InfallingGas",
                "CoolingGas",
                "NewStellarMass",
                "StarFormationRate",
                "QuasarModeBHaccretionMass",
                "SupernovaReheatedMass",
                "SupernovaEjectedMass",
                "Cooling",
                "Heating",
                "Rcool",
                "CoolingLambda",
                "SupernovaOutflowRate",
                "UnstableDiskGasFraction",
            )
        }
    )


def _host_inherit_progenitor(state, halo, descendant, source_time, is_main_branch):
    state = _host_reset_snapshot_accumulators(state)
    halo = halo._replace(
        HaloNr=np.int32(descendant.halo_nr),
        dT=np.float64(source_time - descendant.current_time),
    )
    if int(halo.Type) == 3:
        return state, halo, False
    if int(halo.Type) not in (0, 1):
        return state, halo, True

    if is_main_branch:
        previous_mvir = np.float64(halo.Mvir)
        previous_vvir = np.float64(halo.Vvir)
        previous_vmax = np.float64(halo.Vmax)
        became_satellite = not bool(descendant.is_fof_central)
        capture_infall = became_satellite and int(halo.Type) == 0
        grew = bool(descendant.virial_mass > previous_mvir)
        payload = descendant.payload
        halo = halo._replace(
            Type=np.int32(0 if bool(descendant.is_fof_central) else 1),
            Len=np.int32(payload.Len),
            Mvir=np.float64(descendant.virial_mass),
            deltaMvir=np.float64(descendant.virial_mass - previous_mvir),
            Rvir=np.float64(descendant.virial_radius if grew else halo.Rvir),
            Vvir=np.float64(descendant.virial_velocity if grew else halo.Vvir),
            infallMvir=np.float64(previous_mvir if capture_infall else halo.infallMvir),
            infallVvir=np.float64(previous_vvir if capture_infall else halo.infallVvir),
            infallVmax=np.float64(previous_vmax if capture_infall else halo.infallVmax),
            Pos=np.asarray(payload.Pos, dtype=np.float32),
            Vel=np.asarray(payload.Vel, dtype=np.float32),
            VelDisp=np.float32(payload.VelDisp),
            Vmax=np.float32(payload.Vmax),
            Spin=np.asarray(payload.Spin, dtype=np.float32),
            MostBoundID=np.int64(payload.MostBoundID),
        )
    else:
        was_central = int(halo.Type) == 0
        halo = halo._replace(
            Type=np.int32(2),
            Len=np.int32(0),
            deltaMvir=np.float64(-halo.Mvir),
            Mvir=np.float64(0.0),
            infallMvir=np.float64(halo.Mvir if was_central else halo.infallMvir),
            infallVvir=np.float64(halo.Vvir if was_central else halo.infallVvir),
            infallVmax=np.float64(halo.Vmax if was_central else halo.infallVmax),
        )
    return state, halo, True


def _host_initialise_new_central(descendant):
    payload = descendant.payload
    halo = payload._replace(
        SnapNum=np.int32(descendant.current_snap - 1),
        Type=np.int32(0),
        HaloNr=np.int32(descendant.halo_nr),
        UniqueGalaxyID=np.int64(descendant.unique_galaxy_id),
        UniqueCentralGalaxyID=np.int64(0),
        dT=np.float64(descendant.new_halo_dt),
        deltaMvir=np.float64(0.0),
        CentralMvir=np.float64(0.0),
        infallMvir=np.float64(-1.0),
        infallVvir=np.float64(-1.0),
        infallVmax=np.float64(-1.0),
    )
    return _host_initial_state(), halo


def _first_occupied_progenitor(tree, processed_by_halo, halo_index):
    progenitors = _linked_indices(tree, "FirstProgenitor", "NextProgenitor", halo_index)
    if not progenitors:
        return -1
    first_occupied = progenitors[0]
    if processed_by_halo[first_occupied]:
        return first_occupied
    largest_length = 0
    for progenitor in progenitors:
        if processed_by_halo[progenitor] and int(tree["Len"][progenitor]) > largest_length:
            largest_length = int(tree["Len"][progenitor])
            first_occupied = progenitor
    return first_occupied


def _raw_payload(tree, halo_index, timing, units, particle_mass):
    raw = tree[halo_index]
    snapshot = int(raw["SnapNum"])
    mass = virial_mass(tree, halo_index, particle_mass)
    radius = virial_radius(mass, float(timing.redshift[snapshot]), units)
    velocity = virial_velocity(mass, radius, units)
    return _host_initial_halo()._replace(
        SnapNum=np.int32(snapshot),
        Len=np.int32(raw["Len"]),
        Mvir=np.float64(mass),
        Rvir=np.float64(radius),
        Vvir=np.float64(velocity),
        Pos=np.asarray(raw["Pos"], dtype=np.float32),
        Vel=np.asarray(raw["Vel"], dtype=np.float32),
        VelDisp=np.float32(raw["VelDisp"]),
        Vmax=np.float32(raw["Vmax"]),
        Spin=np.asarray(raw["Spin"], dtype=np.float32),
        MostBoundID=np.int64(raw["MostBoundID"]),
    )


@lru_cache(maxsize=None)
def _compiled_group_runner(member_count, central_index, num_substeps):
    del member_count

    @jax.jit
    def run(states, halos, context, parameters, units, cooling_tables, perturbations):
        return evolve_upstream_sequential_group_final(
            states,
            halos,
            context,
            central_index,
            parameters,
            units,
            cooling_tables,
            num_substeps=num_substeps,
            perturbations=perturbations,
        )

    return run


@lru_cache(maxsize=None)
def _compiled_batched_group_runner(member_count, num_substeps, batch_size):
    del member_count, batch_size

    def evolve_one(
        states,
        halos,
        context,
        central_index,
        parameters,
        units,
        cooling_tables,
        perturbations,
    ):
        return evolve_upstream_sequential_group_final(
            states,
            halos,
            context,
            central_index,
            parameters,
            units,
            cooling_tables,
            num_substeps=num_substeps,
            perturbations=perturbations,
        )

    return jax.jit(jax.vmap(evolve_one, in_axes=(0, 0, 0, 0, None, None, None, None)))


def _perturbations_at_snapshot(perturbations, snapshot, snapshot_count):
    """Select scalar process controls from scalar or per-snapshot schedules."""

    def select(value):
        value = np.asarray(value)
        if value.ndim == 0:
            return value
        if value.ndim == 1 and value.shape[0] == snapshot_count:
            return value[snapshot]
        raise ValueError(
            "tree process perturbations must be scalar or one-dimensional with "
            "one value per snapshot"
        )

    return jax.tree_util.tree_map(select, perturbations)


def _member_bin(member_count, policy):
    if policy == "exact":
        return member_count
    return 1 << (member_count - 1).bit_length()


def _pad_inactive_members(task, target_count):
    """Append inert Type-3 slots without changing the live member order."""

    current_count = int(task.states.HotGas.shape[0])
    if target_count < current_count:
        raise ValueError("target_count cannot discard live group members")
    if target_count == current_count:
        return task.states, task.halos
    padding_count = target_count - current_count
    padding_state = _host_initial_state()
    padding_halo = _host_initial_halo()._replace(
        Type=np.int32(3),
        CentralHalo=np.int32(task.central_index),
        Len=np.int32(0),
        Mvir=np.float64(0.0),
        deltaMvir=np.float64(0.0),
        CentralMvir=np.float64(0.0),
        Rvir=np.float64(0.0),
        Vvir=np.float64(0.0),
        Vmax=np.float32(0.0),
        dT=np.float64(0.0),
    )

    def append(values, padding):
        tail = np.broadcast_to(
            np.asarray(padding),
            (padding_count,) + np.asarray(padding).shape,
        )
        return np.concatenate((values, tail), axis=0)

    return (
        jax.tree_util.tree_map(append, task.states, padding_state),
        jax.tree_util.tree_map(append, task.halos, padding_halo),
    )


def _validate_tree(tree, timing):
    if not isinstance(tree, np.ndarray) or tree.ndim != 1 or tree.size == 0:
        raise TypeError("tree must be a non-empty one-dimensional structured record array")
    if not set(
        (
            "FirstProgenitor",
            "NextProgenitor",
            "FirstHaloInFOFgroup",
            "NextHaloInFOFgroup",
            "SnapNum",
            "Len",
            "M_Crit200",
        )
    ).issubset(tree.dtype.names or ()):
        raise TypeError("tree must be a structured L-Halo record array")
    if np.any(tree["SnapNum"] < 0) or len(timing.scale_factor) <= int(np.max(tree["SnapNum"])):
        raise ValueError("snapshot timing does not cover every halo in the tree")


def _new_tree_workspace(tree, timing, tree_index, global_tree_offset):
    _validate_tree(tree, timing)
    roots_by_snapshot = {}
    for snapshot in sorted({int(value) for value in tree["SnapNum"]}):
        roots_by_snapshot[snapshot] = tuple(
            index
            for index in range(len(tree))
            if int(tree["SnapNum"][index]) == snapshot
            and int(tree["FirstHaloInFOFgroup"][index]) == index
        )
    return _TreeWorkspace(
        tree=tree,
        tree_index=tree_index,
        global_tree_offset=global_tree_offset,
        processed=[[] for _ in range(len(tree))],
        records_by_snapshot={},
        roots_by_snapshot=roots_by_snapshot,
    )


def _prepare_group(
    workspace,
    root,
    timing,
    units,
    particle_mass,
    num_substeps,
    *,
    discard_consumed_progenitors,
    tangent_dimension=None,
):
    tree = workspace.tree
    snapshot = int(tree["SnapNum"][root])
    members = _fof_members(tree, root)
    workspace_states = []
    workspace_halos = []
    workspace_tangents = [] if tangent_dimension is not None else None
    segments = []
    central_catalog_mass = virial_mass(tree, root, particle_mass)

    for descendant_index in members:
        start = len(workspace_states)
        first_occupied = _first_occupied_progenitor(tree, workspace.processed, descendant_index)
        payload = _raw_payload(tree, descendant_index, timing, units, particle_mass)
        descendant = InheritanceDescendant(
            payload=payload,
            halo_nr=np.int32(descendant_index),
            current_snap=np.int32(snapshot),
            current_time=np.float64(timing.lookback_time[snapshot]),
            new_halo_dt=np.float64(
                timing.lookback_time[snapshot - 1] - timing.lookback_time[snapshot]
                if snapshot > 0
                else -1.0
            ),
            virial_mass=np.float64(payload.Mvir),
            virial_radius=np.float64(payload.Rvir),
            virial_velocity=np.float64(payload.Vvir),
            is_fof_central=np.bool_(descendant_index == root),
            unique_galaxy_id=np.int64(
                descendant_index
                + 1_000_000_000 * (workspace.global_tree_offset + workspace.tree_index + 1)
            ),
        )
        progenitors = _linked_indices(tree, "FirstProgenitor", "NextProgenitor", descendant_index)
        for progenitor in progenitors:
            for source in workspace.processed[progenitor]:
                inherited_state, inherited_halo, retained = _host_inherit_progenitor(
                    source.state,
                    source.halo,
                    descendant,
                    float(timing.lookback_time[int(source.halo.SnapNum)]),
                    progenitor == first_occupied,
                )
                if retained:
                    workspace_states.append(inherited_state)
                    workspace_halos.append(inherited_halo)
                    if workspace_tangents is not None:
                        if source.state_tangent is None:
                            raise ValueError("linearized inheritance requires progenitor tangents")
                        workspace_tangents.append(
                            _host_reset_snapshot_accumulator_tangents(source.state_tangent)
                        )
            if discard_consumed_progenitors:
                workspace.processed[progenitor] = []

        if len(workspace_states) == start and descendant_index == root:
            created_state, created_halo = _host_initialise_new_central(descendant)
            workspace_states.append(created_state)
            workspace_halos.append(created_halo)
            if workspace_tangents is not None:
                workspace_tangents.append(
                    _host_zero_state_tangent(created_state, tangent_dimension)
                )

        end = len(workspace_states)
        if end > start:
            central_candidates = [
                index for index in range(start, end) if int(workspace_halos[index].Type) in (0, 1)
            ]
            if len(central_candidates) != 1:
                raise ValueError(
                    f"subhalo {descendant_index} has {len(central_candidates)} local "
                    "Type 0/1 centrals"
                )
            local_central = central_candidates[0]
            for index in range(start, end):
                workspace_halos[index] = workspace_halos[index]._replace(
                    CentralHalo=np.int32(local_central),
                    CentralMvir=np.float64(central_catalog_mass),
                )
        segments.append((descendant_index, start, end))

    if not workspace_states:
        raise ValueError(f"FoF root {root} produced an empty workspace")
    central_candidates = [
        index for index, halo in enumerate(workspace_halos) if int(halo.Type) == 0
    ]
    if len(central_candidates) != 1:
        raise ValueError(f"FoF root {root} has {len(central_candidates)} Type 0 centrals")
    central_index = central_candidates[0]
    if int(workspace_halos[central_index].HaloNr) != root:
        raise ValueError(f"FoF root {root} does not own its Type 0 central")
    central_id = workspace_halos[central_index].UniqueGalaxyID
    workspace_halos = [halo._replace(UniqueCentralGalaxyID=central_id) for halo in workspace_halos]
    previous_snapshot = int(workspace_halos[central_index].SnapNum)
    time_interval = (
        float(timing.lookback_time[previous_snapshot] - timing.lookback_time[snapshot])
        if previous_snapshot >= 0
        else 0.0
    )
    substep_dt = time_interval / num_substeps
    context = StepContext(
        redshift=np.float64(timing.redshift[snapshot]),
        time=np.float64(timing.lookback_time[snapshot]),
        snapshot_number=np.int32(snapshot),
        substep_number=np.int32(0),
        num_substeps=np.int32(num_substeps),
        time_interval=np.float64(time_interval),
        substep_time=np.float64(timing.lookback_time[snapshot] + time_interval - 0.5 * substep_dt),
        substep_dt=np.float64(substep_dt),
    )
    return _PreparedGroup(
        workspace=workspace,
        snapshot=snapshot,
        states=_stack(workspace_states),
        halos=_stack(workspace_halos),
        context=context,
        central_index=central_index,
        segments=tuple(segments),
        state_tangents=(
            None
            if workspace_tangents is None
            else jax.tree_util.tree_map(
                lambda *values: np.moveaxis(np.stack(values), 0, 1),
                *workspace_tangents,
            )
        ),
    )


def _marshal_group(task, result, output_snapshots, state_tangents=None):
    workspace = task.workspace
    snapshot = task.snapshot
    workspace.success = workspace.success and bool(np.asarray(result.success))
    workspace.groups_evolved += 1
    retain_output = output_snapshots is None or snapshot in output_snapshots

    for descendant_index, start, end in task.segments:
        output_segment = []
        for index in range(start, end):
            halo = _record_at(result.final_halos, index)
            if int(halo.Type) == 3:
                continue
            halo = halo._replace(SnapNum=np.int32(snapshot))
            tangent = (
                None
                if state_tangents is None
                else jax.tree_util.tree_map(lambda values: values[:, index], state_tangents)
            )
            record = GalaxyRecord(
                _record_at(result.final_states, index),
                halo,
                descendant_index,
                tangent,
            )
            output_segment.append(record)
            if retain_output:
                workspace.records_by_snapshot.setdefault(snapshot, []).append(record)
        workspace.processed[descendant_index] = output_segment


def _tree_result(workspace):
    return TreeEvolutionResult(
        tuple(tuple(records) for records in workspace.processed),
        {snapshot: tuple(records) for snapshot, records in workspace.records_by_snapshot.items()},
        workspace.groups_evolved,
        workspace.success,
    )


def evolve_lhalo_tree(
    tree: np.ndarray,
    timing: SnapshotTiming,
    *,
    tree_index: int,
    global_tree_offset: int = 0,
    particle_mass: float = 0.0860657,
    num_substeps: int = 10,
    parameters: Sage16Parameters = None,
    units: Sage16Units = None,
    cooling_tables: CoolingTables = None,
    perturbations=None,
    jit_physics: bool = True,
) -> TreeEvolutionResult:
    """Evolve one raw Mini-Millennium tree with upstream inheritance and ordering.

    Tree topology remains ordinary Python because its list lengths and event
    ownership are discrete. Every fixed-shape FoF interval is delegated to the
    immutable JAX physics kernel and may be JIT compiled independently.
    """

    if parameters is None:
        parameters = fiducial_parameters()
    if units is None:
        units = sage16_units()
    if cooling_tables is None:
        cooling_tables = load_cooling_tables()
    if perturbations is None:
        perturbations = process_perturbations()
    if num_substeps <= 0:
        raise ValueError("num_substeps must be positive")
    workspace = _new_tree_workspace(tree, timing, tree_index, global_tree_offset)
    for snapshot in sorted(workspace.roots_by_snapshot):
        for root in workspace.roots_by_snapshot[snapshot]:
            task = _prepare_group(
                workspace,
                root,
                timing,
                units,
                particle_mass,
                num_substeps,
                discard_consumed_progenitors=False,
            )
            if jit_physics:
                runner = _compiled_group_runner(
                    task.states.HotGas.shape[0], task.central_index, num_substeps
                )
                result = runner(
                    task.states,
                    task.halos,
                    task.context,
                    parameters,
                    units,
                    cooling_tables,
                    _perturbations_at_snapshot(
                        perturbations,
                        snapshot,
                        len(timing.scale_factor),
                    ),
                )
            else:
                result = evolve_upstream_sequential_group_interval(
                    task.states,
                    task.halos,
                    task.context,
                    task.central_index,
                    parameters,
                    units,
                    cooling_tables,
                    num_substeps=num_substeps,
                    perturbations=_perturbations_at_snapshot(
                        perturbations,
                        snapshot,
                        len(timing.scale_factor),
                    ),
                )
            result = jax.device_get(result)
            _marshal_group(task, result, None)
    return _tree_result(workspace)


def evolve_lhalo_partition(
    partition,
    timing: SnapshotTiming,
    *,
    tree_indices: Optional[Sequence[int]] = None,
    global_tree_offset: int = 0,
    particle_mass: float = 0.0860657,
    num_substeps: int = 10,
    output_snapshots: Optional[Sequence[int]] = None,
    batch_size: int = 128,
    max_batch_members: int = 512,
    member_binning: str = "exact",
    parameters: Sage16Parameters = None,
    units: Sage16Units = None,
    cooling_tables: CoolingTables = None,
    perturbations=None,
    progress_callback=None,
) -> PartitionEvolutionResult:
    """Evolve independent same-snapshot FoF groups in fixed-shape VMAP batches.

    The host still owns ragged tree inheritance. Batching changes only how
    already assembled, independent FoF workspaces enter the exact JAX kernel.
    ``member_binning='power_of_two'`` appends inert Type-3 records to reduce
    executable shape count; ``'exact'`` retains the reference workspace size.
    """

    if parameters is None:
        parameters = fiducial_parameters()
    if units is None:
        units = sage16_units()
    if cooling_tables is None:
        cooling_tables = load_cooling_tables()
    if perturbations is None:
        perturbations = process_perturbations()
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive Python integer")
    if not isinstance(max_batch_members, int) or max_batch_members <= 0:
        raise ValueError("max_batch_members must be a positive Python integer")
    if not isinstance(num_substeps, int) or num_substeps <= 0:
        raise ValueError("num_substeps must be a positive Python integer")
    if member_binning not in ("exact", "power_of_two"):
        raise ValueError("member_binning must be 'exact' or 'power_of_two'")
    if tree_indices is None:
        tree_indices = tuple(range(partition.tree_count))
    else:
        tree_indices = tuple(int(index) for index in tree_indices)
    if len(set(tree_indices)) != len(tree_indices):
        raise ValueError("tree_indices must not contain duplicates")
    output_snapshot_set = None if output_snapshots is None else frozenset(output_snapshots)

    workspaces = [
        _new_tree_workspace(
            partition.read_tree(tree_index),
            timing,
            tree_index,
            global_tree_offset,
        )
        for tree_index in tree_indices
    ]
    snapshots = sorted(
        {snapshot for workspace in workspaces for snapshot in workspace.roots_by_snapshot}
    )

    for snapshot in snapshots:
        snapshot_perturbations = _perturbations_at_snapshot(
            perturbations,
            snapshot,
            len(timing.scale_factor),
        )
        buckets = {}
        for workspace in workspaces:
            for root in workspace.roots_by_snapshot.get(snapshot, ()):
                task = _prepare_group(
                    workspace,
                    root,
                    timing,
                    units,
                    particle_mass,
                    num_substeps,
                    discard_consumed_progenitors=True,
                )
                key = _member_bin(int(task.states.HotGas.shape[0]), member_binning)
                buckets.setdefault(key, []).append(task)

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "snapshot",
                    "snapshot": snapshot,
                    "groups": sum(len(tasks) for tasks in buckets.values()),
                    "shape_count": len(buckets),
                    "maximum_members": max(buckets, default=0),
                }
            )

        for member_count, tasks in buckets.items():
            # Bound total members per batch so a rare large workspace does not
            # inherit the memory target chosen for the common one-member case.
            member_limited_batch = max(
                1,
                min(batch_size, max_batch_members // member_count),
            )
            occupied_batch = 1 << (len(tasks) - 1).bit_length()
            shape_batch_size = (
                member_limited_batch
                if member_binning == "power_of_two" or member_count < 32
                else min(member_limited_batch, occupied_batch)
            )
            runner = _compiled_batched_group_runner(
                member_count,
                num_substeps,
                shape_batch_size,
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "shape",
                        "snapshot": snapshot,
                        "member_count": member_count,
                        "central_indices": tuple(sorted({task.central_index for task in tasks})),
                        "groups": len(tasks),
                        "batch_size": shape_batch_size,
                    }
                )
            for start in range(0, len(tasks), shape_batch_size):
                batch_started = time.perf_counter()
                active_tasks = tasks[start : start + shape_batch_size]
                padded_tasks = active_tasks + [active_tasks[-1]] * (
                    shape_batch_size - len(active_tasks)
                )
                padded_groups = [_pad_inactive_members(task, member_count) for task in padded_tasks]
                states = _stack([group[0] for group in padded_groups])
                halos = _stack([group[1] for group in padded_groups])
                contexts = _stack([task.context for task in padded_tasks])
                central_indices = np.asarray(
                    [task.central_index for task in padded_tasks], dtype=np.int32
                )
                batched = runner(
                    states,
                    halos,
                    contexts,
                    central_indices,
                    parameters,
                    units,
                    cooling_tables,
                    snapshot_perturbations,
                )
                batched = jax.device_get(batched)
                batch_elapsed = time.perf_counter() - batch_started
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "batch",
                            "snapshot": snapshot,
                            "member_count": member_count,
                            "groups": len(active_tasks),
                            "padded_groups": shape_batch_size,
                            "elapsed_seconds": batch_elapsed,
                        }
                    )
                for index, task in enumerate(active_tasks):
                    result = UpstreamGroupFinalResult(
                        _record_at(batched.final_states, index),
                        _record_at(batched.final_halos, index),
                        batched.success[index],
                    )
                    _marshal_group(task, result, output_snapshot_set)

    records_by_tree = tuple(
        {snapshot: tuple(records) for snapshot, records in workspace.records_by_snapshot.items()}
        for workspace in workspaces
    )
    aggregate = {}
    for records in records_by_tree:
        for snapshot, snapshot_records in records.items():
            aggregate.setdefault(snapshot, []).extend(snapshot_records)
    return PartitionEvolutionResult(
        tree_indices=tuple(tree_indices),
        records_by_tree=records_by_tree,
        records_by_snapshot={snapshot: tuple(records) for snapshot, records in aggregate.items()},
        groups_evolved=sum(workspace.groups_evolved for workspace in workspaces),
        success=all(workspace.success for workspace in workspaces),
    )
