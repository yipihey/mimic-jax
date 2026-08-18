"""Host-side L-Halo traversal around the pure JAX SAGE16 group evolution kernel."""

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16.cooling_tables import CoolingTables, load_cooling_tables
from mimic_jax.sage16.group_evolve import evolve_upstream_sequential_group_interval
from mimic_jax.sage16.inheritance import inherit_progenitor, initialise_new_central
from mimic_jax.sage16.types import (
    GalaxyState,
    HaloForcing,
    Sage16Parameters,
    Sage16Units,
    fiducial_parameters,
    inheritance_descendant,
    initial_halo_forcing,
    sage16_units,
    step_context,
)


class GalaxyRecord(NamedTuple):
    """One surviving galaxy and the descendant subhalo that owns its output segment."""

    state: GalaxyState
    halo: HaloForcing
    source_halo: int


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
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *records)


def _record_at(records, index):
    return jax.tree_util.tree_map(lambda value: value[index], records)


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
    return initial_halo_forcing(
        SnapNum=snapshot,
        Len=int(raw["Len"]),
        Mvir=mass,
        Rvir=radius,
        Vvir=velocity,
        Pos=raw["Pos"],
        Vel=raw["Vel"],
        VelDisp=float(raw["VelDisp"]),
        Vmax=float(raw["Vmax"]),
        Spin=raw["Spin"],
        MostBoundID=int(raw["MostBoundID"]),
    )


@lru_cache(maxsize=None)
def _compiled_group_runner(member_count, central_index, num_substeps):
    del member_count

    @jax.jit
    def run(states, halos, context, parameters, units, cooling_tables):
        return evolve_upstream_sequential_group_interval(
            states,
            halos,
            context,
            central_index,
            parameters,
            units,
            cooling_tables,
            num_substeps=num_substeps,
        )

    return run


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
    if len(timing.scale_factor) <= int(np.max(tree["SnapNum"], initial=-1)):
        raise ValueError("snapshot timing does not cover every halo in the tree")
    if num_substeps <= 0:
        raise ValueError("num_substeps must be positive")

    processed = [[] for _ in range(len(tree))]
    records_by_snapshot: Dict[int, list] = {}
    groups_evolved = 0
    all_success = True
    snapshots = sorted({int(value) for value in tree["SnapNum"]})

    for snapshot in snapshots:
        roots = [
            index
            for index in range(len(tree))
            if int(tree["SnapNum"][index]) == snapshot
            and int(tree["FirstHaloInFOFgroup"][index]) == index
        ]
        for root in roots:
            members = _fof_members(tree, root)
            workspace_states = []
            workspace_halos = []
            segments = []
            central_catalog_mass = virial_mass(tree, root, particle_mass)

            for descendant_index in members:
                start = len(workspace_states)
                first_occupied = _first_occupied_progenitor(tree, processed, descendant_index)
                payload = _raw_payload(tree, descendant_index, timing, units, particle_mass)
                mass = float(payload.Mvir)
                radius = float(payload.Rvir)
                velocity = float(payload.Vvir)
                descendant = inheritance_descendant(
                    payload=payload,
                    halo_nr=descendant_index,
                    current_snap=snapshot,
                    current_time=float(timing.lookback_time[snapshot]),
                    new_halo_dt=(
                        float(timing.lookback_time[snapshot - 1] - timing.lookback_time[snapshot])
                        if snapshot > 0
                        else -1.0
                    ),
                    virial_mass=mass,
                    virial_radius=radius,
                    virial_velocity=velocity,
                    is_fof_central=descendant_index == root,
                    unique_galaxy_id=(
                        descendant_index + 1_000_000_000 * (global_tree_offset + tree_index + 1)
                    ),
                )
                progenitors = _linked_indices(
                    tree, "FirstProgenitor", "NextProgenitor", descendant_index
                )
                for progenitor in progenitors:
                    for source in processed[progenitor]:
                        inherited = inherit_progenitor(
                            source.state,
                            source.halo,
                            descendant,
                            float(timing.lookback_time[int(source.halo.SnapNum)]),
                            progenitor == first_occupied,
                        )
                        if bool(np.asarray(inherited.retained)):
                            workspace_states.append(inherited.state)
                            workspace_halos.append(inherited.halo)

                if len(workspace_states) == start and descendant_index == root:
                    created = initialise_new_central(descendant)
                    workspace_states.append(created.state)
                    workspace_halos.append(created.halo)

                end = len(workspace_states)
                if end > start:
                    central_candidates = [
                        index
                        for index in range(start, end)
                        if int(workspace_halos[index].Type) in (0, 1)
                    ]
                    if len(central_candidates) != 1:
                        raise ValueError(
                            f"subhalo {descendant_index} has {len(central_candidates)} local "
                            "Type 0/1 centrals"
                        )
                    local_central = central_candidates[0]
                    for index in range(start, end):
                        workspace_halos[index] = workspace_halos[index]._replace(
                            CentralHalo=jnp.asarray(local_central, dtype=jnp.int32),
                            CentralMvir=jnp.asarray(central_catalog_mass, dtype=jnp.float64),
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
            workspace_halos = [
                halo._replace(UniqueCentralGalaxyID=central_id) for halo in workspace_halos
            ]
            states = _stack(workspace_states)
            halos = _stack(workspace_halos)
            previous_snapshot = int(workspace_halos[central_index].SnapNum)
            time_interval = (
                float(timing.lookback_time[previous_snapshot] - timing.lookback_time[snapshot])
                if previous_snapshot >= 0
                else 0.0
            )
            context = step_context(
                redshift=float(timing.redshift[snapshot]),
                time=float(timing.lookback_time[snapshot]),
                snapshot_number=snapshot,
                num_substeps=num_substeps,
                time_interval=time_interval,
            )
            if jit_physics:
                runner = _compiled_group_runner(len(workspace_states), central_index, num_substeps)
                result = runner(states, halos, context, parameters, units, cooling_tables)
            else:
                result = evolve_upstream_sequential_group_interval(
                    states,
                    halos,
                    context,
                    central_index,
                    parameters,
                    units,
                    cooling_tables,
                    num_substeps=num_substeps,
                )
            all_success = all_success and bool(np.asarray(result.success))
            groups_evolved += 1

            for descendant_index, start, end in segments:
                output_segment = []
                for index in range(start, end):
                    halo = _record_at(result.final_halos, index)
                    if int(halo.Type) == 3:
                        continue
                    halo = halo._replace(SnapNum=jnp.asarray(snapshot, dtype=jnp.int32))
                    record = GalaxyRecord(
                        _record_at(result.final_states, index), halo, descendant_index
                    )
                    output_segment.append(record)
                    records_by_snapshot.setdefault(snapshot, []).append(record)
                processed[descendant_index] = output_segment

    return TreeEvolutionResult(
        tuple(tuple(records) for records in processed),
        {snapshot: tuple(records) for snapshot, records in records_by_snapshot.items()},
        groups_evolved,
        all_success,
    )
