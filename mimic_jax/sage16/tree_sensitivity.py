"""Chain-rule sensitivities through the exact host/JAX SAGE16 tree evolution.

The merger-tree topology remains an ordinary Python traversal.  Each fixed-shape
FoF interval is linearized with :func:`jax.linearize`, and the resulting state
tangents are carried through the same inheritance map as the primal galaxy
state.  This gives derivatives of the implemented numerical SAGE16 map on its
active threshold/event branch without pretending that discrete topology is
smooth.
"""

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16.cooling_tables import CoolingTables, load_cooling_tables
from mimic_jax.sage16.group_evolve import evolve_upstream_sequential_group_final
from mimic_jax.sage16.perturbations import PROCESS_NAMES, process_perturbations
from mimic_jax.sage16.transfers import UpstreamGroupFinalResult
from mimic_jax.sage16.tree_evolve import (
    GalaxyRecord,
    PartitionEvolutionResult,
    SnapshotTiming,
    _compiled_batched_group_runner,
    _marshal_group,
    _member_bin,
    _new_tree_workspace,
    _pad_inactive_members,
    _prepare_group,
    _record_at,
    _stack,
)
from mimic_jax.sage16.types import (
    Sage16Parameters,
    Sage16Units,
    fiducial_parameters,
    sage16_units,
)


@dataclass(frozen=True)
class LinearizedPartitionEvolutionResult:
    """Exact partition evolution plus tangents attached to every output record.

    Each ``GalaxyRecord.state_tangent`` is a ``GalaxyState`` whose leading axis
    follows ``control_names``.  Parameter controls are ordinary derivatives
    with respect to the named parameter value.  Process controls are
    derivatives with respect to a finite-epoch logarithmic multiplier
    ``rate -> rate * exp(epsilon)``.
    """

    evolution: PartitionEvolutionResult
    control_names: Tuple[str, ...]
    parameter_names: Tuple[str, ...]
    process_names: Tuple[str, ...]
    ln_scale_factor_edges: np.ndarray
    derivative_method: str = "jax.linearize forward chain rule"

    @property
    def records_by_tree(self):
        return self.evolution.records_by_tree

    @property
    def records_by_snapshot(self):
        return self.evolution.records_by_snapshot

    @property
    def tree_indices(self):
        return self.evolution.tree_indices

    @property
    def success(self):
        return self.evolution.success

    @property
    def groups_evolved(self):
        return self.evolution.groups_evolved


def _selected_values(parameters, names):
    if not names:
        return jnp.zeros((0,), dtype=jnp.float64)
    return jnp.stack([jnp.asarray(getattr(parameters, name)) for name in names])


@lru_cache(maxsize=None)
def _compiled_batched_linearized_group_runner(
    member_count,
    num_substeps,
    batch_size,
    parameter_names,
    process_names,
):
    del member_count, batch_size

    def final_states_from_controls(
        states,
        selected_parameters,
        selected_processes,
        halos,
        contexts,
        central_indices,
        base_parameters,
        units,
        cooling_tables,
    ):
        replacements = {
            name: selected_parameters[index] for index, name in enumerate(parameter_names)
        }
        parameters = base_parameters._replace(**replacements)
        perturbations = process_perturbations(
            **{name: selected_processes[index] for index, name in enumerate(process_names)}
        )

        def controlled_one(state, halo, context, central_index):
            return evolve_upstream_sequential_group_final(
                state,
                halo,
                context,
                central_index,
                parameters,
                units,
                cooling_tables,
                num_substeps=num_substeps,
                perturbations=perturbations,
            ).final_states

        return jax.vmap(controlled_one)(states, halos, contexts, central_indices)

    @jax.jit
    def run(
        states,
        state_tangent,
        halos,
        contexts,
        central_indices,
        parameters,
        parameter_tangent,
        process_tangent,
        units,
        cooling_tables,
    ):
        selected_parameters = _selected_values(parameters, parameter_names)
        selected_processes = jnp.zeros((len(process_names),), dtype=jnp.float64)

        def differentiable(states_value, parameter_value, process_value):
            return final_states_from_controls(
                states_value,
                parameter_value,
                process_value,
                halos,
                contexts,
                central_indices,
                parameters,
                units,
                cooling_tables,
            )

        _, tangent = jax.jvp(
            differentiable,
            (states, selected_parameters, selected_processes),
            (state_tangent, parameter_tangent, process_tangent),
        )
        return tangent

    return run


def _validate_controls(parameters, parameter_names, process_names, ln_scale_factor_edges, timing):
    parameter_names = tuple(parameter_names)
    process_names = tuple(process_names)
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("parameter_names must be unique")
    if len(set(process_names)) != len(process_names):
        raise ValueError("process_names must be unique")
    unknown_parameters = set(parameter_names) - set(parameters._fields)
    if unknown_parameters:
        raise ValueError(f"Unknown SAGE16 parameters: {sorted(unknown_parameters)}")
    unknown_processes = set(process_names) - set(PROCESS_NAMES)
    if unknown_processes:
        raise ValueError(f"Unknown SAGE16 processes: {sorted(unknown_processes)}")
    for name in parameter_names:
        value = jnp.asarray(getattr(parameters, name))
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError(f"Parameter {name} is discrete and cannot be linearized")

    if process_names:
        if ln_scale_factor_edges is None:
            raise ValueError("process controls require explicit ln_scale_factor_edges")
        edges = np.asarray(ln_scale_factor_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("ln_scale_factor_edges must be a strictly increasing vector")
        history = np.log(np.asarray(timing.scale_factor, dtype=np.float64))
        tolerance = 32.0 * np.finfo(np.float64).eps
        if edges[0] > history.min() + tolerance or edges[-1] < history.max() - tolerance:
            raise ValueError("ln_scale_factor_edges must cover the supplied tree forcing")
    else:
        edges = np.asarray([], dtype=np.float64)
    if not parameter_names and not process_names:
        raise ValueError("at least one parameter or process control is required")
    return parameter_names, process_names, edges


def _epoch_index(snapshot, timing, edges):
    value = np.log(float(timing.scale_factor[snapshot]))
    return int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, edges.size - 2))


def _direct_tangents(
    parameter_count,
    process_count,
    epoch_count,
    epoch_index,
):
    dimension = parameter_count + process_count * epoch_count
    parameter_tangents = np.zeros((dimension, parameter_count), dtype=np.float64)
    if parameter_count:
        parameter_tangents[:parameter_count] = np.eye(parameter_count, dtype=np.float64)
    process_tangents = np.zeros((dimension, process_count), dtype=np.float64)
    for process_index in range(process_count):
        direction = parameter_count + process_index * epoch_count + epoch_index
        process_tangents[direction, process_index] = 1.0
    return parameter_tangents, process_tangents


def _pad_linearized_group(task, target_count):
    states, halos = _pad_inactive_members(task, target_count)
    current_count = int(task.states.HotGas.shape[0])
    padding_count = target_count - current_count
    if padding_count == 0:
        return states, halos, task.state_tangents
    tangents = jax.tree_util.tree_map(
        lambda values: np.pad(
            values,
            ((0, 0), (0, padding_count)) + ((0, 0),) * (values.ndim - 2),
        ),
        task.state_tangents,
    )
    return states, halos, tangents


def linearize_lhalo_partition(
    partition,
    timing: SnapshotTiming,
    *,
    parameter_names: Sequence[str] = (),
    process_names: Sequence[str] = (),
    ln_scale_factor_edges=None,
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
    progress_callback=None,
) -> LinearizedPartitionEvolutionResult:
    """Evolve a partition and propagate SAGE parameter/process tangents.

    Parameter tangents are exact derivatives of the implemented numerical map
    on the active branch.  A process control multiplies its faithful transfer
    by ``exp(epsilon)`` only in the finite ``ln(a)`` epoch selected by the
    corresponding control.  Event identities and threshold branches follow
    the unperturbed reference run and are therefore piecewise differentiable.
    """

    if parameters is None:
        parameters = fiducial_parameters()
    if units is None:
        units = sage16_units()
    if cooling_tables is None:
        cooling_tables = load_cooling_tables()
    parameter_names, process_names, edges = _validate_controls(
        parameters,
        parameter_names,
        process_names,
        ln_scale_factor_edges,
        timing,
    )
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

    epoch_count = edges.size - 1 if process_names else 0
    tangent_dimension = len(parameter_names) + len(process_names) * epoch_count
    control_names = tuple(parameter_names) + tuple(
        f"{process}:epoch_{epoch}" for process in process_names for epoch in range(epoch_count)
    )
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
                    tangent_dimension=tangent_dimension,
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

        epoch = _epoch_index(snapshot, timing, edges) if process_names else 0
        parameter_tangents, process_tangents = _direct_tangents(
            len(parameter_names),
            len(process_names),
            epoch_count,
            epoch,
        )
        for member_count, tasks in buckets.items():
            member_limited_batch = max(1, min(batch_size, max_batch_members // member_count))
            occupied_batch = 1 << (len(tasks) - 1).bit_length()
            shape_batch_size = (
                member_limited_batch
                if member_binning == "power_of_two" or member_count < 32
                else min(member_limited_batch, occupied_batch)
            )
            runner = _compiled_batched_linearized_group_runner(
                member_count,
                num_substeps,
                shape_batch_size,
                parameter_names,
                process_names,
            )
            primal_runner = _compiled_batched_group_runner(
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
                padded_groups = [_pad_linearized_group(task, member_count) for task in padded_tasks]
                states = _stack([group[0] for group in padded_groups])
                halos = _stack([group[1] for group in padded_groups])
                state_tangents = jax.tree_util.tree_map(
                    lambda *values: np.stack(values, axis=1),
                    *[group[2] for group in padded_groups],
                )
                contexts = _stack([task.context for task in padded_tasks])
                central_indices = np.asarray(
                    [task.central_index for task in padded_tasks], dtype=np.int32
                )
                batched = primal_runner(
                    states,
                    halos,
                    contexts,
                    central_indices,
                    parameters,
                    units,
                    cooling_tables,
                    process_perturbations(),
                )
                output_tangents = []
                for direction in range(tangent_dimension):
                    output_tangents.append(
                        runner(
                            states,
                            jax.tree_util.tree_map(
                                lambda values: values[direction], state_tangents
                            ),
                            halos,
                            contexts,
                            central_indices,
                            parameters,
                            parameter_tangents[direction],
                            process_tangents[direction],
                            units,
                            cooling_tables,
                        )
                    )
                output_tangents = jax.tree_util.tree_map(
                    lambda *values: jnp.stack(values), *output_tangents
                )
                batched, output_tangents = jax.device_get((batched, output_tangents))
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "batch",
                            "snapshot": snapshot,
                            "member_count": member_count,
                            "groups": len(active_tasks),
                            "padded_groups": shape_batch_size,
                            "elapsed_seconds": time.perf_counter() - batch_started,
                        }
                    )
                for index, task in enumerate(active_tasks):
                    result = UpstreamGroupFinalResult(
                        _record_at(batched.final_states, index),
                        _record_at(batched.final_halos, index),
                        batched.success[index],
                    )
                    task_tangents = jax.tree_util.tree_map(
                        lambda values: values[:, index], output_tangents
                    )
                    _marshal_group(
                        task,
                        result,
                        output_snapshot_set,
                        state_tangents=task_tangents,
                    )

    records_by_tree = tuple(
        {snapshot: tuple(records) for snapshot, records in workspace.records_by_snapshot.items()}
        for workspace in workspaces
    )
    aggregate = {}
    for records in records_by_tree:
        for snapshot, snapshot_records in records.items():
            aggregate.setdefault(snapshot, []).extend(snapshot_records)
    evolution = PartitionEvolutionResult(
        tree_indices=tuple(tree_indices),
        records_by_tree=records_by_tree,
        records_by_snapshot={snapshot: tuple(records) for snapshot, records in aggregate.items()},
        groups_evolved=sum(workspace.groups_evolved for workspace in workspaces),
        success=all(workspace.success for workspace in workspaces),
    )
    return LinearizedPartitionEvolutionResult(
        evolution=evolution,
        control_names=control_names,
        parameter_names=parameter_names,
        process_names=process_names,
        ln_scale_factor_edges=edges,
    )


def state_tangent_matrix(
    records: Sequence[GalaxyRecord],
    field: str,
) -> np.ndarray:
    """Return ``(n_record, n_control)`` tangents for one state field."""

    if not records:
        return np.empty((0, 0), dtype=np.float64)
    if field not in records[0].state._fields:
        raise ValueError(f"Unknown SAGE16 state field: {field}")
    if any(record.state_tangent is None for record in records):
        raise ValueError("all records must carry state tangents")
    return np.stack(
        [np.asarray(getattr(record.state_tangent, field)) for record in records],
        axis=0,
    )


def state_field_array(records: Sequence[GalaxyRecord], field: str) -> np.ndarray:
    """Return one ordinary state field from an aligned output-record sequence."""

    if not records:
        return np.asarray([], dtype=np.float64)
    if field not in records[0].state._fields:
        raise ValueError(f"Unknown SAGE16 state field: {field}")
    return np.asarray([getattr(record.state, field) for record in records])
