"""Strict reader for the public mini-SURFS/VELOCIraptor SHARK tree schema."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

_REQUIRED_NODE_FIELDS = (
    "nodeIndex",
    "nodeMass",
    "snapshotNumber",
    "redshift",
    "descendantIndex",
    "descendantSnapshot",
    "mainProgenitorIndex",
    "hostIndex",
    "enclosingIndex",
    "fofIndex",
    "isMainProgenitor",
    "isFoFCentre",
    "isDHaloCentre",
    "isInterpolated",
    "isRemerged",
    "Vvir",
    "maximumCircularVelocity",
    "halfMassRadius",
    "lambda",
    "cnfw",
    "angularMomentum",
    "position",
    "velocity",
)


@dataclass(frozen=True)
class SharkTreeData:
    """In-memory node table plus validated tree boundaries and metadata."""

    path: Path
    nodes: Mapping[str, np.ndarray]
    first_node: np.ndarray
    number_of_nodes: np.ndarray
    final_snapshot: np.ndarray
    output_snapshots: np.ndarray
    output_redshifts: np.ndarray
    box_size_mpc_over_h: float
    particle_mass_msun_over_h: float
    number_of_files: int
    file_number: int
    self_contained: bool
    has_subhalos: bool
    missing_descendant_rows: np.ndarray

    @property
    def number_of_trees(self) -> int:
        return int(self.first_node.size)

    @property
    def number_of_nodes_total(self) -> int:
        return int(self.nodes["nodeIndex"].size)

    @property
    def number_of_missing_descendants(self) -> int:
        """Number of positive descendant IDs unresolved in this input file.

        SHARK's ``skip_missing_descendants`` option removes these subhalos
        while building the in-memory topology.  They are therefore a normal,
        explicit input condition for the public CI tree rather than evidence
        that the complete file failed to load.
        """

        return int(self.missing_descendant_rows.size)

    def tree_slice(self, tree_number: int) -> slice:
        first = int(self.first_node[tree_number])
        return slice(first, first + int(self.number_of_nodes[tree_number]))

    def nodes_at_snapshot(self, snapshot: int) -> np.ndarray:
        return np.flatnonzero(self.nodes["snapshotNumber"] == snapshot)


def load_shark_tree(path, *, require_all_descendants=False) -> SharkTreeData:
    """Load a native SHARK tree and enforce topology/schema invariants.

    Positive descendant IDs that cannot be resolved within the file are
    retained in :attr:`SharkTreeData.missing_descendant_rows`.  This mirrors
    the upstream tree builder's reviewed ``skip_missing_descendants`` branch.
    Set ``require_all_descendants=True`` when validating an input contract
    that promises every descendant subhalo is present.
    """

    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Reading SHARK merger trees requires h5py") from error
    path = Path(path)
    with h5py.File(path, "r") as handle:
        missing = [name for name in _REQUIRED_NODE_FIELDS if f"haloTrees/{name}" not in handle]
        if missing:
            raise ValueError(f"SHARK tree is missing required node fields: {missing}")
        nodes = {name: np.asarray(handle[f"haloTrees/{name}"]) for name in _REQUIRED_NODE_FIELDS}
        first_node = np.asarray(handle["treeIndex/firstNode"], dtype=np.int64)
        number_of_nodes = np.asarray(handle["treeIndex/numberOfNodes"], dtype=np.int64)
        final_snapshot = np.asarray(handle["treeIndex/finalSnapshot"], dtype=np.int32)
        output_snapshots = np.asarray(handle["outputTimes/snapshotNumber"], dtype=np.int32)
        output_redshifts = np.asarray(handle["outputTimes/redshift"], dtype=np.float64)
        box_size = float(np.asarray(handle["simulation"].attrs["boxSize"]))
        particle_mass = float(np.asarray(handle["simulation"].attrs["particleMass"]))
        file_info = handle["fileInfo"].attrs
        halo_info = handle["haloTrees"].attrs
        number_of_files = int(file_info["numberOfFiles"])
        file_number = int(file_info["thisFile"])
        self_contained = bool(halo_info["treesAreSelfContained"])
        has_subhalos = bool(halo_info["treesHaveSubhalos"])

    shapes = {values.shape[0] for values in nodes.values()}
    if len(shapes) != 1:
        raise ValueError("SHARK node fields have inconsistent leading dimensions")
    total = int(number_of_nodes.sum())
    if total != next(iter(shapes)):
        raise ValueError(
            f"Tree boundaries cover {total} nodes but node table has {next(iter(shapes))}"
        )
    expected_first = np.concatenate(([0], np.cumsum(number_of_nodes[:-1])))
    if not np.array_equal(first_node, expected_first):
        raise ValueError("SHARK tree boundaries are not contiguous")
    node_indices = np.asarray(nodes["nodeIndex"], dtype=np.int64)
    if np.unique(node_indices).size != node_indices.size:
        raise ValueError("SHARK nodeIndex values are not unique")
    descendants = np.asarray(nodes["descendantIndex"], dtype=np.int64)
    positive_descendant_rows = np.flatnonzero(descendants >= 0)
    missing_descendant_rows = positive_descendant_rows[
        ~np.isin(descendants[positive_descendant_rows], node_indices)
    ].astype(np.int64, copy=False)
    if require_all_descendants and missing_descendant_rows.size:
        raise ValueError(
            "SHARK tree contains "
            f"{missing_descendant_rows.size} unresolved positive descendant links"
        )
    if output_snapshots.shape != output_redshifts.shape:
        raise ValueError("SHARK output snapshot and redshift arrays differ in length")
    return SharkTreeData(
        path=path,
        nodes=nodes,
        first_node=first_node,
        number_of_nodes=number_of_nodes,
        final_snapshot=final_snapshot,
        output_snapshots=output_snapshots,
        output_redshifts=output_redshifts,
        box_size_mpc_over_h=box_size,
        particle_mass_msun_over_h=particle_mass,
        number_of_files=number_of_files,
        file_number=file_number,
        self_contained=self_contained,
        has_subhalos=has_subhalos,
        missing_descendant_rows=missing_descendant_rows,
    )
