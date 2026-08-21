"""Canonical merger-tree forcing contract and cross-model readiness audit.

This module does not pretend that L-Halo and VELOCIraptor trees encode the
same halo definition.  It exposes the common topology/forcing quantities,
records whether each is native or derived, and reports the remaining semantic
and driver blockers before a model is run on a foreign tree.
"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from mimic_jax.sage16.tree_evolve import SnapshotTiming, virial_mass, virial_radius, virial_velocity
from mimic_jax.sage16.types import Sage16Units, sage16_units
from mimic_jax.shark.tree import SharkTreeData


@dataclass(frozen=True)
class TreeField:
    """One canonical halo-history quantity with provenance."""

    values: np.ndarray
    unit: str
    description: str
    source_fields: Tuple[str, ...]
    origin: str = "native"
    qualification: str = ""

    def __post_init__(self) -> None:
        if self.origin not in ("native", "derived"):
            raise ValueError("Tree field origin must be 'native' or 'derived'")
        values = np.array(self.values, copy=True)
        if values.ndim < 1:
            raise ValueError("Tree fields must have a node axis")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class CanonicalMergerTree:
    """Tree-local topology and halo forcing shared without hiding conventions."""

    source_format: str
    source_path: Path
    tree_index: int
    fields: Mapping[str, TreeField]
    unavailable_fields: Mapping[str, str]
    metadata: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        fields = dict(self.fields)
        unavailable = dict(self.unavailable_fields)
        required = {"node_id", "snapshot", "redshift", "descendant_row", "host_row"}
        missing = required - set(fields)
        if missing:
            raise ValueError(f"Canonical tree is missing structural fields: {sorted(missing)}")
        overlap = set(fields) & set(unavailable)
        if overlap:
            raise ValueError(f"Tree fields cannot be available and unavailable: {sorted(overlap)}")
        sizes = {field.values.shape[0] for field in fields.values()}
        if len(sizes) != 1:
            raise ValueError("All canonical tree fields must share a node axis")
        node_count = next(iter(sizes))
        for key in ("descendant_row", "main_progenitor_row", "host_row"):
            if key not in fields:
                continue
            links = np.asarray(fields[key].values, dtype=np.int64)
            if np.any((links < -1) | (links >= node_count)):
                raise ValueError(f"Canonical {key} contains an out-of-range row index")
        object.__setattr__(self, "fields", MappingProxyType(fields))
        object.__setattr__(self, "unavailable_fields", MappingProxyType(unavailable))
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))

    @property
    def node_count(self) -> int:
        return int(self.fields["node_id"].values.shape[0])

    def has_field(self, name: str) -> bool:
        return name in self.fields


@dataclass(frozen=True)
class ModelTreeRequirements:
    """Forcing fields and non-field semantics required by one model driver."""

    model: str
    native_formats: Tuple[str, ...]
    population_driver_formats: Tuple[str, ...]
    required_fields: Tuple[str, ...]
    semantic_requirements: Tuple[str, ...]


@dataclass(frozen=True)
class TreeCompatibility:
    """Audited readiness of a model on one canonical source tree."""

    model: str
    source_format: str
    field_ready: bool
    population_driver_ready: bool
    native_run: bool
    missing_fields: Tuple[str, ...]
    derived_fields: Tuple[str, ...]
    semantic_blockers: Tuple[str, ...]

    @property
    def fully_runnable(self) -> bool:
        return self.field_ready and self.population_driver_ready and not self.semantic_blockers


SAGE16_TREE_REQUIREMENTS = ModelTreeRequirements(
    model="SAGE16",
    native_formats=("lhalo_binary",),
    population_driver_formats=("lhalo_binary",),
    required_fields=(
        "snapshot",
        "redshift",
        "descendant_row",
        "first_progenitor_row",
        "host_row",
        "particle_count",
        "halo_mass",
        "virial_radius",
        "virial_velocity",
        "maximum_circular_velocity",
        "position",
        "velocity",
        "velocity_dispersion",
        "halo_angular_momentum",
    ),
    semantic_requirements=(
        "SAGE's M200c/particle-count fallback and subhalo ordering must be defined.",
        "The angular-momentum vector must use the convention expected by SAGE's disk-radius law.",
    ),
)

SHARK_LAGOS23_TREE_REQUIREMENTS = ModelTreeRequirements(
    model="SHARK Lagos23",
    native_formats=("shark_velociraptor_hdf5",),
    population_driver_formats=(),
    required_fields=(
        "snapshot",
        "redshift",
        "descendant_row",
        "main_progenitor_row",
        "host_row",
        "halo_mass",
        "virial_velocity",
        "maximum_circular_velocity",
        "position",
        "velocity",
        "halo_angular_momentum",
        "spin_parameter",
        "concentration",
        "half_mass_radius",
        "is_fof_centre",
        "is_interpolated",
    ),
    semantic_requirements=(
        "SHARK halo/subhalo/enclosing ownership and DHalo flags must be defined.",
        "Concentration, half-mass radius, and spin reliability conventions must match SHARK.",
    ),
)


def _tree_field(values, unit, description, source_fields, origin="native", qualification=""):
    return TreeField(
        np.asarray(values),
        unit,
        description,
        tuple(source_fields),
        origin,
        qualification,
    )


def canonical_tree_from_lhalo(
    tree: np.ndarray,
    timing: SnapshotTiming,
    *,
    source_path: Path,
    tree_index: int,
    particle_mass_1e10_msun_over_h: float,
    units: Optional[Sage16Units] = None,
) -> CanonicalMergerTree:
    """Project one L-Halo tree while retaining SAGE's mass convention."""

    if tree.ndim != 1 or not tree.dtype.names:
        raise TypeError("tree must be a one-dimensional L-Halo structured array")
    if units is None:
        units = sage16_units()
    count = len(tree)
    rows = np.arange(count, dtype=np.int64)
    snapshots = np.asarray(tree["SnapNum"], dtype=np.int32)
    redshift = np.asarray(timing.redshift[snapshots], dtype=np.float64)
    masses_internal = np.asarray(
        [virial_mass(tree, row, particle_mass_1e10_msun_over_h) for row in rows],
        dtype=np.float64,
    )
    radii = np.asarray(
        [virial_radius(mass, z, units) for mass, z in zip(masses_internal, redshift)],
        dtype=np.float64,
    )
    velocities = np.asarray(
        [virial_velocity(mass, radius, units) for mass, radius in zip(masses_internal, radii)],
        dtype=np.float64,
    )
    spin_vector = np.asarray(tree["Spin"], dtype=np.float64)
    spin_denominator = np.sqrt(2.0) * velocities * radii
    spin_parameter = np.zeros(count, dtype=np.float64)
    valid_spin = spin_denominator > 0.0
    spin_parameter[valid_spin] = (
        np.linalg.norm(spin_vector[valid_spin], axis=1) / spin_denominator[valid_spin]
    )
    fields = {
        "node_id": _tree_field(rows, "dimensionless", "Tree-local L-Halo row.", ("row",)),
        "native_node_id": _tree_field(
            tree["MostBoundID"],
            "dimensionless",
            "Most-bound particle identifier.",
            ("MostBoundID",),
        ),
        "snapshot": _tree_field(snapshots, "dimensionless", "Snapshot number.", ("SnapNum",)),
        "redshift": _tree_field(
            redshift,
            "dimensionless",
            "Redshift from the simulation scale-factor list.",
            ("SnapNum", "scale_factor"),
            "derived",
        ),
        "descendant_row": _tree_field(
            tree["Descendant"], "dimensionless", "Tree-local descendant row.", ("Descendant",)
        ),
        "first_progenitor_row": _tree_field(
            tree["FirstProgenitor"],
            "dimensionless",
            "First progenitor in native L-Halo ordering.",
            ("FirstProgenitor",),
        ),
        "host_row": _tree_field(
            tree["FirstHaloInFOFgroup"],
            "dimensionless",
            "Tree-local FoF central row.",
            ("FirstHaloInFOFgroup",),
        ),
        "is_fof_centre": _tree_field(
            tree["FirstHaloInFOFgroup"] == rows,
            "dimensionless",
            "Whether the node is the L-Halo FoF central.",
            ("FirstHaloInFOFgroup",),
            "derived",
        ),
        "particle_count": _tree_field(tree["Len"], "particles", "Bound-particle count.", ("Len",)),
        "halo_mass": _tree_field(
            masses_internal * 1.0e10,
            "Msun/h",
            "SAGE virial mass: central M_Crit200 or Len times particle mass.",
            ("M_Crit200", "Len"),
            "derived",
            "The central and subhalo mass branches deliberately follow SAGE16.",
        ),
        "virial_radius": _tree_field(
            radii,
            "Mpc/h",
            "SAGE 200-critical virial radius.",
            ("halo_mass", "redshift", "cosmology"),
            "derived",
        ),
        "virial_velocity": _tree_field(
            velocities,
            "km/s",
            "Circular velocity sqrt(G Mvir/Rvir).",
            ("halo_mass", "virial_radius"),
            "derived",
        ),
        "maximum_circular_velocity": _tree_field(
            tree["Vmax"], "km/s", "Maximum circular velocity.", ("Vmax",)
        ),
        "position": _tree_field(tree["Pos"], "Mpc/h", "Comoving position.", ("Pos",)),
        "velocity": _tree_field(tree["Vel"], "km/s", "Peculiar velocity.", ("Vel",)),
        "velocity_dispersion": _tree_field(
            tree["VelDisp"], "km/s", "Velocity dispersion.", ("VelDisp",)
        ),
        "halo_angular_momentum": _tree_field(
            spin_vector,
            "native L-Halo angular-momentum unit",
            "Vector called Spin by L-Halo and consumed by SAGE's disk-radius law.",
            ("Spin",),
            qualification="Its unit/convention must be transformed before SHARK consumes it.",
        ),
        "spin_parameter": _tree_field(
            spin_parameter,
            "dimensionless",
            "SAGE disk-law spin parameter derived from |Spin|/(sqrt(2) Vvir Rvir).",
            ("Spin", "virial_velocity", "virial_radius"),
            "derived",
            "This is not yet validated against SHARK's halo_lambda reconstruction.",
        ),
        "half_mass_radius": _tree_field(
            tree["SubHalfMass"],
            "native L-Halo length unit",
            "Native subhalo half-mass-radius field.",
            ("SubHalfMass",),
            qualification="The unit and reliability convention require validation before SHARK use.",
        ),
    }
    unavailable = {
        "main_progenitor_row": (
            "L-Halo stores a first-progenitor chain; equivalence to SHARK's mainProgenitorIndex "
            "must be audited"
        ),
        "concentration": "the Mini-Millennium L-Halo record has no concentration field",
        "is_interpolated": "the L-Halo schema does not mark interpolated nodes",
        "is_dhalo_centre": "the L-Halo schema has no DHalo-centre concept",
        "enclosing_row": "the L-Halo schema has only FoF membership links",
    }
    return CanonicalMergerTree(
        source_format="lhalo_binary",
        source_path=Path(source_path),
        tree_index=int(tree_index),
        fields=fields,
        unavailable_fields=unavailable,
        metadata={
            "particle_mass_1e10_msun_over_h": float(particle_mass_1e10_msun_over_h),
            "unresolved_descendants": 0,
        },
    )


def _local_rows(native_ids: np.ndarray, linked_ids: np.ndarray) -> Tuple[np.ndarray, int]:
    lookup = {int(identifier): row for row, identifier in enumerate(native_ids)}
    rows = np.full(linked_ids.shape, -1, dtype=np.int64)
    unresolved = 0
    for row, identifier in enumerate(np.asarray(linked_ids, dtype=np.int64)):
        if identifier < 0:
            continue
        if int(identifier) in lookup:
            rows[row] = lookup[int(identifier)]
        else:
            unresolved += 1
    return rows, unresolved


def canonical_tree_from_shark(data: SharkTreeData, tree_index: int) -> CanonicalMergerTree:
    """Project one public SHARK/VELOCIraptor tree into common forcing fields."""

    selected = data.tree_slice(tree_index)
    nodes = {name: np.asarray(values[selected]) for name, values in data.nodes.items()}
    native_ids = np.asarray(nodes["nodeIndex"], dtype=np.int64)
    descendant, unresolved_descendant = _local_rows(native_ids, nodes["descendantIndex"])
    main_progenitor, unresolved_main = _local_rows(native_ids, nodes["mainProgenitorIndex"])
    host, unresolved_host = _local_rows(native_ids, nodes["hostIndex"])
    enclosing, unresolved_enclosing = _local_rows(native_ids, nodes["enclosingIndex"])
    fields = {
        "node_id": _tree_field(
            native_ids, "dimensionless", "VELOCIraptor node ID.", ("nodeIndex",)
        ),
        "snapshot": _tree_field(
            nodes["snapshotNumber"], "dimensionless", "Snapshot number.", ("snapshotNumber",)
        ),
        "redshift": _tree_field(
            nodes["redshift"], "dimensionless", "Node redshift.", ("redshift",)
        ),
        "descendant_row": _tree_field(
            descendant,
            "dimensionless",
            "Tree-local descendant row resolved from node IDs.",
            ("descendantIndex", "nodeIndex"),
            "derived",
        ),
        "main_progenitor_row": _tree_field(
            main_progenitor,
            "dimensionless",
            "Tree-local main-progenitor row resolved from node IDs.",
            ("mainProgenitorIndex", "nodeIndex"),
            "derived",
        ),
        "host_row": _tree_field(
            host,
            "dimensionless",
            "Tree-local host-subhalo row resolved from node IDs.",
            ("hostIndex", "nodeIndex"),
            "derived",
        ),
        "enclosing_row": _tree_field(
            enclosing,
            "dimensionless",
            "Tree-local enclosing-subhalo row resolved from node IDs.",
            ("enclosingIndex", "nodeIndex"),
            "derived",
        ),
        "halo_mass": _tree_field(
            nodes["nodeMass"], "Msun/h", "Native VELOCIraptor node mass.", ("nodeMass",)
        ),
        "particle_count": _tree_field(
            np.rint(nodes["nodeMass"] / data.particle_mass_msun_over_h).astype(np.int64),
            "particles",
            "Particle count inferred from node mass and simulation particle mass.",
            ("nodeMass", "simulation/particleMass"),
            "derived",
        ),
        "virial_velocity": _tree_field(nodes["Vvir"], "km/s", "Native virial velocity.", ("Vvir",)),
        "maximum_circular_velocity": _tree_field(
            nodes["maximumCircularVelocity"],
            "km/s",
            "Native maximum circular velocity.",
            ("maximumCircularVelocity",),
        ),
        "position": _tree_field(nodes["position"], "Mpc/h", "Comoving position.", ("position",)),
        "velocity": _tree_field(nodes["velocity"], "km/s", "Peculiar velocity.", ("velocity",)),
        "halo_angular_momentum": _tree_field(
            nodes["angularMomentum"],
            "native VELOCIraptor angular-momentum unit",
            "Native halo angular-momentum vector.",
            ("angularMomentum",),
            qualification="A SAGE Spin-vector conversion must be validated before cross-running.",
        ),
        "spin_parameter": _tree_field(
            nodes["lambda"], "dimensionless", "Native halo spin parameter.", ("lambda",)
        ),
        "concentration": _tree_field(
            nodes["cnfw"], "dimensionless", "Native NFW concentration.", ("cnfw",)
        ),
        "half_mass_radius": _tree_field(
            nodes["halfMassRadius"],
            "Mpc/h",
            "Native halo half-mass radius.",
            ("halfMassRadius",),
        ),
        "is_fof_centre": _tree_field(
            nodes["isFoFCentre"].astype(bool),
            "dimensionless",
            "VELOCIraptor FoF-centre flag.",
            ("isFoFCentre",),
        ),
        "is_dhalo_centre": _tree_field(
            nodes["isDHaloCentre"].astype(bool),
            "dimensionless",
            "DHalo-centre flag.",
            ("isDHaloCentre",),
        ),
        "is_interpolated": _tree_field(
            nodes["isInterpolated"].astype(bool),
            "dimensionless",
            "Whether the tree builder interpolated this node.",
            ("isInterpolated",),
        ),
    }
    unavailable = {
        "virial_radius": "the public tree stores Vvir and mass but no directly audited radius",
        "velocity_dispersion": "the public SHARK tree has no velocity-dispersion field",
        "first_progenitor_row": (
            "SHARK stores main progenitor plus descendant links; L-Halo first/next ordering "
            "must be constructed and audited"
        ),
    }
    return CanonicalMergerTree(
        source_format="shark_velociraptor_hdf5",
        source_path=data.path,
        tree_index=int(tree_index),
        fields=fields,
        unavailable_fields=unavailable,
        metadata={
            "particle_mass_msun_over_h": data.particle_mass_msun_over_h,
            "unresolved_descendants": unresolved_descendant,
            "unresolved_main_progenitors": unresolved_main,
            "unresolved_hosts": unresolved_host,
            "unresolved_enclosing": unresolved_enclosing,
        },
    )


def assess_tree_compatibility(
    tree: CanonicalMergerTree,
    requirements: ModelTreeRequirements,
) -> TreeCompatibility:
    """Report field and population-driver readiness without filling gaps silently."""

    missing = tuple(field for field in requirements.required_fields if not tree.has_field(field))
    derived = tuple(
        field
        for field in requirements.required_fields
        if tree.has_field(field) and tree.fields[field].origin == "derived"
    )
    native = tree.source_format in requirements.native_formats
    population_driver_ready = tree.source_format in requirements.population_driver_formats
    semantic_blockers = () if native else requirements.semantic_requirements
    unresolved = int(tree.metadata.get("unresolved_descendants", 0))
    if unresolved:
        semantic_blockers = semantic_blockers + (
            f"{unresolved} descendant links are unresolved in this tree-local projection.",
        )
    return TreeCompatibility(
        model=requirements.model,
        source_format=tree.source_format,
        field_ready=not missing,
        population_driver_ready=population_driver_ready,
        native_run=native,
        missing_fields=missing,
        derived_fields=derived,
        semantic_blockers=semantic_blockers,
    )


def comparison_tree_requirements() -> Tuple[ModelTreeRequirements, ...]:
    """Return the reviewed model requirements used by reports and agents."""

    return (SAGE16_TREE_REQUIREMENTS, SHARK_LAGOS23_TREE_REQUIREMENTS)
