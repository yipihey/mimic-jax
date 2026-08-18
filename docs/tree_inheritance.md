# SAGE16 Tree-State Inheritance

The merger tree is an external input signal, but SAGE does not simply evaluate a fresh galaxy at each tree node. The core inheritance service copies surviving progenitor galaxies into a descendant FoF workspace, resets snapshot accumulators, changes halo ownership/type, and preserves the persistent baryonic and event state. mimic-jax represents those operations explicitly before any pre-timestep physics runs.

## Upstream source and state boundary

The source is [`src/core/inheritance.c`](../src/core/inheritance.c), driven by [`src/core/build_model.c`](../src/core/build_model.c). `inherit_progenitor` is the pure numerical counterpart for one progenitor record. Ragged progenitor lookup, output-buffer ownership, recursion, and unique-ID allocation remain ordinary driver concerns because tree topology is discrete rather than differentiable physics.

Deep-copy inheritance resets exactly the 13 properties declared `init: repeat` by the generated metadata:

`InfallingGas`, `CoolingGas`, `NewStellarMass`, `StarFormationRate`, `QuasarModeBHaccretionMass`, `SupernovaReheatedMass`, `SupernovaEjectedMass`, `Cooling`, `Heating`, `Rcool`, `CoolingLambda`, `SupernovaOutflowRate`, and `UnstableDiskGasFraction`.

Persistent reservoirs, their metals, `HaloBaryonFraction`, `Rheat`, `DiskScaleRadius`, `MergTime`, and merger-history fields are retained. This distinction is executable through `reset_snapshot_accumulators`; it is not reproduced from a handwritten list in downstream notebooks.

## Main-branch halo update

Every retained progenitor first receives the descendant `HaloNr` and object duration `source_time - descendant_time`. A Type-0/1 main-branch record then receives the descendant phase-space payload, length, `Vmax`, and virial mass. Its live mass increment is

`deltaMvir = Mvir,desc - Mvir,prog`.

SAGE updates `Rvir` and `Vvir` only when the descendant virial mass exceeds the progenitor mass. On mass loss, those two structural values remain frozen even though `Mvir` and `deltaMvir` update. This behavior matters to cooling and reincorporation and is preserved exactly.

If the descendant is the FoF central, the inherited record becomes Type 0. Otherwise it becomes Type 1. A Type-0→1 transition records the previous `Mvir`, `Vvir`, and float `Vmax` in the three infall fields before applying the descendant halo state.

## Orphans, consumed records, and new centrals

A non-main Type-0/1 progenitor becomes Type 2: `Mvir` and `Len` become zero, `deltaMvir` is the negative previous virial mass, and a former Type 0 records its infall properties. An existing Type 2 is carried through without repeating that transition. A Type-3 source is marked unretained and never enters the new live workspace.

When a FoF-central descendant has no surviving progenitor, `initialise_new_central` creates a default galaxy and a Type-0 halo from the descendant payload. Its `SnapNum` is `current_snap - 1`, preserving the initial-boundary timing convention, and its `dT` comes from the driver-supplied new-halo interval.

Each descendant subhalo slice must contain exactly one Type-0/1 local central. `set_local_central` stamps that local index onto every member's `CentralHalo` field and reports invalid zero/multiple-central topology instead of guessing. The later FoF driver separately identifies the unique Type-0 central for shared baryonic destinations.

## Differentiability and forcing metadata

The `HaloForcing` PyTree now carries the complete inheritance-relevant identity and phase-space subset: unique galaxy IDs, position, velocity, velocity dispersion, spin, `Vmax`, and most-bound-particle ID in addition to the physics forcing fields. Integer identities and topology/type choices are discrete. On a fixed inheritance branch, persistent galaxy values pass through differentiably; a test verifies unit derivative of inherited cold gas. This is useful for differentiating later evolution with respect to an inherited baryonic state, but it is not a derivative of tree connectivity.

## Executable evidence and tree connection

The compiled oracle compares 22 fields from a main-branch inheritance case, covering persistent and reset galaxy fields, copied snapshot identity, object duration, virial growth, descendant payload values, infall state, and local-central linkage. Every field matches the compiled inheritance service exactly. Python tests additionally cover the full 13-field reset contract, Type-0→1 and orphan transitions, preserved Type 2, discarded Type 3, mass-loss virial freezing, new-central defaults, topology validation, JIT, VMAP, and the fixed-branch state derivative. The existing upstream C inheritance suite also remains an independent gate.

`evolve_lhalo_tree` now supplies the ragged driver boundary: it follows real Mini-Millennium progenitor and FoF lists, applies these maps, stamps `CentralMvir` and central identity, executes the complete schedule, and marshals surviving records. Selected linear and branched trees match upstream output across all public fields at recorded tolerances; see [`mini_millennium_equivalence.md`](mini_millennium_equivalence.md). Full-partition population equivalence remains the next, distinct gate.

Current code: [`inheritance.py`](../mimic_jax/sage16/inheritance.py), [`tree_evolve.py`](../mimic_jax/sage16/tree_evolve.py), and [`types.py`](../mimic_jax/sage16/types.py). Tests: [`test_inheritance.py`](../tests/mimic_jax/test_inheritance.py) and [`test_tree_evolve.py`](../tests/mimic_jax/test_tree_evolve.py).
