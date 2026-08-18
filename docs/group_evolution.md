# Fiducial SAGE16 FoF-Group Evolution

`evolve_upstream_sequential_group_interval` is the first complete JAX composition of the fiducial SAGE16 module schedule for an already inherited FoF workspace. It preserves the upstream numerical method; it is not a higher-order replacement and is not by itself a merger-tree reader.

## The live workspace

Every `GalaxyState` and `HaloForcing` leaf has a leading group-member axis. The tree adapter supplies the FoF central's integer index. Type 3 records remain in the workspace for identity and output bookkeeping but no longer own their transferred reservoirs, so physics and conservation sums exclude them.

The group state cannot be evolved as independent VMAP calls. Satellite stripping, SN feedback, starburst feedback, and some metal yields write into the live FoF central. A later galaxy in the group must see those writes. The group driver therefore uses an ordered `jax.lax.scan` over members while retaining VMAP across independent FoF groups.

## Exact configured schedule

The pre-timestep map runs once:

1. `sage_reionization` over every non-Type-3 member;
2. `sage_prepare_infall_budget` over the complete group;
3. `sage_set_disk_scale_radius` for the Type-0 central;
4. `sage_initialise_merger_clock` over the complete group.

Each substep then runs the two full-halo modules first:

1. apply the fixed infall partition to the FoF central;
2. reincorporate the FoF central's ejected gas.

The dispatcher next uses galaxy-major ordering. Each live record completes this entire chain before the next record starts:

1. satellite stripping;
2. cooling-budget calculation;
3. stored and newly generated radio-mode heating;
4. cooling application;
5. star-formation budget;
6. SN reheating/ejection budget;
7. SF/SN reservoir application;
8. disk instability;
9. disk-triggered quasar mode;
10. disk-instability starburst;
11. delayed disk-SF enrichment.

Finally, `sage_resolve_mergers_and_disruption` scans the live group in source order. Merger events run their configured quasar and starburst consumers immediately, before the resolver inspects the next source. This last phase remains an event map rather than an ODE step.

## Time and numerical meaning

The interval driver holds halo forcing piecewise constant. The shared `StepContext` provides scheduler time and the configured substep count, while rate-based physics uses each member's inherited `halo.dT / num_substeps`. Infall partitions one precomputed snapshot budget. Satellite stripping recomputes a finite fraction of live excess gas. Threshold maps and merger events retain their upstream finite-update meanings.

This method is therefore named `upstream_sequential`. Alternative RK or adaptive integrations may later operate only on an explicitly rate-based subset and will always be compared against this reference path.

## Explicit outputs

Preparation, every galaxy-major process, full-halo supply, and the event phase return immutable transfer diagnostics. `UpstreamGroupHistoryResult` contains the prepared state, every substep state and live halo type, final state, diagnostics, and a merger-resolution success flag. These records are the basis for conservation ledgers, fractional process perturbations, timestep refinement, and later historical response matrices.

## Evidence and current boundary

The compiled C oracle executes the same two-member-plus-Type-3 schedule for two substeps. It compares 41 intermediate/final group fields. Thirty-nine are bit-for-bit identical; the two central hot-metal values differ by one float32 ULP and use the explicit `atol=2e-8` mixed-precision accumulation tolerance. The full controlled oracle now compares 187 fields across the implementation, with 178 exact.

Tests also require eager/JIT equality, VMAP over independent groups, exact Type-3 skipping, object-local clock decrement, group baryon conservation, and a zero derivative of the closed baryon residual. The latter exposed and fixed an inactive zero-dynamical-time division in the star-formation branch without changing any forward SAGE16 result.

The remaining end-to-end boundary is the format-specific merger-tree adapter: assembling inherited descendant workspaces over a complete tree, writing catalogue records, and comparing those histories and catalogues with a complete upstream Mini-Millennium run. No Mini-Millennium equivalence or population-science claim is made yet.

Code: [`mimic_jax/sage16/group_evolve.py`](../mimic_jax/sage16/group_evolve.py). Tests: [`tests/mimic_jax/test_group_evolve.py`](../tests/mimic_jax/test_group_evolve.py). Compiled reference: [`models/sage16/modules/_tests/test_unit_mimic_jax_reference.c`](../models/sage16/modules/_tests/test_unit_mimic_jax_reference.c).
