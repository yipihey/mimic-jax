# Mimic-JAX Initial SAGE16 Implementation Plan

**Status:** Active fork-local implementation plan; initial quiescent slice implemented and gated 2026-08-18.
**Baseline:** MIMIC `69590cc60dcb7b8b6510ee0b16b1ed921a6c4853` (the fork and upstream were identical when this plan was written).
**Scope:** An additive JAX physics package; no changes to MIMIC's C core, model metadata, tree readers, or runtime module ABI.

## Evidence from SAGE16

The fiducial run uses four `pre_timestep` modules, then ten fixed substeps. Each substep executes `galaxy_physics` followed by `satellite_mergers`. Full-halo modules run first, merger events dispatch immediately, and by-galaxy modules then run galaxy-major. The actual state is the 32 fields in `models/sage16/model_properties.yaml`: persistent baryonic reservoirs and diagnostics plus snapshot-scoped transport fields. Persistent SAGE-parity reservoirs are `float`; transport budgets such as `InfallingGas`, `CoolingGas`, `NewStellarMass`, `SupernovaReheatedMass`, and `SupernovaEjectedMass` are `double`.

The prescriptions are not all one mathematical kind. Cooling, star formation, feedback, reincorporation, and AGN modules calculate finite-substep budgets. Infall is prepared once and partitioned across substeps. Satellite stripping recomputes and removes one fraction of the current excess every substep. Merger/disruption handling is an ordered event map with immediate downstream consumers. The JAX design must preserve these distinctions rather than label every operation an ODE rate.

## Initial Vertical Slice

1. Add an installable `mimic_jax` Python package with immutable JAX PyTrees for all 32 SAGE16 galaxy fields, halo/tree forcing, fiducial parameters, unit constants, and explicit process-transfer records. Keep upstream field and parameter names where they are public scientific vocabulary.
2. Port the closed quiescent disk chain in fiducial order: `sage_calculate_star_formation`, `sage_calculate_supernova_feedback`, `sage_apply_star_formation_supernova`, and `sage_apply_metal_enrichment`. Preserve float-storage boundaries, exact thresholds, renormalisation, central/satellite destinations, and the delayed disk-yield application.
3. Add the explicit hot-to-cold cooling apply transfer and ejected-to-hot reincorporation transfer. These extend the usable reservoir cycle without prematurely translating the tabulated cooling-budget or radio-mode kernels.
4. Test process outputs against a compiled C reference oracle derived from the live MIMIC SAGE16 formulas, with the baseline commit and tolerances recorded. Run the existing MIMIC module tests independently so the reference side is known-good.
5. Add executable mass and metal ledgers. Closed transfers must conserve baryonic mass and pre-existing metals; enrichment must expose `Yield * NewStellarMass` as a metal source rather than masquerading as conservation.
6. Demonstrate `jax.jit`, `jax.vmap`, and `jax.grad` on the faithful subset. Validate an automatic derivative of final `StellarMass` with respect to `SfrEfficiency` against a centred finite difference away from thresholds.
7. Add a small `lax.scan` history driver for independent galaxies and expose exact discrete reverse-mode sensitivities to per-step process budgets. Do not imply that shared-central galaxy-major coupling or merger events are vectorised until their scatter/event maps are implemented and tested.

## Fractional Responses as a Core Data Product

Public parameter sensitivities default to `d ln(O) / d ln(theta)`, with explicit validity masks and an opt-in reference-scale convention for zero, signed, or otherwise non-logarithmic quantities. Historical sensitivities perturb faithful finite-substep transfers as `r -> r exp(epsilon)` and report finite-epoch `d ln(O) / d epsilon`, using explicit `ln(a)` and redshift edges. Response results carry names, units, fiducial values, normalization, sign, and derivative method and can be saved without reconstructing metadata in a notebook.

The first scientific application program is [`../mimic_jax_scientific_program.md`](../mimic_jax_scientific_program.md). It remains gated on complete Mini-Millennium equivalence; controlled-subset response examples must not be presented as population conclusions.

## Deferred Fidelity Work

The Sutherland-Dopita cooling-table interpolation and cooling budget are now implemented. The next slices, in order, are radio-mode AGN heating, infall/reionization, shared-central satellite stripping, disk instability/quasar/starburst, and finally ordered merger/disruption event maps plus tree inheritance. Full-tree and final-catalogue equivalence is not claimed until these processes and the C tree interface are present.

## Initial Milestone Evidence

The first slice landed in commits `9f6d27d2` and `1ce7a135`. Twenty-one Python tests cover the complete state, process formulas, conservation, transforms, normalization safety, scan histories, and finite-difference validation. The next cooling-budget slice adds six focused Python tests. Relevant upstream C suites pass, including an executable oracle that compares 30 fields from cooling-table, cooling-budget/application, reincorporation, and quiescent SF/SN/enrichment cases. Twenty-seven controlled CPU fields are exact; three cooling-budget fields use `rtol=1e-13`, `atol=0` for C-libm/XLA transcendental ordering. These are process-level results, not full-tree or Mini-Millennium equivalence.

## Gates

- MIMIC's existing C build and relevant SAGE16 unit tests pass unchanged.
- JAX process equivalence cases state `rtol` and `atol`; unexplained differences fail.
- Conservation tests cover both central and satellite destinations.
- JIT and VMAP results match eager execution; gradients match finite differences on smooth branches.
- Benchmarks separate first-call compilation from warmed execution and make no upstream speedup claim before comparable end-to-end workloads exist.
- Documentation records source functions, equations, conservation behavior, differentiability class, and current limitations for every implemented process.
