# Mimic-JAX Initial SAGE16 Implementation Plan

**Status:** Active fork-local implementation plan; all fiducial process modules, tree-state inheritance maps, and the isolated merger-event phase implemented and process-gated 2026-08-18.
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

## Numerical Integration as an Explicit Choice

The exact upstream module sequence is the reference method and is called `upstream_sequential`; it is not casually labeled forward Euler. Rate-times-`dt` prescriptions, partitioned snapshot budgets, bounded finite transfers, threshold maps, and merger events retain distinct representations. Alternative integrators may act only on an explicitly extracted continuous-rate subset and never replace the zero-deviation equivalence path.

The controlled central slice now exposes piecewise-constant halo forcing with independent baryonic substeps, `1, 2, 4, 8` refinement studies, empirical-order metadata, conservation residuals, positivity checks, and finite-step-to-timescale ratios. Full scientific conclusions remain gated on tree-complete Mini-Millennium comparisons. The contract and future fixed/adaptive-method tests are in [`../numerical_integration.md`](../numerical_integration.md).

## Deferred Fidelity Work

The Sutherland-Dopita cooling table/budget, fiducial Bondi radio-mode heating, reionization, group infall preparation, disk-radius setup, merger-clock initialization, signed infall application, shared-central Type-1 satellite stripping, disk instability, disk-triggered quasar mode, collisional starburst, correctly delayed disk enrichment, clock decrement, the ordered merger/disruption event map with immediate consumers, and descendant inheritance maps are now implemented. The next slice is the complete group/timestep orchestration that interleaves shared-central galaxy-major writes with the event phase, followed by the Mini-Millennium tree adapter. Full-tree and final-catalogue equivalence is not claimed until that driver and tree interface are present.

## Initial Milestone Evidence

The executable oracle now compares 146 fields from inheritance, reionization/infall, disk-radius and merger-clock setup, satellite stripping, cooling-table, cooling-budget/radio-mode/application, reincorporation, quiescent SF/SN, disk instability, quasar mode, starburst feedback, delayed enrichment, the composed post-quiescent chain, and a disruption-plus-merger event chain. One hundred thirty-nine controlled CPU fields are exact; cooling-budget and composed radio-mode fields use `rtol=1e-13`, while the FoF group infall sum uses `rtol=1e-15`, all with `atol=0`. Python tests additionally cover snapshot reset and fixed-branch inheritance; disk-radius precision and satellite freezing; clock sentinel, ceiling, and particle-floor behavior; live event order and target redirection; ownership transfer and the explicit disruption BH sink; group conservation and its derivative; stripping's geometric substep dependence; signed source/sink ledgers and derivatives; quasar/starburst conservation and fractional responses; exact sequential scan; substep refinement metadata; and positivity. These are process-level results, not full-tree or Mini-Millennium equivalence.

## Gates

- MIMIC's existing C build and relevant SAGE16 unit tests pass unchanged.
- JAX process equivalence cases state `rtol` and `atol`; unexplained differences fail.
- Conservation tests cover both central and satellite destinations.
- JIT and VMAP results match eager execution; gradients match finite differences on smooth branches.
- Benchmarks separate first-call compilation from warmed execution and make no upstream speedup claim before comparable end-to-end workloads exist.
- Documentation records source functions, equations, conservation behavior, differentiability class, and current limitations for every implemented process.
