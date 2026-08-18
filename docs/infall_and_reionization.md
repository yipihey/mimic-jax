# Fiducial SAGE16 Reionization and Infall

The baryon-supply path follows three upstream modules with deliberately different scopes: [`sage_reionization.c`](../models/sage16/modules/sage_reionization/sage_reionization.c), [`sage_prepare_infall_budget.c`](../models/sage16/modules/sage_prepare_infall_budget/sage_prepare_infall_budget.c), and [`sage_apply_infall.c`](../models/sage16/modules/sage_apply_infall/sage_apply_infall.c).

## Reionization

For every non-Type-3 halo, SAGE evaluates the Gnedin filtering-mass prescription with the Kravtsov fitting formula and sets `HaloBaryonFraction = GlobalBaryonFraction * modifier(Mvir, z)`. The UV background turns on at `z = 8` and reionization completes at `z = 7` in this prescription. The characteristic mass also depends on `Omega`, `OmegaLambda`, code-unit `H(z)`, and `G`. Upstream passes `Mvir` through a float helper argument even though the halo field is double; mimic-jax preserves that precision boundary.

The modifier is smooth within each scale-factor branch and for positive halo mass. The `z = 8` and `z = 7` branch boundaries, the near-zero-mass guard, and Type-3 exclusion are thresholded or discrete. `GlobalBaryonFraction` remains an ordinary positive differentiable parameter on an active branch.

## Snapshot infall budget

`prepare_infall_budget` receives all galaxies in one FoF group and an explicit central index. It sums stellar, BH, cold, hot, ejected, and ICS baryons over non-Type-3 members. Satellite ejected gas and ICS, including their tracked metals, are transferred to the central; satellite hot gas is deliberately retained, including for Type-2 orphans. The central then receives the finite snapshot budget

`InfallingGas = HaloBaryonFraction * Mvir_central - total_group_baryons`.

If the central still has the `-1` baryon-fraction sentinel, the module substitutes `GlobalBaryonFraction`. Consolidation is a closed ownership change across the group. The calculated `InfallingGas` is a transport budget and can be negative; it is not itself a persistent mass reservoir.

## Per-substep application

`apply_infall` applies `InfallingGas / num_substeps` to the central on every substep. Positive infall is pristine external mass added to `HotGas`. Negative infall is an external sink: it removes `EjectedGas` first, then `HotGas`, carrying each reservoir's metallicity and stopping at zero if the requested removal exceeds available gas. The explicit `InfallTransfer` records requested mass, external source, gas and metal sinks, and any unfulfilled removal.

The named `infall` process perturbation multiplies the signed fixed budget by `exp(epsilon)` before division across substeps. It preserves the sign and upstream priority/caps. A positive historical response therefore asks what changes if cosmological supply is fractionally larger; on a negative-budget epoch it asks what changes if the modeled baryon loss is fractionally stronger. Metadata must retain the signed fiducial budget so those cases are not conflated.

The compiled oracle checks one reionization value, group consolidation and budget fields, positive partitioning, and the negative ejected-to-hot cascade. Of the new controlled fields, all but the group budget are exact; the group budget uses `rtol=1e-15`, `atol=0` for the final-bit difference between an XLA reduction and the scalar C addition order.
