# SAGE16 Reservoirs and Explicit Transfers

mimic-jax keeps upstream SAGE16 prescriptions recognizable and separates two questions that procedural code often combines: what physical transfer is requested, and how that transfer changes stored reservoirs. A transfer is a finite-substep amount unless the upstream module genuinely defines a rate.

## Implemented process slice

| Process | Upstream implementation | Explicit transfer | Reservoir action |
| --- | --- | --- | --- |
| Cooling budget | [`sage_calculate_cooling_budget.c`](../models/sage16/modules/sage_calculate_cooling_budget/sage_calculate_cooling_budget.c) | `CoolingBudget(gas, radius, cooling_lambda)` | calculates transport only |
| Cooling application | [`sage_apply_cooling.c`](../models/sage16/modules/sage_apply_cooling/sage_apply_cooling.c) | `CoolingTransfer(gas, metals)` | hot to cold |
| Reincorporation | [`sage_reincorporation.c`](../models/sage16/modules/sage_reincorporation/sage_reincorporation.c) | `ReincorporationTransfer(gas, metals)` | ejected to hot |
| Quiescent star formation | [`sage_calculate_star_formation.c`](../models/sage16/modules/sage_calculate_star_formation/sage_calculate_star_formation.c) | `StarFormationBudget.NewStellarMass` | cold to long-lived stars after recycling |
| SN reheating/ejection | [`sage_calculate_supernova_feedback.c`](../models/sage16/modules/sage_calculate_supernova_feedback/sage_calculate_supernova_feedback.c) | reheated and ejected fields in `StarFormationBudget` | cold to hot to ejected |
| SF/SN application | [`sage_apply_star_formation_supernova.c`](../models/sage16/modules/sage_apply_star_formation_supernova/sage_apply_star_formation_supernova.c) | `StarFormationTransfer` | commits mass and pre-existing metals |
| Disk-SF enrichment | [`sage_apply_metal_enrichment.c`](../models/sage16/modules/sage_apply_metal_enrichment/sage_apply_metal_enrichment.c) | `MetalEnrichmentTransfer` | explicit new-metal source to cold/hot gas |

For quiescent star formation, SAGE uses `r_eff = StarFormingDiskFactor * DiskScaleRadius`, `M_cold,crit = 0.19 V_vir r_eff`, and `dot(M)_star = SfrEfficiency (M_cold - M_cold,crit) / (r_eff / V_vir)` above the threshold. The calculated finite-substep stellar mass is then passed to SN feedback, which renormalizes star formation and reheating together if their sum exceeds the available cold gas. mimic-jax preserves that calculation/apply order and the intermediate double-precision budgets.

The committed long-lived stellar mass is `(1 - RecycleFraction) * NewStellarMass`. Reheated material moves from the local galaxy's cold disk to the FoF central's hot atmosphere. Ejected material then moves from that central hot atmosphere to its ejected reservoir. This distinction matters for satellites, and the conservation tests cover the combined local-plus-central system.

The disk yield is not a conserved transfer: `Yield * NewStellarMass` is newly produced metal mass. mimic-jax returns it explicitly as `produced_metals`, then records its split between cold disk gas and the central hot atmosphere.

## Process perturbations

The faithful calculation is always the zero-perturbation path. Sensitivity experiments wrap a named transfer or requested budget as `m -> m exp(epsilon)` before the ordinary SAGE capacity limits and downstream application. At `epsilon = 0`, the compiled C equivalence cases remain bit-identical. The currently implemented names are `cooling`, `star_formation`, `sn_reheating`, `sn_ejection`, and `reincorporation`; AGN and BH channels will be added only with their faithful physics implementations.

This construction means a process response has a direct interpretation: `d ln(O) / d epsilon = -0.4` means that making that process 1% stronger during the selected finite epoch changes the final positive observable by approximately -0.4% near the fiducial history.

Current code: [`mimic_jax/sage16/transfers.py`](../mimic_jax/sage16/transfers.py), [`mimic_jax/sage16/processes/`](../mimic_jax/sage16/processes/), and [`mimic_jax/sage16/perturbations.py`](../mimic_jax/sage16/perturbations.py).

Cooling-table interpolation and the two cooling regimes are documented separately in [`cooling.md`](cooling.md).
