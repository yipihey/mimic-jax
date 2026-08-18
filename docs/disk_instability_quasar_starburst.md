# Disk Instability, Quasar Mode, and Starbursts

These three fiducial SAGE16 modules form one ordered finite-transfer chain. mimic-jax exposes each calculation separately, but its `upstream_sequential` reference step composes them in the exact configured order and delays quiescent disk enrichment until they have finished.

## Disk instability: structure first, gas trigger second

The upstream sources are [`sage_disk_instability.c`](../models/sage16/modules/sage_disk_instability/sage_disk_instability.c) and [`sage_disk_instability_physics.h`](../models/sage16/shared/sage_disk_instability_physics.h). SAGE forms

`M_disk = M_cold + (M_star - M_bulge)`

and

`M_crit = min[M_disk, V_max^2 (StarFormingDiskFactor R_disk) / G]`.

The unstable mass `M_disk - M_crit` is divided between gas and stellar disk in their current mass fractions. Unstable stellar mass and its pre-existing metals move into the bulge components; total stellar mass and total stellar metals do not change. Unstable gas is not moved by this module. Instead, `UnstableDiskGasFraction` carries the finite trigger to the immediately following quasar and starburst modules.

[`apply_disk_instability`](../mimic_jax/sage16/processes/disk_instability.py) preserves the float32 expression and component-write boundaries. Its `disk_instability` response channel scales the live unstable mass by `exp(epsilon)` before the physical disk-mass cap. On the active uncapped branch, a response says what changes if the instability-driven structural response is 1% stronger in that epoch.

## Quasar mode: BH growth followed by threshold winds

The upstream sources are [`sage_quasar_mode.c`](../models/sage16/modules/sage_quasar_mode/sage_quasar_mode.c) and [`sage_agn_physics.h`](../models/sage16/shared/sage_agn_physics.h). For trigger efficiency `e`, cold-gas BH accretion is

`Delta M_BH = min[M_cold, BlackHoleGrowthRate e M_cold / (1 + (280 / V_vir)^2)]`.

That mass leaves cold gas and enters `BlackHoleMass`; the associated cold metals are an explicit tracked-metal sink because SAGE16 has no BH-metal reservoir. The accumulated `QuasarModeBHaccretionMass` diagnostic receives the same mass.

Quasar energy is `QuasarModeEfficiency * 0.1 * Delta M_BH * (c / UnitVelocity)^2`. If it exceeds the binding energy of the post-accretion cold reservoir, all remaining cold gas and metals move to ejected gas. If it exceeds the combined cold-plus-hot binding energy, all hot gas and metals move as well. These are exact threshold decisions, not continuous partial-wind rates.

The `quasar_mode` response channel scales the BH accretion request and therefore the energy that powers the coupled wind. It does not independently tune BH growth and wind energy. A derivative is local to the current cold/hot ejection branch; crossing an energy threshold requires a finite perturbation study.

## Collisional starburst: a finite burst and immediate yield

The upstream sources are [`sage_starburst_feedback.c`](../models/sage16/modules/sage_starburst_feedback/sage_starburst_feedback.c) and [`sage_starburst_physics.h`](../models/sage16/shared/sage_starburst_physics.h). A disk-instability trigger uses burst efficiency `e_burst = e`; a merger event uses `e_burst = 0.56 e^0.7`. The requested burst mass is `e_burst M_cold`. SAGE calculates SN reheating and ejection with the familiar feedback parameters, renormalizes formed stars and reheated gas together when they exceed available cold gas, and sends reheated/ejected material to the FoF central reservoirs.

After instantaneous recycling, long-lived burst stars enter both `StellarMass` and `BulgeMass`. Pre-existing metals follow every transfer. New metals `Yield * formed_stars` are added immediately by the burst prescription, unlike the separately delayed quiescent disk yield. [`StarburstTransfer`](../mimic_jax/sage16/transfers.py) records the pre-existing metal moves and new-metal source separately.

For the disk-instability channel, mass is evaluated once per trigger, while `StarFormationRate` and `SupernovaOutflowRate` divide by the full object interval `halo.dT`. They do not divide by `halo.dT / num_substeps`. This is a finite map with snapshot-averaged diagnostics, not an ODE rate-times-substep prescription.

The `starburst` response channel scales the requested burst stellar mass before SAGE's cold-gas balance. The named `sn_reheating` and `sn_ejection` channels scale the corresponding feedback requests in both the quiescent and burst paths.

## Exact composed order

For each galaxy in the fiducial galaxy-major pass, the post-cooling portion is

`quiescent SF budget -> SN budget -> reservoir application -> disk instability -> quasar mode -> starburst feedback -> quiescent disk enrichment`.

Delaying the quiescent yield matters: quasar accretion/winds and the burst see the pre-yield metallicities, while the disk yield is deposited into whatever cold/hot reservoirs remain afterward. The compiled oracle includes a composed case specifically to guard this order.

Merger-triggered calls use the same quasar and starburst kernels, but their ordered event dispatch and post-minor-merger instability recheck remain part of the next merger/event milestone.

## Conservation, differentiation, and evidence

Disk instability changes components but not top-level totals. Quasar mode conserves baryonic mass across gas and BH reservoirs while declaring the BH-associated metal sink. A starburst conserves baryonic mass and increases total tracked metals by its explicit yield source. Tests cover central and satellite feedback destinations, JIT, finite-difference agreement, and derivative-level baryon conservation.

[`scripts/check_mimic_jax_equivalence.py`](../scripts/check_mimic_jax_equivalence.py) compares isolated disk-instability, quasar, and starburst cases plus the composed chain with compiled MIMIC. All 41 fields added by this slice match exactly. The result remains process-level evidence, not full-tree or Mini-Millennium equivalence.
