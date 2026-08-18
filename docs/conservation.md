# Conservation as Executable SAGE16 Physics

Conservation checks operate on physical reservoirs, not every field in `GalaxyState`. `BulgeMass` is a component of `StellarMass`; transport buffers are pending moves; SFR and energy fields are diagnostics. Counting them as additional mass would create a false invariant.

The closed baryonic total used by the initial slice is `M_b = M_cold + M_hot + M_ejected + M_star + M_ICS + M_BH`. The corresponding pre-existing metal total is `M_Z = M_Z,cold + M_Z,hot + M_Z,ejected + M_Z,star + M_Z,ICS`.

| Operation | Baryonic behavior | Metal behavior |
| --- | --- | --- |
| Reionization | no immediate reservoir change; changes a later baryon target | unchanged |
| Infall-budget consolidation | conserved across the complete FoF group | conserved for ejected gas and ICS transfers, subject to upstream validity clamps |
| Positive infall application | explicit external baryon source to hot gas | pristine source; no metals added |
| Negative infall application | explicit external sink from ejected then hot gas | removes metals at each source reservoir's metallicity |
| Satellite stripping | conserved across satellite and FoF central hot gas | conserved by crediting the central with exactly the metal mass debited from the satellite |
| Cooling-budget calculation | unchanged; writes transport/diagnostic fields only | unchanged |
| Cooling application | conserved, hot to cold | conserved, hot to cold at hot-gas metallicity |
| Radio-mode BH accretion | conserved, hot gas to BH | decreases tracked metals by explicit `hot_metals_accreted`; SAGE has no BH-metal reservoir |
| Reincorporation | conserved, ejected to hot | conserved, ejected to hot at ejected-gas metallicity |
| Star formation plus recycling | conserved across cold gas and long-lived stars | pre-existing metals conserved across cold gas and stars |
| SN reheating/ejection | conserved across local cold and central hot/ejected reservoirs | pre-existing metals conserved across the same destinations |
| Disk instability | top-level baryonic reservoirs unchanged; only the stellar bulge component changes | top-level stellar metals unchanged; only the bulge-metal component changes |
| Quasar BH growth and wind | conserved across cold/hot/ejected gas and BH mass | wind conserves tracked metals; BH accretion removes explicit `cold_metals_accreted` because SAGE has no BH-metal reservoir |
| Collisional starburst | conserved across local cold/stars and central hot/ejected reservoirs | pre-existing metals conserved; total metals increase by exactly `Yield * formed_stars` within float-storage tolerance |
| Stellar enrichment | baryonic mass unchanged | increases total metals by exactly `Yield * NewStellarMass`, subject to upstream float-storage rounding |
| Merger ownership | conserved across live source and target, then source ownership becomes Type 3 | conserved across corresponding target reservoirs |
| Satellite disruption | conserved after subtracting the explicit lost satellite BH mass | conserved; cold/hot metals enter target hot gas and stellar metals enter target ICS |

Tests assert both the invariants and the source term. Central and satellite cases are separate because satellite reheating and ejection mutate the FoF central. Tolerances reflect sequential writes into upstream-compatible float32 reservoirs: current controlled tests use absolute baryonic tolerance `2e-6` per galaxy and metal tolerance `2e-7` per galaxy. The compiled C oracle uses exact equality for its controlled CPU cases.

Consumed merger/disruption sources remain stored in the workspace but become Type 3, so summing every record would double-count transferred material. `active_group_baryonic_mass` and `active_group_metal_mass` sum only non-Type-3 owners. A failing invariant is evidence of either a coding error or an incomplete system boundary; it is not fixed by increasing a tolerance without a numerical explanation.

Conservation is also tested after differentiation. For the radio-mode hot-to-BH transfer, the derivative of `delta HotGas + delta BlackHoleMass` with respect to `RadioModeEfficiency` is zero on the controlled smooth branch. For satellite stripping, the derivative of combined satellite-plus-central hot gas with respect to its fractional process perturbation is zero. The corresponding derivatives of total baryonic mass through quasar-mode and starburst fractional perturbations are also zero on controlled smooth branches. The event-map test differentiates the live ownership residual with respect to source stellar mass and obtains zero away from classification thresholds. Numerical refinement studies use the same ledgers through [`mimic_jax/numerics.py`](../mimic_jax/numerics.py), so integration error, declared sources/sinks, and parameter response remain separate quantities.

Current code: [`mimic_jax/sage16/conservation.py`](../mimic_jax/sage16/conservation.py). Tests: [`tests/mimic_jax/test_processes.py`](../tests/mimic_jax/test_processes.py), [`tests/mimic_jax/test_satellite_stripping.py`](../tests/mimic_jax/test_satellite_stripping.py), [`tests/mimic_jax/test_instability_quasar_starburst.py`](../tests/mimic_jax/test_instability_quasar_starburst.py), and [`tests/mimic_jax/test_mergers.py`](../tests/mimic_jax/test_mergers.py).
