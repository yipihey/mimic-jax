# Conservation as Executable SAGE16 Physics

Conservation checks operate on physical reservoirs, not every field in `GalaxyState`. `BulgeMass` is a component of `StellarMass`; transport buffers are pending moves; SFR and energy fields are diagnostics. Counting them as additional mass would create a false invariant.

The closed baryonic total used by the initial slice is `M_b = M_cold + M_hot + M_ejected + M_star + M_ICS + M_BH`. The corresponding pre-existing metal total is `M_Z = M_Z,cold + M_Z,hot + M_Z,ejected + M_Z,star + M_Z,ICS`.

| Operation | Baryonic behavior | Metal behavior |
| --- | --- | --- |
| Cooling application | conserved, hot to cold | conserved, hot to cold at hot-gas metallicity |
| Reincorporation | conserved, ejected to hot | conserved, ejected to hot at ejected-gas metallicity |
| Star formation plus recycling | conserved across cold gas and long-lived stars | pre-existing metals conserved across cold gas and stars |
| SN reheating/ejection | conserved across local cold and central hot/ejected reservoirs | pre-existing metals conserved across the same destinations |
| Stellar enrichment | baryonic mass unchanged | increases total metals by exactly `Yield * NewStellarMass`, subject to upstream float-storage rounding |

Tests assert both the invariants and the source term. Central and satellite cases are separate because satellite reheating and ejection mutate the FoF central. Tolerances reflect sequential writes into upstream-compatible float32 reservoirs: current controlled tests use absolute baryonic tolerance `2e-6` per galaxy and metal tolerance `2e-7` per galaxy. The compiled C oracle uses exact equality for its controlled CPU cases.

Future infall, stripping, BH radiation, disruption, and merger slices must label external sources, sinks, or ownership changes rather than weakening the ledger. A failing invariant is evidence of either a coding error or an incomplete system boundary; it is not fixed by increasing a tolerance without a numerical explanation.

Current code: [`mimic_jax/sage16/conservation.py`](../mimic_jax/sage16/conservation.py). Tests: [`tests/mimic_jax/test_processes.py`](../tests/mimic_jax/test_processes.py).
