# Fiducial SAGE16 Satellite Stripping

The JAX implementation of Type-1 satellite hot-gas stripping is a direct finite-transfer map, not a generic rate equation. Its public result contains both updated immutable galaxy states and the gas/metal transfer, so ownership and conservation are inspectable without reconstructing procedural writes.

## Upstream correspondence

The source is [`sage_satellite_stripping.c`](../models/sage16/modules/sage_satellite_stripping/sage_satellite_stripping.c). For a Type-1 satellite with hot gas, SAGE16 forms the live baryonic total

`M_b,sat = M_star + M_cold + M_hot + M_ejected + M_BH + M_ICS`

and the reionization-limited allowance

`M_allowed = f_b,halo M_vir`.

If the satellite's stored `HaloBaryonFraction` is not positive, the module uses `GlobalBaryonFraction`. On each substep it requests

`Delta M_strip = max(0, M_b,sat - M_allowed) / N_substep`,

then caps the request by the available satellite hot gas. Metals follow the capped hot-gas metallicity, with an independent source-metal cap. The same committed gas and metal values are subtracted from the satellite and added to the FoF central.

[`apply_satellite_stripping`](../mimic_jax/sage16/processes/satellite_stripping.py) preserves the upstream float32 reservoir sum and write boundaries while retaining double precision for the requested transfer.

## Ordering is part of the model

The fiducial configuration first runs full-FoF infall and reincorporation, then enters the galaxy-major local-module pass. For each galaxy, stripping is immediately followed by cooling-budget calculation, radio-mode heating, cooling, and the remaining local physics. The FoF central is normally visited before its satellites. Consequently, hot gas deposited into the central by a later satellite cannot cool in that central until the next substep.

A future full-group JAX driver must preserve this interleaving. Applying stripping to every satellite in a separate vectorized pass before cooling would change SAGE16 even if the isolated stripping formula were identical.

## Exact numerical behavior

The excess is recomputed after every transfer. If no other process changes the satellite, `N` repeated calls leave

`M_b,sat - M_allowed = (M_b,sat^0 - M_allowed) (1 - 1/N)^N`.

The fraction of the initial excess stripped during the interval is therefore

`1 - (1 - 1/N)^N`,

which is one for `N=1` and approaches `1 - 1/e` rather than one as `N` grows. The map does not use `halo.dT`. This is inherited SAGE behavior and is preserved in the reference method. It is a numerical dependence to quantify, not something to silently replace while claiming equivalence.

## Conservation and MIMIC parity

Gas is a closed transfer across the satellite-plus-central boundary. MIMIC also credits the central with exactly the metal mass debited from the satellite. That symmetric update fixes a classic SAGE cap-regime metal-loss edge case while remaining identical on the validated Mini-Millennium baseline described in the upstream module. mimic-jax follows MIMIC, its direct upstream for equivalence.

The JAX tests check combined baryon and metal totals at float-storage tolerances, including the gas/metal-cap branch. They also differentiate the combined hot-gas total with respect to the process perturbation and require a zero derivative.

## Fractional process response

Sensitivity experiments multiply the requested live excess by `exp(epsilon_satellite_stripping)` before ordinary caps. At zero this is exactly the faithful calculation. On a smooth uncapped branch,

`d ln(O) / d epsilon_satellite_stripping`

means: “A 1% increase in satellite stripping in this finite epoch produces approximately this percentage change in the positive observable.” Type selection, the excess threshold, and reservoir caps remain exact and are not smoothed.

The current controlled central-history example reports a zero reference row because it contains no satellite. Population historical responses require the future group/tree driver that preserves the galaxy-major ordering above.

## Evidence and scope

[`tests/mimic_jax/test_satellite_stripping.py`](../tests/mimic_jax/test_satellite_stripping.py) covers the formula, gates, caps, geometric `N` dependence, JIT, finite-difference response validation, and conservation before and after differentiation. [`scripts/check_mimic_jax_equivalence.py`](../scripts/check_mimic_jax_equivalence.py) compares the four controlled output reservoirs with the compiled C module exactly.

This is process-level equivalence. It is not yet a full-tree or Mini-Millennium result.
