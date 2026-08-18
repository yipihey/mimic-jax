# Fiducial SAGE16 Cooling

The cooling implementation follows two upstream components: [`cooling_tables.c`](../models/sage16/modules/sage_calculate_cooling_budget/cooling_tables.c) interpolates the Sutherland-Dopita tables, and [`sage_calculate_cooling_budget.c`](../models/sage16/modules/sage_calculate_cooling_budget/sage_calculate_cooling_budget.c) calculates a finite-substep gas budget. [`sage_apply_cooling.c`](../models/sage16/modules/sage_apply_cooling/sage_apply_cooling.c) later commits that budget after AGN heating has had an opportunity to reduce it.

`load_cooling_tables` is ordinary Python I/O. It reads column six of the eight model-owned `stripped_*.cie` files as float32, matching upstream's `%f` parse, then promotes the values to float64 JAX arrays. The pure numerical function receives those arrays explicitly, so there is no hidden mutable table global inside a JIT kernel.

For a hot reservoir, SAGE sets `T_vir = 35.9 V_vir^2`, evaluates `Lambda(log10(T_vir), log10(Z_hot))`, derives the density at which gas cools in one halo dynamical time, and computes `R_cool` for an isothermal hot-gas density profile. If `R_cool > R_vir`, the cold-accretion budget is `M_hot / (R_vir / V_vir) * dt`. Otherwise the hot-halo budget is `(M_hot / R_vir) * [R_cool / (2 R_vir / V_vir)] * dt`. The result is capped at `M_hot` and stored in `CoolingGas`; no reservoir moves until `apply_cooling`.

The table interpolation is piecewise linear in `log10(T)` and `log10(Z)`, followed by `10^logLambda`. Metallicity is clamped to the eight-table range. Temperature below `log10(T/K)=4` is clamped. Upstream's current high-temperature implementation fixes the table index to the last interval but continues to extrapolate beyond `8.5`; mimic-jax preserves this behavior for equivalence rather than silently converting it to a high-temperature clamp.

Within a temperature/metallicity cell and away from the cold-accretion switch and gas cap, the budget is smooth. Table knots, the `R_cool = R_vir` regime boundary, empty reservoirs, and the cap are piecewise or threshold boundaries and are reported as such.

The compiled reference oracle currently compares four interpolation probes and three cooling-budget fields. Interpolation probes are exact on the controlled CPU case; `CoolingGas`, `Rcool`, and `CoolingLambda` agree within `rtol=1e-13`, reflecting double-precision transcendental operation ordering across C libm and XLA. The combined calculate/apply test also checks baryonic and metal conservation.
