# Fiducial SAGE16 Radio-Mode Heating

The JAX kernel follows [`sage_radio_mode_heating.c`](../models/sage16/modules/sage_radio_mode_heating/sage_radio_mode_heating.c). The fiducial Mini-Millennium configuration selects `AGNrecipe = 2`, the Bondi-like prescription, and `RadioModeEfficiency = 0.08`.

The module first reduces the current finite cooling budget using the heating radius stored from earlier substeps. If `Rheat < Rcool`, it keeps `(1 - Rheat / Rcool) CoolingGas`; otherwise it suppresses the current budget completely. It then calculates radio-mode BH accretion, applies the Eddington, available-hot-gas, and available-cooling-heating caps, transfers accreted mass from hot gas to the BH, removes the corresponding hot-gas metals, accumulates a heating-energy diagnostic, and may grow `Rheat`.

A subtle but deliberate SAGE ordering is preserved: newly generated heating does not subtract `AGNheating` from `CoolingGas` again in the same call. It changes `Rheat`, which affects later substeps. This is why the explicit transfer record distinguishes `cooling_after_prior_heating`, `heating_mass`, and `heating_radius_after`.

Radio-mode accretion conserves modeled baryonic mass across `HotGas -> BlackHoleMass`, subject to upstream float32 storage rounding. SAGE has no black-hole metal reservoir, so `hot_metals_accreted` is an explicit sink from the tracked metal ledger rather than a false metal-conservation claim.

The named historical perturbation `agn_heating` multiplies the selected upstream radio-mode accretion rate as `rate -> rate exp(epsilon)` before the unchanged Eddington and physical caps. It consequently perturbs the coupled BH-growth/heating prescription without breaking their upstream relation. At zero perturbation the C reference path is unchanged.

The process is smooth within a fixed accretion recipe and away from the Eddington, hot-gas, cooling-heating, and radius-update caps. `AGNrecipe` selection, complete prior-heating suppression, and cold-cloud triggering are discrete or thresholded. The implementation avoids undefined derivatives in behaviorally inactive branches while retaining identical forward state updates.

The compiled oracle composes the real C cooling-budget and radio-mode modules and compares `CoolingGas`, `BlackHoleMass`, `HotGas`, `MetalsHotGas`, `Rheat`, and `Heating`. The controlled chain agrees within `rtol=1e-13`, `atol=0`; local tests also validate JIT execution, finite-difference response, baryon and metal ledgers, and a finite multi-epoch AGN response.
