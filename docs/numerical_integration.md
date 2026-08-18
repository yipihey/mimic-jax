# Numerical Integration of Fiducial SAGE16

The reference numerical method in mimic-jax is the upstream SAGE16 execution itself. Alternative methods are experiments on the same physical prescriptions; they do not replace the reference path and cannot be used to claim upstream equivalence.

## What upstream actually does

The fiducial Mini-Millennium configuration sets `SubSteps: 10`. At the start of a tree interval, four pre-timestep modules run in configured order: reionization, FoF infall-budget preparation, Type-0 disk-radius update, and merger-clock initialization. Every substep then executes the configured modules in order. A later module sees reservoir writes made by every earlier module in that substep. The merger/disruption phase follows the ordinary galaxy-physics phase and emits discrete events immediately to its quasar and starburst consumers before continuing the satellite scan.

This is best described as an explicit sequential or operator-split update, not simply as one forward-Euler evaluation. For the currently implemented central slice, one substep is

`fixed-budget infall application -> reincorporation -> cooling budget -> stored-AGN suppression and new radio-mode heating -> cooling application -> star formation -> SN feedback -> reservoir application -> disk instability -> quasar mode -> starburst feedback -> delayed disk enrichment`.

The prescriptions have different numerical meanings:

| Kind | SAGE16 examples | Timestep behavior |
| --- | --- | --- |
| Rate multiplied by object substep duration | cooling budget, quiescent star formation, reincorporation, radio-mode BH accretion | uses `halo.dT / num_substeps`, followed by thresholds and source-reservoir caps |
| Snapshot budget partitioned across substeps | cosmological infall | prepares `InfallingGas` once, then applies `InfallingGas / num_substeps` |
| Recomputed fractional finite transfer | satellite stripping | recomputes the current excess and removes `excess / num_substeps`; over a fixed interval the stripped fraction is `1 - (1 - 1/N)^N`, not exactly one |
| Downstream finite transfer | cooling application, SN reheating/ejection, recycling, metal enrichment | consumes an earlier transport budget and commits bounded reservoir moves or explicit sources |
| Thresholded finite map | disk instability, quasar feedback, starburst feedback | applies a finite redistribution after a trigger; it is not automatically an ODE right-hand side |
| Discrete event/jump map | merger or disruption and its immediate consumers | preserves event order and tree identity; it is not passed to a continuous integrator |

Disk-radius and merger-clock initialization are neither rate integrations nor reservoir transfers. They are pre-timestep state maps: the first derives Type-0 disk structure from current halo forcing, while the second sets or resets persistent event clocks from type and sentinel state. Subcycling does not rerun either map.

`halo.dT` is the per-object tree-interval duration. `mimic_object_substep_dt` divides it by the configured substep count. Diagnostic rates such as SFR, cooling energy rate, and heating energy rate may divide committed amounts by the full `halo.dT`; those diagnostics should not be mistaken for the timestep used to calculate each transfer.

The disk-instability burst is a concrete example: the structural trigger is recomputed each substep, but its `StarFormationRate` and `SupernovaOutflowRate` increments divide by the full `halo.dT`, not the substep duration. The burst mass is a finite trigger-driven transfer. It must not be treated as a rate-times-`dt` term in a higher-order integrator.

## Reference API and forcing resolution

`upstream_sequential_central_step` implements the exact module order for the currently ported per-central subset. The FoF-wide `prepare_infall_budget` operation runs once before it, not inside the substep map. `subcycle_upstream_sequential_central(..., num_substeps=N)` holds the halo forcing and prepared infall budget fixed over one tree interval and repeats the reference map `N` times, while each rate-based module uses `halo.dT / N` and infall applies one `InfallingGas / N` partition. This is the initial, explicitly labeled `piecewise_constant` forcing assumption.

The merger-tree sampling and baryonic integration resolution are separate concepts. Future forcing interpolation must be selected explicitly and recorded in outputs. Piecewise-constant and linear interpolation will be compared before anything more elaborate is considered. Interpolation must not alter event times or invent smooth tree topology. The implemented merger phase derives each source's substep duration independently, decrements its float clock, and preserves event sequence as a discrete scan; it is outside every continuous higher-order integrator experiment.

Satellite stripping supplies an especially important numerical counterexample to a generic ODE interpretation. Its transfer is independent of `halo.dT`; changing only `num_substeps` changes the finite interval map, approaching a stripped excess fraction of `1 - 1/e` as `N` grows. This inherited behavior is retained in `upstream_sequential`. It must be measured separately from convergence of genuinely rate-times-`dt` processes, and it must not be silently “corrected” in an equivalence run.

## Executable numerical diagnostics

`timestep_refinement_study` runs successively refined substep counts, stores observables and absolute differences from the finest requested run, and estimates empirical convergence orders from consecutive error ratios. The finest run is labeled a provisional reference; it is not silently treated as the exact solution. Results store the method and forcing-interpolation names and can be saved as an NPZ archive.

`conservation_residual` evaluates `delta(total) - (sources - sinks)`. `step_to_timescale_ratio` evaluates the dimensionless finite-step diagnostic `|transfer| / source`, equivalent to `dt / tau` when `tau = source / |rate|`. Large values identify process/reservoir pairs that deserve timestep-refinement attention. Empty sources with nonzero transfers return infinity rather than hiding an invalid ledger.

Tests already exercise `1, 2, 4, 8` substeps on the controlled implemented central chain, require nonnegative stored reservoirs, check the baryon ledger, preserve JIT execution, and differentiate through the subcycled path. These tests establish the machinery, not Mini-Millennium convergence.

## Alternative integrators: strict boundary

Forward Euler, Heun or midpoint RK2, RK4, adaptive embedded Runge-Kutta, symmetric splitting, and conservative positivity-preserving methods are candidate numerical experiments only where the port exposes a mathematically legitimate continuous rate system. They must not be applied blindly to finite caps, snapshot partitions, merger events, or thresholded jump maps.

The first higher-order experiment will therefore define and test a continuous-rate subset explicitly, compare it with the upstream sequential map at matched forcing, and report conservation, positivity, derivative behavior, right-hand-side evaluations, and wall-clock cost. Adaptive accept/reject decisions will be documented as piecewise or non-smooth. Modified Patankar-type methods will be investigated only if ordinary schemes exhibit material positivity or conservation failures in SAGE16 regimes.

## Required Mini-Millennium evidence

After full pipeline equivalence, the numerical application will measure final reservoirs and familiar population statistics under `N, 2N, 4N, 8N` substeps; separate tree-forcing interpolation from baryonic subcycling; quantify sensible module-order alternatives; and compare numerical shifts with parameter-response magnitudes. A useful conclusion may be that the upstream scheme is already converged, that only specific mass/redshift regimes need refinement, or that event/tree discreteness dominates. The tests, rather than the choice of a sophisticated solver, determine the conclusion.

Current code: [`mimic_jax/sage16/evolve.py`](../mimic_jax/sage16/evolve.py) and [`mimic_jax/numerics.py`](../mimic_jax/numerics.py). Tests: [`tests/mimic_jax/test_numerics.py`](../tests/mimic_jax/test_numerics.py).

The runnable [`examples/sage16_timestep_refinement.py`](../examples/sage16_timestep_refinement.py) prints the controlled slice at `1, 2, 4, 8` substeps, its provisional-reference differences, and baryon residuals. Its final line states the scope explicitly so the example cannot be mistaken for a Mini-Millennium convergence result.
