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

`halo.dT` is the per-object tree-interval duration inherited as `source progenitor time - descendant time`. It can differ among records in one live FoF workspace. `mimic_object_substep_dt` divides it by the configured substep count. The shared `context.time_interval` is derived from the FoF central's retained progenitor snapshot and is used for scheduler metadata, not as a substitute for object-local physics time. Diagnostic rates such as SFR, cooling energy rate, and heating energy rate may divide committed amounts by the full `halo.dT`; those diagnostics should not be mistaken for the timestep used to calculate each transfer.

The disk-instability burst is a concrete example: the structural trigger is recomputed each substep, but its `StarFormationRate` and `SupernovaOutflowRate` increments divide by the full `halo.dT`, not the substep duration. The burst mass is a finite trigger-driven transfer. It must not be treated as a rate-times-`dt` term in a higher-order integrator.

## Reference API and forcing resolution

`upstream_sequential_central_step` remains a compact single-central test slice. `evolve_upstream_sequential_group_interval` is the complete fiducial FoF schedule: it runs all four pre-timestep maps once, repeats the full-halo and galaxy-major physics in exact order, and then executes the ordered event phase. It holds halo forcing fixed over one tree interval, while each rate-based module uses its object's `halo.dT / N` and infall applies one `InfallingGas / N` partition. This is the initial, explicitly labeled `piecewise_constant` forcing assumption. See [`group_evolution.md`](group_evolution.md) for the live shared-central ordering.

The merger-tree sampling and baryonic integration resolution are separate concepts. Future forcing interpolation must be selected explicitly and recorded in outputs. Piecewise-constant and linear interpolation will be compared before anything more elaborate is considered. Interpolation must not alter event times or invent smooth tree topology. The implemented merger phase derives each source's substep duration independently, decrements its float clock, and preserves event sequence as a discrete scan; it is outside every continuous higher-order integrator experiment.

Satellite stripping supplies an especially important numerical counterexample to a generic ODE interpretation. Its transfer is independent of `halo.dT`; changing only `num_substeps` changes the finite interval map, approaching a stripped excess fraction of `1 - 1/e` as `N` grows. This inherited behavior is retained in `upstream_sequential`. It must be measured separately from convergence of genuinely rate-times-`dt` processes, and it must not be silently “corrected” in an equivalence run.

## Executable numerical diagnostics

`timestep_refinement_study` runs successively refined substep counts, stores observables and absolute differences from the finest requested run, and estimates empirical convergence orders from consecutive error ratios. The finest run is labeled a provisional reference; it is not silently treated as the exact solution. Results store the method and forcing-interpolation names and can be saved as an NPZ archive.

`conservation_residual` evaluates `delta(total) - (sources - sinks)`. `step_to_timescale_ratio` evaluates the dimensionless finite-step diagnostic `|transfer| / source`, equivalent to `dt / tau` when `tau = source / |rate|`. Large values identify process/reservoir pairs that deserve timestep-refinement attention. Empty sources with nonzero transfers return infinity rather than hiding an invalid ledger.

Tests already exercise `1, 2, 4, 8` substeps on the controlled central chain, require nonnegative stored reservoirs, check the baryon ledger, preserve JIT execution, and differentiate through the subcycled path. Group-level tests additionally cover exact pre-timestep placement, object-local durations, live Type-3 ownership, JIT, VMAP across independent groups, baryon conservation, and its zero derivative. These tests establish the machinery, not Mini-Millennium convergence.

## Continuous and hybrid formulation

The first alternative-integration slice is now executable. `Sage16OdeState` contains the quiescent central reservoirs, while `Sage16HybridState` adds the persistent BH, structural, event-clock, and `Rheat` history variables needed for the wider hybrid system. The initial fixed-forcing RHS includes cooling, quiescent star formation with instantaneous recycling, SN reheating/ejection, reincorporation, and their metal flows. The hybrid layer additionally exposes prepared infall, radio-mode BH growth and AGN-regulated cooling, merger-clock countdown, and the continuous fixed-forcing limit of satellite stripping.

AGN illustrates why this is a hybrid rather than an ODE-only design. Its mass transport is rate based, but SAGE updates the stored heating radius with a monotone projection after evaluating new heating. `Rheat` is therefore part of the Markov state, while its upstream update remains an explicit projection. Disk instability, quasar/starburst triggers, mergers, and disruption remain finite maps or events. See [`sage16_hybrid_system.md`](sage16_hybrid_system.md) for the complete process classification.

`integrate_fixed_step` supplies forward Euler, Heun RK2, and RK4 for a declared continuous RHS. It does not automatically reinterpret event maps as rates. The controlled smooth-rate experiment compares those methods with repeated upstream-order rate modules under identical fixed halo forcing. Its tests recover first-, second-, and fourth-order convergence respectively, preserve the continuous baryon invariant to floating-point precision, and retain JIT and gradient support.

## Adaptive continuous-flow integration

`integrate_adaptive` implements the embedded Dormand–Prince 5(4) pair. Each attempted step estimates a weighted root-mean-square local error with component scales `atol + rtol * max(|x_old|, |x_new|)`. A separate stability cap evaluates the tolerance-scaled Jacobian `D^-1 (df/dx) D` and requires `h ||D^-1 J D||_infinity` to remain below the configured factor. Scaling is essential: a raw Jacobian norm changes if a reservoir, radius, or metallicity is merely expressed in different units.

Adaptive steps operate only between declared external boundaries. Finite SAGE maps, merger-tree forcing changes, merger/disruption events, disk-instability projections, and the monotone `Rheat` update are never inserted at internal Runge–Kutta stages. `integrate_sage16_ode_adaptive` and `integrate_sage16_hybrid_flow_adaptive` expose this contract directly. The padded result records accepted and rejected steps, RHS evaluations, local-error norms, scaled-Jacobian norms, and a status code suitable for JIT and batched execution. Step-size and accept/reject choices are stop-gradient decisions, so derivatives describe the selected numerical branch rather than smoothing controller switches.

The first Mini-Millennium experiment samples 64 trees at snapshot 63, obtains 52 candidate central-galaxy intervals, and retains 27 whose four mass reservoirs stay positive and whose 4,096-step reference trajectory crosses neither the quiescent-star-formation nor cooling-regime threshold. All 27 complete at every tested tolerance from `1e-3` to `1e-9`. At `rtol=1e-7`, comparison with the independent RK4 reference gives median/maximum per-galaxy reservoir errors of `2.10e-9`/`6.00e-7`, a maximum stellar-mass difference of `2.54e-9 dex`, and a maximum baryon residual of `2.22e-16`. The derivative with respect to `SfrEfficiency` agrees with symmetric finite differences at three perturbation sizes to `8.31e-7` relative error. At `rtol=1e-9`, the median reservoir error falls to `6.16e-11` and the maximum stellar-mass difference to `8.77e-10 dex`.

This evidence supports convergence of the separated continuous flows under fixed halo forcing; it is not yet a full-tree adaptive-convergence claim. The 25 excluded boundary- or threshold-crossing candidates require root/event localization and bounded reservoir maps before they can be included without changing the mathematical problem. Population stellar-mass functions also require adaptive flow integration across every tree interval with forcing and genuine maps held on one common event schedule. The machine-readable experiment is produced by [`analyze_mini_millennium_adaptive.py`](../scripts/analyze_mini_millennium_adaptive.py) and appears in the [science-program report](../reports/mini-millennium-sage16-science-program/index.md).

## Alternative integrators: strict boundary

Forward Euler, Heun or midpoint RK2, RK4, adaptive embedded Runge-Kutta, symmetric splitting, and conservative positivity-preserving methods are numerical experiments only where the port exposes a mathematically legitimate continuous rate system. They must not be applied blindly to finite caps, snapshot partitions, merger events, or thresholded jump maps.

The fixed-step and adaptive experiments define their continuous-rate subset explicitly, compare at matched forcing, and report conservation, positivity, derivative behavior, and right-hand-side evaluations. A complete population run and steady-state cost benchmark remain pending. Modified Patankar-type methods will be investigated only if ordinary schemes exhibit material positivity or conservation failures in SAGE16 regimes.

## Mini-Millennium substep evidence

The first population experiment now evolves 500 trees spread across the complete input partition at 5, 10, 20, 40, and 80 substeps under identical piecewise-constant tree forcing. It resolves 25 stellar-mass-function bins over `8.05 <= log10(Mstar/Msun) <= 10.55`. Relative to the provisional 80-substep endpoint, the fiducial 10-substep run differs by 4.81% in the median resolved bin and 57.8% at maximum. The 40-to-80 median difference remains 4.23%, and total stellar mass changes by 81.5% between 10 and 80 substeps. The sequence therefore does not support a claim that the complete upstream schedule is converged under naive `SubSteps` refinement.

This result is not interpreted as evidence that a single Euler-like discretization is inaccurate. `SubSteps` controls both ordinary rate-times-`dt` updates and the number of times SAGE recomputes finite stripping, threshold, instability, and event maps. Refinement can therefore change the discrete model realization rather than approach one continuous solution. The smooth-interval adaptive experiment above now confirms that the separated continuous flows converge when genuine finite/event maps are held outside the integrator. Tree-forcing interpolation, boundary localization, sensible module-order alternatives, positivity across boundary maps, and cost at matched accuracy remain to be evaluated on the complete population.

### Why the stellar-mass residual looks like ringing

The alternating excesses and deficits in the stellar mass function are along the **stellar-mass coordinate**, not an oscillation measured through cosmic time. A matched-catalogue comparison of the same 500 trees at 10 and 80 substeps finds all 599 final galaxy identities in both catalogues, no galaxy type changes, and a positive median displacement `log10(Mstar_10 / Mstar_80) = 0.00980 dex`. The coarse run produces the larger stellar mass for 92.3% of positive-mass matches. Common identities account for the complete difference in total stellar mass.

For a small mass-coordinate displacement `delta m`, number conservation gives the leading response

`delta phi(m) approximately -d[phi(m) delta m(m)] / dm`.

Consequently, coherent motion through a steep and finite-sample mass distribution creates adjacent positive and negative lobes at the locations of the original distribution's slopes and sampling features. Increasing the mass shift increases their amplitude without moving those locations. This explains why the residual curves can retain approximately the same phase as the substep count changes. It is transport across mass bins, not by itself evidence for a time-domain numerical mode.

The diagnostic also separates Type-0 centrals and satellites. With the report's 0.05-dex Gaussian-CDF estimator, their median absolute 10-to-80 differences are 5.31% and 5.68%, respectively, and neither identities nor final types change. Satellite stripping may still affect central descendants through their progenitors, but direct satellite classification changes cannot be the sole cause. Increasing the estimator bandwidth changes the amplitude but does not remove the underlying matched-galaxy mass displacement; the positive smoothing kernel therefore reveals and attenuates bin transport rather than generating a Gibbs phenomenon.

One deliberately timestep-sensitive, 88-halo tree supplies an initial module ablation. Its final 10-to-80-substep stellar-mass ratio is 9.477 in the fiducial schedule and 9.538 when satellite stripping is suppressed. The ratio falls to 1.012 when disk instability is suppressed, 1.012 when the quasar/starburst consumers of that instability are suppressed, and 1.047 when AGN heating is suppressed. In this case the large amplification therefore comes from the coupled disk-instability, burst/BH-growth, and later AGN-regulation chain, not from satellite stripping. This is a selected case study rather than a population-average process attribution.

The wider population still requires the cleaner hybrid experiment that refines declared flows while holding event maps fixed. Ordering of finite infall packets with thresholded star formation, merger timing, and the stored AGN-heating-radius projection can contribute to the remaining percent-level displacement. The report therefore labels the population refinement result a warning rather than a convergence-order measurement.

Current code: [`evolve.py`](../mimic_jax/sage16/evolve.py), [`ode.py`](../mimic_jax/sage16/ode.py), [`hybrid.py`](../mimic_jax/sage16/hybrid.py), and [`numerics.py`](../mimic_jax/numerics.py). Tests: [`test_numerics.py`](../tests/mimic_jax/test_numerics.py), [`test_ode.py`](../tests/mimic_jax/test_ode.py), and [`test_hybrid.py`](../tests/mimic_jax/test_hybrid.py).

The runnable [`examples/sage16_timestep_refinement.py`](../examples/sage16_timestep_refinement.py) prints the controlled slice at `1, 2, 4, 8` substeps, its provisional-reference differences, and baryon residuals. Its final line states the scope explicitly so the example cannot be mistaken for a Mini-Millennium convergence result.
