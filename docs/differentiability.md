# Differentiability of Fiducial SAGE16

Differentiability is a property of a particular SAGE16 branch and numerical execution, not a blanket label for the whole model. mimic-jax first reproduces upstream decisions exactly and then reports what kind of derivative is available.

| Operation | Classification | Consequence |
| --- | --- | --- |
| Type-0 disk-radius formula away from zero spin | smooth | local halo-spin and virial-property responses are meaningful |
| Disk-radius Type/fallback guards | thresholded/discrete | retain satellite freezing and the exact small-virial fallback |
| Dynamical-friction merger-clock formula on one resolved branch | smooth | local clock responses are meaningful before the float write |
| Merger-clock sentinel, particle floor, target, and zero crossing | thresholded/discrete event | do not interpret AD as differentiating merger identity or occurrence |
| Reionization modifier within one scale-factor branch | smooth | mass, redshift, cosmology, and baryon-fraction responses are locally meaningful |
| Reionization era/mass/Type guards | thresholded/discrete | retain exact `z=8`, `z=7`, mass, and Type-3 decisions |
| Positive infall partition | smooth within float-storage resolution | fractional supply responses pass through the fixed snapshot budget |
| Negative infall priority and reservoir depletion | piecewise smooth | derivatives describe the active ejected/hot depletion branch |
| Type-1 satellite stripping above the excess threshold | piecewise smooth | fractional process responses describe the active live-excess branch; Type and excess decisions remain exact |
| Cooling-table interpolation within one table interval | piecewise smooth | derivative exists within the interval; knots change the local formula |
| Cooling and reincorporation formulas away from caps | smooth | ordinary JAX derivatives are meaningful |
| Fiducial Bondi radio-mode rate within fixed caps | smooth | parameter and process responses are meaningful on the active branch |
| Eddington, hot-gas, and heating-mass caps | piecewise smooth | gradients describe the selected local cap |
| Monotone `Rheat` history update | piecewise-smooth projection | `Rheat` is explicit Markov state; derivatives pass through the active max branch |
| `AGNrecipe` and cold-cloud triggering | discrete/thresholded | reproduce selection exactly; do not differentiate the integer recipe |
| Reservoir caps such as `min(requested, available)` | piecewise smooth | derivative changes at saturation |
| Star-formation critical cold-gas test | thresholded | each branch is differentiable; the threshold itself is not |
| SN nonnegative clamp and cold-gas renormalization | piecewise smooth | gradients describe the active local branch |
| Metallicity zero/floor guards | thresholded | derivatives can be discontinuous at empty-reservoir boundaries |
| Disk-instability criterion and stellar/gas split | thresholded/piecewise smooth | exact stable/unstable decision and caps are retained; fractional response is local to the active unstable branch |
| Quasar BH growth before wind thresholds | piecewise smooth | low-velocity suppression and cold-gas cap are differentiable on each active branch |
| Quasar cold/hot wind decisions | thresholded | the all-reservoir ejection decisions are exact and are not smoothed |
| Starburst formation, recycling, and SN transport | piecewise smooth | responses are local to the active trigger, balance, ejection-cap, and yield branches |
| Fixed-identity merger/disruption ownership map | piecewise smooth jump | progenitor-to-descendant derivatives are available for the selected event |
| Merger/disruption choice, target identity, and major/minor classification | discrete event | preserved as a live ordered jump map; conditional derivatives do not differentiate event occurrence |
| Merger-tree topology and galaxy identity | discrete | not differentiated by the initial model |
| Persistent-state inheritance on a fixed branch | smooth identity/copy map | later observables can be differentiated with respect to inherited baryonic state |
| Fixed-topology FoF group schedule | piecewise smooth ordered scan | reverse mode includes live satellite-to-central writes; group membership, Type, and event identity remain fixed |

The complete fixed-topology group schedule is compatible with JAX transformations at its pure-process boundary. Ordered `lax.scan` preserves live shared-central writes within a group; VMAP applies across independent groups. Tests exercise `jax.jit`, `jax.vmap`, `jax.grad`, `jax.jacfwd`, and `jax.jacrev` as appropriate, evaluate derivatives away from thresholds, and compare process/parameter derivatives with symmetric finite differences at several perturbation sizes. A derivative near a branch boundary should be reported with the active branch and, when scientifically relevant, a finite perturbation study across the boundary.

Persistent SAGE reservoirs are float32 for parity. JAX can differentiate through a float64 calculation followed by a float32 write, but finite differences of the stored value eventually hit float32 resolution. This is one reason validation uses several perturbation sizes rather than declaring the smallest step the most accurate.

Numerical integration is an additional differentiability boundary. The exact
upstream sequential scan supports reverse-mode differentiation on its active
branches. Euler, Heun RK2, and RK4 are now available for explicitly declared
continuous flows and retain JAX differentiation. Projections are kept as maps,
not hidden inside the RHS. Future adaptive accept/reject control and alternative
splitting order may add discrete choices; their forward accuracy and derivative
semantics will be reported separately. See
[`numerical_integration.md`](numerical_integration.md) and the complete
[`hybrid-system classification`](sage16_hybrid_system.md).

Hard catalog selections and hard histogram bins deserve separate care. A galaxy crossing a mass-bin edge changes a conventional stellar mass function discretely, while pathwise AD through hard integer bin membership is zero almost everywhere and undefined at the edge. The first Mini-Millennium stellar-mass-function response will therefore require a separately validated population estimator or finite-volume treatment whose relation to the upstream hard-bin plot is measured and documented. mimic-jax will not silently report the derivative of fixed bin assignments as the derivative of galaxy abundance.
