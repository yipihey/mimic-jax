# Differentiability of Fiducial SAGE16

Differentiability is a property of a particular SAGE16 branch and numerical execution, not a blanket label for the whole model. mimic-jax first reproduces upstream decisions exactly and then reports what kind of derivative is available.

| Operation | Classification | Consequence |
| --- | --- | --- |
| Reionization modifier within one scale-factor branch | smooth | mass, redshift, cosmology, and baryon-fraction responses are locally meaningful |
| Reionization era/mass/Type guards | thresholded/discrete | retain exact `z=8`, `z=7`, mass, and Type-3 decisions |
| Positive infall partition | smooth within float-storage resolution | fractional supply responses pass through the fixed snapshot budget |
| Negative infall priority and reservoir depletion | piecewise smooth | derivatives describe the active ejected/hot depletion branch |
| Type-1 satellite stripping above the excess threshold | piecewise smooth | fractional process responses describe the active live-excess branch; Type and excess decisions remain exact |
| Cooling-table interpolation within one table interval | piecewise smooth | derivative exists within the interval; knots change the local formula |
| Cooling and reincorporation formulas away from caps | smooth | ordinary JAX derivatives are meaningful |
| Fiducial Bondi radio-mode rate within fixed caps | smooth | parameter and process responses are meaningful on the active branch |
| Eddington, hot-gas, heating-mass, and `Rheat` caps | piecewise smooth | gradients describe the selected local cap |
| `AGNrecipe` and cold-cloud triggering | discrete/thresholded | reproduce selection exactly; do not differentiate the integer recipe |
| Reservoir caps such as `min(requested, available)` | piecewise smooth | derivative changes at saturation |
| Star-formation critical cold-gas test | thresholded | each branch is differentiable; the threshold itself is not |
| SN nonnegative clamp and cold-gas renormalization | piecewise smooth | gradients describe the active local branch |
| Metallicity zero/floor guards | thresholded | derivatives can be discontinuous at empty-reservoir boundaries |
| Disk-instability criterion and stellar/gas split | thresholded/piecewise smooth | exact stable/unstable decision and caps are retained; fractional response is local to the active unstable branch |
| Quasar BH growth before wind thresholds | piecewise smooth | low-velocity suppression and cold-gas cap are differentiable on each active branch |
| Quasar cold/hot wind decisions | thresholded | the all-reservoir ejection decisions are exact and are not smoothed |
| Starburst formation, recycling, and SN transport | piecewise smooth | responses are local to the active trigger, balance, ejection-cap, and yield branches |
| Merger/disruption choice and major/minor classification | discrete event | preserve as an ordered jump map; do not replace with a sigmoid for equivalence |
| Merger-tree topology and galaxy identity | discrete | not differentiated by the initial model |

The current baryon-supply/satellite-stripping/cooling/radio-mode/quiescent/instability/quasar/starburst slice is compatible with `jax.jit`, `jax.vmap`, `jax.grad`, `jax.jacfwd`, and `jax.jacrev` at its pure-process boundary. Tests evaluate derivatives away from thresholds and compare them with symmetric finite differences at several perturbation sizes. A derivative near a branch boundary should be reported with the active branch and, when scientifically relevant, a finite perturbation study across the boundary.

Persistent SAGE reservoirs are float32 for parity. JAX can differentiate through a float64 calculation followed by a float32 write, but finite differences of the stored value eventually hit float32 resolution. This is one reason validation uses several perturbation sizes rather than declaring the smallest step the most accurate.

Numerical integration is an additional differentiability boundary. The exact upstream sequential scan supports reverse-mode differentiation on its active branches. Future adaptive accept/reject control and alternative splitting order may add discrete choices; their forward accuracy and derivative semantics will be reported separately. See [`numerical_integration.md`](numerical_integration.md).

Hard catalog selections and hard histogram bins deserve separate care. A galaxy crossing a mass-bin edge changes a conventional stellar mass function discretely, while pathwise AD through hard integer bin membership is zero almost everywhere and undefined at the edge. The first Mini-Millennium stellar-mass-function response will therefore require a separately validated population estimator or finite-volume treatment whose relation to the upstream hard-bin plot is measured and documented. mimic-jax will not silently report the derivative of fixed bin assignments as the derivative of galaxy abundance.
