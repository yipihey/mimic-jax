# Differentiability of Fiducial SAGE16

Differentiability is a property of a particular SAGE16 branch and numerical execution, not a blanket label for the whole model. mimic-jax first reproduces upstream decisions exactly and then reports what kind of derivative is available.

| Operation | Classification | Consequence |
| --- | --- | --- |
| Cooling-table interpolation within one table interval | piecewise smooth | derivative exists within the interval; knots change the local formula |
| Cooling and reincorporation formulas away from caps | smooth | ordinary JAX derivatives are meaningful |
| Reservoir caps such as `min(requested, available)` | piecewise smooth | derivative changes at saturation |
| Star-formation critical cold-gas test | thresholded | each branch is differentiable; the threshold itself is not |
| SN nonnegative clamp and cold-gas renormalization | piecewise smooth | gradients describe the active local branch |
| Metallicity zero/floor guards | thresholded | derivatives can be discontinuous at empty-reservoir boundaries |
| Disk-instability criterion | thresholded | exact branch behavior must precede any optional approximation |
| Quasar and starburst prescriptions after a trigger | piecewise smooth | continuous within the selected event branch |
| Merger/disruption choice and major/minor classification | discrete event | preserve as an ordered jump map; do not replace with a sigmoid for equivalence |
| Merger-tree topology and galaxy identity | discrete | not differentiated by the initial model |

The current quiescent slice is compatible with `jax.jit`, `jax.vmap`, `jax.grad`, `jax.jacfwd`, and `jax.jacrev`. Tests evaluate derivatives away from thresholds and compare them with symmetric finite differences at several perturbation sizes. A derivative near a branch boundary should be reported with the active branch and, when scientifically relevant, a finite perturbation study across the boundary.

Persistent SAGE reservoirs are float32 for parity. JAX can differentiate through a float64 calculation followed by a float32 write, but finite differences of the stored value eventually hit float32 resolution. This is one reason validation uses several perturbation sizes rather than declaring the smallest step the most accurate.

Hard catalog selections and hard histogram bins deserve separate care. A galaxy crossing a mass-bin edge changes a conventional stellar mass function discretely, while pathwise AD through hard integer bin membership is zero almost everywhere and undefined at the edge. The first Mini-Millennium stellar-mass-function response will therefore require a separately validated population estimator or finite-volume treatment whose relation to the upstream hard-bin plot is measured and documented. mimic-jax will not silently report the derivative of fixed bin assignments as the derivative of galaxy abundance.
