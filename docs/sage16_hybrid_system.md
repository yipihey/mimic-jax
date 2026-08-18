# SAGE16 as a Hybrid Reservoir System

Fiducial SAGE16 is neither one monolithic ODE nor an arbitrary collection of
mutations. The smallest faithful mathematical description is a hybrid system:

\[
\dot{x}=f(x,h(t),\theta), \qquad x^+=J(x^-,h,\theta),
\]

with algebraic calculations and bounded projections where the upstream model
uses them. The merger tree supplies the external forcing \(h(t)\). This page
classifies the actual configured Mini-Millennium pipeline; it does not infer a
new galaxy-formation model from the module names.

Two implementations remain deliberately distinct:

- `upstream_sequential` reproduces the configured SAGE16 maps, their order,
  float storage, finite caps, and event scan. It is the equivalence reference.
- the hybrid formulation exposes the continuous limit of prescriptions that
  have one, retains projections and jumps explicitly, and permits alternative
  time integration. Agreement with the reference is a result to measure, not
  an assumption.

## Persistent Markov state

The complete upstream record has 32 fields. The hybrid state keeps the fields
that must persist to predict later physical evolution:

| State | Physical role |
| --- | --- |
| `ColdGas`, `HotGas`, `EjectedGas` | gas reservoirs |
| `StellarMass`, `BulgeMass`, `ICS` | total stars, bulge component, and intracluster stars |
| `BlackHoleMass` | central black-hole reservoir |
| `MetalsColdGas`, `MetalsHotGas`, `MetalsEjectedGas` | gas-phase metal reservoirs |
| `MetalsStellarMass`, `MetalsBulgeMass`, `MetalsICS` | stellar metal mass and components |
| `HaloBaryonFraction` | reionization-modified baryon fraction used by infall and stripping |
| `Rheat` | stored radio-mode heating radius; makes AGN cooling suppression Markovian |
| `DiskScaleRadius` | current central disk structure, retained by satellites |
| `MergTime` | event clock with continuous countdown between merger-tree events |
| `TimeOfLastMajorMerger`, `TimeOfLastMinorMerger` | retained event history |

`InfallingGas` is a prepared interval control, not a reservoir. `CoolingGas`,
`NewStellarMass`, and `UnstableDiskGasFraction` are finite transport budgets or
event triggers. SFR, cooling/heating powers, outflow rates, and similar fields
are output quadratures. They remain in the exact 32-field `GalaxyState`, while
`Sage16HybridState` makes the mathematical distinction explicit.

## Complete process classification

The upstream algorithms below refer to the modules configured in
[`sage16_mini-millennium.yaml`](../models/sage16/input/sage16_mini-millennium.yaml)
and to tree inheritance in `src/core`.

| Process | Upstream algorithm | Physical interpretation | Required state/history | Mathematical type |
| --- | --- | --- | --- | --- |
| Tree inheritance and snapshot reset | Copy surviving progenitors into the descendant workspace, update type/halo ownership, preserve persistent fields, and reset interval accumulators | Connects tree nodes and assigns galaxies to the new FoF topology | progenitor records, descendant identity, tree links | forcing update + topology jump + bookkeeping projection |
| Reionization | Evaluate the mass/redshift modifier and store `HaloBaryonFraction` | Reduces the baryon allotment of small haloes | `Mvir`, redshift, cosmology | algebraic forcing map |
| Infall-budget preparation | Consolidate satellite ejecta and ICS into the FoF central, total live group baryons, set `InfallingGas = f_b Mvir - M_baryon` | Establishes the interval baryon target and ownership of external reservoirs | live FoF reservoirs, central `Mvir`, baryon fraction | ownership projection + algebraic budget constraint |
| Infall application | Apply `InfallingGas/N`; add positive budgets as pristine hot gas, remove negative budgets from ejected then hot gas | Cosmological supply or correction to the halo allotment | prepared signed budget, source metallicities | external forcing + piecewise flow + reservoir-boundary projection |
| Disk scale radius | Type 0 recalculates disk size from halo spin/virial quantities; satellites retain their value | Structural forcing for star formation and instability | halo spin, `Rvir`, `Vvir`; retained satellite radius | algebraic projection with stored memory |
| Merger-clock initialization | Initialize a sentinel clock from dynamical friction, reset centrals, and handle new orphans | Schedules a future satellite event | halo type, orbital/virial forcing, stored `MergTime` | conditional projection |
| Merger-clock countdown | Subtract each satellite's object-local substep duration | Continuous approach to an event surface | `MergTime`, object `dT` | flow, \(\dot M_{\rm ergtime}=-1\) |
| Reincorporation | Compute a rate proportional to ejected mass above the velocity threshold, multiply by object `dt`, cap by ejecta | Returns ejected gas and metals to the central hot halo | ejected reservoir, `Rvir`, `Vvir` | thresholded flow + positivity constraint |
| Type-1 satellite stripping | Recompute excess baryons and transfer `excess/N`, capped by satellite hot gas, to the FoF central | Removes a satellite hot atmosphere and deposits it in the group atmosphere | satellite and central reservoirs, type, `Mvir`, baryon fraction, interval subdivision | conservative group flow in the \(N\to\infty\) limit; exact finite split in reference mode |
| Cooling budget | Calculate cooling radius/function and `CoolingGas = rate * dt`, capped by hot gas | Radiative hot-to-cold supply | hot gas/metals and halo forcing | algebraic rate evaluation + finite capacity projection |
| Prior AGN cooling suppression | Multiply the cooling budget by `1-Rheat/Rcool`, or zero it for `Rheat>=Rcool` | Persistent radio-mode feedback suppresses later cooling | stored `Rheat`, current `Rcool`, cooling budget | algebraic constraint/projection |
| Radio-mode BH accretion and heating | Evaluate the selected rate, apply Eddington/hot-gas/heating caps, transfer hot gas to the BH, propose a larger heating radius | Grows the BH and establishes future cooling suppression | BH, hot gas/metals, halo forcing, cooling function, `Rheat` | flow + capacity constraint + monotone history projection |
| Cooling application | Transfer surviving cooling gas and co-moving metals from hot to cold | Condensation onto the disk | cooling budget, hot metallicity | conservative flow limit; bounded finite reference transfer |
| Quiescent star formation | Evaluate the Kennicutt-like rate above the cold-gas threshold | Converts disk gas into stars | cold gas, disk radius, `Vvir` | thresholded flow |
| Instantaneous recycling | Retain `(1-RecycleFraction)` in stars and return the remainder in the gas bookkeeping | Short-lived stellar mass return | new stellar budget, recycle fraction | simultaneous flow coupling |
| SN reheating and ejection | Calculate reheating and energy-limited ejection; rescale SF/reheating if cold supply is insufficient; cap ejection by central hot gas | Transfers cold to hot and hot to ejecta | SF budget, local cold, central hot, `Vvir` | piecewise flow + coupled capacity projection |
| Quiescent metal enrichment | Add `Yield * NewStellarMass`, split between cold and hot | Newly synthesised metals | SF budget, halo mass, gas reservoirs | explicit source flow, applied later in module order |
| Disk instability | Compare disk mass with the stability threshold, move unstable stellar mass to the bulge, emit an unstable-gas fraction | Instantaneous structural response of an unstable disk | stellar/bulge components, cold gas, disk radius, `Vmax` | threshold + finite projection/event trigger |
| Quasar mode | On instability or merger, accrete finite cold gas; energy thresholds can move all remaining cold/hot gas to ejecta | Rapid BH growth and quasar wind | trigger, cold/hot gas and metals, BH, `Vvir` | finite event map + threshold projections |
| Collisional starburst | On instability or merger, consume finite cold gas and immediately apply recycling, SN feedback, and yield | Triggered burst and coupled feedback | trigger/event ratio, reservoirs, halo forcing | finite event map |
| Disruption | Transfer cold+hot gas to central hot and stars+ICS to central ICS; discard source BH and mark source consumed | Satellite disruption and topology change | source/target states, current interpolated subhalo mass | topology jump with explicit BH sink |
| Merger | At clock crossing, transfer source reservoirs, classify major/minor, run quasar/starburst, and possibly recheck instability | Coalescence and immediate triggered physics | two galaxy states, target identity, event clock, halo forcing | topology jump + ordered finite event maps |

## What moved into the continuous formulation

The original small ODE subset was an engineering milestone, not a mathematical
boundary. The hybrid RHS now adds:

- **radio-mode AGN flow**: hot gas accreted by the BH, its tracked-metal sink,
  prior-`Rheat` cooling suppression, and the candidate next heating radius;
- **prepared infall forcing**: the signed interval budget becomes a
  piecewise-constant external rate, with ejected-first negative removal;
- **satellite stripping**: a group-coupled conservative flow whose fixed-forcing
  limit removes `1-exp(-1)` of an initial excess in one tree interval;
- **merger-clock flow**: satellite clocks evolve continuously between events.

These are exposed by [`hybrid.py`](../mimic_jax/sage16/hybrid.py). Cooling,
quiescent star formation/recycling, SN feedback, reincorporation, and metal
flows remain in [`ode.py`](../mimic_jax/sage16/ode.py).

### AGN memory is state, not hidden history

For fixed current state and halo forcing, SAGE first applies

\[
\dot M_{\rm cool,allowed}=
\begin{cases}
(1-R_{\rm heat}/R_{\rm cool})\dot M_{\rm cool},&R_{\rm heat}<R_{\rm cool},\\
0,&R_{\rm heat}\ge R_{\rm cool},
\end{cases}
\]

then calculates radio-mode BH accretion and heating. Current heating is not
subtracted from allowed cooling a second time. Instead it defines

\[
R_{\rm heat}^{+}=\max\left[
R_{\rm heat}^{-},
\frac{\dot M_{\rm heat}}{\dot M_{\rm cool,allowed}}R_{\rm cool}
\right]
\]

on the active branch. Thus the faithful representation is a mass flow plus a
monotone projection for the augmented state `Rheat`; no hidden past is needed.

### Infall is forcing plus a flow

The exact reference prepares one finite budget from the tree node and current
FoF inventory, then partitions it over `N` substeps. The continuous counterpart
holds

\[
\dot M_{\rm infall}=M_{\rm infall,budget}/\Delta t_{\rm tree}
\]

fixed over that interval. Negative forcing switches from the ejected reservoir
to the hot reservoir at an exhaustion event. Deriving supply from a smoothly
interpolated baryon target would be a different forcing experiment, not
upstream equivalence.

### Stripping is a flow, but its timescale is numerical

With fixed forcing, upstream repeatedly removes `excess/N`. Its continuous
limit is

\[
\dot M_{\rm hot,sat}=-\frac{(M_{\rm baryon,sat}-M_{\rm allowed})_+}
{\Delta t_{\rm tree}},
\]

with an equal central-halo term. This is structurally conservative, but the
effective timescale is the tree interval; SAGE16 supplies no independent
physical stripping timescale. The exact geometric finite map stays in
reference mode.

## What remains a projection or event

Disk instability is a threshold followed by a finite component transfer and
an immediate gas trigger. It can be viewed as a zero-timescale relaxation, but
SAGE16 supplies no instability timescale and the trigger immediately drives
quasar and burst maps. Choosing a finite relaxation time would add physics, so
mimic-jax does not invent one in the fiducial hybrid model.

Mergers and disruptions genuinely change ownership and topology. The public
fixed-identity ownership maps are differentiable JAX functions; event
detection, target identity, and major/minor classification remain discrete.
For a fixed merger event, reverse sensitivities can branch to both progenitors
without pretending that topology is continuous.

## Adaptive integration respects the hybrid boundary

`integrate_sage16_hybrid_flow_adaptive` applies embedded Dormand–Prince 5(4) error control only to the continuous RHS between externally fixed boundaries. The controller also limits the step using a tolerance-scaled state Jacobian. It deliberately leaves the `Rheat` monotone projection, disk instability, quasar/starburst maps, forcing changes, and topology events to the caller's physical schedule; evaluating those maps at rejected or intermediate Runge–Kutta stages would define a different model.

On 27 branch-smooth fixed-forcing intervals drawn from 64 Mini-Millennium trees, all tested tolerances from `1e-3` through `1e-9` complete successfully. At `rtol=1e-7`, the median/maximum reservoir errors against a 4,096-step RK4 reference are `2.10e-9`/`6.00e-7`, the maximum stellar-mass error is `2.54e-9 dex`, and baryon closure is `2.22e-16`. Twenty-five additional candidates cross a reservoir boundary, the star-formation threshold, or the cooling-regime threshold and remain outside this claim until explicit event localization is implemented. See [`numerical_integration.md`](numerical_integration.md) for the method and limitations.

## Conservation and differentiability

The RHS is assembled from named transfers. Its baryon derivative equals the
explicit external infall source/sink. Its tracked-metal derivative equals
stellar production plus external metal flow minus metals swallowed by
radio-mode BH accretion, because SAGE16 has no BH-metal reservoir. Satellite
stripping conserves combined satellite+central baryons and metals.

Tests check those equalities and their parameter derivatives, one-step AGN and
infall correspondence away from finite capacity boundaries, the stripping
continuous limit, the `Rheat` projection, and derivatives through a fixed
merger ownership event. At thresholds, caps, projections, and topology choices,
derivatives remain branch-local and are not silently smoothed.

Current tests: [`test_hybrid.py`](../tests/mimic_jax/test_hybrid.py). Exact
integration ordering and convergence evidence are described in
[`numerical_integration.md`](numerical_integration.md).
