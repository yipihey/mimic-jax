# SHARK Integration, Continuous Reformulation, and SAGE Comparison Plan

**Status:** Active — the pinned Lagos23 controlled process/event kernels, reference intervals, continuous counterpart, managed native population backend, and shared observables are implemented. Independent topology-owning JAX population evolution and per-ID equivalence remain open.
**Date:** 2026-08-20
**Scope:** A sibling SHARK implementation in `mimic_jax/shark`, exact comparison against upstream SHARK, a separately labelled continuous/hybrid formulation, and model-neutral observable comparisons with SAGE16.

---

## Objective and non-negotiable distinction

The objective is to put SHARK and SAGE16 on the same scientific footing in mimic-jax without blending their prescriptions or redefining either model. Each model retains its own immutable state, parameters, forcing adapter, process functions, hybrid events, upstream reference evolution, and validation evidence. Shared numerics, sensitivities, reports, and model-neutral observable definitions are reused.

There are two distinct SHARK calculations:

1. **Reference SHARK:** the exact upstream ordering, adaptive ODE, finite budgets, caps, projections, random choices, and topology-changing events. This is the only target used for upstream-equivalence claims.
2. **Continuous/hybrid SHARK:** the same prescriptions expressed as continuous flows wherever that interpretation is mathematically faithful, augmented with memory states where possible, plus explicit projections and event maps where continuity would misrepresent the model. This mode is tested against timestep refinement and the reference mode but is never silently substituted for it.

“Fully continuous” therefore means that every physical mass, metal, and angular-momentum transport which admits a rate representation is represented as a rate. It does not mean treating a merger, stochastic orientation draw, topology change, or inequality projection as an ODE.

## Pinned upstream and reproducible oracle

The initial audit and executable oracle target is `ICRAR/shark` revision `5af50d8fa7a040883409b10171c645e1db4e5fb2` on the upstream `devel` branch. The last tagged release, `v2.0.0` (`ddbd1df9bdae58af0877854cf4f0c11e4c5a4d9d`), remains the published baseline; every fixture records which revision generated it. A future revision change is a deliberate scientific migration requiring regenerated fixtures and reviewed deltas.

The upstream executable builds locally with AppleClang, GSL 2.8, HDF5 2.2.0, and Boost 1.92.0. The first reference population uses the same public CI fixture as upstream:

| Input | SHA-256 | Size |
|---|---|---:|
| `redshifts.txt` | `816a885a6e73d6d9022fffeb8667acfe2b0719a6cb0da2d696abe61500b135b9` | 2,530 bytes |
| `tree_199.0.hdf5` | `c072a937941fefb9aac441fc319ff030ceb666af4a07f1b88c0f02c5d76a3f43` | 26,479,838 bytes |

The audited Lagos et al. (2023) sample completed on that tree with fixed seed `123456`, 7,553 output galaxies at snapshot 199, and upstream-recorded revision/dirty-state metadata. Downloaded trees and full catalogues remain outside git; selected compact fixtures, checksums, summaries, and figures may be committed.

## What upstream actually integrates

SHARK is not a purely finite-update model. `BasicPhysicalModel<19>` already evolves disk and burst baryon cycling using GSL's adaptive Cash–Karp Runge–Kutta method (`gsl_odeiv2_step_rkck`). Upstream passes zero absolute tolerance and `execution.ode_solver_precision` as the relative tolerance; the Lagos23 sample sets it to `0.05`. The driver resets the initial trial step to the full snapshot interval. The ODE is embedded in a sequential hybrid update whose per-halo order is galaxy-merger events, disk-instability events, per-galaxy cooling preparation plus ODE evolution, and subhalo-merger/type events.

The 19 ODE variables are six masses, their six metal masses, two star-formation episode trackers, and five total angular momenta. The complete ordered state now exists as `mimic_jax.shark.SharkState`, and the exact upstream stoichiometric assembly exists as `shark_rhs_from_rates`.

## Process classification

The table classifies the actual upstream algorithm at the pinned revision. “Continuous counterpart” is a design decision, not an equivalence claim.

| Process | Upstream algorithm and ordering | Physical interpretation | Required state/history | Mathematical type | Continuous/hybrid treatment |
|---|---|---|---|---|---|
| Tree/halo evolution | Snapshot-indexed VELOCIraptor/SURFS merger-tree records assembled before galaxy evolution | External dark-matter assembly and environment | Halo/subhalo topology, masses, radii, velocities, spin, snapshot time | Forcing plus discrete topology | Interpolate only explicitly selected continuous halo quantities; preserve links and type changes as events |
| Halo baryon accretion | `GasCooling::cooling_rate` adds `min(accreted_mass, f_b M_h-M_b)` to hot gas | Cosmological gas supply | Accreted budget and group baryon inventory | Finite forcing budget plus inequality projection | Reference map unchanged; continuous source from interpolated halo accretion with the same baryon-cap complementarity condition |
| Baryon-fraction limit | Excess inside-halo baryons moved from hot to ejected gas | Enforce universal halo baryon ceiling | Group baryon total | Projection/constraint | Explicit conservative projection; optional complementarity flow is experimental, never reference |
| Reionization | Lacey16 or Sobacchi13 virial-velocity/redshift threshold returns zero cooling | Suppression of baryon supply/cooling | Halo velocity and redshift | Piecewise forcing gate | Exact threshold in RHS/forcing; no automatic smoothing |
| Reincorporation | Central-only finite `M_ejected/tau_reinc * dt`, with special zero and long-timescale branches, applied before cooling | Return of ejected material | Ejected mass, halo mass, redshift-dependent halo properties | Rate realized as capped finite transfer | Continuous ejected-to-hot flow with source availability; reference finite transfer retained |
| Cooling: Croton06 | Cooling radius and rate are computed, AGN modifies the rate, then a capped hot-to-cold-halo transfer occurs before the 19-state ODE; the ODE moves cold-halo gas to the galaxy | Hot-halo cooling and delivery to ISM | Hot/cold-halo mass, metals, AM, halo structure, cooling table | Algebraic rate plus two-stage finite/ODE flow | One conservative hot-halo → ISM flow in continuous mode; reference hot → cold-halo staging and ODE drain retained separately |
| Cooling: Benson10 | Stores vectors of temperature, mass, cooling time, and interval and recomputes their integral | History-dependent available cooling time | Cooling-history integral and current thermodynamic state | Memory-dependent rate | Promote the accumulated integral to a Markov state after algebraic equivalence is proven; retain reset rules as events |
| Hot-mode BH accretion | Rate computed from cooling/AGN prescription; `rate*dt` is capped by hot gas and transferred to BH before the ODE | Radio/jet-mode black-hole growth | BH mass/spin/accretion state and hot gas/metals | Rate realized as capped finite transfer | Continuous hot-gas → BH mass/metals flow with Eddington/availability constraints; exact cap in reference mode |
| AGN heating radius | Croton16/Lagos23 keeps the maximum historical `rheat`; Lagos23 activates memory below a configured redshift | Persistent exclusion of previously heated halo gas | `rheat`, cooling radius/rate, jet power, hydrostatic flag | Monotone memory projection | Augment state with `rheat`; evolve candidate heating and apply explicit `max` projection. It is Markovian but piecewise differentiable |
| Hydrostatic-halo decision | Lagos23 tests a cooling/free-fall criterion; satellite state can inherit central halo status | Eligibility for jet feedback | Halo thermodynamics and group state | Thresholded algebraic constraint | Exact gate; changes are recorded as discrete active-set transitions |
| Disk molecular partition and star formation | Radial quadrature of BR06/GD14/K13/KMT09/KD12 surface-density law supplies SFR and optionally angular-momentum formation rate | HI/H2 partition and quiescent/burst star formation | Cold gas, stars, radii, metallicity, redshift, rotation | Nonlinear continuous flow with internal quadrature | Pinned Lagos23 BR06 law is implemented with pure JAX quadrature; alternative upstream laws remain distinct future configuration variants |
| Instantaneous recycling and enrichment | Recycling fraction and fixed/evolving yield enter the 19-state ODE | Prompt stellar mass return and metal production | Cold-gas metallicity and SFR | Continuous flow plus explicit metal source | Implemented structurally in `shark_rhs_from_rates` |
| Stellar reheating/ejection | Selected loading law returns galaxy/halo mass and angular-momentum loadings used by the ODE | SN-driven ISM → hot → ejected transport | SFR, halo/galaxy velocity, redshift | Piecewise-smooth continuous flow | Lagos23's selected Lagos13 loading law and structural routing are implemented; other upstream configuration variants are not part of this equivalence claim |
| QSO gas outflow | QSO loadings from BH luminosity and galaxy properties enter the ODE; ejected fraction goes to hot gas and lost fraction leaves the halo | Quasar-driven outflow | BH state, SFR, gas metallicity, bulge structure | Piecewise continuous flow | Flow in the augmented RHS; `lost_gas` is an explicit tracked reservoir/sink boundary |
| Angular-momentum exchange | Cooling and SF/SN angular-momentum rates enter the 19-state ODE; sizes are recomputed from specific AM afterward | Size and AM evolution of gas and stars | Five total-AM reservoirs, masses, velocities | Conservative continuous flow plus algebraic projection | Flow implemented; size reconstruction remains explicit algebraic output/projection |
| Molecular/atomic output | Recomputed after each snapshot by radial integration | Observable HI/H2 partition | Gas/stars, radii, metallicity | Algebraic diagnostic | Shared observable layer; not an independent evolved mass unless conservation tests justify promotion |
| Disk instability | Toomre-like threshold; finite disk-to-bulge transfer, size/AM update, burst ODE, BH growth | Secular bulge growth and starburst trigger | Disk/bulge mass, gas, AM, size, BH | Threshold plus jump/projection followed by continuous burst | Exact event map in reference mode; optional finite-timescale relaxation only as a separately named experiment |
| Galaxy merger clock | Analytic timescale assigned/decremented until coalescence | Dynamical friction delay | Orbit, primary/satellite structure, clock | Continuous clock with terminal event | Clock ODE plus root/event; reference decrement/threshold retained |
| Galaxy merger | Two galaxies are combined, morphology/size/history/BH states updated, possible burst follows | Topology-changing coalescence | Both galaxy states, halo, orbit, histories | Jump map plus burst flow | Explicit differentiable map on each smooth branch; never forced into a single-galaxy ODE |
| Subhalo merger/type transition | Disappearing subhalos transfer galaxies and assign central/satellite/orphan types | Tree topology evolution | Halo/subhalo links and galaxy ownership | Discrete event | Explicit event map |
| Ram-pressure stripping | Root/threshold calculations remove hot-halo and optionally ISM gas gradually or instantaneously and deposit it in the central system | Environmental gas transport | Satellite/central gas profiles, orbit/pressure, stripping radii | Constraint-controlled finite transfer | Continuous stripping flow between active-set changes; exact instantaneous branches and disruption remain events |
| Tidal stripping/disruption | Stellar/gas components are removed under tidal criteria and deposited into central/stellar-halo reservoirs | Environmental mass loss and intrahalo light | Satellite structure, host tidal field | Threshold plus finite transfer/event | Conservative jump for reference; continuous relaxation only where upstream explicitly supplies a gradual rate |
| BH seeding | Seed planted when halo/BH threshold is crossed | First black-hole creation | Halo mass, BH presence | Thresholded source event | Explicit jump map |
| BH starburst growth | Gas-driven accretion and timescale calculation during merger/instability burst | Quasar-mode BH growth | Bulge gas, velocity, BH mass/spin | Capped rate plus event-triggered episode | Continuous during the burst episode; episode onset/termination are events |
| BH spin and accretion-disk orientation | Volonteri07/Griffin19 updates include nonlinear accretion/merger maps and random orientation choices | Spin evolution and radiative/mechanical efficiencies | BH mass/spin, accretion episode, RNG key | Flow/map with stochastic discrete forcing | JAX PRNG key is explicit state/forcing; deterministic maps are differentiated conditionally on the sampled branch |
| Galaxy/halo sizes and velocities | Algebraic reconstruction from masses, AM, halo profile, and root solves | Structural equilibrium | Masses, AM, halo concentration/profile | Algebraic constraint/projection | Pure functions evaluated at RHS stages only where upstream semantics require; otherwise post-flow projection |
| Star-formation/BH histories and global ledgers | Snapshot accumulators and output histories updated after evolution | Scientific diagnostics/provenance | Per-snapshot counters/history arrays | Quadrature/diagnostic state | Explicit quadrature states or output diagnostics; never feed physics unless upstream does |

## State architecture

The complete target is a hierarchy rather than one flat universal vector:

- `SharkFlowState`: the implemented 19-variable disk or burst flow state.
- `SharkGalaxyState`: disk and bulge stars/gas/metals/AM/sizes, BH mass/metals/spin/accretion episode, burst channels, merger clock, histories, and output diagnostics.
- `SharkSubhaloState`: hot/cold-halo/ejected/lost/stripped reservoirs, their metals and AM, cooling-history integral, heating radius, infall properties, and environmental active-set state.
- `SharkHaloState`: group baryon ledger, hydrostatic state, excess jet energy, central ownership, and halo histories.
- `SharkForcing`: interpolable halo quantities plus discrete tree links/events and an explicit PRNG key for stochastic prescriptions.

The 19-variable flow state is reused for disk and starburst episodes through explicit projection/insertion maps, matching upstream. Larger states are introduced only when their owning process is implemented and tested.

## Validation ladder

No population-level or scientific comparison claim advances past its unmet gate.

| Gate | Evidence | Tolerance policy |
|---|---|---|
| V0 source identity | Pinned revision, clean/dirty state, compiler/dependencies, input/config checksums, seed | Exact metadata |
| V1 equation assembly | Controlled values compared to upstream 19-equation evaluator; mass/AM residual zero and metal residual equals explicit yield source | Float64 roundoff; derivative residuals also tested |
| V2 individual prescriptions | C++ oracle fixtures for every selected SF, feedback, cooling, AGN, reincorporation, environmental, size, and merger branch | Per-function absolute/relative tolerances justified from upstream precision and quadrature |
| V3 reference interval | JAX reference mode versus upstream for controlled galaxies at one interval, including intermediate transfers | Stored-field precision/ULP-aware; thresholds categorized separately |
| V4 full CI tree | Per-galaxy matched properties and global histories on the public upstream CI tree | Distribution of residuals, matched IDs, threshold flips, unmatched objects; never visual-only |
| V5 time convergence | `dt`, `dt/2`, `dt/4`, `dt/8` plus tight adaptive reference for representative regimes and population observables | Observed order, conservation, positivity, and event-time convergence reported separately |
| V6 common-forcing model comparison | SAGE16 and SHARK use the same canonical halo forcing, cosmology, volume, cuts, IMF convention, and observable definitions | Comparison uncertainty includes numerical and finite-volume effects; it is not an equivalence test |

V0 and V1 are implemented. V2 covers controlled BR06 star-formation/angular-momentum, Lagos13 stellar-feedback, reincorporation, Sobacchi13 reionisation, Croton06 cooling, deterministic Lagos23 AGN/QSO branches, Griffin19 spin in the burst sequence, finite cooling/BH caps, and the event maps used by the pinned configuration. The C++ harness is linked to the pinned upstream library; it does not duplicate the equations in Python. Four BR06 cases agree within `5e-6` relative (the JAX implementation uses deterministic 128-node quadrature rather than upstream's configured 5%-tolerance GSL integral). Feedback and realized reincorporation cases are exact in float64, reionisation branches agree exactly, and the packaged 8 × 227 Cloudy table, cooling, and deterministic AGN/QSO cases agree at float64 precision. V3 passes for the controlled ordered disk interval and event-triggered starburst at explicitly recorded tolerances. V5 passes for the controlled continuous disk flow. V4 (independent per-ID JAX replay of the complete CI tree) and V6 (common-forcing SAGE comparison) remain open.

## Numerical convergence program

Upstream reference mode reproduces GSL Cash–Karp with its configured tolerance and update order. The continuous/hybrid mode uses shared mimic-jax Euler, Heun RK2, RK4, and adaptive Dormand–Prince only where the RHS is defined. Convergence is separated into four errors:

1. radial quadrature error inside molecular partition/SFR;
2. baryonic ODE time-integration error;
3. forcing interpolation error between tree snapshots;
4. event localization and projection/order error.

Representative tests span low-mass SN-regulated centrals, reincorporating systems, cooling-dominated disks, massive AGN-regulated centrals, satellites undergoing gradual stripping, starbursts, and merger intervals. Required outputs include reservoir histories, HI/H2, stellar mass, SFR, metallicities, BH mass/spin, sizes, quenched state, conservation residuals, active-set/event times, and RHS evaluations. Population tests report the fractional change in familiar observables, not only state-vector norms.

The implemented coupled-flow tests establish first-, second-, and fourth-order behavior for Euler, Heun, and RK4 and agreement of adaptive Dormand–Prince with a refined RK4 reference. The published report uses the actual oracled, table-driven Croton06 cooling, BR06 radial star formation, and Lagos13 feedback under explicitly fixed halo structure. AGN heating memory is an explicit monotone projection and hybrid processes are event maps, so they are tested for branch/order/equivalence rather than assigned a fictitious smooth ODE convergence order. Event localization and population-observable convergence across the entire CI tree remain part of V4/V5.

## Common observable contract

Common plots must be projections of one model-neutral definition, not similar-looking model-local scripts. Each definition records mass units, `h`, IMF convention, galaxy selection, volume, bin edges, zero handling, and whether a component sum is used.

| Observable | SAGE16 catalogue projection | SHARK catalogue projection | Status/qualification |
|---|---|---|---|
| Stellar mass function | `StellarMass` | `mstars_disk + mstars_bulge` | Direct after unit and IMF alignment |
| Cosmic SFR density | `StarFormationRate` | `sfr_disk + sfr_burst` | Direct after rate/unit and volume alignment |
| Cold-gas fraction | `ColdGas` | `mgas_disk + mgas_bulge` | Direct total-neutral comparison; phase definitions documented |
| HI and H2 mass functions | Not separated in fiducial SAGE16 output | `matom_*`, `mmol_*` | SHARK-only until a scientifically defined SAGE post-processing counterpart exists |
| Gas mass–metallicity | `MetalsColdGas / ColdGas` | `(mgas_metals_disk + mgas_metals_bulge)/(mgas_disk + mgas_bulge)` | Solar normalization/calibration metadata required |
| Stellar metallicity | `(MetalsStellarMass + MetalsBulgeMass)/StellarMass` | `(mstars_metals_disk + mstars_metals_bulge)/(mstars_disk + mstars_bulge)` | Direct mass-weighted comparison |
| BH–bulge relation | `BlackHoleMass`, `BulgeMass` | `m_bh`, `mstars_bulge` | Direct after units |
| Quenched fraction | sSFR from SFR and stellar mass | sSFR from total SFR and stellar mass | Same threshold and aperture convention required |
| Baryonic Tully–Fisher | stellar plus cold gas and chosen velocity | stellar plus cold gas and chosen velocity | Velocity proxy is not assumed identical; observable adapter must state it |
| Size–mass | `DiskScaleRadius` plus bulge fraction | `rstar_disk`, `rstar_bulge` | Component definitions differ and remain visible |
| Halo/stellar relation | `Mvir`, `StellarMass` | `mvir_hosthalo`, total stars | Central/satellite and host/subhalo selection aligned |
| AGN/BH activity | cooling/heating and BH accretion trackers | hot/burst accretion, bolometric luminosity, mechanical power, spin | SHARK exposes materially richer outputs |
| Environmental stripping | SAGE satellite/ICS reservoirs | hot/ISM stripped gas, tidally stripped stars, stellar halo | SHARK-specific decompositions retained |
| Angular momentum | Limited disk radius/spin information | component-specific stellar, atomic, molecular, and bulge AM | SHARK-specific primary diagnostic |

Additional SHARK outputs that must not be discarded include disk/bulge decomposition, atomic/molecular gas, burst channels by trigger, cold-halo and lost-gas reservoirs, hot/ISM stripped material, stellar halo, component angular momenta and sizes, BH spin, AGN radiative/mechanical power, hydrostatic-halo state, and star-formation/BH histories.

## Same-forcing comparison strategy

Native upstream parity and inter-model prediction comparison require different datasets. Upstream SHARK parity uses its CI mini-SURFS/VELOCIraptor tree; upstream SAGE parity uses Mini-Millennium/L-Halo trees. A scientific SHARK-versus-SAGE comparison must not confuse those simulation differences with model differences.

The comparison gate therefore requires a canonical halo-history representation with model-specific adapters. At minimum it carries stable topology, snapshot cosmic time/redshift, halo/subhalo mass and growth, virial quantities, position/velocity/spin, central/satellite ownership, and infall properties. Any SHARK input absent from the SAGE tree is either derived by an explicitly tested shared convention, supplied as controlled forcing, or marks that observable/experiment unavailable. Both models then run on the same selected histories and the same effective volume/cosmology. Native-tree reports remain useful but are labelled within-model validation, not direct model comparison.

## Implementation phases

### Phase 1 — Executable foundation (implemented in this slice)

- pin and build upstream SHARK;
- run the public CI tree with the Lagos23 configuration and fixed seed;
- encode the exact 19-state PyTree and physical descriptions;
- encode exact flow stoichiometry with named rates;
- expose fixed/adaptive shared integration entry points;
- test mass, metal-source, and angular-momentum conservation, including derivatives;
- test `jit`, `vmap`, `grad`, `jacfwd`, and `jacrev`;
- establish controlled first-/second-/fourth-order convergence and adaptive agreement.

### Phase 2 — Exact Lagos23 prescription layer (implemented for the pinned configuration)

Port and oracle-test cosmology/unit transforms, cooling tables and Croton06 cooling, BR06 radial molecular/SFR quadrature, Lagos13 feedback, reincorporation, Sobacchi13 reionization, Lagos23 hot-mode BH/jet feedback, QSO wind loading, size/AM reconstruction, recycling, and every cap/projection in the sample configuration. Generate fixtures from an instrumented clean upstream executable rather than duplicating expected values in Python.

The BR06, Lagos13, reincorporation, Sobacchi13, Cloudy-table/Croton06, deterministic Lagos23 AGN/QSO, BH seed/growth/spin, disk-instability, merger, stripping, and ordered burst functions are implemented for `sample_lagos23.cfg`. Reincorporation has distinct exact-finite and continuous-rate APIs; the continuous mass/metal/angular-momentum transport is conservative, while upstream's later hot-halo angular-momentum reset remains a separate projection. Cooling likewise retains exact cold-halo staging while continuous mode routes hot gas directly to the ISM. Lagos23 heating memory is an augmented Markov state plus running-maximum projection, not mislabeled as a smooth ODE. The reference Lagos13 implementation deliberately preserves upstream's current `age(redshift_power) ** redshift_power` normalization—and therefore its lack of dependence on the supplied galaxy redshift—rather than silently changing model semantics. Alternative SHARK configuration families are future model variants, not silently implied by Lagos23 coverage.

### Phase 3 — Complete controlled reference hybrid evolution (implemented); population replay open

Galaxy/subhalo/halo states, exact scheduled event order, disk instability and starburst, mergers, environmental stripping, BH growth/spin, ledgers, output adapters, and an exact managed native population backend are implemented. The remaining gate is a topology-owning JAX driver that matches the public CI tree galaxy by stable ID and global history by snapshot.

### Phase 4 — Continuous/hybrid reformulation (controlled intervals implemented)

All legitimate Lagos23 transports in the controlled interval use the augmented RHS; heating radius is state plus projection; finite budgets, thresholds, stochastic forcing, and topology changes remain explicit. Benson10 is a different cooling configuration and is not part of the pinned Lagos23 claim. Whole-tree terminal-event localization and the resulting population differences remain to be quantified.

### Phase 5 — Population convergence and performance

Run tolerance/timestep/forcing/event-location ladders; compare accuracy at equal RHS calls and wall time; report compilation separately; map stiffness/timescale ratios; validate derivatives through fixed and adaptive paths and document accept/reject non-smoothness.

### Phase 6 — Shared observables and SAGE comparison

Land model-neutral catalogue projections and plot definitions; reproduce each model's familiar native plots first; then run both on canonical common forcing and publish paired predictions, differences, fractional responses, convergence/error bands, and provenance-rich comparison reports.

### Phase 7 — SHARK-only science and differentiable diagnostics

Add HI/H2, AM/size, environment, BH spin/AGN power, burst-channel, and history diagnostics; parameter elasticities; historical process responses; calibration examples; and response-theory analyses without weakening upstream equivalence.

## Acceptance criteria for “science grade”

- Every scientific result names the model revision, parameter set, forcing dataset, seed, numerical mode, precision/tolerances, and event treatment.
- Every process has an upstream source/function reference and at least one branch-covering oracle test before it participates in an equivalence claim.
- Conserved transfers close structurally in values and derivatives; external sources/sinks are named.
- No negative mass is repaired silently. Reference projections reproduce upstream; continuous-mode positivity failures are surfaced and used to control the step or method.
- Population parity matches stable galaxy IDs and reports unmatched objects and threshold flips separately.
- Convergence is quantitative and includes population observables, not just trajectories.
- SAGE-versus-SHARK comparisons use shared forcing or prominently state that they do not isolate model physics.
- Large upstream data and catalogues stay outside git; published reports retain checksums and compact derived artifacts.
- “Implemented,” “validated,” “equivalent,” and “converged” are used only for the gate actually passed.

## Immediate next gate

The next gate is not another isolated prescription. It is the independent topology-owning JAX population driver on the public CI merger tree: replay all 20,174 trees, match stable galaxy IDs, report unmatched objects and threshold flips, and compare every shared observable. Until that passes, the managed native backend is the exact population reference and the JAX implementation is claimed equivalent only at the prescription and controlled-interval levels. The subsequent inter-model science gate runs SAGE16 and SHARK on common halo forcing so tree/cosmology differences are not mistaken for baryonic-model differences.
