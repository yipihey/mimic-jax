# Common Differentiable SAM Protocol Plan

**Status:** Active — ecosystem audit, three-model protocol, native Sapphire bridge, and controlled response reports implemented
**Date:** 2026-08-21
**Scope:** A minimal model-neutral execution and response boundary for SAGE16, SHARK Lagos23, and native Sapphire Pandya23, designed to admit further models without a universal-physics ontology

## Scientific question

Mimic-jax is not another from-scratch differentiable SAM. Its distinctive purpose is to put established production SAMs and adjacent native differentiable models into a common, explicit continuous/hybrid analysis representation without erasing their model-specific physics:

\[
\mathrm{SAM}=\mathrm{flows}+\mathrm{forcing}+\mathrm{events}+\mathrm{constraints}.
\]

That representation should let us ask whether models that reproduce similar galaxy populations do so through the same baryon flows, parameter responses, memory times, and feedback modes. SAGE16 and SHARK remain independently traceable to their upstream implementations; Sapphire remains executed by its own pinned package and solver. The common layer supplies semantics and analysis, not replacement prescriptions.

## Ecosystem audit

The audit used pinned source checkouts, their tests and documentation, and the current Sapphire paper. No external source code is copied into mimic-jax.

| Project | Audited revision | License at revision | Boundary used here |
|---|---|---|---|
| MIMIC/SAGE16 | `69590cc60dcb7b8b6510ee0b16b1ed921a6c4853` | GPL-3.0 | Faithful reference and continuous/hybrid adapter |
| SHARK Lagos23 | `5af50d8fa7a040883409b10171c645e1db4e5fb2` | GPL-3.0 | Faithful reference and continuous/hybrid adapter |
| Sapphire | `ee50e858e3427de50368c32205001248849b8be0` | MIT | Architecture, numerical validation, inference, and execution audit |
| Diffmah | `180c6ce6947993f5c3024587eca010937ec1ef4f` | BSD-3-Clause | Optional differentiable halo-forcing interface |
| Diffstar | `035e1cfae64e9b38cc46f2d30b09260fb1aacbcf` | BSD-3-Clause | Optional SFH comparison/emulator interface |
| DiffstarPop | `69f652642a41eddbd07489a045b75119928768e2` | BSD-3-Clause | Optional differentiable population-summary interface |
| DSPS | `2ca3ce4285e16a330b83248208bcafa086036e9e` | BSD-3-Clause | Optional SFH/metallicity-to-photometry interface |
| Galacticus | `0f386049a46fce7fce1c459a8c53cd79fa5e8c0b` | GPL-3.0 | Architectural/numerical reference and later independent analytic cases |

The audited Sapphire checkout uses JAX PyTrees and differentiates a seven-variable logarithmic state containing stellar, ISM, CGM, thermal-energy, and metal reservoirs. The August 2026 paper describes an eight-variable extension including turbulent CGM energy, so results and feature comparisons must identify the code/paper version rather than treating “Sapphire” as one immutable formulation. Its production path uses Diffrax Tsit5 with PID control and direct adjoints; a pedagogical Bogacki–Shampine RK23 implementation remains useful for exposition. The application has no general event/jump interface at this revision and treats merger differentiation as future work. Those facts reinforce mimic-jax's hybrid/event distinction rather than reveal a missing generic solver to rebuild.

Sapphire's inference examples use Optax optimization, automatic Hessian/Fisher estimates, and NumPyro sampling. Its differentiable Gaussian-kernel summaries are directly relevant to mass-function and scaling-relation calibration. These are mature generic patterns to reuse or interoperate with; mimic-jax should concentrate new code on model semantics, established-model validation, conservation, and physically annotated responses.

| Capability | Sapphire | mimic-jax before this plan | Adopt or reuse? | mimic-jax distinction |
|---|---|---|---|---|
| Model target | Purpose-built differentiable regulator | Faithful SAGE16 and SHARK formulations | Compare scientifically; do not recreate Sapphire | Established SAM correspondence and direct cross-SAM analysis |
| State | Flat logarithmic JAX state for a seven-reservoir thermal model in the audited checkout | Model-owned immutable PyTrees, including complete stored/reference states and explicit continuous subsets | Retain semantic model states; investigate logarithmic coordinates as an optional numerical transform | State fields remain traceable to upstream models and include event/history variables where required |
| RHS | One configurable nonlinear RHS closure | Named SAGE/SHARK rates and conservative routing | Adopt the disciplined configured-model boundary | Flows, finite transfers, events, and constraints remain distinct |
| Integration | Diffrax Tsit5/PID/direct adjoint plus a pedagogical RK23 solver | Validated fixed-step and adaptive Dormand–Prince methods used for equivalence and convergence | Use Diffrax optionally rather than expanding a general solver library | Upstream numerical realization remains a first-class reference method |
| Batching | `vmap`, sharding, multi-device execution | `jit`, `vmap`, scan-based histories and population paths | Reuse JAX-native batching patterns | Same analyses can operate on multiple established SAMs |
| Derivatives | Jacobians, optimization, Hessian/Fisher estimates, HMC | Elasticities, finite-epoch responses, Jacobians, response theory, local fitting and MCMC demonstrations | Prefer Optax/NumPyro and tested ecosystem machinery for generic inference | Responses carry physical process, reservoir, epoch, model, and provenance metadata |
| Discontinuities | No general event/jump API in the audited application; merger differentiation is future work | Explicit SAGE and SHARK hybrid maps and threshold classifications | Do not force events into an ODE | Hybrid established-model evolution is central |
| Conservation | Scientific budgets are checked through the model, without a common structural ledger API | Separate executable SAGE and SHARK mass/metal/AM ledgers | Expose one ledger protocol | Conservation and derivative conservation are model-independent diagnostics |
| Testing | Numerical convergence/gradient experiments and a light repository test surface in the audited checkout | Process-level oracles, upstream parity, conservation, integration, catalogue and report tests | Adopt Sapphire's gradient-versus-tolerance study design | Legacy equivalence gates remain mandatory |
| Reporting | Configuration, NPZ output, plots and logs | Markdown/JSON/Quarto run and comparison reports | Preserve mimic-jax reporting | Self-documenting cross-model claims and explicit unavailable states |

### Adjacent packages

- **Diffmah** is the preferred optional differentiable halo-assembly forcing provider. An adapter should translate its `mah_singlehalo`/`mah_halopop` outputs into a model-owned forcing object; real merger trees remain the reference input.
- **Diffstar and DiffstarPop** are complementary parametric SFH and population models. They are useful comparison/emulator tools and sources of differentiable population summaries, not SAM implementations to copy into mimic-jax.
- **DSPS** is the preferred optional differentiable output layer from SAM SFH and metallicity histories to SEDs and photometry. Mimic-jax should retain template, IMF, dust, and filter provenance and should not reimplement stellar population synthesis.
- **Galacticus** is an architectural and numerical reference for adaptive ODE evolution plus explicit tree events. Its GPL implementation is not copied. Analytic reservoir cases from its public validation suite are useful later independent tests.

## Minimal protocol

The protocol is deliberately smaller than a universal SAM ontology. A configured model exposes:

1. model identity, upstream revision, formulation, and claim qualification;
2. typed metadata for state, forcing, parameter, process, and observable coordinates;
3. `rhs(time, state, forcing, parameters, log_process_perturbations)`;
4. `rhs_and_rates(...)`, retaining the model-native named rate result;
5. conserved quantities with units and explicit source/sink interpretation;
6. descriptors for model-owned flow, forcing, event, constraint, and projection operators;
7. immutable parameter access/replacement, including nested parameter paths;
8. common elasticity and physically annotated local-response entry points.

The protocol does **not** require identical reservoir names, the same parameterization, or an invented one-to-one process map. Cross-model comparisons must declare the matched physical question and any qualification.

## Implementation slices

### Slice 1 — Semantic protocol and configured adapters

**Implemented 2026-08-21.**

- Add immutable variable, process, capability, model, conservation, and configured-model records.
- Add a registry with `load_model("sage16")` and `load_model("shark")`.
- Wrap the existing SAGE16 continuous central subset and SHARK Lagos23 controlled disk subset without changing either kernel.
- Expose finite events and constraints as model-owned semantic operators rather than pretending they share a universal call signature.

### Slice 2 — Common response outputs

**Implemented 2026-08-21 for local state/process and state/parameter Jacobians.**

- Extend local state-space results with model, formulation, state/input/output labels, units, halo mass, redshift, and linearization-point metadata.
- Add stable/neutral/unstable mode classification, damping times, oscillation periods, and reservoir eigenvectors.
- Make nested model parameters available to the existing dimensionless response API without breaking flat SAGE parameter records.

### Slice 3 — Shared validation

**Implemented 2026-08-21 at the configured continuous-subset boundary.** Population and hybrid-event validation retain their independent model gates.

- Run the same protocol tests on both configured models.
- Test structural baryon and metal conservation and their parameter/control derivatives.
- Validate representative elasticities and local responses against finite differences or nonlinear perturbations.
- Preserve every upstream parity test; protocol adaptation must be numerically invisible to native calls.

### Slice 4 — First controlled SAGE–SHARK response comparison

**Implemented 2026-08-21 as a qualified local experiment.** The [response-foundation report](../../reports/sage16-shark-response-foundation/index.md) matches the main reservoir inventory, disk scale, velocity, metallicity, cooling, and reincorporation supply; converts both models to Gyr; applies an explicit state similarity scaling; and validates the local prediction against nonlinear evolution. It keeps full same-tree population isolation and matched AGN regulation marked not evaluated.

Match halo/structural conditions and a clearly declared continuous central/disk boundary. Compare baryon flows, process perturbations, coupled response times, and cooling-to-SFR response. This is a controlled local experiment, not a population-equivalence claim. The report must state every unmatched assumption.

### Slice 5 — Native Sapphire interoperability

**Implemented 2026-08-21 for one pinned controlled trajectory and local response.** The isolated-runtime bridge preserves Sapphire's modern JAX/Diffrax environment, exports a versioned checksum-bound artifact, validates native AD against finite differences, tests tolerance refinement and open-system budgets, and loads through `load_model("sapphire")`. The dedicated [integration plan](MIMIC-JAX-SAPPHIRE-INTEGRATION-PLAN.md) records the exact boundary and remaining population gates.

### Slice 6 — Interoperability experiments

- Prototype Diffmah forcing without replacing real trees.
- Prototype one DSPS history-to-photometry path and verify a SAM-parameter-to-flux derivative.
- Reproduce one analytic Galacticus reservoir test independently.
- Extend Sapphire comparison from the controlled native fixture to weighted population summaries only where forcing, selection, state, and observables can be matched without relabeling different physics.

## Dependency decisions

- Keep JAX as the only required runtime dependency in this phase.
- Keep Diffrax, Diffmah, Diffstar, DSPS, Optax, and NumPyro optional and lazily imported.
- Record external package versions and scientific assets in provenance.
- Raise an actionable unavailable-capability result when an optional dependency or physical mapping is absent; never substitute an arbitrary approximation silently.

## Acceptance criteria

- SAGE16, SHARK, and Sapphire load through one public registry and expose the same protocol concepts, with native/external execution differences explicit.
- Native kernel outputs and upstream-equivalence tests remain unchanged.
- Every response array identifies model, formulation, coordinates, units, operating point, and derivative method.
- Common conservation diagnostics work for both models and retain external sources/sinks explicitly.
- Cross-model reports distinguish matched physics, qualified comparisons, and unavailable quantities.
- Generic solver/inference/SPS functionality is delegated to mature packages where practical.
- No result claims that SAGE and SHARK regulate galaxies similarly or differently until common forcing and observable gates pass.

## First report target

**SAGE and SHARK can generate similar galaxy populations—but do they regulate those galaxies in the same way?**

The first report will compare familiar outputs before moving to local baryon flows, fractional responses, response times, cooling-to-SFR transfer, and AGN regulation. Practitioner-facing headings will lead with the astrophysical question. Jacobians, poles, and transfer functions will appear as the machinery underneath the result.

The controlled local precursor is complete. The remaining decisive report gate is a full common-forcing population experiment with both models' hybrid event schedules and closed AGN loops in scope.

## Primary references

- [Sapphire source](https://github.com/virajpandya/sapphire) and [current paper](https://arxiv.org/abs/2604.06318)
- [Diffmah](https://github.com/ArgonneCPAC/diffmah)
- [Diffstar](https://github.com/ArgonneCPAC/diffstar) and [DiffstarPop](https://github.com/ArgonneCPAC/diffstarpop)
- [DSPS](https://github.com/ArgonneCPAC/dsps)
- [Galacticus](https://github.com/galacticusorg/galacticus) and [merger-tree evolver documentation](https://galacticus.readthedocs.io/en/stable/physics/mergerTreeEvolver.html)
- [MIMIC](https://github.com/darrencroton/mimic) and [SHARK](https://github.com/ICRAR/shark)
