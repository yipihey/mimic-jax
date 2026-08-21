# SHARK Lagos23 in mimic-jax

`mimic_jax.shark` puts the public SHARK Lagos23 configuration on the same explicit numerical footing as the SAGE16 work while keeping the two models scientifically separate.

## Two calculations, two claims

The package exposes two deliberately distinct paths:

1. **Managed native reference.** `run_reference_shark` invokes a pinned upstream SHARK executable, verifies the public CI inputs, writes an effective configuration and provenance manifest, and loads the resulting catalogue through `SharkCatalogue`. This is the exact population reference.
2. **JAX continuous/hybrid model.** Pure JAX functions implement the Lagos23 continuous transfers, augmented BH/AGN memory state, finite constraints, projections, and event maps. These functions support `jit`, vectorization, and automatic differentiation. They are validated against the native library at prescription, controlled-interval, and exhaustive realized-population RHS level.

The full-tree gate is no longer unknown. An opt-in trace of the complete 20,174-tree native run records every state at which SHARK requests its disk or starburst derivative. One `jax.jit(jax.vmap)` kernel independently recalculates all 5,709,080 realized states. Three of 62,799,880 named-rate values exceed the predeclared `rtol=1.1e-4` strict gate; all three are BR06 star-formation rates, the maximum relative difference is `1.2295e-4`, and none exceeds the explicit `rtol=1.5e-4` quadrature warning band. All 108,472,520 routing comparisons pass the strict gate. The native driver still supplies the variable-cardinality topology and realized states, so this is an exhaustive **population physics shadow replay**, not yet a topology-owning per-ID JAX catalogue. The report displays both conclusions rather than collapsing them into one green claim.

## Mathematical structure

Between events the disk or burst reservoirs obey

\[
\dot x = f_{\rm Lagos23}(x,h,\theta),
\]

where the 19-state upstream flow contains six masses, six metal masses, two episode trackers, and five total angular momenta. Continuous mode augments this state with black-hole mass, metals and spin, the AGN heating radius, and excess jet power.

Processes that are not smooth flows stay visible:

- halo infall and availability caps are finite forcing maps;
- the baryon-fraction ceiling and heating-radius running maximum are projections;
- BH seeding, disk instability, mergers, ram/tidal stripping, and starburst onset are event maps;
- the merger clock is continuous between a terminal threshold event;
- stochastic BH orientation is explicit sampled forcing, not hidden mutable state.

This is a hybrid model, not an assertion that every line of SHARK is an ODE.

## Practitioner workflow

Run the pinned native reference after building SHARK:

```bash
python scripts/run_shark_reference.py \
  --shark-source /path/to/shark \
  --shark-executable /path/to/shark/build/shark \
  --config /path/to/sample_lagos23.cfg \
  --tree /path/to/tree_199.0.hdf5 \
  --redshifts /path/to/redshifts.txt \
  --output /path/to/reference-output \
  --seed 123456
```

The command refuses a revision mismatch, records input checksums, separates the output from the repository, and writes `shark-reference-manifest.json`.

For a controlled interval, choose the semantics explicitly:

```python
from mimic_jax.shark import (
    evolve_shark_continuous_interval,
    evolve_shark_reference_interval,
    lagos23_model_parameters,
)

parameters = lagos23_model_parameters()
reference = evolve_shark_reference_interval(state, forcing, parameters)
continuous = evolve_shark_continuous_interval(
    state,
    forcing,
    parameters,
    method="rk4",
    num_substeps=8,
)
```

`evolve_shark_hybrid_interval` surrounds either interval calculation with an explicit upstream-ordered schedule for mergers, starbursts, instability, and environmental events.

### Reproduce the exhaustive population replay

The 3 GB raw trace is a temporary validation product and is not committed. Apply the reviewed, opt-in instrumentation to a separate checkout of the pinned upstream revision, rebuild, and set one environment variable when running the normal Lagos23 configuration:

```bash
git -C /path/to/shark apply /path/to/mimic-jax/scripts/shark_full_population_rhs_trace.patch
cmake -S /path/to/shark -B /path/to/shark/build -DCMAKE_BUILD_TYPE=Release
cmake --build /path/to/shark/build -j
MIMIC_JAX_SHARK_RHS_TRACE=/scratch/full-population-rhs.bin \
  /path/to/shark/build/shark /path/to/effective-shark.cfg

python scripts/evaluate_shark_full_tree_parity.py \
  /scratch/full-population-rhs.bin \
  /path/to/tree_199.0.hdf5 \
  /scratch/full-population-parity.json \
  --instrumented-output-root /path/to/traced/mini-SURFS/lagos23-trace \
  --reference-output-root /path/to/clean/mini-SURFS/lagos23-reference
```

The evaluator memory-maps the trace and streams fixed-size batches, so the 3 GB file is never copied into device memory. It also verifies that tracing did not perturb native science output: all 5,332,172 values in 1,462 galaxy datasets across 17 output snapshots are bitwise identical to the clean native run. The published JSON stores that noninterference result alongside the trace checksum, input-tree checksum, coverage, strict and warning tolerances, per-rate maxima and exception counts, and separate compilation/steady-state timings without putting the raw trace in git. The evaluator enables JAX 64-bit mode explicitly; the library API fails fast if parity is requested in 32-bit mode.

## What is tested

- direct C++-library oracles for BR06 molecular star formation, Lagos13 feedback, reincorporation, Sobacchi13 reionisation, Croton06 cooling, and deterministic Lagos23 AGN/QSO rates;
- complete controlled upstream disk-interval and event-triggered starburst comparisons, including BH growth/spin;
- mass, metal-source, and angular-momentum conservation in values and derivatives;
- Euler, Heun, RK4, and adaptive integration, including observed convergence order;
- `jax.jit`, `jax.vmap`, `jax.grad`, `jax.jacfwd`, and `jax.jacrev` on the applicable smooth branches;
- strict public-CI tree schema and native catalogue observables;
- exhaustive replay of every realized disk/starburst RHS evaluation in the public-CI population;
- explicit merger, instability, stripping, and event-order maps.

Exact tolerances and measured residuals live in the [SHARK report](../reports/shark-continuous-foundation/index.md) and its adjacent machine-readable `report.json`.

## Comparison with SAGE16

The [model-comparison contract](model_comparison.md) supplies common definitions for stellar mass function, cosmic SFR density, gas fraction, gas and stellar metallicity, BH–bulge, quenched fraction, and stellar-to-halo mass. SHARK-only products such as atomic/molecular mass functions, component sizes/angular momentum, BH spin, resolved environmental reservoirs, and burst channels remain available rather than being reduced to SAGE's state. The [interoperability audit](../reports/sage16-shark-interoperability-audit/index.md) records which definitions are direct, qualified, or unavailable.

Native SHARK mini-SURFS and native SAGE Mini-Millennium results are useful within-model validation datasets, but not a clean model-versus-model experiment. A causal SAGE–SHARK comparison requires common halo forcing, cosmology, volume, selections, and units.

## Scope

The current JAX claim covers the prescriptions, controlled event branches, and every realized continuous disk/starburst state selected by the pinned `sample_lagos23.cfg`; the exact native backend covers the complete population algorithm. It does not mean the topology-owning JAX population driver or every alternative SHARK configuration—such as every optional star-formation, cooling, reionisation, or feedback family—has passed. Those remain separate gates and require their own direct oracles before equivalence claims.
