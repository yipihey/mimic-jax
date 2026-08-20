# SHARK Lagos23 in mimic-jax

`mimic_jax.shark` puts the public SHARK Lagos23 configuration on the same explicit numerical footing as the SAGE16 work while keeping the two models scientifically separate.

## Two calculations, two claims

The package exposes two deliberately distinct paths:

1. **Managed native reference.** `run_reference_shark` invokes a pinned upstream SHARK executable, verifies the public CI inputs, writes an effective configuration and provenance manifest, and loads the resulting catalogue through `SharkCatalogue`. This is the exact population reference.
2. **JAX continuous/hybrid model.** Pure JAX functions implement the Lagos23 continuous transfers, augmented BH/AGN memory state, finite constraints, projections, and event maps. These functions support `jit`, vectorization, and automatic differentiation. They are validated against the native library at prescription and controlled-interval level.

An independent JAX replay of all 20,174 public-CI merger trees has not yet passed. The report therefore does not call the two population calculations equivalent or draw a JAX population curve over the native stellar mass function.

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
    n_steps=8,
)
```

`evolve_shark_hybrid_interval` surrounds either interval calculation with an explicit upstream-ordered schedule for mergers, starbursts, instability, and environmental events.

## What is tested

- direct C++-library oracles for BR06 molecular star formation, Lagos13 feedback, reincorporation, Sobacchi13 reionisation, Croton06 cooling, and deterministic Lagos23 AGN/QSO rates;
- complete controlled upstream disk-interval and event-triggered starburst comparisons, including BH growth/spin;
- mass, metal-source, and angular-momentum conservation in values and derivatives;
- Euler, Heun, RK4, and adaptive integration, including observed convergence order;
- `jax.jit`, `jax.vmap`, `jax.grad`, `jax.jacfwd`, and `jax.jacrev` on the applicable smooth branches;
- strict public-CI tree schema and native catalogue observables;
- explicit merger, instability, stripping, and event-order maps.

Exact tolerances and measured residuals live in the [SHARK report](../reports/shark-continuous-foundation/index.md) and its adjacent machine-readable `report.json`.

## Comparison with SAGE16

The shared observable layer supplies common definitions for stellar mass function, cosmic SFR density, gas fraction, gas and stellar metallicity, BH–bulge, quenched fraction, and stellar-to-halo mass. SHARK-only products such as atomic/molecular mass functions, component sizes/angular momentum, BH spin, resolved environmental reservoirs, and burst channels remain available rather than being reduced to SAGE's state.

Native SHARK mini-SURFS and native SAGE Mini-Millennium results are useful within-model validation datasets, but not a clean model-versus-model experiment. A causal SAGE–SHARK comparison requires common halo forcing, cosmology, volume, selections, and units.

## Scope

The current JAX claim covers the prescriptions and controlled event branches selected by the pinned `sample_lagos23.cfg`; the exact native backend covers the complete population algorithm. It does not mean the topology-owning JAX population driver or every alternative SHARK configuration—such as every optional star-formation, cooling, reionisation, or feedback family—has passed. Those are separate gates and require their own direct oracles before equivalence claims.
