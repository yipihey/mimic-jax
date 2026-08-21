# Comparing SAGE16, SHARK, Sapphire, and future SAMs

Mimic-jax compares models through three explicit contracts rather than pairwise plotting code:

1. a **canonical galaxy catalogue** for observable reductions;
2. a **canonical merger-tree forcing record** for auditing whether the same histories can drive each model.
3. a **configured SAM protocol** for continuous flows, named process perturbations, conservation, parameter responses, and local dynamical response.

The [SAGE16--SHARK interoperability audit](../reports/sage16-shark-interoperability-audit/index.md) establishes substantial output overlap between the two production SAM ports, but it does **not** yet claim that either JAX population model can run the other model's native trees. The [three-model response report](../reports/three-model-response-foundation/index.md) adds a pinned native Sapphire calculation and makes its distinct smooth-central forcing and event scope explicit.

## Common dynamical boundary

`load_model` provides one deliberately small execution and analysis surface:

```python
from mimic_jax import load_model

sage = load_model("sage16")
shark = load_model("shark")
sapphire = load_model("sapphire")

print(sage.metadata.process_control_names)
print(shark.metadata.capability("continuous_rhs"))
print(sapphire.metadata.capability("events"))
```

SAGE16 and SHARK expose their live in-process RHS, native named rates, conserved quantities, parameter responses, Jacobians, and local response through `ConfiguredSamModel`. Sapphire exposes the same semantic metadata and scientific products through a native external artifact because its current Python/JAX requirements conflict with the validated SAGE16/SHARK environment. The state and forcing coordinates remain model-owned: SAGE is not given SHARK reservoirs, SHARK is not reduced to SAGE's parameterization, and Sapphire's CGM is not silently renamed as another model's hot reservoir.

The common process controls are logarithmic multipliers. A cooling control `epsilon` means `cooling -> cooling * exp(epsilon)`, so its small-perturbation interpretation is a fractional change in cooling. This lets the same response code ask how fractional SFR responds to cooling, star formation, SN feedback, or reincorporation without pretending the underlying prescriptions are identical.

Local response objects record the model, formulation, state/input/output names, units, redshift, halo mass, derivative method, and native time convention. `state_space_in_gyr` must be applied before comparing response times or frequency responses across models. `scale_state_space` applies an explicitly supplied similarity scaling when mixed mass, metal, and angular-momentum coordinates make matrix exponentials poorly conditioned; it preserves poles and the complete input-output transfer function.

The [first common-response report](../reports/sage16-shark-response-foundation/index.md) uses this boundary for a matched local disk experiment. It validates the frozen linear response against nonlinear evolution and keeps same-tree population isolation and matched closed-loop AGN regulation marked **not evaluated**.

The [three-model report](../reports/three-model-response-foundation/index.md) deliberately compares different supply boundaries rather than hiding them. SAGE16 and SHARK receive fractional cooling perturbations; Sapphire receives a fractional dark-matter accretion perturbation that propagates through its CGM. Both ask how strongly SFR follows supply variability, but only the latter contains the upstream atmospheric filtering in that plotted transfer.

## Catalogue boundary

`ComparisonCatalogue` stores physical masses in `Msun`, total SFR in `Msun/yr`, the native comoving volume in `(Mpc/h)^3`, and an explicit `hubble_h`. Each `CatalogueField` records:

- the native source field or component sum;
- its canonical unit and physical meaning;
- whether it is direct, derived, a component sum, or model-specific;
- any qualification needed before comparing it with another model.

Unavailable physics has an explicit reason. SAGE, for example, does not acquire invented atomic and molecular reservoirs merely because SHARK outputs them.

```python
from mimic_jax.catalogue import observable_capabilities
from mimic_jax.observables import catalogue_mass_function
from mimic_jax.sage16 import load_sage_comparison_catalogue
from mimic_jax.shark import load_shark_catalogue, shark_comparison_catalogue

sage = load_sage_comparison_catalogue(
    sage_partitions,
    snapshot=63,
    redshift=0.0,
    hubble_h=0.73,
    effective_volume_mpc_over_h_cubed=62.5**3,
    dataset="Mini-Millennium",
)
shark = shark_comparison_catalogue(
    load_shark_catalogue(shark_output),
    snapshot=199,
    dataset="mini-SURFS",
)

edges = [8.0, 8.25, 8.5, 8.75, 9.0]
sage_smf = catalogue_mass_function(sage, "stellar_mass", bin_edges=edges)
shark_smf = catalogue_mass_function(shark, "stellar_mass", bin_edges=edges)
```

Both mass functions now use the same selection, bins, unit conversion, volume normalization, and zero handling. That makes the reduction comparable; it does not remove differences in the native simulations.

The audit caught one concrete semantic error while establishing this boundary: SAGE's `MetalsStellarMass` is already the total stellar metal mass. `MetalsBulgeMass` is a subset and must not be added again.

## Current observable overlap

Both model adapters currently support one canonical definition for:

- stellar, cold-gas, cold-baryonic, and black-hole mass functions;
- cosmic SFR density;
- SFR--stellar-mass relation;
- quenched fraction under a declared sSFR threshold;
- cold-gas fraction, with model phase definitions visible;
- mass-weighted cold-gas and stellar metallicity;
- black-hole--bulge relation;
- stellar-to-host-halo relation, with halo-mass conventions visible;
- hot, ejected, and diffuse-stellar reservoir relations, with ownership boundaries visible;
- a qualified baryonic Tully--Fisher projection;
- positions for clustering/selection work, with orphan and simulation conventions visible;
- a qualified stellar disk size relation.

SHARK additionally supplies native HI/H2, component sizes and angular momentum, BH spin, AGN luminosity/mechanical power, detailed environmental reservoirs, and burst channels. These remain first-class SHARK outputs. Maximal overlap means sharing valid definitions, not discarding richer physics.

## Observation boundary

Observations use their own declared target conventions. The shared Baldry et al. (2008) stellar mass function loader receives a target `hubble_h` and IMF choice once; it is not silently transformed using whichever model is plotted. Other arrays currently embedded in legacy SAGE plotting modules must be extracted with citations, units, IMF, aperture, calibration, covariance, and validity ranges before they become shared observational products.

## Merger-tree boundary

`CanonicalMergerTree` projects one native tree into tree-local topology and halo forcing. Every `TreeField` records whether it was native or derived. `assess_tree_compatibility` then separates three questions:

1. Are all required forcing fields present?
2. Are the field definitions scientifically compatible with the target model?
3. Does a topology-owning JAX population driver accept this tree format?

The production-SAM topology answers today are:

| Source trees | SAGE16 JAX | SHARK Lagos23 JAX |
| --- | --- | --- |
| L-Halo / Mini-Millennium | native population driver ready | missing SHARK halo conventions; driver open |
| VELOCIraptor / mini-SURFS | missing SAGE/L-Halo conventions; driver open | forcing fields ready; independent JAX topology driver open |

SAGE-on-SHARK needs an audited L-Halo progenitor ordering, virial-radius and velocity-dispersion convention, plus a conversion of VELOCIraptor angular momentum into the vector used by SAGE's disk-radius law. SHARK-on-L-Halo needs at least concentration, interpolation/DHalo semantics, main-progenitor/ownership rules, and validated spin/half-mass-radius conversions.

The exhaustive SHARK population RHS replay is complementary evidence: it validates every realized continuous physics evaluation selected by native SHARK. It does not yet make JAX the owner of variable-cardinality topology and event scheduling.

Sapphire v0.130 consumes smooth main-progenitor histories containing dark-matter accretion, halo mass, virial radius, virial velocity, and NFW concentration. The audited Pandya23 model evolves independent central galaxies and has no general merger/satellite event topology. Adapting a common smooth main branch is feasible; calling that a full-tree topology comparison would not be.

## Reproduce the audit

The command accepts explicit paths because SHARK reference catalogues and large trees are deliberately not committed:

```bash
python scripts/audit_sage_shark_interoperability.py \
  --sage-catalogues output/sage16-mini-millennium/model_*.hdf5 \
  --lhalo-tree simulations/mini-millennium/snapshots/trees_063.0 \
  --scale-factors simulations/mini-millennium/mini-millennium.a_list \
  --shark-catalogue /path/to/199/0/galaxies.hdf5 \
  --shark-tree /path/to/tree_199.0.hdf5 \
  --output reports/sage16-shark-interoperability-audit
```

The report writes durable Markdown, a standard report manifest, compact numerical arrays, and `assets/model-comparison-audit.json`. The latter exposes field provenance, observable capabilities, tree blockers, and claim boundaries to agents without requiring them to reverse-engineer figures.

## Acceptance gate for a controlled population comparison

A population science comparison isolates baryonic model choices only after participating models use:

- the same canonical halo histories and event topology;
- the same cosmology, volume, and redshift outputs;
- documented conversions for every model-required halo property;
- the same galaxy selections, IMF and observable definitions;
- independently converged numerical modes;
- propagated finite-volume and numerical uncertainties.

For Sapphire, the gate additionally requires defensible population weights because its released inference workflow samples smooth central histories rather than providing an automatically number-density-complete merger-tree catalogue. Until these gates pass, native Mini-Millennium, mini-SURFS, and Sapphire controlled-history results demonstrate interoperability and within-model validation, not a causal three-model difference.
