# Native Sapphire Integration Plan

**Status:** Active — native runtime, artifact bridge, common metadata/response API, validation fixture, and first three-model report implemented
**Date:** 2026-08-21
**Scope:** Make Sapphire Pandya23 a third configured model without copying its physics or destabilizing the validated SAGE16/SHARK runtime

## Decision

Sapphire is integrated as a native external model, not reimplemented inside mimic-jax. The audited Sapphire v0.130 checkout requires Python 3.12+, JAX/JAXlib 0.9+, Diffrax, Equinox, and its released cooling/tree assets, while mimic-jax currently validates SAGE16 and SHARK on JAX 0.4.x and supports Python 3.9+. Installing both into one environment would either violate their declared requirements or force an unrelated JAX migration. A versioned subprocess/artifact boundary preserves both projects' numerical environments.

The resulting execution path is:

\[
\text{mimic-jax case}
\rightarrow
\text{isolated native Sapphire v0.130}
\rightarrow
\{x(t),\dot x,r,A,B,J_\theta,\text{provenance}\}
\rightarrow
\text{common analysis/reporting}.
\]

No Sapphire physical equation is copied into mimic-jax. The bridge imports `sapphire.models.pandya23`, runs its Diffrax Tsit5 solver, calls its native auxiliary-rate function, and uses native `jax.jacfwd`. Mimic-jax converts the internal logarithmic coordinate derivative to physical units and records that transform explicitly.

## Pinned scientific source

| Item | Value |
|---|---|
| Repository | <https://github.com/virajpandya/sapphire> |
| Revision | `ee50e858e3427de50368c32205001248849b8be0` |
| Package version | `0.130` |
| License | MIT |
| Released data | `sapphire-data.tar.gz` from release `v0.130` |
| Audited state | Seven logarithmic coordinates: stellar, ISM, CGM and thermal-energy reservoirs plus their three metal reservoirs |
| Native solver | Diffrax Tsit5 with PID control and direct adjoint configuration |

The August 2026 paper discusses an eight-variable extension with turbulent CGM energy; the pinned code fixture contains seven coordinates. Reports must identify the code revision instead of conflating paper variants.

## Common model semantics

`load_model("sapphire")` is available without installing Sapphire and returns its complete semantic manifest. Native execution is enabled only when an explicit `SapphireNativeBackend` supplies the isolated Python executable, pinned source checkout, and official data directory.

The common surface includes:

- state, forcing, parameter, process, observable, and capability metadata;
- native trajectory, physical RHS, auxiliary rates, and solver statistics;
- physical-coordinate state Jacobian;
- fractional halo-forcing input Jacobian;
- native fixed-state parameter Jacobians and end-to-end parameter-to-final-observable derivatives through the adaptive trajectory;
- common local transfer, mode, and characteristic-time analysis;
- common parameter-response normalization from supplied native derivatives;
- reconstructed open-system baryon and metal budgets;
- explicit unavailable/not-applicable states for events and topology that Pandya23 does not contain.

Sapphire's CGM reservoir is not silently renamed as another model's hot reservoir, and its smooth independent-central forcing is not silently promoted to a merger-tree topology driver.

## Artifact contract

Every native run produces `artifact.json` plus checksum-bound `arrays.npz` under schema `mimic-jax-sapphire-native/v1`. The JSON records the complete controlled case, coordinate labels and units, model/source/data revisions, solver and tolerance settings, derivative conventions and validation, convergence result, conservation boundary, command, software, hardware, and array checksum. Large scientific arrays remain in NPZ and are referenced by name, shape, and dtype.

The committed controlled fixture under `tests/data/sapphire/native-v0.130-controlled/` is a native result, not an emulation. It allows ordinary mimic-jax CI to validate the adapter without downloading Sapphire's roughly 0.8 GB data release or installing a conflicting JAX environment.

## Validation gates

The controlled native fixture must satisfy all of the following:

1. exact source revision and package-version identity;
2. artifact checksum, coordinate, shape, and dtype validation;
3. local state/input/fixed-state parameter automatic derivatives versus symmetric finite differences with relative L2 error below `1e-7`;
4. end-to-end parameter derivatives through the adaptive trajectory versus symmetric finite differences at five perturbation sizes, with relative L2 error below `5e-3` at the declared `1e-4` reference step;
5. requested `1e-8` adaptive solve versus `1e-10` solve with maximum final-state fractional difference below `1e-4`;
6. baryon and metal residual divided by the larger source/sink throughput below `1e-12`;
7. JAX x64 required for response analysis because the physical state mixes masses with CGM energies near `1e58 erg`.

The end-to-end gate is deliberately separate. Diffrax's adaptive PID controller makes discrete accept/reject choices, so its complete numerical path is only piecewise smooth even though the Pandya23 RHS is differentiable. The committed artifact records the finite-difference-size scan instead of applying the much tighter local-RHS gate to a different mathematical object.

These gates validate the bridge and the controlled calculation. They do not establish agreement with SAGE16 or SHARK, nor population-level validation of Sapphire's published examples.

## Comparison boundary

The initial three-model report compares only quantities with explicit semantics. SAGE16/SHARK cooling-to-SFR response and Sapphire halo-accretion-to-SFR response are shown together only with the input-boundary difference visible: Sapphire includes propagation through its atmosphere, whereas the established-model curves perturb cooling directly. Local damping times are reported at their named operating points and are not described as matched-galaxy differences.

The next population evidence gate is to adapt a common smooth main-progenitor history into each model's forcing boundary, attach defensible population weights, and compare the shared SMHM, gas-fraction, metallicity, and SFMS observables. Full branch topology remains model-specific; current Sapphire has no merger/satellite/BH event system to compare.

## Practitioner workflow

Metadata is always available:

```python
from mimic_jax import load_model

sapphire = load_model("sapphire")
print(sapphire.metadata.to_dict())
```

After cloning the pinned source with `gh`, creating a separate Python 3.12 environment, installing Sapphire there, and extracting the official v0.130 data asset, configure the backend explicitly:

```python
sapphire = load_model(
    "sapphire",
    python_executable="/path/to/sapphire-venv/bin/python",
    source_repository="/path/to/sapphire",
    data_path="/path/to/sapphire/data",
)
artifact = sapphire.run_local_case(output_directory="output/sapphire-native-case")
response = sapphire.local_response(artifact=artifact)
```

The backend refuses revision drift or a missing cooling table before starting the scientific run.

## Remaining work

- validate a native Sapphire population against its released TNG/CDHMAH examples and published summary statistics;
- add a smooth common-history forcing adapter without treating it as merger-tree topology equivalence;
- map genuinely shared observables and selection/weighting metadata across all three models;
- investigate an upstream-friendly process-perturbation hook in Sapphire rather than copying its RHS;
- compare parameter responses only after explicitly mapping physical parameter roles and normalization conventions;
- add DSPS/Diffmah interoperability once the three-model forcing and observable gates are established.
