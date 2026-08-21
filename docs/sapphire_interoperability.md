# Sapphire Interoperability

Mimic-jax treats [Sapphire](https://github.com/virajpandya/sapphire) as a native third differentiable model, not as physics to copy. The common adapter preserves Sapphire's own Pandya23 equations, logarithmic state, Diffrax integration, parameter conventions, cooling data, and JAX derivatives while exposing the result to the same response, conservation, comparison, and reporting concepts used for SAGE16 and SHARK.

## Why execution is isolated

The pinned Sapphire v0.130 package requires Python 3.12+, JAX/JAXlib 0.9+, Diffrax, Equinox, and its released scientific data. Mimic-jax currently validates the SAGE16 and SHARK implementations with JAX 0.4.x and supports older Python versions. Combining them in one environment would violate declared dependency ranges or turn a model adapter into an unrelated JAX migration.

The solution is a small versioned native-runtime boundary:

```text
mimic-jax case
  -> isolated Sapphire Python/JAX environment
  -> native Pandya23 RHS + Diffrax solve + native JAX derivatives
  -> artifact.json + arrays.npz
  -> mimic-jax response, conservation, comparison, and reporting
```

The artifact is not an emulator. It contains the native physical trajectory, auxiliary rates, physical-coordinate RHS, local state/input/parameter Jacobians, an end-to-end parameter-to-final-observable derivative through the adaptive solve, symmetric finite differences at multiple perturbation sizes, tolerance-refinement results, open-system budget residuals, and complete provenance.

## Install the native backend

Clone and pin Sapphire using GitHub CLI, create a separate Python 3.12 environment, install the pinned source, and download its official v0.130 data asset outside the mimic-jax repository:

```bash
gh repo clone virajpandya/sapphire /path/to/sapphire -- --branch v0.130
git -C /path/to/sapphire checkout ee50e858e3427de50368c32205001248849b8be0

uv venv --python 3.12 /path/to/sapphire-venv
uv pip install --python /path/to/sapphire-venv/bin/python -e /path/to/sapphire

mkdir -p /path/to/sapphire-release
gh release download v0.130 -R virajpandya/sapphire \
  -p sapphire-data.tar.gz -D /path/to/sapphire-release
tar -xzf /path/to/sapphire-release/sapphire-data.tar.gz \
  -C /path/to/sapphire-release
```

The released archive contains a `sapphire/data/` directory. Pass that directory—not the tarball—to mimic-jax.

## Load metadata or run native physics

Metadata and capability inspection do not require Sapphire to be installed:

```python
from mimic_jax import load_model

sapphire = load_model("sapphire")
print(sapphire.metadata.state_variables)
print(sapphire.metadata.capability("events"))
```

Native execution requires all backend paths explicitly:

```python
from mimic_jax import load_model

sapphire = load_model(
    "sapphire",
    python_executable="/path/to/sapphire-venv/bin/python",
    source_repository="/path/to/sapphire",
    data_path="/path/to/sapphire-release/sapphire/data",
)
artifact = sapphire.run_local_case(output_directory="output/sapphire-native-case")
```

The backend checks the exact source revision and SD93 cooling table before starting. It refuses a different checkout rather than producing an ambiguously versioned result.

## Analyze the result

Sapphire's physical state combines masses with CGM thermal energy near `1e58 erg`, so local response and mode calculations require JAX x64. Set `JAX_ENABLE_X64=1` before Python starts:

```bash
JAX_ENABLE_X64=1 uv run python your_analysis.py
```

Then use the common concepts:

```python
from mimic_jax import characteristic_modes, load_model, scale_state_space
from mimic_jax.sapphire import SapphireNativeArtifact

artifact = SapphireNativeArtifact.load("output/sapphire-native-case")
sapphire = load_model("sapphire")

response = sapphire.local_response(artifact=artifact)
response = scale_state_space(response, artifact.state)
modes = characteristic_modes(response)
budgets = sapphire.conservation_balances(artifact)
```

The native parameter coordinates need care. Sapphire's `A_*` values are log10 normalizations in configuration and are exponentiated before the RHS; several fiducial parameters are zero or signed slopes. A logarithmic elasticity is therefore not universally meaningful. `sapphire.parameter_response(...)` uses the derivative of the complete native trajectory, applies the common normalization rules, and requires explicit positive reference scales when a log elasticity is invalid. The artifact retains the fixed-state local derivative separately so local dynamics and accumulated history are never conflated.

The validation tolerances also distinguish these objects. Local RHS and observable derivatives agree with symmetric finite differences to better than `1e-7` relative L2 error. The end-to-end derivative passes through an adaptive PID controller with discrete accepted/rejected step paths; it is checked at five parameter-coordinate perturbation sizes and currently agrees to `1.27e-3` at the declared `1e-4` reference step. This is reported as a separate `5e-3` gate rather than disguising adaptive-control nonsmoothness as a local-Jacobian failure.

## What is and is not comparable

The current common surface supports:

- stars, ISM, CGM, CGM thermal energy, and their metal reservoirs;
- smooth halo forcing through accretion rate, halo mass, virial radius, virial velocity, and concentration;
- native cooling, star formation, instantaneous recycling, ISM wind, CGM outflow, energy, and metal rates;
- SMHM, gas-fraction, MZR, SFMS, and related central-galaxy summaries when a suitable weighted population is supplied;
- state, halo-input, local parameter, and full-trajectory parameter response matrices;
- local transfer functions, coupled modes, response times, adaptive convergence, and open-system budgets.

The audited Pandya23 model does not contain general merger/satellite topology, black holes, an AGN loop, or a separate ejected reservoir. Mimic-jax reports these as unavailable or not applicable; it does not add replacement prescriptions. Sapphire's `M_cgm` is also retained as CGM mass rather than automatically equated with a SAGE/SHARK hot-gas field.

The [three-model response report](../reports/three-model-response-foundation/index.md) is the executable starting point. It shows the native validation and a qualified supply-response comparison. The next population gate is a documented smooth main-progenitor forcing adapter plus defensible selection and weighting, followed by shared SMHM, gas-fraction, metallicity, and SFMS comparisons.

## Reproduce the committed controlled fixture

With the native backend installed, the fixture is generated through the public adapter. Use x64 in both environments:

```bash
JAX_ENABLE_X64=1 uv run python - <<'PY'
from mimic_jax import load_model

sapphire = load_model(
    "sapphire",
    python_executable="/path/to/sapphire-venv/bin/python",
    source_repository="/path/to/sapphire",
    data_path="/path/to/sapphire-release/sapphire/data",
)
sapphire.run_local_case(
    output_directory="tests/data/sapphire/native-v0.130-controlled"
)
PY

JAX_ENABLE_X64=1 uv run python \
  scripts/generate_three_model_response_report.py
```

The committed fixture and report record the source revision, released cooling-table checksum, full case, tolerances, software, hardware, and derivative/convergence evidence. See the Sapphire README and Pandya et al. (2023, 2026) for the physical model and required citation/acknowledgement guidance.
