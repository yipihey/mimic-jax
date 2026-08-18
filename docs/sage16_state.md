# SAGE16 State in mimic-jax

The immutable `GalaxyState` PyTree contains the 32 fields declared by [`models/sage16/model_properties.yaml`](../models/sage16/model_properties.yaml), in metadata order and with the same public names. This is the complete model state even though the initial JAX process slice currently reads or writes only part of it.

Persistent SAGE reservoirs remain `float32` because upstream deliberately preserves the published SAGE structure's float storage. Calculation and transport fields declared `double` upstream remain `float64`. Enable JAX 64-bit mode before constructing a state; mimic-jax fails clearly instead of silently truncating these fields.

| Field | JAX storage | Physical role | Lifecycle |
| --- | --- | --- | --- |
| `HaloBaryonFraction` | float64 | Reionization-modified baryon fraction for the halo | Persistent, initialized on first use |
| `InfallingGas` | float64 | Cosmological infall budget for the current snapshot | Snapshot transport |
| `CoolingGas` | float64 | Hot gas budget allowed to cool during the substep | Substep transport |
| `ColdGas` | float32 | Cold disk gas available for star formation | Persistent reservoir |
| `HotGas` | float32 | Hot halo atmosphere | Persistent reservoir |
| `EjectedGas` | float32 | Gas outside the hot halo after feedback ejection | Persistent reservoir |
| `StellarMass` | float32 | Total bound stellar mass | Persistent reservoir |
| `BulgeMass` | float32 | Bulge subset of `StellarMass` | Persistent component, not an additional mass reservoir |
| `ICS` | float32 | Intracluster stellar mass | Persistent reservoir |
| `NewStellarMass` | float64 | Newly formed stellar mass passed through the SF/SN/yield chain | Substep transport |
| `StarFormationRate` | float32 | Snapshot-accumulated star formation diagnostic | Reset each snapshot |
| `MetalsStellarMass` | float32 | Metals in total bound stars | Persistent metal reservoir |
| `MetalsBulgeMass` | float32 | Bulge subset of `MetalsStellarMass` | Persistent component, not an additional metal reservoir |
| `MetalsColdGas` | float32 | Metals in cold disk gas | Persistent metal reservoir |
| `MetalsHotGas` | float32 | Metals in the hot halo | Persistent metal reservoir |
| `MetalsICS` | float32 | Metals in intracluster stars | Persistent metal reservoir |
| `MetalsEjectedGas` | float32 | Metals in ejected gas | Persistent metal reservoir |
| `BlackHoleMass` | float32 | Central black-hole mass | Persistent reservoir |
| `QuasarModeBHaccretionMass` | float32 | Snapshot quasar-mode BH accretion diagnostic | Reset each snapshot |
| `SupernovaReheatedMass` | float64 | Calculated cold-to-hot SN transport budget | Substep transport |
| `SupernovaEjectedMass` | float64 | Calculated hot-to-ejected SN transport budget | Substep transport |
| `Cooling` | float64 | Snapshot-accumulated cooling-energy diagnostic | Reset each snapshot |
| `Heating` | float64 | Snapshot-accumulated AGN-heating diagnostic | Reset each snapshot |
| `Rcool` | float64 | Cooling radius calculated in the current substep | Substep transport |
| `CoolingLambda` | float64 | Sutherland-Dopita cooling function value for the current substep | Substep diagnostic |
| `Rheat` | float32 | Heating radius retained from AGN feedback | Persistent diagnostic state |
| `SupernovaOutflowRate` | float32 | Snapshot-accumulated reheating/outflow diagnostic | Reset each snapshot |
| `DiskScaleRadius` | float32 | Exponential disk scale length | Persistent structure state |
| `MergTime` | float32 | Dynamical-friction merger clock | Persistent event state; `999.9` means unset |
| `TimeOfLastMajorMerger` | float32 | Cosmic time of the last major merger | Persistent event history |
| `TimeOfLastMinorMerger` | float32 | Cosmic time of the last minor merger | Persistent event history |
| `UnstableDiskGasFraction` | float64 | Cold-gas fraction selected by disk instability for downstream burst/AGN consumers | Substep transport |

`HaloForcing` is separate because merger-tree properties are external forcing, not baryonic state. It currently includes the halo identity/type fields and every instantaneous property read by the inspected SAGE16 modules. `Sage16Parameters`, `Sage16Units`, and `StepContext` are also immutable PyTrees, so the numerical boundary is explicit: `(state, halo, context, parameters, units) -> transfers and new state`.

The baryonic ledger counts `ColdGas + HotGas + EjectedGas + StellarMass + ICS + BlackHoleMass`. It does not add `BulgeMass`, because the bulge is already included in `StellarMass`, and it does not add transport buffers. The metal ledger follows the same rule and does not double-count `MetalsBulgeMass`.

Current code: [`mimic_jax/sage16/types.py`](../mimic_jax/sage16/types.py). Upstream structural source: [`models/sage16/model_properties.yaml`](../models/sage16/model_properties.yaml).
