"""Semantic and native-runtime adapter for Sapphire's Pandya23 model."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from mimic_jax.sam import (
    Capability,
    ConservationBalance,
    ConservationQuantity,
    ProcessMetadata,
    SamModelMetadata,
    VariableMetadata,
)
from mimic_jax.sapphire.artifact import SapphireNativeArtifact
from mimic_jax.sensitivity import LOG_ELASTICITY, parameter_response_from_derivatives

SAPPHIRE_UPSTREAM_REPOSITORY = "https://github.com/virajpandya/sapphire"
SAPPHIRE_UPSTREAM_REVISION = "ee50e858e3427de50368c32205001248849b8be0"
SAPPHIRE_UPSTREAM_VERSION = "0.130"

SAPPHIRE_STATE_NAMES = (
    "M_star",
    "M_ism",
    "M_cgm",
    "Eth_cgm",
    "MZ_star",
    "MZ_ism",
    "MZ_cgm",
)
SAPPHIRE_FORCING_NAMES = ("Mdot_in_dm", "Mvir", "Rvir", "Vvir", "NFW_c")
SAPPHIRE_PARAMETER_NAMES = (
    "A_M",
    "alpha0_M",
    "alphaz_M",
    "beta_M",
    "A_E",
    "alpha0_E",
    "alphaz_E",
    "beta_E",
    "A_SF",
    "alpha0_SF",
    "alphaz_SF",
    "beta_SF",
    "A_Z",
    "alpha0_Z",
    "alphaz_Z",
    "beta_Z",
)


class SapphireBackendUnavailableError(RuntimeError):
    """Raised when native Sapphire execution was requested without a valid backend."""


@dataclass(frozen=True)
class SapphireLocalCase:
    """Controlled halo experiment delegated to native Sapphire.

    The forcing values are held constant over the interval.  This makes the
    local response interpretable but is not a replacement for Sapphire's TNG
    or Diffmah/CDHMAH history inputs.
    """

    initial_state: Mapping[str, float]
    forcing: Mapping[str, float]
    parameters: Mapping[str, float]
    start_time_gyr: float = 4.0
    end_time_gyr: float = 8.0
    sample_count: int = 65
    relative_tolerance: float = 1.0e-8
    absolute_tolerance: float = 1.0e-8
    maximum_steps: int = 65536

    def __post_init__(self) -> None:
        for label, values, required in (
            ("state", self.initial_state, SAPPHIRE_STATE_NAMES),
            ("forcing", self.forcing, SAPPHIRE_FORCING_NAMES),
            ("parameter", self.parameters, SAPPHIRE_PARAMETER_NAMES),
        ):
            if set(values) != set(required):
                raise ValueError(
                    f"Sapphire {label} names must be {required}, received {tuple(values)}"
                )
        if self.start_time_gyr <= 0.0 or self.end_time_gyr <= self.start_time_gyr:
            raise ValueError("Sapphire local-case times must satisfy 0 < start < end")
        if self.sample_count < 2:
            raise ValueError("Sapphire local cases require at least two saved times")
        if any(float(value) <= 0.0 for value in self.initial_state.values()):
            raise ValueError("Sapphire logarithmic state coordinates must be strictly positive")
        if any(float(value) <= 0.0 for value in self.forcing.values()):
            raise ValueError("Sapphire halo-forcing coordinates must be strictly positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mimic-jax-sapphire-case/v1",
            "initial_state": dict(self.initial_state),
            "forcing": dict(self.forcing),
            "parameters": dict(self.parameters),
            "start_time_gyr": self.start_time_gyr,
            "end_time_gyr": self.end_time_gyr,
            "sample_count": self.sample_count,
            "solver": {
                "rtol": self.relative_tolerance,
                "atol": self.absolute_tolerance,
                "max_steps": self.maximum_steps,
            },
        }


def fiducial_sapphire_case() -> SapphireLocalCase:
    """Return a transparent, positive controlled case using Sapphire defaults."""

    return SapphireLocalCase(
        initial_state={
            "M_star": 1.0e9,
            "M_ism": 2.0e9,
            "M_cgm": 8.0e10,
            "Eth_cgm": 4.0e58,
            "MZ_star": 1.0e7,
            "MZ_ism": 2.0e7,
            "MZ_cgm": 1.6e8,
        },
        forcing={
            "Mdot_in_dm": 70.0,
            "Mvir": 1.0e12,
            "Rvir": 190.0,
            "Vvir": 155.0,
            "NFW_c": 8.0,
        },
        parameters={
            "A_M": 0.0,
            "alpha0_M": -1.0,
            "alphaz_M": 0.0,
            "beta_M": 0.0,
            "A_E": -1.0,
            "alpha0_E": -1.0,
            "alphaz_E": 0.0,
            "beta_E": 0.0,
            "A_SF": 0.8,
            "alpha0_SF": -1.8,
            "alphaz_SF": 0.0,
            "beta_SF": -0.7,
            "A_Z": -1.7,
            "alpha0_Z": -0.3,
            "alphaz_Z": 0.0,
            "beta_Z": 0.0,
        },
    )


@dataclass(frozen=True)
class SapphireNativeBackend:
    """Isolated Python runtime and assets used for native Sapphire execution."""

    python_executable: Path
    source_repository: Path
    data_path: Path
    bridge_script: Path | None = None

    def __post_init__(self) -> None:
        # Keep a virtual-environment launcher as a symlink. Resolving it would
        # invoke the base interpreter and silently discard the environment's
        # installed Sapphire/JAX packages.
        object.__setattr__(self, "python_executable", Path(self.python_executable).absolute())
        object.__setattr__(self, "source_repository", Path(self.source_repository).resolve())
        object.__setattr__(self, "data_path", Path(self.data_path).resolve())
        if self.bridge_script is not None:
            object.__setattr__(self, "bridge_script", Path(self.bridge_script).resolve())

    def validate(self) -> None:
        """Fail before a scientific run if runtime, source, data, or revision drifted."""

        if not self.python_executable.is_file():
            raise SapphireBackendUnavailableError(
                f"Sapphire Python executable does not exist: {self.python_executable}"
            )
        if not (self.source_repository / "sapphire" / "models" / "pandya23.py").is_file():
            raise SapphireBackendUnavailableError(
                f"Sapphire source tree is incomplete: {self.source_repository}"
            )
        if not (self.data_path / "coolfunc" / "newcool.dat").is_file():
            raise SapphireBackendUnavailableError(
                "Sapphire SD93 cooling table is missing; expected "
                f"{self.data_path / 'coolfunc' / 'newcool.dat'}"
            )
        revision = subprocess.run(
            ["git", "-C", str(self.source_repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != SAPPHIRE_UPSTREAM_REVISION:
            raise SapphireBackendUnavailableError(
                "Sapphire source revision mismatch: "
                f"expected {SAPPHIRE_UPSTREAM_REVISION}, received {revision}"
            )

    def run(self, case: SapphireLocalCase, output_directory) -> SapphireNativeArtifact:
        """Execute the pinned native model and return a verified artifact."""

        self.validate()
        bridge = self.bridge_script
        if bridge is None:
            bridge = Path(__file__).resolve().parents[2] / "scripts" / "run_sapphire_bridge.py"
        if not bridge.is_file():
            raise SapphireBackendUnavailableError(f"Sapphire bridge script is missing: {bridge}")
        output_directory = Path(output_directory).resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mimic-jax-sapphire-case-") as temporary:
            case_path = Path(temporary) / "case.json"
            case_path.write_text(
                json.dumps(case.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            subprocess.run(
                [
                    str(self.python_executable),
                    str(bridge),
                    "--source",
                    str(self.source_repository),
                    "--data",
                    str(self.data_path),
                    "--case",
                    str(case_path),
                    "--output",
                    str(output_directory),
                ],
                check=True,
            )
        artifact = SapphireNativeArtifact.load(output_directory)
        if artifact.manifest["model"]["revision"] != SAPPHIRE_UPSTREAM_REVISION:
            raise ValueError("Native Sapphire artifact does not match the configured revision")
        return artifact


@dataclass(frozen=True)
class SapphireConfiguredModel:
    """Configured third-model boundary backed by native Sapphire artifacts."""

    metadata: SamModelMetadata
    backend: SapphireNativeBackend | None = None

    @property
    def default_parameters(self):
        return fiducial_sapphire_case().parameters

    def run_local_case(
        self, case: SapphireLocalCase | None = None, *, output_directory
    ) -> SapphireNativeArtifact:
        if self.backend is None:
            raise SapphireBackendUnavailableError(
                "Native Sapphire execution needs an isolated backend. Configure "
                "load_model('sapphire', python_executable=..., source_repository=..., "
                "data_path=...)."
            )
        return self.backend.run(
            fiducial_sapphire_case() if case is None else case, output_directory
        )

    def local_response(self, *, artifact: SapphireNativeArtifact) -> Any:
        return artifact.local_response()

    def observable_value(self, artifact: SapphireNativeArtifact, observable_name: str) -> float:
        return artifact.observable(observable_name)

    def rate_value(self, artifact: SapphireNativeArtifact, process_name: str) -> float:
        return artifact.rate(process_name)

    def conserved_quantities(
        self, artifact: SapphireNativeArtifact
    ) -> tuple[ConservationQuantity, ...]:
        """Expose open-system mass ledgers reconstructed from native coordinates."""

        state = artifact.state
        return (
            ConservationQuantity(
                "baryons",
                jnp.asarray(state[:3]).sum(),
                "Msun",
                "Mass in Sapphire's stellar, ISM, and CGM reservoirs.",
                ("halo inflow",),
                ("CGM outflow",),
            ),
            ConservationQuantity(
                "metals",
                jnp.asarray(state[4:]).sum(),
                "Msun",
                "Metal mass in stellar, ISM, and CGM reservoirs.",
                ("stellar yield", "enriched halo inflow"),
                ("enriched CGM outflow",),
            ),
        )

    def conservation_balances(
        self, artifact: SapphireNativeArtifact
    ) -> tuple[ConservationBalance, ...]:
        """Return native open-boundary residuals exported by the bridge."""

        residuals = artifact.arrays["conservation_residuals"]
        return (
            ConservationBalance(
                "baryons",
                residuals[0],
                artifact.rate("Mdot_in_halo") * 1.0e9,
                artifact.rate("Mdot_out_halo") * 1.0e9,
                "Msun/Gyr",
                "Residual after native halo inflow and CGM outflow.",
            ),
            ConservationBalance(
                "metals",
                residuals[1],
                (artifact.rate("MZdot_yield") + artifact.rate("MZdot_in_halo")) * 1.0e9,
                artifact.rate("MZdot_out_halo") * 1.0e9,
                "Msun/Gyr",
                "Residual after yield, enriched inflow, and enriched CGM outflow.",
            ),
        )

    def parameter_response(
        self,
        artifact: SapphireNativeArtifact,
        *,
        parameter_names,
        observable_names,
        normalization: str = LOG_ELASTICITY,
        observable_scales=None,
        parameter_scales=None,
        invalid: str = "raise",
    ):
        """Normalize native end-to-end trajectory derivatives with common semantics."""

        parameter_names = tuple(parameter_names)
        observable_names = tuple(observable_names)
        parameter_indices = tuple(artifact.parameter_names.index(name) for name in parameter_names)
        observable_indices = tuple(
            artifact.observable_names.index(name) for name in observable_names
        )
        raw = artifact.trajectory_parameter_output_jacobian()[
            np.ix_(observable_indices, parameter_indices)
        ]
        parameter_values = artifact.arrays["parameter_values"][list(parameter_indices)]
        observable_values = artifact.arrays["observable_values"][list(observable_indices)]
        observable_metadata = {
            item["name"]: item for item in artifact.manifest["coordinates"]["observable"]
        }
        parameter_metadata = {
            item["name"]: item for item in artifact.manifest["coordinates"]["parameter"]
        }
        return parameter_response_from_derivatives(
            raw,
            observable_values,
            parameter_values,
            parameter_names=parameter_names,
            observable_names=observable_names,
            observable_units=tuple(observable_metadata[name]["unit"] for name in observable_names),
            parameter_units=tuple(parameter_metadata[name]["unit"] for name in parameter_names),
            normalization=normalization,
            observable_scales=observable_scales,
            parameter_scales=parameter_scales,
            invalid=invalid,
            derivative_method=artifact.manifest["derivatives"].get(
                "trajectory_method", artifact.manifest["derivatives"]["method"]
            ),
            model=self.metadata.label,
            formulation=self.metadata.formulation,
            qualification=(
                self.metadata.qualification
                + " Response differentiates final observables through the native adaptive trajectory."
            ),
        )


def _metadata() -> SamModelMetadata:
    state_units = {
        "M_star": "Msun",
        "M_ism": "Msun",
        "M_cgm": "Msun",
        "Eth_cgm": "erg",
        "MZ_star": "Msun",
        "MZ_ism": "Msun",
        "MZ_cgm": "Msun",
    }
    state_descriptions = {
        "M_star": "Long-lived stellar mass.",
        "M_ism": "Interstellar gas mass.",
        "M_cgm": "Circumgalactic gas mass; not assumed identical to another SAM's hot reservoir.",
        "Eth_cgm": "Thermal energy of the circumgalactic medium.",
        "MZ_star": "Metal mass locked in long-lived stars.",
        "MZ_ism": "Metal mass in the interstellar medium.",
        "MZ_cgm": "Metal mass in the circumgalactic medium.",
    }
    forcing_units = {
        "Mdot_in_dm": "Msun/yr",
        "Mvir": "Msun",
        "Rvir": "proper kpc",
        "Vvir": "km/s",
        "NFW_c": "dimensionless",
    }
    processes = (
        ProcessMetadata(
            "halo_infall",
            "Halo inflow",
            "forcing",
            "UV/preventive-feedback-modulated cosmological supply to the CGM.",
            "piecewise smooth",
            "sapphire.models.pandya23.setup/integrator",
            (),
            ("M_cgm",),
        ),
        ProcessMetadata(
            "cooling",
            "CGM cooling",
            "flow",
            "Thermal-energy cooling coupled to CGM-to-ISM mass transfer.",
            "piecewise smooth with a dynamical-time limiter",
            "sapphire.models.pandya23.setup/integrator",
            ("M_cgm", "Eth_cgm"),
            ("M_ism",),
        ),
        ProcessMetadata(
            "star_formation",
            "Star formation",
            "flow",
            "ISM consumption into long-lived stellar mass with instantaneous recycling.",
            "smooth for positive state",
            "sapphire.models.pandya23.setup/integrator",
            ("M_ism",),
            ("M_star",),
        ),
        ProcessMetadata(
            "ism_wind",
            "ISM wind",
            "flow",
            "FIRE-calibrated mass, energy, and metal loading from ISM to CGM.",
            "piecewise smooth because energy loading is capped",
            "sapphire.models.pandya23.setup/integrator",
            ("M_ism",),
            ("M_cgm", "Eth_cgm"),
        ),
        ProcessMetadata(
            "cgm_outflow",
            "CGM outflow",
            "flow",
            "Loss from an over-pressurized CGM on a halo dynamical time.",
            "thresholded at positive excess thermal energy",
            "sapphire.models.pandya23.setup/integrator",
            ("M_cgm", "Eth_cgm"),
            (),
        ),
        ProcessMetadata(
            "metal_enrichment",
            "Metal enrichment",
            "flow",
            "Stellar yield and advective metal transport through stars, ISM, and CGM.",
            "piecewise smooth",
            "sapphire.models.pandya23.setup/integrator",
        ),
    )
    return SamModelMetadata(
        name="sapphire",
        label="Sapphire Pandya23",
        upstream_repository=SAPPHIRE_UPSTREAM_REPOSITORY,
        upstream_revision=SAPPHIRE_UPSTREAM_REVISION,
        formulation="native seven-state Pandya23 continuous central-galaxy model",
        qualification=(
            "Native Sapphire v0.130 is delegated to an isolated runtime. The common adapter "
            "does not add mergers, satellites, black holes, or a separate ejected reservoir."
        ),
        time_unit="Gyr",
        time_unit_in_gyr=1.0,
        state_variables=tuple(
            VariableMetadata(
                name,
                name.replace("_", " "),
                state_units[name],
                "metal_reservoir" if name.startswith("MZ") else "reservoir",
                state_descriptions[name],
            )
            for name in SAPPHIRE_STATE_NAMES
        ),
        forcing_variables=tuple(
            VariableMetadata(
                name,
                name.replace("_", " "),
                forcing_units[name],
                "forcing",
                "Smooth halo-history input consumed by native Sapphire.",
            )
            for name in SAPPHIRE_FORCING_NAMES
        ),
        parameter_variables=tuple(
            VariableMetadata(
                name,
                name.replace("_", " "),
                "native Sapphire coordinate",
                "parameter",
                (
                    "Sapphire configuration parameter. A_* normalizations are log10 values in "
                    "configuration and exponentiated before the RHS; slopes remain linear."
                ),
            )
            for name in SAPPHIRE_PARAMETER_NAMES
        ),
        observable_variables=(
            VariableMetadata("stellar_mass", "stellar mass", "Msun", "observable", "M_star."),
            VariableMetadata("ism_mass", "ISM mass", "Msun", "observable", "M_ism."),
            VariableMetadata(
                "cgm_mass", "CGM mass", "Msun", "observable", "M_cgm without relabeling it hot gas."
            ),
            VariableMetadata(
                "star_formation_rate",
                "star-formation rate",
                "Msun/yr",
                "observable",
                "Native instantaneous Mdot_sfr.",
            ),
            VariableMetadata(
                "stellar_metallicity",
                "stellar metallicity",
                "mass fraction",
                "observable",
                "MZ_star / M_star.",
            ),
            VariableMetadata(
                "ism_metallicity",
                "ISM metallicity",
                "mass fraction",
                "observable",
                "MZ_ism / M_ism.",
            ),
        ),
        processes=processes,
        capabilities=(
            Capability(
                "continuous_rhs",
                "supported",
                "Native Pandya23 seven-state logarithmic RHS, exported in physical coordinates.",
            ),
            Capability(
                "adaptive_integration",
                "supported",
                "Native Diffrax Tsit5 with PID error control and explicit tolerances.",
            ),
            Capability(
                "events",
                "unavailable",
                "The audited Sapphire model has no general merger/satellite jump-map interface.",
            ),
            Capability(
                "constraints",
                "model_specific",
                "Cooling, energy loading, UV suppression, and CGM outflow include native caps/thresholds.",
            ),
            Capability(
                "conservation",
                "partial",
                "Mass and metal open-system budgets can be reconstructed from native rates; no structural ledger API is claimed upstream.",
            ),
            Capability(
                "full_tree_physics_parity",
                "not_applicable",
                "Execution delegates to native Sapphire rather than porting Pandya23 equations.",
            ),
            Capability(
                "independent_topology_driver",
                "not_applicable",
                "The current model evolves independent smooth central-halo histories without merger topology.",
            ),
            Capability(
                "common_tree_forcing",
                "partial",
                "Smooth main-progenitor halo histories can be adapted; branch topology is outside this model.",
            ),
            Capability(
                "population_observables",
                "partial",
                "SMHM, gas fraction, MZR, SFMS, and related summaries are native; number-density statistics require a weighted population sample.",
            ),
        ),
    )


def configured_sapphire(**options) -> SapphireConfiguredModel:
    """Return Sapphire metadata and, when configured, a native subprocess backend."""

    backend = options.pop("backend", None)
    runtime_fields = ("python_executable", "source_repository", "data_path")
    supplied = {name: options.pop(name, None) for name in runtime_fields}
    bridge_script = options.pop("bridge_script", None)
    if options:
        raise TypeError(f"Unknown Sapphire model options: {sorted(options)}")
    if backend is not None and any(value is not None for value in supplied.values()):
        raise TypeError("Pass either backend= or native Sapphire runtime fields, not both")
    if backend is None and any(value is not None for value in supplied.values()):
        if any(value is None for value in supplied.values()):
            missing = tuple(name for name, value in supplied.items() if value is None)
            raise TypeError(f"Incomplete native Sapphire backend; missing {missing}")
        backend = SapphireNativeBackend(bridge_script=bridge_script, **supplied)
    return SapphireConfiguredModel(metadata=_metadata(), backend=backend)
