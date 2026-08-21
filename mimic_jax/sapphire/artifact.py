"""Versioned native-Sapphire result artifacts.

Sapphire currently requires a newer Python/JAX stack than mimic-jax.  The
artifact contract keeps native execution in an isolated environment while
making its states, rates, Jacobians, trajectories, and provenance available
to the same analysis and reporting code used for established SAMs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np

from mimic_jax.linear_response import (
    AnnotatedStateSpace,
    LinearizationPoint,
    LocalStateSpace,
    ResponseCoordinate,
    annotate_state_space,
)

SAPPHIRE_ARTIFACT_SCHEMA = "mimic-jax-sapphire-native/v1"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coordinates(records) -> tuple[ResponseCoordinate, ...]:
    return tuple(
        ResponseCoordinate(
            name=record["name"],
            label=record.get("label", record["name"]),
            unit=record["unit"],
            description=record.get("description", ""),
        )
        for record in records
    )


@dataclass(frozen=True)
class SapphireNativeArtifact:
    """One checksum-verified native Sapphire trajectory and local response."""

    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    directory: Path | None = None

    def __post_init__(self) -> None:
        if self.manifest.get("schema_version") != SAPPHIRE_ARTIFACT_SCHEMA:
            raise ValueError(
                "Unsupported Sapphire artifact schema " f"{self.manifest.get('schema_version')!r}"
            )
        expected = self.manifest.get("arrays", {})
        if set(expected) != set(self.arrays):
            raise ValueError(
                "Sapphire artifact array keys do not match its manifest: "
                f"expected {tuple(sorted(expected))}, received {tuple(sorted(self.arrays))}"
            )
        for name, metadata in expected.items():
            array = np.asarray(self.arrays[name])
            if list(array.shape) != metadata["shape"]:
                raise ValueError(
                    f"Sapphire artifact array {name!r} has shape {array.shape}, "
                    f"expected {tuple(metadata['shape'])}"
                )
            if str(array.dtype) != metadata["dtype"]:
                raise ValueError(
                    f"Sapphire artifact array {name!r} has dtype {array.dtype}, "
                    f"expected {metadata['dtype']}"
                )

    @classmethod
    def load(cls, directory) -> "SapphireNativeArtifact":
        """Load and verify the JSON/NPZ artifact pair in ``directory``."""

        directory = Path(directory).resolve()
        manifest_path = directory / "artifact.json"
        arrays_path = directory / "arrays.npz"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_checksum = manifest.get("array_file", {}).get("sha256")
        actual_checksum = _sha256(arrays_path)
        if expected_checksum != actual_checksum:
            raise ValueError(
                "Sapphire artifact checksum mismatch: "
                f"manifest has {expected_checksum!r}, arrays have {actual_checksum!r}"
            )
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        return cls(manifest=manifest, arrays=arrays, directory=directory)

    @property
    def state(self) -> np.ndarray:
        """Physical state at the local linearization point."""

        return self.arrays["linearization_state"]

    @property
    def derivative(self) -> np.ndarray:
        """Native-Pandya23 derivative expressed in physical units per Gyr."""

        return self.arrays["state_derivative"]

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(item["name"] for item in self.manifest["coordinates"]["state"])

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(item["name"] for item in self.manifest["coordinates"]["observable"])

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(item["name"] for item in self.manifest["coordinates"]["input"])

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(item["name"] for item in self.manifest["coordinates"]["parameter"])

    def observable(self, name: str) -> float:
        """Return one physically named local observable."""

        try:
            index = self.observable_names.index(name)
        except ValueError as error:
            raise KeyError(
                f"Unknown Sapphire observable {name!r}; choose from {self.observable_names}"
            ) from error
        return float(self.arrays["observable_values"][index])

    def rate(self, name: str) -> float:
        """Return one native auxiliary rate with its units retained in the manifest."""

        names = tuple(item["name"] for item in self.manifest["coordinates"]["rate"])
        try:
            index = names.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown Sapphire rate {name!r}; choose from {names}") from error
        return float(self.arrays["rate_values"][index])

    def local_response(self) -> AnnotatedStateSpace:
        """Return the native JAX linearization through mimic-jax response tools."""

        if not jax.config.x64_enabled:
            raise RuntimeError(
                "Sapphire response analysis requires JAX x64 because the native state mixes "
                "galaxy masses with CGM energies near 1e58 erg. Set JAX_ENABLE_X64=1 before "
                "starting Python."
            )
        context = self.manifest["linearization_point"]
        matrices = LocalStateSpace(
            state_jacobian=self.arrays["state_jacobian"],
            input_jacobian=self.arrays["input_jacobian"],
            output_jacobian=self.arrays["output_jacobian"],
            direct_input_jacobian=self.arrays["direct_input_jacobian"],
        )
        annotated = annotate_state_space(
            matrices,
            point=LinearizationPoint(
                model="Sapphire Pandya23",
                formulation=self.manifest["model"]["formulation"],
                time=float(context["time_gyr"]),
                time_unit="Gyr",
                time_unit_in_gyr=1.0,
                redshift=float(context["redshift"]),
                halo_mass=float(context["halo_mass_msun"]),
                halo_mass_unit="Msun",
                qualification=self.manifest["qualification"],
            ),
            state_coordinates=_coordinates(self.manifest["coordinates"]["state"]),
            input_coordinates=_coordinates(self.manifest["coordinates"]["input"]),
            output_coordinates=_coordinates(self.manifest["coordinates"]["observable"]),
        )
        return AnnotatedStateSpace(
            matrices=annotated.matrices,
            point=annotated.point,
            state_coordinates=annotated.state_coordinates,
            input_coordinates=annotated.input_coordinates,
            output_coordinates=annotated.output_coordinates,
            derivative_method=self.manifest["derivatives"].get(
                "local_method", self.manifest["derivatives"]["method"]
            ),
            state_scales=self.state,
        )

    def parameter_output_jacobian(self) -> np.ndarray:
        """Return the instantaneous output derivative at fixed final state."""

        return self.arrays["parameter_output_jacobian"]

    def trajectory_parameter_output_jacobian(self) -> np.ndarray:
        """Return end-to-end output derivatives through native Diffrax."""

        return self.arrays["trajectory_parameter_output_jacobian"]

    @property
    def trajectory_parameter_validation(self) -> tuple[np.ndarray, np.ndarray]:
        """Return finite-difference steps and AD discrepancies for the adaptive solve."""

        return (
            self.arrays["trajectory_parameter_output_jacobian_finite_difference_steps"],
            self.arrays["trajectory_parameter_output_jacobian_finite_difference_errors"],
        )

    @property
    def derivative_validation(self) -> Mapping[str, float]:
        """Native AD-versus-symmetric-finite-difference validation metrics."""

        return self.manifest["derivatives"]["validation"]

    @property
    def convergence_fraction(self) -> np.ndarray:
        """Requested-tolerance final state relative to a stricter native solve."""

        return self.arrays["convergence_fraction"]


def write_sapphire_artifact(directory, manifest, arrays) -> SapphireNativeArtifact:
    """Write a deterministic artifact pair and return its verified representation."""

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    normalized_arrays = {name: np.asarray(value) for name, value in sorted(arrays.items())}
    arrays_path = directory / "arrays.npz"
    np.savez_compressed(arrays_path, **normalized_arrays)
    resolved = dict(manifest)
    resolved["schema_version"] = SAPPHIRE_ARTIFACT_SCHEMA
    resolved["arrays"] = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in normalized_arrays.items()
    }
    resolved["array_file"] = {
        "path": "arrays.npz",
        "sha256": _sha256(arrays_path),
        "size_bytes": arrays_path.stat().st_size,
    }
    (directory / "artifact.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return SapphireNativeArtifact.load(directory)
