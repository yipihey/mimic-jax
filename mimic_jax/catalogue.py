"""Model-neutral galaxy catalogue contract for inter-SAM comparisons.

The contract stores physical quantities, their provenance, and explicit
unavailability reasons.  Model adapters remain responsible for translating
native output fields; observable functions consume only this representation.
This keeps a plot from silently changing units, component sums, or selections
when a third semi-analytic model is added.
"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

import numpy as np


class FieldUnavailableError(KeyError):
    """Raised when a requested physical quantity is absent by model design."""


@dataclass(frozen=True)
class CatalogueField:
    """One canonical catalogue quantity and its native-field provenance."""

    values: np.ndarray
    unit: str
    description: str
    source_fields: Tuple[str, ...]
    projection: str = "direct"
    qualification: str = ""

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        if values.ndim < 1:
            raise ValueError("Catalogue fields must have a galaxy axis")
        if not self.description or not self.source_fields:
            raise ValueError("Catalogue fields require a description and source fields")
        if self.projection not in ("direct", "component_sum", "derived", "model_specific"):
            raise ValueError(f"Unknown catalogue-field projection {self.projection!r}")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class ComparisonCatalogue:
    """Canonical physical galaxy quantities plus comparison metadata.

    All masses use physical ``Msun`` and star-formation rates use physical
    ``Msun/yr``.  ``effective_volume_mpc_over_h_cubed`` deliberately retains
    the native comoving ``(Mpc/h)^3`` convention, so number-density reductions
    apply the recorded ``hubble_h`` exactly once.
    """

    model: str
    dataset: str
    snapshot: int
    redshift: float
    hubble_h: float
    effective_volume_mpc_over_h_cubed: float
    fields: Mapping[str, CatalogueField]
    unavailable_fields: Mapping[str, str]
    source_paths: Tuple[Path, ...] = ()
    metadata: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        if not self.model or not self.dataset:
            raise ValueError("Comparison catalogues require model and dataset names")
        if not np.isfinite(self.redshift) or self.redshift < 0.0:
            raise ValueError("Catalogue redshift must be finite and non-negative")
        if not np.isfinite(self.hubble_h) or self.hubble_h <= 0.0:
            raise ValueError("Catalogue hubble_h must be finite and positive")
        if (
            not np.isfinite(self.effective_volume_mpc_over_h_cubed)
            or self.effective_volume_mpc_over_h_cubed <= 0.0
        ):
            raise ValueError("Catalogue effective volume must be finite and positive")
        fields = dict(self.fields)
        unavailable = dict(self.unavailable_fields)
        overlap = set(fields) & set(unavailable)
        if overlap:
            raise ValueError(f"Fields cannot be both available and unavailable: {sorted(overlap)}")
        required = {"galaxy_id", "galaxy_type"}
        missing = required - set(fields)
        if missing:
            raise ValueError(f"Comparison catalogue is missing identity fields: {sorted(missing)}")
        sizes = {field.values.shape[0] for field in fields.values()}
        if len(sizes) != 1:
            raise ValueError("All comparison catalogue fields must share a galaxy axis")
        object.__setattr__(self, "fields", MappingProxyType(fields))
        object.__setattr__(self, "unavailable_fields", MappingProxyType(unavailable))
        object.__setattr__(self, "source_paths", tuple(Path(path) for path in self.source_paths))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))

    @property
    def galaxy_count(self) -> int:
        """Number of galaxy rows in the catalogue."""

        return int(self.fields["galaxy_id"].values.shape[0])

    def has_field(self, name: str) -> bool:
        """Return whether a canonical physical quantity is available."""

        return name in self.fields

    def field(self, name: str) -> CatalogueField:
        """Return a canonical field or raise with its scientific reason."""

        if name in self.fields:
            return self.fields[name]
        reason = self.unavailable_fields.get(name, "the model adapter does not define this field")
        raise FieldUnavailableError(f"{name!r} is unavailable for {self.model}: {reason}")

    def values(self, name: str) -> np.ndarray:
        """Return the immutable values for one canonical field."""

        return self.field(name).values


@dataclass(frozen=True)
class ObservableSpec:
    """Shared observable definition and the fields needed to evaluate it."""

    key: str
    label: str
    required_fields: Tuple[str, ...]
    unit: str
    definition: str
    qualification: str = ""


@dataclass(frozen=True)
class ObservableCapability:
    """Availability of one observable for one concrete catalogue."""

    key: str
    status: str
    reason: str
    missing_fields: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("direct", "qualified", "unavailable"):
            raise ValueError(f"Unknown observable capability status {self.status!r}")


COMMON_OBSERVABLES = (
    ObservableSpec(
        "stellar_mass_function",
        "Stellar mass function",
        ("stellar_mass",),
        "Mpc^-3 dex^-1",
        "Number density in bins of total physical stellar mass.",
    ),
    ObservableSpec(
        "cold_gas_mass_function",
        "Cold-gas mass function",
        ("cold_gas_mass",),
        "Mpc^-3 dex^-1",
        "Number density in bins of total star-forming cold-gas mass.",
        "Cold-phase definitions are similar but not mathematically identical across SAMs.",
    ),
    ObservableSpec(
        "baryonic_mass_function",
        "Cold baryonic mass function",
        ("baryonic_mass",),
        "Mpc^-3 dex^-1",
        "Number density in bins of M_star+M_cold.",
        "This declared cold-baryon definition excludes hot, ejected, and diffuse reservoirs.",
    ),
    ObservableSpec(
        "black_hole_mass_function",
        "Black-hole mass function",
        ("black_hole_mass",),
        "Mpc^-3 dex^-1",
        "Number density in bins of central black-hole mass.",
    ),
    ObservableSpec(
        "cosmic_sfr_density",
        "Cosmic star-formation-rate density",
        ("stellar_mass", "star_formation_rate"),
        "Msun yr^-1 Mpc^-3",
        "Sum of total SFR for the declared stellar-mass selection per physical volume.",
    ),
    ObservableSpec(
        "quenched_fraction",
        "Quenched fraction",
        ("stellar_mass", "star_formation_rate"),
        "dimensionless",
        "Fraction below a declared total-sSFR threshold in stellar-mass bins.",
    ),
    ObservableSpec(
        "sfr_stellar_mass_relation",
        "SFR--stellar-mass relation",
        ("stellar_mass", "star_formation_rate"),
        "log10(Msun/yr)",
        "Median total SFR in total-stellar-mass bins.",
    ),
    ObservableSpec(
        "cold_gas_fraction",
        "Cold-gas fraction",
        ("stellar_mass", "cold_gas_mass"),
        "dimensionless",
        "M_cold/(M_cold+M_star) using each model's total star-forming cold reservoir.",
        "Cold-phase definitions are similar but not mathematically identical across SAMs.",
    ),
    ObservableSpec(
        "gas_metallicity",
        "Cold-gas metallicity",
        ("stellar_mass", "cold_gas_mass", "cold_gas_metal_mass"),
        "log10 metal mass fraction",
        "Mass-weighted total metal fraction M_Z,cold/M_cold.",
        "This is not an oxygen-line calibration; comparison to 12+log(O/H) needs a convention.",
    ),
    ObservableSpec(
        "stellar_metallicity",
        "Stellar metallicity",
        ("stellar_mass", "stellar_metal_mass"),
        "log10 metal mass fraction",
        "Mass-weighted total stellar metal fraction M_Z,star/M_star.",
    ),
    ObservableSpec(
        "black_hole_bulge_relation",
        "Black-hole--bulge relation",
        ("bulge_stellar_mass", "black_hole_mass"),
        "log10(Msun)",
        "Median physical black-hole mass in bins of physical bulge stellar mass.",
    ),
    ObservableSpec(
        "stellar_halo_relation",
        "Stellar-to-host-halo relation",
        ("galaxy_type", "stellar_mass", "host_halo_mass"),
        "log10(Msun)",
        "Median total stellar mass in host-halo-mass bins for a declared type selection.",
        "Host-mass definitions and central selection must be aligned.",
    ),
    ObservableSpec(
        "hot_gas_relation",
        "Hot-gas--halo relation",
        ("host_halo_mass", "hot_gas_mass"),
        "log10(Msun)",
        "Median tracked hot gas in host-halo-mass bins.",
        "Reservoir ownership across satellites and FoF groups must be aligned.",
    ),
    ObservableSpec(
        "ejected_gas_relation",
        "Ejected-gas--halo relation",
        ("host_halo_mass", "ejected_gas_mass"),
        "log10(Msun)",
        "Median feedback-ejected reservoir in host-halo-mass bins.",
        "The location and ownership of the ejected reservoir differ between models.",
    ),
    ObservableSpec(
        "diffuse_stellar_relation",
        "Diffuse-stellar--halo relation",
        ("host_halo_mass", "intracluster_stellar_mass"),
        "log10(Msun)",
        "Median diffuse intrahalo/intracluster stellar mass in host-halo-mass bins.",
        "SAGE ICS and SHARK stellar-halo boundaries arise from different stripping prescriptions.",
    ),
    ObservableSpec(
        "baryonic_tully_fisher",
        "Baryonic Tully--Fisher relation",
        ("baryonic_mass", "maximum_circular_velocity"),
        "log10(Msun)",
        "Median cold baryonic mass as a function of a declared circular-velocity proxy.",
        "Vmax ownership for satellites/orphans and the observational velocity proxy must be declared.",
    ),
    ObservableSpec(
        "spatial_clustering",
        "Spatial clustering / selections",
        ("position", "stellar_mass"),
        "dimensionless",
        "Correlation statistics from comoving positions under a declared galaxy selection.",
        "Requires the same simulation volume, cosmology, periodic boundary, and selection.",
    ),
    ObservableSpec(
        "atomic_gas_mass_function",
        "Atomic-gas mass function",
        ("atomic_gas_mass",),
        "Mpc^-3 dex^-1",
        "Number density in bins of model-predicted atomic hydrogen mass.",
    ),
    ObservableSpec(
        "molecular_gas_mass_function",
        "Molecular-gas mass function",
        ("molecular_gas_mass",),
        "Mpc^-3 dex^-1",
        "Number density in bins of model-predicted molecular hydrogen mass.",
    ),
    ObservableSpec(
        "stellar_size_relation",
        "Stellar size--mass relation",
        ("stellar_mass", "stellar_disk_radius"),
        "log10(kpc)",
        "Median stellar disk radius in total-stellar-mass bins.",
        "SAGE and SHARK radius definitions differ and must remain visible.",
    ),
    ObservableSpec(
        "black_hole_spin_relation",
        "Black-hole spin relation",
        ("black_hole_mass", "black_hole_spin"),
        "dimensionless",
        "Median dimensionless BH spin in black-hole-mass bins.",
    ),
    ObservableSpec(
        "agn_mechanical_power_relation",
        "AGN mechanical-power relation",
        ("black_hole_mass", "agn_mechanical_power"),
        "1e40 erg/s",
        "Median mechanical jet power in black-hole- or halo-mass bins.",
    ),
    ObservableSpec(
        "cooling_rate_relation",
        "Cooling-rate relation",
        ("host_halo_mass", "cooling_rate"),
        "Msun/yr",
        "Median instantaneous cooling mass rate in host-halo-mass bins.",
        "SAGE's public Cooling output is an energy proxy, not this mass rate.",
    ),
    ObservableSpec(
        "stellar_angular_momentum_relation",
        "Stellar angular-momentum relation",
        ("stellar_mass", "stellar_angular_momentum"),
        "km/s Mpc/h",
        "Median specific disk-stellar angular momentum in stellar-mass bins.",
    ),
    ObservableSpec(
        "burst_fraction_relation",
        "Burst contribution to star formation",
        ("star_formation_rate", "burst_star_formation_rate"),
        "dimensionless",
        "Burst SFR divided by total SFR under the model's trigger definitions.",
    ),
)


def observable_capabilities(
    catalogue: ComparisonCatalogue,
    specs: Sequence[ObservableSpec] = COMMON_OBSERVABLES,
) -> Tuple[ObservableCapability, ...]:
    """Assess common observables without substituting missing model physics."""

    capabilities = []
    for spec in specs:
        missing = tuple(field for field in spec.required_fields if not catalogue.has_field(field))
        if missing:
            reasons = [catalogue.unavailable_fields.get(field, "not provided") for field in missing]
            capabilities.append(
                ObservableCapability(
                    spec.key,
                    "unavailable",
                    "; ".join(f"{field}: {reason}" for field, reason in zip(missing, reasons)),
                    missing,
                )
            )
        elif spec.qualification or any(
            catalogue.field(field).qualification for field in spec.required_fields
        ):
            qualifications = [spec.qualification] + [
                catalogue.field(field).qualification for field in spec.required_fields
            ]
            capabilities.append(
                ObservableCapability(
                    spec.key,
                    "qualified",
                    " ".join(value for value in qualifications if value),
                )
            )
        else:
            capabilities.append(
                ObservableCapability(spec.key, "direct", "All required canonical fields exist.")
            )
    return tuple(capabilities)
