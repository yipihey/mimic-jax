"""Ordinary-Python adapter from native SHARK HDF5 to common observables."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mimic_jax.catalogue import CatalogueField, ComparisonCatalogue
from mimic_jax.observables import (
    BinnedFraction,
    BinnedRelation,
    MassFunction,
    binned_relation,
    binned_selected_fraction,
    mass_function,
)

StellarMassFunction = MassFunction


@dataclass(frozen=True)
class SharkCatalogue:
    """Selected native fields plus model-neutral component sums.

    Native masses are ``Msun/h`` and native star-formation rates are
    ``Msun/Gyr/h``.  Component sums remain in those native units so no hidden
    IMF or ``h`` convention enters the adapter.
    """

    path: Path
    galaxy_id: np.ndarray
    galaxy_type: np.ndarray
    stellar_mass: np.ndarray
    bulge_stellar_mass: np.ndarray
    cold_gas_mass: np.ndarray
    atomic_gas_mass: np.ndarray
    molecular_gas_mass: np.ndarray
    star_formation_rate: np.ndarray
    burst_star_formation_rate: np.ndarray
    black_hole_mass: np.ndarray
    host_halo_mass: np.ndarray
    cold_gas_metal_mass: np.ndarray
    stellar_metal_mass: np.ndarray
    hubble_h: float
    effective_volume: float
    redshift: float
    upstream_revision: str
    upstream_version: str
    seed: int
    extras: Mapping[str, np.ndarray]

    @property
    def star_formation_rate_msun_per_year(self):
        """Return total SFR in physical ``Msun/yr``."""

        return self.star_formation_rate / self.hubble_h / 1.0e9

    @property
    def stellar_mass_msun(self):
        """Return total stellar mass in physical ``Msun``."""

        return self.stellar_mass / self.hubble_h

    @property
    def cold_gas_mass_msun(self):
        """Return disk+bulge cold gas in physical ``Msun``."""

        return self.cold_gas_mass / self.hubble_h

    @property
    def black_hole_mass_msun(self):
        """Return black-hole mass in physical ``Msun``."""

        return self.black_hole_mass / self.hubble_h

    @property
    def bulge_stellar_mass_msun(self):
        """Return bulge stellar mass in physical ``Msun``."""

        return self.bulge_stellar_mass / self.hubble_h

    @property
    def specific_star_formation_rate_per_year(self):
        """Return total SFR divided by total stellar mass in ``yr^-1``."""

        result = np.full(self.stellar_mass.shape, np.nan, dtype=np.float64)
        positive = self.stellar_mass > 0.0
        result[positive] = self.star_formation_rate[positive] / 1.0e9 / self.stellar_mass[positive]
        return result


_REQUIRED_FIELDS = (
    "id_galaxy",
    "type",
    "mstars_disk",
    "mstars_bulge",
    "mgas_disk",
    "mgas_bulge",
    "matom_disk",
    "matom_bulge",
    "mmol_disk",
    "mmol_bulge",
    "sfr_disk",
    "sfr_burst",
    "m_bh",
    "mvir_hosthalo",
    "mgas_metals_disk",
    "mgas_metals_bulge",
    "mstars_metals_disk",
    "mstars_metals_bulge",
)

_EXTRA_FIELDS = (
    "bh_accretion_rate_hh",
    "bh_accretion_rate_sb",
    "bh_spin",
    "bolometric_luminosity_agn",
    "cooling_rate",
    "mechanical_power_agn",
    "mhot",
    "mreheated",
    "mlost",
    "mhot_stripped",
    "mism_stripped",
    "mstars_tidally_stripped",
    "mstellar_halo",
    "position_x",
    "position_y",
    "position_z",
    "rstar_disk",
    "rstar_bulge",
    "specific_angular_momentum_disk_star",
    "specific_angular_momentum_disk_gas",
    "specific_angular_momentum_disk_gas_atom",
    "specific_angular_momentum_disk_gas_mol",
    "vmax_subhalo",
    "velocity_x",
    "velocity_y",
    "velocity_z",
)


def _decode_scalar(value: Any) -> str:
    scalar = np.asarray(value).item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8").rstrip("\x00")
    return str(scalar)


def load_shark_catalogue(path) -> SharkCatalogue:
    """Load a native upstream SHARK catalogue with strict schema checks."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - dependency is in repository requirements
        raise RuntimeError("Reading SHARK output requires h5py") from error

    path = Path(path)
    with h5py.File(path, "r") as handle:
        missing = [name for name in _REQUIRED_FIELDS if f"galaxies/{name}" not in handle]
        if missing:
            raise ValueError(f"SHARK catalogue is missing required fields: {missing}")
        fields = {
            name: np.asarray(handle[f"galaxies/{name}"], dtype=np.float64)
            for name in _REQUIRED_FIELDS
        }
        fields["id_galaxy"] = np.asarray(handle["galaxies/id_galaxy"], dtype=np.int64)
        fields["type"] = np.asarray(handle["galaxies/type"], dtype=np.int32)
        lengths = {values.shape for values in fields.values()}
        if len(lengths) != 1 or not lengths or len(next(iter(lengths))) != 1:
            raise ValueError("Required SHARK galaxy fields must be equal-length vectors")
        extras = {
            name: np.asarray(handle[f"galaxies/{name}"])
            for name in _EXTRA_FIELDS
            if f"galaxies/{name}" in handle
        }
        hubble_h = float(np.asarray(handle["cosmology/h"]))
        effective_volume = float(np.asarray(handle["run_info/effective_volume"]))
        redshift = float(np.asarray(handle["run_info/redshift"]))
        revision = _decode_scalar(handle["run_info/shark_git_revision"])
        version = _decode_scalar(handle["run_info/shark_version"])
        seed = int(np.asarray(handle["run_info/seed"]))

    if hubble_h <= 0.0 or effective_volume <= 0.0:
        raise ValueError("SHARK catalogue has invalid cosmology or effective volume metadata")
    return SharkCatalogue(
        path=path,
        galaxy_id=fields["id_galaxy"],
        galaxy_type=fields["type"],
        stellar_mass=fields["mstars_disk"] + fields["mstars_bulge"],
        bulge_stellar_mass=fields["mstars_bulge"],
        cold_gas_mass=fields["mgas_disk"] + fields["mgas_bulge"],
        atomic_gas_mass=fields["matom_disk"] + fields["matom_bulge"],
        molecular_gas_mass=fields["mmol_disk"] + fields["mmol_bulge"],
        star_formation_rate=fields["sfr_disk"] + fields["sfr_burst"],
        burst_star_formation_rate=fields["sfr_burst"],
        black_hole_mass=fields["m_bh"],
        host_halo_mass=fields["mvir_hosthalo"],
        cold_gas_metal_mass=fields["mgas_metals_disk"] + fields["mgas_metals_bulge"],
        stellar_metal_mass=fields["mstars_metals_disk"] + fields["mstars_metals_bulge"],
        hubble_h=hubble_h,
        effective_volume=effective_volume,
        redshift=redshift,
        upstream_revision=revision,
        upstream_version=version,
        seed=seed,
        extras=extras,
    )


def shark_comparison_catalogue(
    catalogue: SharkCatalogue,
    *,
    dataset: str,
    snapshot: int,
) -> ComparisonCatalogue:
    """Project native SHARK output into the model-neutral physical catalogue."""

    mass_factor = 1.0 / catalogue.hubble_h

    def field(values, unit, description, source_fields, projection="direct", qualification=""):
        return CatalogueField(
            np.asarray(values),
            unit,
            description,
            tuple(source_fields),
            projection,
            qualification,
        )

    fields = {
        "galaxy_id": field(
            catalogue.galaxy_id,
            "dimensionless",
            "Persistent native SHARK galaxy identifier.",
            ("id_galaxy",),
        ),
        "galaxy_type": field(
            catalogue.galaxy_type,
            "dimensionless",
            "SHARK type: 0 central, 1 resolved satellite, 2 orphan.",
            ("type",),
        ),
        "stellar_mass": field(
            catalogue.stellar_mass * mass_factor,
            "Msun",
            "Total disk plus bulge stellar mass.",
            ("mstars_disk", "mstars_bulge"),
            "component_sum",
        ),
        "bulge_stellar_mass": field(
            catalogue.bulge_stellar_mass * mass_factor,
            "Msun",
            "Bulge stellar mass.",
            ("mstars_bulge",),
        ),
        "cold_gas_mass": field(
            catalogue.cold_gas_mass * mass_factor,
            "Msun",
            "Total disk plus bulge cold gas.",
            ("mgas_disk", "mgas_bulge"),
            "component_sum",
            "SHARK partitions cold gas into atomic and molecular components internally.",
        ),
        "baryonic_mass": field(
            (catalogue.stellar_mass + catalogue.cold_gas_mass) * mass_factor,
            "Msun",
            "Declared cold baryonic mass: total stars plus cold gas.",
            ("mstars_disk", "mstars_bulge", "mgas_disk", "mgas_bulge"),
            "derived",
        ),
        "atomic_gas_mass": field(
            catalogue.atomic_gas_mass * mass_factor,
            "Msun",
            "Total disk plus bulge atomic hydrogen mass.",
            ("matom_disk", "matom_bulge"),
            "component_sum",
        ),
        "molecular_gas_mass": field(
            catalogue.molecular_gas_mass * mass_factor,
            "Msun",
            "Total disk plus bulge molecular hydrogen mass.",
            ("mmol_disk", "mmol_bulge"),
            "component_sum",
        ),
        "star_formation_rate": field(
            catalogue.star_formation_rate / catalogue.hubble_h / 1.0e9,
            "Msun/yr",
            "Total disk plus burst star-formation rate.",
            ("sfr_disk", "sfr_burst"),
            "component_sum",
        ),
        "burst_star_formation_rate": field(
            catalogue.burst_star_formation_rate / catalogue.hubble_h / 1.0e9,
            "Msun/yr",
            "Star-formation rate in merger- and disk-instability-triggered bursts.",
            ("sfr_burst",),
            "model_specific",
        ),
        "black_hole_mass": field(
            catalogue.black_hole_mass * mass_factor,
            "Msun",
            "Central black-hole mass.",
            ("m_bh",),
        ),
        "host_halo_mass": field(
            catalogue.host_halo_mass * mass_factor,
            "Msun",
            "Host-halo virial mass.",
            ("mvir_hosthalo",),
            qualification="SHARK's native virial-mass convention follows the input catalogue.",
        ),
        "cold_gas_metal_mass": field(
            catalogue.cold_gas_metal_mass * mass_factor,
            "Msun",
            "Total metal mass in disk plus bulge cold gas.",
            ("mgas_metals_disk", "mgas_metals_bulge"),
            "component_sum",
        ),
        "stellar_metal_mass": field(
            catalogue.stellar_metal_mass * mass_factor,
            "Msun",
            "Total metal mass in disk plus bulge stars.",
            ("mstars_metals_disk", "mstars_metals_bulge"),
            "component_sum",
        ),
    }
    optional_mass_fields = {
        "hot_gas_mass": ("mhot", "Hot halo-gas mass."),
        "ejected_gas_mass": (
            "mreheated",
            "SHARK reheated/ejected reservoir; its boundary differs from SAGE's EjectedGas.",
        ),
        "lost_gas_mass": ("mlost", "Gas explicitly lost beyond the tracked halo reservoirs."),
        "intracluster_stellar_mass": (
            "mstellar_halo",
            "Diffuse stellar-halo component.",
        ),
    }
    for key, (native, description) in optional_mass_fields.items():
        if native in catalogue.extras:
            fields[key] = field(
                np.asarray(catalogue.extras[native]) * mass_factor,
                "Msun",
                description,
                (native,),
                qualification=(
                    "Reservoir boundaries must be aligned before interpreting a SAGE comparison."
                    if key == "ejected_gas_mass"
                    else ""
                ),
            )
    if "bh_spin" in catalogue.extras:
        fields["black_hole_spin"] = field(
            catalogue.extras["bh_spin"],
            "dimensionless",
            "Dimensionless black-hole spin.",
            ("bh_spin",),
            "model_specific",
        )
    if "bolometric_luminosity_agn" in catalogue.extras:
        fields["agn_bolometric_luminosity"] = field(
            catalogue.extras["bolometric_luminosity_agn"],
            "1e40 erg/s",
            "AGN bolometric luminosity in the native SHARK output convention.",
            ("bolometric_luminosity_agn",),
            "model_specific",
        )
    if "mechanical_power_agn" in catalogue.extras:
        fields["agn_mechanical_power"] = field(
            catalogue.extras["mechanical_power_agn"],
            "1e40 erg/s",
            "AGN mechanical power in the native SHARK output convention.",
            ("mechanical_power_agn",),
            "model_specific",
        )
    if "cooling_rate" in catalogue.extras:
        fields["cooling_rate"] = field(
            np.asarray(catalogue.extras["cooling_rate"]) / catalogue.hubble_h / 1.0e9,
            "Msun/yr",
            "Instantaneous cooling rate of the hot-halo component.",
            ("cooling_rate",),
        )
    if "rstar_disk" in catalogue.extras:
        fields["stellar_disk_radius"] = field(
            np.asarray(catalogue.extras["rstar_disk"]) / catalogue.hubble_h * 1.0e3,
            "kpc",
            "SHARK stellar disk half-mass radius.",
            ("rstar_disk",),
            qualification="This is a half-mass radius, not SAGE's exponential scale radius.",
        )
    if "specific_angular_momentum_disk_star" in catalogue.extras:
        fields["stellar_angular_momentum"] = field(
            catalogue.extras["specific_angular_momentum_disk_star"],
            "km/s Mpc/h",
            "Specific angular momentum of disk stars in SHARK's native convention.",
            ("specific_angular_momentum_disk_star",),
            "model_specific",
        )
    if "vmax_subhalo" in catalogue.extras:
        fields["maximum_circular_velocity"] = field(
            catalogue.extras["vmax_subhalo"],
            "km/s",
            "Maximum circular velocity of the owning subhalo or inherited value.",
            ("vmax_subhalo",),
        )
    position_fields = ("position_x", "position_y", "position_z")
    if all(name in catalogue.extras for name in position_fields):
        fields["position"] = field(
            np.stack([catalogue.extras[name] for name in position_fields], axis=1),
            "Mpc/h",
            "Comoving galaxy position; SHARK samples type-2 orphans in the host NFW halo.",
            position_fields,
            "derived",
            "Orphan positions include SHARK's stochastic NFW sampling convention.",
        )
    velocity_fields = ("velocity_x", "velocity_y", "velocity_z")
    if all(name in catalogue.extras for name in velocity_fields):
        fields["velocity"] = field(
            np.stack([catalogue.extras[name] for name in velocity_fields], axis=1),
            "km/s",
            "Galaxy peculiar velocity.",
            velocity_fields,
            "derived",
        )

    unavailable = {}
    for key, reason in (
        ("black_hole_spin", "the selected SHARK output omitted bh_spin"),
        ("agn_bolometric_luminosity", "the selected SHARK output omitted AGN luminosity"),
        ("agn_mechanical_power", "the selected SHARK output omitted AGN mechanical power"),
        ("stellar_disk_radius", "the selected SHARK output omitted rstar_disk"),
        ("stellar_angular_momentum", "the selected SHARK output omitted disk stellar AM"),
        ("maximum_circular_velocity", "the selected SHARK output omitted vmax_subhalo"),
        ("position", "the selected SHARK output omitted one or more position components"),
        ("velocity", "the selected SHARK output omitted one or more velocity components"),
        ("cooling_rate", "the selected SHARK output omitted cooling_rate"),
    ):
        if key not in fields:
            unavailable[key] = reason
    return ComparisonCatalogue(
        model="SHARK Lagos23",
        dataset=dataset,
        snapshot=int(snapshot),
        redshift=float(catalogue.redshift),
        hubble_h=float(catalogue.hubble_h),
        effective_volume_mpc_over_h_cubed=float(catalogue.effective_volume),
        fields=fields,
        unavailable_fields=unavailable,
        source_paths=(catalogue.path,),
        metadata={
            "upstream_revision": catalogue.upstream_revision,
            "upstream_version": catalogue.upstream_version,
            "seed": catalogue.seed,
            "native_mass_unit": "Msun/h",
            "native_sfr_unit": "Msun/Gyr/h",
        },
    )


def shark_stellar_mass_function(catalogue: SharkCatalogue, *, bin_edges) -> StellarMassFunction:
    """Evaluate the same histogram/volume convention used by SAGE16 reports."""

    return mass_function(
        catalogue.stellar_mass_msun,
        volume_mpc_over_h_cubed=catalogue.effective_volume,
        hubble_h=catalogue.hubble_h,
        bin_edges=bin_edges,
    )


def shark_atomic_gas_mass_function(catalogue: SharkCatalogue, *, bin_edges) -> StellarMassFunction:
    """Return the total disk+bulge HI mass function in the shared convention."""

    return mass_function(
        catalogue.atomic_gas_mass / catalogue.hubble_h,
        volume_mpc_over_h_cubed=catalogue.effective_volume,
        hubble_h=catalogue.hubble_h,
        bin_edges=bin_edges,
    )


def shark_molecular_gas_mass_function(
    catalogue: SharkCatalogue, *, bin_edges
) -> StellarMassFunction:
    """Return the total disk+bulge H2 mass function in the shared convention."""

    return mass_function(
        catalogue.molecular_gas_mass / catalogue.hubble_h,
        volume_mpc_over_h_cubed=catalogue.effective_volume,
        hubble_h=catalogue.hubble_h,
        bin_edges=bin_edges,
    )


def _selection(catalogue, centrals_only):
    return (
        catalogue.galaxy_type == 0
        if centrals_only
        else np.ones_like(catalogue.galaxy_type, dtype=bool)
    )


def shark_quenched_fraction(
    catalogue: SharkCatalogue,
    *,
    bin_edges,
    specific_sfr_threshold_per_year=1.0e-11,
    centrals_only=False,
) -> BinnedFraction:
    """Return quenched fraction versus physical stellar mass."""

    selected_sample = _selection(catalogue, centrals_only)
    stellar_mass = catalogue.stellar_mass_msun[selected_sample]
    positive = stellar_mass > 0.0
    log_stellar_mass = np.full(stellar_mass.shape, np.nan)
    log_stellar_mass[positive] = np.log10(stellar_mass[positive])
    quenched = (
        catalogue.specific_star_formation_rate_per_year[selected_sample]
        < specific_sfr_threshold_per_year
    )
    return binned_selected_fraction(log_stellar_mass, quenched, bin_edges=bin_edges)


def shark_cold_gas_fraction_relation(
    catalogue: SharkCatalogue, *, bin_edges, centrals_only=False
) -> BinnedRelation:
    """Return cold-gas baryonic fraction versus physical stellar mass."""

    selected = _selection(catalogue, centrals_only)
    stellar = catalogue.stellar_mass_msun[selected]
    gas = catalogue.cold_gas_mass_msun[selected]
    predictor = np.full(stellar.shape, np.nan)
    predictor[stellar > 0.0] = np.log10(stellar[stellar > 0.0])
    denominator = stellar + gas
    fraction = np.full(stellar.shape, np.nan)
    fraction[denominator > 0.0] = gas[denominator > 0.0] / denominator[denominator > 0.0]
    return binned_relation(predictor, fraction, bin_edges=bin_edges)


def shark_gas_metallicity_relation(
    catalogue: SharkCatalogue, *, bin_edges, centrals_only=False
) -> BinnedRelation:
    """Return ``log10(M_Z,cold/M_cold)`` versus physical stellar mass."""

    selected = _selection(catalogue, centrals_only)
    stellar = catalogue.stellar_mass_msun[selected]
    cold = catalogue.cold_gas_mass[selected]
    metals = catalogue.cold_gas_metal_mass[selected]
    predictor = np.full(stellar.shape, np.nan)
    predictor[stellar > 0.0] = np.log10(stellar[stellar > 0.0])
    metallicity = np.full(stellar.shape, np.nan)
    positive = (cold > 0.0) & (metals > 0.0)
    metallicity[positive] = np.log10(metals[positive] / cold[positive])
    return binned_relation(predictor, metallicity, bin_edges=bin_edges)


def shark_black_hole_bulge_relation(
    catalogue: SharkCatalogue, *, bin_edges, centrals_only=False
) -> BinnedRelation:
    """Return physical black-hole mass versus physical bulge stellar mass."""

    selected = _selection(catalogue, centrals_only)
    bulge = catalogue.bulge_stellar_mass_msun[selected]
    black_hole = catalogue.black_hole_mass_msun[selected]
    predictor = np.full(bulge.shape, np.nan)
    response = np.full(black_hole.shape, np.nan)
    predictor[bulge > 0.0] = np.log10(bulge[bulge > 0.0])
    response[black_hole > 0.0] = np.log10(black_hole[black_hole > 0.0])
    return binned_relation(predictor, response, bin_edges=bin_edges)


def shark_black_hole_spin_relation(
    catalogue: SharkCatalogue, *, bin_edges, centrals_only=False
) -> BinnedRelation:
    """Return BH spin versus physical black-hole mass where spin is available."""

    if "bh_spin" not in catalogue.extras:
        raise ValueError("SHARK catalogue does not contain galaxies/bh_spin")
    selected = _selection(catalogue, centrals_only)
    mass = catalogue.black_hole_mass_msun[selected]
    spin = np.asarray(catalogue.extras["bh_spin"], dtype=np.float64)[selected]
    predictor = np.full(mass.shape, np.nan)
    predictor[mass > 0.0] = np.log10(mass[mass > 0.0])
    return binned_relation(predictor, spin, bin_edges=bin_edges)


def shark_stellar_size_relation(
    catalogue: SharkCatalogue, *, bin_edges, centrals_only=False
) -> BinnedRelation:
    """Return disk stellar half-mass radius versus physical stellar mass."""

    if "rstar_disk" not in catalogue.extras:
        raise ValueError("SHARK catalogue does not contain galaxies/rstar_disk")
    selected = _selection(catalogue, centrals_only)
    mass = catalogue.stellar_mass_msun[selected]
    radius = np.asarray(catalogue.extras["rstar_disk"], dtype=np.float64)[selected]
    predictor = np.full(mass.shape, np.nan)
    response = np.full(radius.shape, np.nan)
    predictor[mass > 0.0] = np.log10(mass[mass > 0.0])
    # Native radii are Mpc/h. Convert to physical kpc.
    response[radius > 0.0] = np.log10(radius[radius > 0.0] / catalogue.hubble_h * 1.0e3)
    return binned_relation(predictor, response, bin_edges=bin_edges)


def shark_cosmic_star_formation_rate_density(catalogue: SharkCatalogue) -> float:
    """Return total SFR density in ``Msun yr^-1 (Mpc/h)^-3``.

    The volume remains the catalogue's native comoving ``(Mpc/h)^3`` volume;
    SFR is converted from ``Msun/Gyr/h`` to physical ``Msun/yr``.  Keeping the
    mixed but familiar convention explicit avoids a hidden factor of ``h^3``.
    """

    return float(
        np.sum(catalogue.star_formation_rate)
        / catalogue.hubble_h
        / 1.0e9
        / catalogue.effective_volume
    )


def shark_stellar_metallicity_relation(
    catalogue: SharkCatalogue, *, bin_edges, centrals_only=False
) -> BinnedRelation:
    """Return ``log10(M_Z,star/M_star)`` versus physical stellar mass."""

    selected = _selection(catalogue, centrals_only)
    stellar = catalogue.stellar_mass[selected]
    metals = catalogue.stellar_metal_mass[selected]
    physical_stellar = catalogue.stellar_mass_msun[selected]
    predictor = np.full(stellar.shape, np.nan)
    predictor[physical_stellar > 0.0] = np.log10(physical_stellar[physical_stellar > 0.0])
    metallicity = np.full(stellar.shape, np.nan)
    positive = (stellar > 0.0) & (metals > 0.0)
    metallicity[positive] = np.log10(metals[positive] / stellar[positive])
    return binned_relation(predictor, metallicity, bin_edges=bin_edges)


def shark_stellar_to_halo_mass_relation(
    catalogue: SharkCatalogue, *, bin_edges, centrals_only=True
) -> BinnedRelation:
    """Return physical stellar mass versus physical host-halo mass."""

    selected = _selection(catalogue, centrals_only)
    halo = catalogue.host_halo_mass[selected] / catalogue.hubble_h
    stellar = catalogue.stellar_mass_msun[selected]
    predictor = np.full(halo.shape, np.nan)
    response = np.full(stellar.shape, np.nan)
    predictor[halo > 0.0] = np.log10(halo[halo > 0.0])
    response[stellar > 0.0] = np.log10(stellar[stellar > 0.0])
    return binned_relation(predictor, response, bin_edges=bin_edges)
