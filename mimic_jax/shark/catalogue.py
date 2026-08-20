"""Ordinary-Python adapter from native SHARK HDF5 to common observables."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mimic_jax.observables import (
    BinnedFraction,
    BinnedRelation,
    binned_relation,
    binned_selected_fraction,
)
from mimic_jax.sage16.population import StellarMassFunction, stellar_mass_function


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
    "bh_spin",
    "bolometric_luminosity_agn",
    "mechanical_power_agn",
    "mhot",
    "mreheated",
    "mlost",
    "mhot_stripped",
    "mism_stripped",
    "mstars_tidally_stripped",
    "mstellar_halo",
    "rstar_disk",
    "rstar_bulge",
    "specific_angular_momentum_disk_star",
    "specific_angular_momentum_disk_gas",
    "specific_angular_momentum_disk_gas_atom",
    "specific_angular_momentum_disk_gas_mol",
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


def shark_stellar_mass_function(catalogue: SharkCatalogue, *, bin_edges) -> StellarMassFunction:
    """Evaluate the same histogram/volume convention used by SAGE16 reports."""

    return stellar_mass_function(
        catalogue.stellar_mass / 1.0e10,
        volume_mpc_over_h_cubed=catalogue.effective_volume,
        hubble_h=catalogue.hubble_h,
        bin_edges=bin_edges,
    )


def shark_atomic_gas_mass_function(catalogue: SharkCatalogue, *, bin_edges) -> StellarMassFunction:
    """Return the total disk+bulge HI mass function in the shared convention."""

    return stellar_mass_function(
        catalogue.atomic_gas_mass / 1.0e10,
        volume_mpc_over_h_cubed=catalogue.effective_volume,
        hubble_h=catalogue.hubble_h,
        bin_edges=bin_edges,
    )


def shark_molecular_gas_mass_function(
    catalogue: SharkCatalogue, *, bin_edges
) -> StellarMassFunction:
    """Return the total disk+bulge H2 mass function in the shared convention."""

    return stellar_mass_function(
        catalogue.molecular_gas_mass / 1.0e10,
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
