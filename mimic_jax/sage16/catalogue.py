"""Explicit conversion from internal SAGE16 records to upstream catalogue quantities."""

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from mimic_jax.catalogue import CatalogueField, ComparisonCatalogue
from mimic_jax.sage16.tree_evolve import (
    GalaxyRecord,
    SnapshotTiming,
    virial_mass,
    virial_radius,
    virial_velocity,
)
from mimic_jax.sage16.types import Sage16Units, sage16_units

SECONDS_PER_YEAR = 3.155e7
SECONDS_PER_MEGAYEAR = 3.155e13
SOLAR_MASS_G = 1.989e33

_COMPARISON_REQUIRED_FIELDS = (
    "UniqueGalaxyID",
    "Type",
    "StellarMass",
    "BulgeMass",
    "ColdGas",
    "HotGas",
    "EjectedGas",
    "ICS",
    "StarFormationRate",
    "BlackHoleMass",
    "CentralMvir",
    "Mvir",
    "MetalsColdGas",
    "MetalsStellarMass",
    "DiskScaleRadius",
    "Vmax",
    "Pos",
    "Vel",
    "Cooling",
    "Heating",
    "SupernovaOutflowRate",
)


def _float32_scale(value, factor):
    return np.float32(float(np.float32(value)) * factor)


def record_to_catalogue(
    record: GalaxyRecord,
    tree: np.ndarray,
    timing: SnapshotTiming,
    units: Sage16Units = None,
    *,
    particle_mass: float = 0.0860657,
):
    """Return all 42 public SAGE16 fields using upstream output conversions.

    This boundary is intentionally ordinary Python: unit conversion and HDF5
    marshalling are not part of the differentiable physical model.
    """

    if units is None:
        units = sage16_units()
    state = record.state
    halo = record.halo
    galaxy_type = int(halo.Type)
    snapshot = int(halo.SnapNum)
    dtime = float(halo.dT)
    if dtime != -1.0:
        dtime *= float(units.UnitTime_in_s) / SECONDS_PER_MEGAYEAR
    mass_rate_factor = (
        float(units.UnitMass_in_g) / float(units.UnitTime_in_s) * SECONDS_PER_YEAR / SOLAR_MASS_G
    )
    power_factor = float(units.UnitEnergy_in_cgs) / float(units.UnitTime_in_s)
    source_mass = virial_mass(tree, record.source_halo, particle_mass)
    source_radius = virial_radius(source_mass, float(timing.redshift[snapshot]), units)
    source_velocity = virial_velocity(source_mass, source_radius, units)

    def luminosity(value):
        value = float(value)
        if value != 0.0:
            value *= power_factor
        if value != 0.0:
            value = math.log10(value)
        return value

    return {
        "SnapNum": np.int32(snapshot),
        "Type": np.int32(galaxy_type),
        "UniqueGalaxyID": np.int64(halo.UniqueGalaxyID),
        "UniqueCentralGalaxyID": np.int64(halo.UniqueCentralGalaxyID),
        "dT": np.float64(dtime),
        "Len": np.int32(halo.Len),
        "Mvir": np.float64(halo.Mvir),
        "deltaMvir": np.float64(halo.deltaMvir),
        "CentralMvir": np.float64(halo.CentralMvir),
        "Rvir": np.float64(float(halo.Rvir) if galaxy_type == 2 else source_radius),
        "Vvir": np.float64(float(halo.Vvir) if galaxy_type == 2 else source_velocity),
        "infallMvir": np.float64(halo.infallMvir if galaxy_type != 0 else 0.0),
        "infallVvir": np.float64(halo.infallVvir if galaxy_type != 0 else 0.0),
        "infallVmax": np.float64(halo.infallVmax if galaxy_type != 0 else 0.0),
        "Pos": np.asarray(halo.Pos, dtype=np.float32),
        "Vel": np.asarray(halo.Vel, dtype=np.float32),
        "VelDisp": np.float32(halo.VelDisp),
        "Vmax": np.float32(halo.Vmax),
        "Spin": np.asarray(halo.Spin, dtype=np.float32),
        "MostBoundID": np.int64(halo.MostBoundID),
        "HaloBaryonFraction": np.float64(state.HaloBaryonFraction),
        "ColdGas": np.float32(state.ColdGas),
        "HotGas": np.float32(state.HotGas),
        "EjectedGas": np.float32(state.EjectedGas),
        "StellarMass": np.float32(state.StellarMass),
        "BulgeMass": np.float32(state.BulgeMass),
        "ICS": np.float32(state.ICS),
        "StarFormationRate": _float32_scale(state.StarFormationRate, mass_rate_factor),
        "MetalsStellarMass": np.float32(state.MetalsStellarMass),
        "MetalsBulgeMass": np.float32(state.MetalsBulgeMass),
        "MetalsColdGas": np.float32(state.MetalsColdGas),
        "MetalsHotGas": np.float32(state.MetalsHotGas),
        "MetalsICS": np.float32(state.MetalsICS),
        "MetalsEjectedGas": np.float32(state.MetalsEjectedGas),
        "BlackHoleMass": np.float32(state.BlackHoleMass),
        "QuasarModeBHaccretionMass": np.float32(state.QuasarModeBHaccretionMass),
        "Cooling": np.float64(luminosity(state.Cooling)),
        "Heating": np.float64(luminosity(state.Heating)),
        "SupernovaOutflowRate": _float32_scale(state.SupernovaOutflowRate, mass_rate_factor),
        "DiskScaleRadius": np.float32(state.DiskScaleRadius),
        "TimeOfLastMajorMerger": _float32_scale(
            state.TimeOfLastMajorMerger,
            float(units.UnitTime_in_s) / SECONDS_PER_MEGAYEAR / 1000.0,
        ),
        "TimeOfLastMinorMerger": _float32_scale(
            state.TimeOfLastMinorMerger,
            float(units.UnitTime_in_s) / SECONDS_PER_MEGAYEAR / 1000.0,
        ),
    }


def load_sage_comparison_catalogue(
    paths: Sequence[Path],
    *,
    snapshot: int,
    hubble_h: float,
    effective_volume_mpc_over_h_cubed: float,
    redshift: float,
    dataset: str,
) -> ComparisonCatalogue:
    """Project one or more native MIMIC/SAGE HDF5 files into common units.

    Volume, ``h``, and redshift are required because the current native SAGE
    output records fields and parameter metadata but does not embed the full
    simulation-volume/cosmology contract.  Requiring them avoids a hidden
    Mini-Millennium default in comparisons to other simulations.
    """

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - repository requirements include h5py
        raise RuntimeError("Reading SAGE HDF5 catalogues requires h5py") from error

    paths = tuple(Path(path) for path in paths)
    if not paths:
        raise ValueError("At least one SAGE catalogue path is required")
    group_name = f"Snap{int(snapshot):03d}"
    chunks = []
    parameters = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            dataset_path = f"{group_name}/Galaxies"
            if dataset_path not in handle:
                raise ValueError(f"{path} does not contain {dataset_path}")
            galaxies = np.asarray(handle[dataset_path])
            missing = [
                name for name in _COMPARISON_REQUIRED_FIELDS if name not in galaxies.dtype.names
            ]
            if missing:
                raise ValueError(f"SAGE catalogue {path} is missing required fields: {missing}")
            chunks.append(galaxies)
            if "RunProperties/Parameters" in handle:
                current = {
                    bytes(row["param_name"]).decode("utf-8"): bytes(row["value"]).decode("utf-8")
                    for row in np.asarray(handle["RunProperties/Parameters"])
                }
                if parameters is None:
                    parameters = current
                elif parameters != current:
                    raise ValueError("SAGE catalogue partitions use different parameter sets")
    galaxies = np.concatenate(chunks)
    mass_factor = 1.0e10 / float(hubble_h)

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
            galaxies["UniqueGalaxyID"],
            "dimensionless",
            "Persistent SAGE galaxy identifier.",
            ("UniqueGalaxyID",),
        ),
        "galaxy_type": field(
            galaxies["Type"],
            "dimensionless",
            "SAGE type: 0 central, 1 resolved satellite, 2 orphan.",
            ("Type",),
        ),
        "stellar_mass": field(
            galaxies["StellarMass"] * mass_factor,
            "Msun",
            "Total stellar mass, including the bulge component.",
            ("StellarMass",),
        ),
        "bulge_stellar_mass": field(
            galaxies["BulgeMass"] * mass_factor,
            "Msun",
            "Bulge stellar mass, a subset of total stellar mass.",
            ("BulgeMass",),
        ),
        "cold_gas_mass": field(
            galaxies["ColdGas"] * mass_factor,
            "Msun",
            "Total SAGE cold star-forming gas reservoir.",
            ("ColdGas",),
            qualification="SAGE does not natively split this reservoir into HI and H2.",
        ),
        "baryonic_mass": field(
            (galaxies["StellarMass"] + galaxies["ColdGas"]) * mass_factor,
            "Msun",
            "Declared cold baryonic mass: total stars plus cold gas.",
            ("StellarMass", "ColdGas"),
            "derived",
        ),
        "hot_gas_mass": field(
            galaxies["HotGas"] * mass_factor,
            "Msun",
            "Hot halo gas assigned to the galaxy record.",
            ("HotGas",),
        ),
        "ejected_gas_mass": field(
            galaxies["EjectedGas"] * mass_factor,
            "Msun",
            "Gas in SAGE's feedback-ejected reservoir.",
            ("EjectedGas",),
        ),
        "intracluster_stellar_mass": field(
            galaxies["ICS"] * mass_factor,
            "Msun",
            "Intracluster stellar mass.",
            ("ICS",),
        ),
        "star_formation_rate": field(
            galaxies["StarFormationRate"],
            "Msun/yr",
            "Snapshot-averaged total star-formation rate.",
            ("StarFormationRate",),
        ),
        "black_hole_mass": field(
            galaxies["BlackHoleMass"] * mass_factor,
            "Msun",
            "Central black-hole mass.",
            ("BlackHoleMass",),
        ),
        "host_halo_mass": field(
            galaxies["CentralMvir"] * mass_factor,
            "Msun",
            "FoF central halo virial mass stamped onto each member.",
            ("CentralMvir",),
            qualification="SAGE uses its 200-critical virial-mass convention.",
        ),
        "subhalo_mass": field(
            galaxies["Mvir"] * mass_factor,
            "Msun",
            "Galaxy-owning halo/subhalo virial mass; orphans have zero.",
            ("Mvir",),
            qualification="SAGE uses M200c for centrals and particle mass for subhalos.",
        ),
        "cold_gas_metal_mass": field(
            galaxies["MetalsColdGas"] * mass_factor,
            "Msun",
            "Total metal mass in the cold-gas reservoir.",
            ("MetalsColdGas",),
        ),
        "stellar_metal_mass": field(
            galaxies["MetalsStellarMass"] * mass_factor,
            "Msun",
            "Total stellar metal mass, already including bulge metals.",
            ("MetalsStellarMass",),
            qualification=(
                "MetalsBulgeMass is a subset and must not be added to MetalsStellarMass."
            ),
        ),
        "stellar_disk_radius": field(
            galaxies["DiskScaleRadius"] / float(hubble_h) * 1.0e3,
            "kpc",
            "SAGE exponential disk scale radius.",
            ("DiskScaleRadius",),
            qualification="This is a scale radius, not SHARK's disk half-mass radius.",
        ),
        "maximum_circular_velocity": field(
            galaxies["Vmax"],
            "km/s",
            "Maximum circular velocity of the owning subhalo or inherited infall value.",
            ("Vmax",),
        ),
        "position": field(
            galaxies["Pos"],
            "Mpc/h",
            "Comoving galaxy position inherited from the owning halo.",
            ("Pos",),
        ),
        "velocity": field(
            galaxies["Vel"],
            "km/s",
            "Galaxy peculiar velocity inherited from the owning halo.",
            ("Vel",),
        ),
        "sage_cooling_power": field(
            galaxies["Cooling"],
            "log10(erg/s)",
            "SAGE cumulative cooling-energy proxy converted to logarithmic power.",
            ("Cooling",),
            "model_specific",
            "This is not an instantaneous cooling mass rate.",
        ),
        "sage_heating_power": field(
            galaxies["Heating"],
            "log10(erg/s)",
            "SAGE cumulative AGN-heating proxy converted to logarithmic power.",
            ("Heating",),
            "model_specific",
            "This is not the same output as SHARK mechanical jet power.",
        ),
        "supernova_outflow_rate": field(
            galaxies["SupernovaOutflowRate"],
            "Msun/yr",
            "SAGE supernova-driven outflow rate.",
            ("SupernovaOutflowRate",),
            "model_specific",
        ),
    }
    unavailable = {
        "atomic_gas_mass": "fiducial SAGE16 outputs only total cold gas",
        "molecular_gas_mass": "fiducial SAGE16 outputs only total cold gas",
        "black_hole_spin": "fiducial SAGE16 does not evolve black-hole spin",
        "agn_bolometric_luminosity": "fiducial SAGE16 outputs cooling/heating proxies instead",
        "agn_mechanical_power": "fiducial SAGE16 outputs a heating proxy, not jet power",
        "stellar_angular_momentum": "fiducial SAGE16 does not output component angular momentum",
        "cooling_rate": "SAGE's public Cooling field is an energy proxy, not a mass rate",
        "burst_star_formation_rate": "fiducial SAGE16 outputs only total snapshot-averaged SFR",
    }
    return ComparisonCatalogue(
        model="SAGE16",
        dataset=dataset,
        snapshot=int(snapshot),
        redshift=float(redshift),
        hubble_h=float(hubble_h),
        effective_volume_mpc_over_h_cubed=float(effective_volume_mpc_over_h_cubed),
        fields=fields,
        unavailable_fields=unavailable,
        source_paths=paths,
        metadata={"parameters": parameters or {}, "native_mass_unit": "1e10 Msun/h"},
    )
