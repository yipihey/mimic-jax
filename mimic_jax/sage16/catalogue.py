"""Explicit conversion from internal SAGE16 records to upstream catalogue quantities."""

import math

import numpy as np

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
