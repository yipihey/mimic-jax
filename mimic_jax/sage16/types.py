"""Immutable PyTree representations of the complete fiducial SAGE16 state."""

from typing import Any, Dict, NamedTuple

import jax.numpy as jnp

from mimic_jax.sage16.precision import require_x64

Array = Any


class GalaxyState(NamedTuple):
    """All 32 fields from ``models/sage16/model_properties.yaml`` in upstream order."""

    HaloBaryonFraction: Array
    InfallingGas: Array
    CoolingGas: Array
    ColdGas: Array
    HotGas: Array
    EjectedGas: Array
    StellarMass: Array
    BulgeMass: Array
    ICS: Array
    NewStellarMass: Array
    StarFormationRate: Array
    MetalsStellarMass: Array
    MetalsBulgeMass: Array
    MetalsColdGas: Array
    MetalsHotGas: Array
    MetalsICS: Array
    MetalsEjectedGas: Array
    BlackHoleMass: Array
    QuasarModeBHaccretionMass: Array
    SupernovaReheatedMass: Array
    SupernovaEjectedMass: Array
    Cooling: Array
    Heating: Array
    Rcool: Array
    CoolingLambda: Array
    Rheat: Array
    SupernovaOutflowRate: Array
    DiskScaleRadius: Array
    MergTime: Array
    TimeOfLastMajorMerger: Array
    TimeOfLastMinorMerger: Array
    UnstableDiskGasFraction: Array


class HaloForcing(NamedTuple):
    """Instantaneous tree/halo quantities read by one or more SAGE16 processes."""

    Type: Array
    CentralHalo: Array
    HaloNr: Array
    SnapNum: Array
    Len: Array
    Mvir: Array
    deltaMvir: Array
    CentralMvir: Array
    Rvir: Array
    Vvir: Array
    infallMvir: Array
    infallVvir: Array
    infallVmax: Array
    Vmax: Array
    Spin: Array
    dT: Array


class Sage16Parameters(NamedTuple):
    """The complete parameter set in the shipped fiducial SAGE16 run."""

    GlobalBaryonFraction: Array
    SfrEfficiency: Array
    StarFormingDiskFactor: Array
    FeedbackReheatingEpsilon: Array
    FeedbackEjectionEfficiency: Array
    ReIncorporationFactor: Array
    AGNrecipe: Array
    RadioModeEfficiency: Array
    BlackHoleGrowthRate: Array
    QuasarModeEfficiency: Array
    RecycleFraction: Array
    Yield: Array
    FracZleaveDisk: Array
    ThresholdMajorMerger: Array
    ThresholdSatDisruption: Array


class Sage16Units(NamedTuple):
    """MIMIC's fixed internal reference units and derived constants."""

    Hubble_h: Array
    Omega: Array
    OmegaLambda: Array
    UnitLength_in_cm: Array
    UnitTime_in_s: Array
    UnitVelocity_in_cm_per_s: Array
    UnitMass_in_g: Array
    UnitDensity_in_cgs: Array
    UnitEnergy_in_cgs: Array
    G: Array
    Hubble: Array


class StepContext(NamedTuple):
    """Immutable subset of MIMIC's ``ModuleContext`` used by physics kernels."""

    redshift: Array
    time: Array
    snapshot_number: Array
    substep_number: Array
    num_substeps: Array
    time_interval: Array
    substep_time: Array
    substep_dt: Array


_GALAXY_FLOAT64_FIELDS = {
    "HaloBaryonFraction",
    "InfallingGas",
    "CoolingGas",
    "NewStellarMass",
    "SupernovaReheatedMass",
    "SupernovaEjectedMass",
    "Cooling",
    "Heating",
    "Rcool",
    "CoolingLambda",
    "UnstableDiskGasFraction",
}

_GALAXY_DEFAULTS = {
    "HaloBaryonFraction": -1.0,
    "InfallingGas": 0.0,
    "CoolingGas": 0.0,
    "ColdGas": 0.0,
    "HotGas": 0.0,
    "EjectedGas": 0.0,
    "StellarMass": 0.0,
    "BulgeMass": 0.0,
    "ICS": 0.0,
    "NewStellarMass": 0.0,
    "StarFormationRate": 0.0,
    "MetalsStellarMass": 0.0,
    "MetalsBulgeMass": 0.0,
    "MetalsColdGas": 0.0,
    "MetalsHotGas": 0.0,
    "MetalsICS": 0.0,
    "MetalsEjectedGas": 0.0,
    "BlackHoleMass": 0.0,
    "QuasarModeBHaccretionMass": 0.0,
    "SupernovaReheatedMass": 0.0,
    "SupernovaEjectedMass": 0.0,
    "Cooling": 0.0,
    "Heating": 0.0,
    "Rcool": 0.0,
    "CoolingLambda": 0.0,
    "Rheat": 0.0,
    "SupernovaOutflowRate": 0.0,
    "DiskScaleRadius": 0.0,
    "MergTime": 999.9,
    "TimeOfLastMajorMerger": 0.0,
    "TimeOfLastMinorMerger": 0.0,
    "UnstableDiskGasFraction": 0.0,
}


def initial_galaxy_state(**overrides: float) -> GalaxyState:
    """Construct the metadata-defined SAGE16 initial state with optional field overrides."""

    require_x64()
    unknown = set(overrides) - set(_GALAXY_DEFAULTS)
    if unknown:
        raise TypeError(f"Unknown SAGE16 galaxy fields: {sorted(unknown)}")

    values: Dict[str, Array] = {}
    for name, default in _GALAXY_DEFAULTS.items():
        dtype = jnp.float64 if name in _GALAXY_FLOAT64_FIELDS else jnp.float32
        values[name] = jnp.asarray(overrides.get(name, default), dtype=dtype)
    return GalaxyState(**values)


def initial_halo_forcing(**overrides: float) -> HaloForcing:
    """Construct a valid scalar halo forcing record for process and VMAP tests."""

    require_x64()
    defaults = {
        "Type": 0,
        "CentralHalo": 0,
        "HaloNr": 0,
        "SnapNum": 63,
        "Len": 1000,
        "Mvir": 100.0,
        "deltaMvir": 0.0,
        "CentralMvir": 100.0,
        "Rvir": 0.2,
        "Vvir": 200.0,
        "infallMvir": -1.0,
        "infallVvir": -1.0,
        "infallVmax": -1.0,
        "Vmax": 200.0,
        "Spin": (0.0, 0.0, 0.0),
        "dT": 0.01,
    }
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise TypeError(f"Unknown SAGE16 halo forcing fields: {sorted(unknown)}")
    defaults.update(overrides)

    return HaloForcing(
        Type=jnp.asarray(defaults["Type"], dtype=jnp.int32),
        CentralHalo=jnp.asarray(defaults["CentralHalo"], dtype=jnp.int32),
        HaloNr=jnp.asarray(defaults["HaloNr"], dtype=jnp.int32),
        SnapNum=jnp.asarray(defaults["SnapNum"], dtype=jnp.int32),
        Len=jnp.asarray(defaults["Len"], dtype=jnp.int32),
        Mvir=jnp.asarray(defaults["Mvir"], dtype=jnp.float64),
        deltaMvir=jnp.asarray(defaults["deltaMvir"], dtype=jnp.float64),
        CentralMvir=jnp.asarray(defaults["CentralMvir"], dtype=jnp.float64),
        Rvir=jnp.asarray(defaults["Rvir"], dtype=jnp.float64),
        Vvir=jnp.asarray(defaults["Vvir"], dtype=jnp.float64),
        infallMvir=jnp.asarray(defaults["infallMvir"], dtype=jnp.float64),
        infallVvir=jnp.asarray(defaults["infallVvir"], dtype=jnp.float64),
        infallVmax=jnp.asarray(defaults["infallVmax"], dtype=jnp.float64),
        Vmax=jnp.asarray(defaults["Vmax"], dtype=jnp.float32),
        Spin=jnp.asarray(defaults["Spin"], dtype=jnp.float32),
        dT=jnp.asarray(defaults["dT"], dtype=jnp.float64),
    )


def fiducial_parameters() -> Sage16Parameters:
    """Return values from ``sage16_mini-millennium.yaml`` as differentiable scalars."""

    require_x64()
    scalar = lambda value: jnp.asarray(value, dtype=jnp.float64)
    return Sage16Parameters(
        GlobalBaryonFraction=scalar(0.17),
        SfrEfficiency=scalar(0.05),
        StarFormingDiskFactor=scalar(3.0),
        FeedbackReheatingEpsilon=scalar(3.0),
        FeedbackEjectionEfficiency=scalar(0.3),
        ReIncorporationFactor=scalar(0.15),
        AGNrecipe=jnp.asarray(2, dtype=jnp.int32),
        RadioModeEfficiency=scalar(0.08),
        BlackHoleGrowthRate=scalar(0.015),
        QuasarModeEfficiency=scalar(0.005),
        RecycleFraction=scalar(0.43),
        Yield=scalar(0.025),
        FracZleaveDisk=scalar(0.0),
        ThresholdMajorMerger=scalar(0.3),
        ThresholdSatDisruption=scalar(1.0),
    )


def sage16_units(hubble_h: float = 0.73, omega: float = 0.25, omega_lambda: float = 0.75):
    """Reproduce ``set_units()`` for MIMIC's fixed internal reference basis."""

    require_x64()
    scalar = lambda value: jnp.asarray(value, dtype=jnp.float64)
    unit_length = scalar(3.08568e24)
    unit_mass = scalar(1.989e43)
    unit_velocity = scalar(1.0e5)
    unit_time = unit_length / unit_velocity
    gravity_cgs = scalar(6.672e-8)
    hubble_cgs = scalar(3.2407789e-18)
    return Sage16Units(
        Hubble_h=scalar(hubble_h),
        Omega=scalar(omega),
        OmegaLambda=scalar(omega_lambda),
        UnitLength_in_cm=unit_length,
        UnitTime_in_s=unit_time,
        UnitVelocity_in_cm_per_s=unit_velocity,
        UnitMass_in_g=unit_mass,
        UnitDensity_in_cgs=unit_mass / unit_length**3,
        UnitEnergy_in_cgs=unit_mass * unit_length**2 / unit_time**2,
        G=gravity_cgs / unit_length**3 * unit_mass * unit_time**2,
        Hubble=hubble_cgs * unit_time,
    )


def step_context(
    *,
    redshift: float = 0.0,
    time: float = 13.8,
    snapshot_number: int = 63,
    substep_number: int = 0,
    num_substeps: int = 1,
    time_interval: float = 0.01,
) -> StepContext:
    """Construct MIMIC-equivalent substep timing for a scalar halo evolution call."""

    require_x64()
    substep_dt = time_interval / num_substeps
    substep_time = (time + time_interval) - (substep_number + 0.5) * substep_dt
    return StepContext(
        redshift=jnp.asarray(redshift, dtype=jnp.float64),
        time=jnp.asarray(time, dtype=jnp.float64),
        snapshot_number=jnp.asarray(snapshot_number, dtype=jnp.int32),
        substep_number=jnp.asarray(substep_number, dtype=jnp.int32),
        num_substeps=jnp.asarray(num_substeps, dtype=jnp.int32),
        time_interval=jnp.asarray(time_interval, dtype=jnp.float64),
        substep_time=jnp.asarray(substep_time, dtype=jnp.float64),
        substep_dt=jnp.asarray(substep_dt, dtype=jnp.float64),
    )
