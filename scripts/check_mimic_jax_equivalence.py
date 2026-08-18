#!/usr/bin/env python3
"""Compile SAGE16 C reference cases and compare their outputs with mimic-jax."""

import argparse
import math
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mimic_jax.sage16 import (  # noqa: E402
    apply_collisional_starburst,
    apply_cooling,
    apply_disk_instability,
    apply_disk_instability_starburst,
    apply_infall,
    apply_metal_enrichment,
    apply_quasar_mode,
    apply_radio_mode_heating,
    apply_reincorporation,
    apply_reionization,
    apply_satellite_stripping,
    apply_star_formation_supernova,
    calculate_cooling_budget,
    calculate_star_formation_budget,
    calculate_supernova_feedback_budget,
    fiducial_parameters,
    inherit_progenitor,
    inheritance_descendant,
    initial_galaxy_state,
    initial_halo_forcing,
    initialise_merger_clocks,
    load_cooling_tables,
    metal_dependent_cooling_rate,
    prepare_infall_budget,
    process_perturbations,
    resolve_mergers_and_disruption,
    sage16_units,
    set_disk_scale_radius,
    step_context,
)

REFERENCE_PREFIX = "MIMIC_JAX_REFERENCE "
REFERENCE_TEST = "models/sage16/modules/_tests/test_unit_mimic_jax_reference.c"
REFERENCE_LOG = "tests/unit/build/test_unit_mimic_jax_reference.run.log"
CASE_TOLERANCES = {
    "cooling_budget": (1.0e-13, 0.0),
    "infall_budget": (1.0e-15, 0.0),
    "radio_mode": (1.0e-13, 0.0),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-c-log",
        action="store_true",
        help="Reuse the last compiled C test log instead of rebuilding the reference executable",
    )
    return parser.parse_args()


def run_c_reference(repository: Path) -> None:
    environment = os.environ.copy()
    environment["TEST_SUMMARY"] = "1"
    subprocess.run(
        ["tests/unit/run_tests.sh", REFERENCE_TEST],
        cwd=repository,
        env=environment,
        check=True,
    )


def parse_c_reference(path: Path):
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        marker = line.find(REFERENCE_PREFIX)
        if marker < 0:
            continue
        tokens = line[marker + len(REFERENCE_PREFIX) :].split()
        values = dict(token.split("=", 1) for token in tokens)
        case = values.pop("case")
        records[case] = {name: float(value) for name, value in values.items()}
    required = {
        "cooling",
        "cooling_budget",
        "cooling_interpolation",
        "disk_instability",
        "disk_radius",
        "infall_budget",
        "infall_negative",
        "infall_positive",
        "inheritance",
        "merger_clock",
        "merger_event",
        "reincorporation",
        "radio_mode",
        "reionization",
        "satellite_stripping",
        "post_quiescent_chain",
        "quasar_mode",
        "starburst",
        "star_formation_budget",
        "star_formation_final",
    }
    missing = required - set(records)
    if missing:
        raise RuntimeError(f"Missing C reference records: {sorted(missing)}")
    return records


def select(record, names):
    return {name: float(getattr(record, name)) for name in names}


def calculate_jax_reference():
    parameters = fiducial_parameters()
    units = sage16_units()
    cooling_tables = load_cooling_tables()

    log_z_sun = math.log10(0.02)
    interpolation = {
        "midpoint": float(metal_dependent_cooling_rate(5.025, log_z_sun - 0.75, cooling_tables)),
        "low_temperature": float(metal_dependent_cooling_rate(3.0, log_z_sun, cooling_tables)),
        "high_temperature": float(metal_dependent_cooling_rate(9.0, log_z_sun, cooling_tables)),
        "primordial": float(metal_dependent_cooling_rate(5.5, -10.0, cooling_tables)),
    }

    reionization_state = apply_reionization(
        initial_galaxy_state(),
        initial_halo_forcing(Mvir=1.0, Rvir=0.1, Vvir=100.0, dT=0.01),
        step_context(redshift=2.0, num_substeps=10, time_interval=0.01),
        parameters,
        units,
    ).state

    disk_radius = set_disk_scale_radius(
        initial_galaxy_state(DiskScaleRadius=0.123),
        initial_halo_forcing(Type=0, Spin=(100.0, 150.0, 200.0), Vvir=200.0, Rvir=0.2),
    ).state

    clock_states = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values),
        initial_galaxy_state(MergTime=5.0),
        initial_galaxy_state(MergTime=999.9, StellarMass=5.0, ColdGas=2.0),
        initial_galaxy_state(MergTime=999.9, StellarMass=3.0, ColdGas=1.0),
        initial_galaxy_state(MergTime=999.9),
    )
    clock_halos = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values),
        initial_halo_forcing(Type=0, Len=1000, Mvir=100.0, Rvir=0.5, Vvir=200.0),
        initial_halo_forcing(Type=1, Len=200, Mvir=20.0, Rvir=0.2, Vvir=100.0),
        initial_halo_forcing(
            Type=2,
            Len=0,
            Mvir=5.0,
            Rvir=0.1,
            Vvir=50.0,
            CentralHalo=1,
        ),
        initial_halo_forcing(Type=3, Len=200, Mvir=20.0, Rvir=0.2, Vvir=100.0),
    )
    merger_clock = initialise_merger_clocks(clock_states, clock_halos, units).states

    event_states = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values),
        initial_galaxy_state(
            MergTime=999.9,
            ColdGas=10.0,
            HotGas=5.0,
            EjectedGas=1.0,
            StellarMass=10.0,
            BulgeMass=2.0,
            ICS=0.5,
            BlackHoleMass=0.01,
            MetalsColdGas=0.2,
            MetalsHotGas=0.1,
            MetalsEjectedGas=0.02,
            MetalsStellarMass=0.2,
            MetalsBulgeMass=0.04,
            MetalsICS=0.01,
            DiskScaleRadius=0.0001,
            TimeOfLastMajorMerger=-1.0,
            TimeOfLastMinorMerger=-1.0,
        ),
        initial_galaxy_state(
            MergTime=0.4,
            ColdGas=1.0,
            HotGas=1.0,
            EjectedGas=0.2,
            StellarMass=1.0,
            BulgeMass=0.2,
            ICS=0.1,
            BlackHoleMass=0.03,
            MetalsColdGas=0.02,
            MetalsHotGas=0.02,
            MetalsEjectedGas=0.004,
            MetalsStellarMass=0.02,
            MetalsBulgeMass=0.004,
            MetalsICS=0.002,
        ),
        initial_galaxy_state(
            MergTime=-0.1,
            ColdGas=1.0,
            HotGas=0.5,
            EjectedGas=0.1,
            StellarMass=2.0,
            BulgeMass=0.3,
            ICS=0.05,
            BlackHoleMass=0.02,
            MetalsColdGas=0.02,
            MetalsHotGas=0.01,
            MetalsEjectedGas=0.002,
            MetalsStellarMass=0.04,
            MetalsBulgeMass=0.006,
            MetalsICS=0.001,
        ),
    )
    event_halos = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values),
        initial_halo_forcing(
            Type=0,
            Mvir=100.0,
            Rvir=0.2,
            Vvir=300.0,
            Vmax=300.0,
            dT=0.1,
            Len=1000,
        ),
        initial_halo_forcing(
            Type=1,
            CentralHalo=0,
            Mvir=2.0,
            Rvir=0.1,
            Vvir=100.0,
            Vmax=100.0,
            dT=0.1,
            Len=20,
        ),
        initial_halo_forcing(
            Type=1,
            CentralHalo=0,
            Mvir=1.0,
            Rvir=0.1,
            Vvir=100.0,
            Vmax=100.0,
            dT=0.1,
            Len=20,
        ),
    )
    merger_event = resolve_mergers_and_disruption(
        event_states,
        event_halos,
        step_context(time=13.8, time_interval=0.1),
        parameters,
        units,
    )

    inheritance_source_state = initial_galaxy_state(
        ColdGas=4.0,
        StarFormationRate=6.0,
        Rheat=0.2,
        MergTime=2.0,
        InfallingGas=9.0,
    )
    inheritance_source_halo = initial_halo_forcing(
        SnapNum=4,
        Type=0,
        CentralHalo=-1,
        HaloNr=7,
        UniqueGalaxyID=4444,
        UniqueCentralGalaxyID=3333,
        dT=1.0,
        Len=500,
        Mvir=100.0,
        deltaMvir=4.0,
        Rvir=1.0,
        Vvir=200.0,
        infallMvir=-1.0,
        infallVvir=-1.0,
        infallVmax=-1.0,
        Pos=(1.0, 2.0, 3.0),
        Vel=(2.0, 3.0, 4.0),
        Spin=(3.0, 4.0, 5.0),
        Vmax=210.0,
        VelDisp=90.0,
        MostBoundID=100,
    )
    inheritance_payload = initial_halo_forcing(
        SnapNum=5,
        Len=1234,
        Mvir=150.0,
        Rvir=1.5,
        Vvir=250.0,
        Pos=(10.0, 11.0, 12.0),
        Vel=(20.0, 21.0, 22.0),
        Spin=(0.1, 0.11, 0.12),
        Vmax=300.0,
        VelDisp=120.0,
        MostBoundID=987654321,
    )
    inherited = inherit_progenitor(
        inheritance_source_state,
        inheritance_source_halo,
        inheritance_descendant(payload=inheritance_payload),
        14.0,
        True,
    )
    inherited_halo = inherited.halo._replace(CentralHalo=jnp.asarray(0, dtype=jnp.int32))

    central = initial_galaxy_state(
        HaloBaryonFraction=0.17,
        StellarMass=5.0,
        ColdGas=3.0,
        HotGas=8.0,
        EjectedGas=1.0,
        ICS=0.5,
        BlackHoleMass=0.1,
        MetalsEjectedGas=0.02,
        MetalsICS=0.01,
    )
    satellite = initial_galaxy_state(
        HaloBaryonFraction=0.17,
        HotGas=3.0,
        EjectedGas=2.0,
        ICS=1.5,
        MetalsHotGas=0.06,
        MetalsEjectedGas=0.04,
        MetalsICS=0.03,
    )
    infall_states = jax.tree_util.tree_map(lambda *values: jnp.stack(values), central, satellite)
    infall_halos = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values),
        initial_halo_forcing(Type=0, Mvir=100.0, Rvir=0.2, Vvir=200.0, dT=0.01),
        initial_halo_forcing(Type=2, Mvir=0.0, Rvir=0.1, Vvir=100.0, dT=0.01),
    )
    infall_budget = prepare_infall_budget(infall_states, infall_halos, 0, parameters).states
    positive_infall = apply_infall(
        initial_galaxy_state(InfallingGas=12.0, HotGas=5.0),
        step_context(num_substeps=4),
    ).state
    negative_infall = apply_infall(
        initial_galaxy_state(
            InfallingGas=-8.0,
            EjectedGas=3.0,
            MetalsEjectedGas=0.06,
            HotGas=10.0,
            MetalsHotGas=0.2,
        ),
        step_context(),
    ).state

    stripping = apply_satellite_stripping(
        initial_galaxy_state(
            HaloBaryonFraction=0.17,
            StellarMass=0.4,
            ColdGas=0.3,
            HotGas=5.0,
            EjectedGas=0.2,
            BlackHoleMass=0.05,
            ICS=0.1,
            MetalsHotGas=0.1,
        ),
        initial_galaxy_state(HotGas=100.0, MetalsHotGas=2.0),
        initial_halo_forcing(Type=1, Mvir=10.0),
        step_context(num_substeps=10),
        parameters,
    )

    cooling_budget_state = initial_galaxy_state(HotGas=8.0, MetalsHotGas=0.16)
    cooling_budget_halo = initial_halo_forcing(Rvir=0.2, Vvir=200.0, dT=0.01)
    cooling_budget = calculate_cooling_budget(
        cooling_budget_state,
        cooling_budget_halo,
        step_context(num_substeps=10, time_interval=0.01),
        units,
        cooling_tables,
    ).state

    radio_initial = initial_galaxy_state(
        HotGas=8.0,
        MetalsHotGas=0.16,
        BlackHoleMass=0.01,
        Rheat=0.01,
    )
    radio_halo = initial_halo_forcing(Rvir=0.2, Vvir=200.0, dT=0.01)
    radio_budget = calculate_cooling_budget(
        radio_initial,
        radio_halo,
        step_context(num_substeps=10, time_interval=0.01),
        units,
        cooling_tables,
    ).state
    radio_mode = apply_radio_mode_heating(
        radio_budget,
        radio_halo,
        step_context(num_substeps=10, time_interval=0.01),
        parameters,
        units,
    ).state

    cooling_state = initial_galaxy_state(
        ColdGas=2.0,
        HotGas=8.0,
        MetalsColdGas=0.04,
        MetalsHotGas=0.16,
        CoolingGas=1.5,
    )
    cooling_halo = initial_halo_forcing(Vvir=200.0, dT=0.01)
    cooled = apply_cooling(cooling_state, cooling_halo).state

    reincorporation_state = initial_galaxy_state(
        HotGas=2.0,
        EjectedGas=4.0,
        MetalsHotGas=0.04,
        MetalsEjectedGas=0.08,
    )
    reincorporation_halo = initial_halo_forcing(
        Vvir=100.0,
        Rvir=0.2,
        dT=0.01,
    )
    reincorporated = apply_reincorporation(
        reincorporation_state,
        reincorporation_halo,
        step_context(num_substeps=10, time_interval=0.01),
        parameters,
    ).state

    disk_instability_state = initial_galaxy_state(
        ColdGas=5.0,
        StellarMass=10.0,
        BulgeMass=2.0,
        MetalsStellarMass=0.2,
        MetalsBulgeMass=0.04,
        DiskScaleRadius=0.003,
    )
    disk_instability_halo = initial_halo_forcing(
        Mvir=100.0,
        Rvir=0.2,
        Vvir=200.0,
        Vmax=200.0,
        dT=0.1,
    )
    disk_instability = apply_disk_instability(
        disk_instability_state,
        disk_instability_halo,
        parameters,
        units,
    )

    quasar_state = initial_galaxy_state(
        ColdGas=10.0,
        HotGas=5.0,
        EjectedGas=1.0,
        MetalsColdGas=0.2,
        MetalsHotGas=0.1,
        MetalsEjectedGas=0.02,
        BlackHoleMass=0.01,
        UnstableDiskGasFraction=0.5,
    )
    quasar_halo = initial_halo_forcing(Mvir=100.0, Rvir=0.2, Vvir=300.0, dT=0.1)
    quasar = apply_quasar_mode(quasar_state, quasar_halo, parameters, units)

    starburst_state = initial_galaxy_state(
        ColdGas=10.0,
        HotGas=5.0,
        EjectedGas=1.0,
        StellarMass=5.0,
        BulgeMass=1.0,
        MetalsColdGas=0.2,
        MetalsHotGas=0.1,
        MetalsEjectedGas=0.02,
        MetalsStellarMass=0.1,
        MetalsBulgeMass=0.02,
        UnstableDiskGasFraction=0.2,
    )
    starburst_halo = initial_halo_forcing(Mvir=100.0, Rvir=0.2, Vvir=300.0, dT=0.1)
    starburst = apply_collisional_starburst(
        starburst_state,
        starburst_state,
        starburst_halo,
        starburst_halo,
        starburst_state.UnstableDiskGasFraction,
        1,
        starburst_halo.dT,
        parameters,
        units,
    )

    star_formation_state = initial_galaxy_state(
        ColdGas=10.0,
        HotGas=5.0,
        EjectedGas=1.0,
        StellarMass=2.0,
        MetalsColdGas=0.2,
        MetalsHotGas=0.1,
        MetalsEjectedGas=0.01,
        MetalsStellarMass=0.04,
        DiskScaleRadius=0.01,
    )
    star_formation_halo = initial_halo_forcing(
        Mvir=100.0,
        Rvir=0.2,
        Vvir=150.0,
        dT=0.0001,
    )
    context = step_context(time_interval=0.0001)
    star_formation = calculate_star_formation_budget(
        star_formation_state,
        star_formation_halo,
        context,
        parameters,
    )
    budget = calculate_supernova_feedback_budget(
        star_formation_state,
        star_formation_halo,
        parameters,
        units,
        star_formation,
    )
    applied = apply_star_formation_supernova(
        star_formation_state,
        star_formation_state,
        star_formation_halo,
        parameters,
        budget,
    )
    enriched = apply_metal_enrichment(
        applied.galaxy,
        applied.central,
        star_formation_halo,
        True,
        parameters,
    ).galaxy

    composed_state = initial_galaxy_state(
        ColdGas=10.0,
        HotGas=5.0,
        EjectedGas=1.0,
        StellarMass=2.0,
        BulgeMass=0.5,
        MetalsColdGas=0.2,
        MetalsHotGas=0.1,
        MetalsEjectedGas=0.01,
        MetalsStellarMass=0.04,
        MetalsBulgeMass=0.01,
        DiskScaleRadius=0.0001,
    )
    composed_halo = initial_halo_forcing(
        Mvir=100.0,
        Rvir=0.2,
        Vvir=150.0,
        Vmax=150.0,
        dT=0.0001,
    )
    composed_context = step_context(time_interval=0.0001)
    composed_sf = calculate_star_formation_budget(
        composed_state,
        composed_halo,
        composed_context,
        parameters,
    )
    composed_sn = calculate_supernova_feedback_budget(
        composed_state,
        composed_halo,
        parameters,
        units,
        composed_sf,
    )
    composed_applied = apply_star_formation_supernova(
        composed_state,
        composed_state,
        composed_halo,
        parameters,
        composed_sn,
    )
    composed_instability = apply_disk_instability(
        composed_applied.galaxy,
        composed_halo,
        parameters,
        units,
    )
    composed_quasar = apply_quasar_mode(
        composed_instability.state,
        composed_halo,
        parameters,
        units,
    )
    composed_starburst = apply_disk_instability_starburst(
        composed_quasar.state,
        composed_quasar.state,
        composed_halo,
        composed_halo,
        parameters,
        units,
        process_perturbations(),
    )
    composed = apply_metal_enrichment(
        composed_starburst.galaxy,
        composed_starburst.central,
        composed_halo,
        True,
        parameters,
    ).galaxy

    return {
        "reionization": select(reionization_state, ("HaloBaryonFraction",)),
        "disk_radius": select(disk_radius, ("DiskScaleRadius",)),
        "merger_clock": {
            "CentralMergTime": float(merger_clock.MergTime[0]),
            "SatelliteMergTime": float(merger_clock.MergTime[1]),
            "OrphanMergTime": float(merger_clock.MergTime[2]),
            "Type3MergTime": float(merger_clock.MergTime[3]),
        },
        "merger_event": {
            "CentralColdGas": float(merger_event.states.ColdGas[0]),
            "CentralHotGas": float(merger_event.states.HotGas[0]),
            "CentralEjectedGas": float(merger_event.states.EjectedGas[0]),
            "CentralStellarMass": float(merger_event.states.StellarMass[0]),
            "CentralBulgeMass": float(merger_event.states.BulgeMass[0]),
            "CentralICS": float(merger_event.states.ICS[0]),
            "CentralBlackHoleMass": float(merger_event.states.BlackHoleMass[0]),
            "CentralMetalsColdGas": float(merger_event.states.MetalsColdGas[0]),
            "CentralMetalsHotGas": float(merger_event.states.MetalsHotGas[0]),
            "CentralMetalsEjectedGas": float(merger_event.states.MetalsEjectedGas[0]),
            "CentralMetalsStellarMass": float(merger_event.states.MetalsStellarMass[0]),
            "CentralMetalsBulgeMass": float(merger_event.states.MetalsBulgeMass[0]),
            "CentralMetalsICS": float(merger_event.states.MetalsICS[0]),
            "CentralQuasarAccretion": float(merger_event.states.QuasarModeBHaccretionMass[0]),
            "CentralSFR": float(merger_event.states.StarFormationRate[0]),
            "CentralOutflowRate": float(merger_event.states.SupernovaOutflowRate[0]),
            "CentralUnstableFraction": float(merger_event.states.UnstableDiskGasFraction[0]),
            "LastMinor": float(merger_event.states.TimeOfLastMinorMerger[0]),
            "LastMajor": float(merger_event.states.TimeOfLastMajorMerger[0]),
            "DisruptedType": float(merger_event.halos.Type[1]),
            "MergedType": float(merger_event.halos.Type[2]),
            "DisruptedClock": float(merger_event.states.MergTime[1]),
            "MergedClock": float(merger_event.states.MergTime[2]),
        },
        "inheritance": {
            "ColdGas": float(inherited.state.ColdGas),
            "StarFormationRate": float(inherited.state.StarFormationRate),
            "Rheat": float(inherited.state.Rheat),
            "MergTime": float(inherited.state.MergTime),
            "InfallingGas": float(inherited.state.InfallingGas),
            "Type": float(inherited_halo.Type),
            "SnapNum": float(inherited_halo.SnapNum),
            "HaloNr": float(inherited_halo.HaloNr),
            "CentralHalo": float(inherited_halo.CentralHalo),
            "dT": float(inherited_halo.dT),
            "Len": float(inherited_halo.Len),
            "Mvir": float(inherited_halo.Mvir),
            "deltaMvir": float(inherited_halo.deltaMvir),
            "Rvir": float(inherited_halo.Rvir),
            "Vvir": float(inherited_halo.Vvir),
            "infallMvir": float(inherited_halo.infallMvir),
            "infallVvir": float(inherited_halo.infallVvir),
            "infallVmax": float(inherited_halo.infallVmax),
            "Vmax": float(inherited_halo.Vmax),
            "Pos0": float(inherited_halo.Pos[0]),
            "Spin2": float(inherited_halo.Spin[2]),
            "MostBoundID": float(inherited_halo.MostBoundID),
        },
        "infall_budget": {
            "InfallingGas": float(infall_budget.InfallingGas[0]),
            "EjectedGas": float(infall_budget.EjectedGas[0]),
            "MetalsEjectedGas": float(infall_budget.MetalsEjectedGas[0]),
            "ICS": float(infall_budget.ICS[0]),
            "MetalsICS": float(infall_budget.MetalsICS[0]),
            "SatelliteEjectedGas": float(infall_budget.EjectedGas[1]),
            "SatelliteICS": float(infall_budget.ICS[1]),
            "SatelliteHotGas": float(infall_budget.HotGas[1]),
        },
        "infall_positive": select(positive_infall, ("HotGas",)),
        "infall_negative": select(
            negative_infall,
            ("EjectedGas", "MetalsEjectedGas", "HotGas", "MetalsHotGas"),
        ),
        "satellite_stripping": {
            "SatelliteHotGas": float(stripping.satellite.HotGas),
            "SatelliteMetalsHotGas": float(stripping.satellite.MetalsHotGas),
            "CentralHotGas": float(stripping.central.HotGas),
            "CentralMetalsHotGas": float(stripping.central.MetalsHotGas),
        },
        "cooling_interpolation": interpolation,
        "cooling_budget": select(
            cooling_budget,
            ("CoolingGas", "Rcool", "CoolingLambda"),
        ),
        "radio_mode": {
            "CoolingGasBefore": float(radio_budget.CoolingGas),
            **select(
                radio_mode,
                (
                    "CoolingGas",
                    "BlackHoleMass",
                    "HotGas",
                    "MetalsHotGas",
                    "Rheat",
                    "Heating",
                ),
            ),
        },
        "cooling": select(
            cooled,
            ("ColdGas", "HotGas", "MetalsColdGas", "MetalsHotGas", "Cooling"),
        ),
        "reincorporation": select(
            reincorporated,
            ("HotGas", "EjectedGas", "MetalsHotGas", "MetalsEjectedGas"),
        ),
        "disk_instability": select(
            disk_instability.state,
            ("BulgeMass", "MetalsBulgeMass", "UnstableDiskGasFraction"),
        ),
        "quasar_mode": select(
            quasar.state,
            (
                "ColdGas",
                "HotGas",
                "EjectedGas",
                "MetalsColdGas",
                "MetalsHotGas",
                "MetalsEjectedGas",
                "BlackHoleMass",
                "QuasarModeBHaccretionMass",
                "UnstableDiskGasFraction",
            ),
        ),
        "starburst": select(
            starburst.galaxy,
            (
                "ColdGas",
                "HotGas",
                "EjectedGas",
                "StellarMass",
                "BulgeMass",
                "MetalsColdGas",
                "MetalsHotGas",
                "MetalsEjectedGas",
                "MetalsStellarMass",
                "MetalsBulgeMass",
                "StarFormationRate",
                "SupernovaOutflowRate",
                "UnstableDiskGasFraction",
            ),
        ),
        "star_formation_budget": select(
            budget,
            ("NewStellarMass", "SupernovaReheatedMass", "SupernovaEjectedMass"),
        ),
        "star_formation_final": select(
            enriched,
            (
                "ColdGas",
                "HotGas",
                "EjectedGas",
                "StellarMass",
                "MetalsColdGas",
                "MetalsHotGas",
                "MetalsEjectedGas",
                "MetalsStellarMass",
                "StarFormationRate",
                "SupernovaOutflowRate",
                "NewStellarMass",
            ),
        ),
        "post_quiescent_chain": select(
            composed,
            (
                "ColdGas",
                "HotGas",
                "EjectedGas",
                "StellarMass",
                "BulgeMass",
                "BlackHoleMass",
                "MetalsColdGas",
                "MetalsHotGas",
                "MetalsEjectedGas",
                "MetalsStellarMass",
                "MetalsBulgeMass",
                "NewStellarMass",
                "UnstableDiskGasFraction",
                "StarFormationRate",
                "SupernovaOutflowRate",
                "QuasarModeBHaccretionMass",
            ),
        ),
    }


def compare_records(c_reference, jax_reference) -> None:
    discrepancies = []
    compared = 0
    exact = 0
    for case, c_values in c_reference.items():
        jax_values = jax_reference[case]
        if set(c_values) != set(jax_values):
            discrepancies.append(
                f"{case}: field mismatch C={sorted(c_values)} JAX={sorted(jax_values)}"
            )
            continue
        for name, c_value in c_values.items():
            compared += 1
            jax_value = jax_values[name]
            relative_tolerance, absolute_tolerance = CASE_TOLERANCES.get(case, (0.0, 0.0))
            if jax_value == c_value:
                exact += 1
            elif not math.isclose(
                jax_value,
                c_value,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            ):
                discrepancies.append(
                    f"{case}.{name}: C={c_value:.17g}, JAX={jax_value:.17g}, "
                    f"abs_diff={abs(jax_value - c_value):.3e}, "
                    f"rtol={relative_tolerance:.1e}, atol={absolute_tolerance:.1e}"
                )
    if discrepancies:
        raise AssertionError("C/JAX equivalence failed:\n" + "\n".join(discrepancies))
    print(f"MIMIC-JAX equivalence: {compared} fields match compiled SAGE16 " f"({exact} exactly)")
    print(
        "Tolerances: cooling budget and radio mode rtol=1e-13; "
        "group infall budget rtol=1e-15; every other controlled field exact"
    )


def main() -> int:
    arguments = parse_arguments()
    repository = Path(__file__).resolve().parents[1]
    if not arguments.reuse_c_log:
        run_c_reference(repository)
    c_reference = parse_c_reference(repository / REFERENCE_LOG)
    compare_records(c_reference, calculate_jax_reference())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
