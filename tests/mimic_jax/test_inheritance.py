"""Tree-descendant inheritance and snapshot-reset parity contracts."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16 import (
    inherit_progenitor,
    inheritance_descendant,
    initial_galaxy_state,
    initial_halo_forcing,
    initialise_new_central,
    reset_snapshot_accumulators,
    set_local_central,
)


def _stack(*records):
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *records)


def _source_halo(**overrides):
    values = {
        "SnapNum": 4,
        "Type": 0,
        "CentralHalo": -1,
        "HaloNr": 7,
        "UniqueGalaxyID": 4444,
        "UniqueCentralGalaxyID": 3333,
        "dT": 1.0,
        "Len": 500,
        "Mvir": 100.0,
        "deltaMvir": 4.0,
        "Rvir": 1.0,
        "Vvir": 200.0,
        "infallMvir": -1.0,
        "infallVvir": -1.0,
        "infallVmax": -1.0,
        "Pos": (1.0, 2.0, 3.0),
        "Vel": (2.0, 3.0, 4.0),
        "Spin": (3.0, 4.0, 5.0),
        "Vmax": 210.0,
        "VelDisp": 90.0,
        "MostBoundID": 100,
    }
    values.update(overrides)
    return initial_halo_forcing(**values)


def _descendant(*, central=True, mass=150.0):
    payload = initial_halo_forcing(
        SnapNum=5,
        Len=1234,
        Mvir=mass,
        Rvir=1.5,
        Vvir=250.0,
        Pos=(10.0, 11.0, 12.0),
        Vel=(20.0, 21.0, 22.0),
        Spin=(0.1, 0.11, 0.12),
        Vmax=300.0,
        VelDisp=120.0,
        MostBoundID=987654321,
    )
    return inheritance_descendant(
        payload=payload,
        virial_mass=mass,
        is_fof_central=central,
    )


def test_snapshot_reset_covers_exactly_the_generated_repeat_fields():
    state = initial_galaxy_state(
        HaloBaryonFraction=0.17,
        ColdGas=4.0,
        Rheat=0.2,
        DiskScaleRadius=0.03,
        MergTime=2.0,
        TimeOfLastMajorMerger=5.0,
        InfallingGas=9.0,
        CoolingGas=8.0,
        NewStellarMass=7.0,
        StarFormationRate=6.0,
        QuasarModeBHaccretionMass=5.0,
        SupernovaReheatedMass=4.0,
        SupernovaEjectedMass=3.0,
        Cooling=2.0,
        Heating=1.0,
        Rcool=0.9,
        CoolingLambda=0.8,
        SupernovaOutflowRate=0.7,
        UnstableDiskGasFraction=0.6,
    )
    reset = reset_snapshot_accumulators(state)
    for name in (
        "InfallingGas",
        "CoolingGas",
        "NewStellarMass",
        "StarFormationRate",
        "QuasarModeBHaccretionMass",
        "SupernovaReheatedMass",
        "SupernovaEjectedMass",
        "Cooling",
        "Heating",
        "Rcool",
        "CoolingLambda",
        "SupernovaOutflowRate",
        "UnstableDiskGasFraction",
    ):
        np.testing.assert_array_equal(getattr(reset, name), 0.0)
    for name in (
        "HaloBaryonFraction",
        "ColdGas",
        "Rheat",
        "DiskScaleRadius",
        "MergTime",
        "TimeOfLastMajorMerger",
    ):
        np.testing.assert_array_equal(getattr(reset, name), getattr(state, name))


def test_main_branch_deep_copy_semantics_update_descendant_forcing():
    state = initial_galaxy_state(ColdGas=4.0, StarFormationRate=6.0, Rheat=0.2)
    result = inherit_progenitor(state, _source_halo(), _descendant(), 14.0, True)
    assert bool(result.retained)
    assert not bool(result.created)
    np.testing.assert_array_equal(result.state.ColdGas, np.float32(4.0))
    np.testing.assert_array_equal(result.state.StarFormationRate, np.float32(0.0))
    np.testing.assert_array_equal(result.state.Rheat, np.float32(0.2))
    np.testing.assert_array_equal(result.halo.Type, 0)
    np.testing.assert_array_equal(result.halo.SnapNum, 4)
    np.testing.assert_array_equal(result.halo.HaloNr, 42)
    np.testing.assert_array_equal(result.halo.dT, 4.0)
    np.testing.assert_array_equal(result.halo.Mvir, 150.0)
    np.testing.assert_array_equal(result.halo.deltaMvir, 50.0)
    np.testing.assert_array_equal(result.halo.Rvir, 1.5)
    np.testing.assert_array_equal(result.halo.Vvir, 250.0)
    np.testing.assert_array_equal(result.halo.Vmax, np.float32(300.0))
    np.testing.assert_array_equal(result.halo.MostBoundID, 987654321)


def test_type0_to_type1_and_orphan_transitions_capture_infall_state():
    state = initial_galaxy_state()
    satellite = inherit_progenitor(state, _source_halo(), _descendant(central=False), 14.0, True)
    np.testing.assert_array_equal(satellite.halo.Type, 1)
    np.testing.assert_array_equal(satellite.halo.infallMvir, 100.0)
    np.testing.assert_array_equal(satellite.halo.infallVvir, 200.0)
    np.testing.assert_array_equal(satellite.halo.infallVmax, 210.0)

    orphan = inherit_progenitor(state, _source_halo(), _descendant(central=False), 14.0, False)
    np.testing.assert_array_equal(orphan.halo.Type, 2)
    np.testing.assert_array_equal(orphan.halo.Mvir, 0.0)
    np.testing.assert_array_equal(orphan.halo.deltaMvir, -100.0)
    np.testing.assert_array_equal(orphan.halo.Len, 0)
    np.testing.assert_array_equal(orphan.halo.infallMvir, 100.0)


def test_type2_is_preserved_type3_is_discarded_and_smaller_descendant_freezes_virial_shape():
    state = initial_galaxy_state(ColdGas=2.0, StarFormationRate=5.0)
    type2 = inherit_progenitor(
        state,
        _source_halo(Type=2, Mvir=75.0, deltaMvir=-25.0, Len=0),
        _descendant(central=False),
        14.0,
        False,
    )
    assert bool(type2.retained)
    np.testing.assert_array_equal(type2.halo.Type, 2)
    np.testing.assert_array_equal(type2.halo.Mvir, 75.0)
    np.testing.assert_array_equal(type2.halo.deltaMvir, -25.0)

    type3 = inherit_progenitor(state, _source_halo(Type=3), _descendant(), 14.0, True)
    assert not bool(type3.retained)

    smaller = inherit_progenitor(state, _source_halo(), _descendant(mass=90.0), 14.0, True)
    np.testing.assert_array_equal(smaller.halo.Mvir, 90.0)
    np.testing.assert_array_equal(smaller.halo.deltaMvir, -10.0)
    np.testing.assert_array_equal(smaller.halo.Rvir, 1.0)
    np.testing.assert_array_equal(smaller.halo.Vvir, 200.0)


def test_new_central_uses_payload_defaults_and_previous_snapshot_sentinel():
    result = initialise_new_central(_descendant())
    assert bool(result.retained)
    assert bool(result.created)
    np.testing.assert_array_equal(result.halo.Type, 0)
    np.testing.assert_array_equal(result.halo.SnapNum, 4)
    np.testing.assert_array_equal(result.halo.HaloNr, 42)
    np.testing.assert_array_equal(result.halo.UniqueGalaxyID, 111000222)
    np.testing.assert_array_equal(result.halo.dT, 2.5)
    np.testing.assert_array_equal(result.state.MergTime, np.float32(999.9))
    np.testing.assert_array_equal(result.state.HaloBaryonFraction, -1.0)


def test_local_central_links_require_exactly_one_type0_or_type1():
    valid = set_local_central(
        _stack(
            _source_halo(Type=1),
            _source_halo(Type=2),
            _source_halo(Type=2),
        )
    )
    assert bool(valid.valid)
    np.testing.assert_array_equal(valid.central_index, 0)
    np.testing.assert_array_equal(valid.halos.CentralHalo, (0, 0, 0))

    invalid = set_local_central(_stack(_source_halo(Type=0), _source_halo(Type=1)))
    assert not bool(invalid.valid)
    np.testing.assert_array_equal(invalid.central_index, -1)


def test_inheritance_supports_jit_vmap_and_state_gradients():
    state = initial_galaxy_state(ColdGas=4.0, StarFormationRate=6.0)
    halo = _source_halo()
    descendant = _descendant()
    eager = inherit_progenitor(state, halo, descendant, 14.0, True)
    compiled = jax.jit(inherit_progenitor)(state, halo, descendant, 14.0, True)
    np.testing.assert_array_equal(compiled.halo.Mvir, eager.halo.Mvir)

    states = _stack(state, state)
    halos = _stack(halo, halo)
    descendants = _stack(descendant, descendant)
    batched = jax.vmap(inherit_progenitor)(
        states,
        halos,
        descendants,
        jnp.asarray((14.0, 14.0)),
        jnp.asarray((True, True)),
    )
    np.testing.assert_array_equal(batched.state.ColdGas, (4.0, 4.0))

    def inherited_cold(cold_gas):
        current = initial_galaxy_state(ColdGas=cold_gas)
        return inherit_progenitor(current, halo, descendant, 14.0, True).state.ColdGas

    derivative = jax.grad(inherited_cold)(jnp.asarray(4.0, dtype=jnp.float32))
    np.testing.assert_array_equal(derivative, np.float32(1.0))
