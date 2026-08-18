"""Real Mini-Millennium tree-to-catalogue SAGE16 equivalence regression."""

from pathlib import Path

import numpy as np

from mimic_jax.io import open_lhalo_partition
from mimic_jax.sage16 import (
    evolve_lhalo_tree,
    load_scale_factors,
    record_to_catalogue,
    snapshot_timing,
)

TEST_DATA = Path(__file__).parents[1] / "data" / "input"
SCALE_FACTORS = (
    Path(__file__).parents[2] / "simulations" / "mini-millennium" / "mini-millennium.a_list"
)


# Produced by upstream MIMIC at the repository commit represented by this
# checkout, using sage16_mini-millennium.yaml on partition 0, tree 1575.
UPSTREAM_Z0 = {
    "SnapNum": 63,
    "Type": 0,
    "UniqueGalaxyID": 1576000000005,
    "UniqueCentralGalaxyID": 1576000000005,
    "dT": 192.3052167940173,
    "Len": 21,
    "Mvir": 1.2049193382263184,
    "deltaMvir": 0.08606564998626709,
    "CentralMvir": 1.2049193382263184,
    "Rvir": 0.03728199218746071,
    "Vvir": 37.28201692835741,
    "infallMvir": 0.0,
    "infallVvir": 0.0,
    "infallVmax": 0.0,
    "Pos": (5.628704071044922, 24.563262939453125, 20.95482063293457),
    "Vel": (191.90882873535156, -5.847516059875488, 316.0148620605469),
    "VelDisp": 17.84486961364746,
    "Vmax": 38.51534652709961,
    "Spin": (-0.03654450178146362, 0.03398793935775757, 0.07384028285741806),
    "MostBoundID": 2126780,
    "HaloBaryonFraction": 0.03432487456499495,
    "ColdGas": 0.029922185465693474,
    "HotGas": 0.0,
    "EjectedGas": 0.011029256507754326,
    "StellarMass": 0.00040726797305978835,
    "BulgeMass": 0.0,
    "ICS": 0.0,
    "StarFormationRate": 0.029563724994659424,
    "MetalsStellarMass": 1.0718354559458021e-07,
    "MetalsBulgeMass": 0.0,
    "MetalsColdGas": 1.71914798556827e-05,
    "MetalsHotGas": -1.6045052158960874e-24,
    "MetalsICS": 0.0,
    "MetalsEjectedGas": 5.537054903470562e-07,
    "BlackHoleMass": 0.0,
    "QuasarModeBHaccretionMass": 0.0,
    "Cooling": 35.7174149180091,
    "Heating": 0.0,
    "SupernovaOutflowRate": 0.08869117498397827,
    "DiskScaleRadius": 0.0011956276139244437,
    "TimeOfLastMajorMerger": 0.0,
    "TimeOfLastMinorMerger": 0.0,
}


def test_linear_real_tree_matches_all_upstream_catalogue_fields_at_z0():
    tree = open_lhalo_partition(TEST_DATA / "trees_063.0").read_tree(1575)
    timing = snapshot_timing(load_scale_factors(SCALE_FACTORS))
    result = evolve_lhalo_tree(tree, timing, tree_index=1575)
    assert result.success
    assert result.groups_evolved == 6
    assert len(result.records_by_snapshot[63]) == 1
    actual = record_to_catalogue(result.records_by_snapshot[63][0], tree, timing)

    assert actual.keys() == UPSTREAM_Z0.keys()
    for name, expected in UPSTREAM_Z0.items():
        observed = np.asarray(actual[name])
        expected = np.asarray(expected)
        if observed.dtype.kind in "iu":
            np.testing.assert_array_equal(observed, expected, err_msg=name)
        elif observed.dtype.itemsize == 4:
            np.testing.assert_allclose(
                observed,
                expected,
                rtol=2.0e-6,
                atol=2.0e-6,
                err_msg=name,
            )
        else:
            np.testing.assert_allclose(
                observed,
                expected,
                rtol=2.0e-12,
                atol=2.0e-12,
                err_msg=name,
            )


def test_branched_real_tree_matches_upstream_merger_and_group_outcome_at_z0():
    tree = open_lhalo_partition(TEST_DATA / "trees_063.0").read_tree(61)
    timing = snapshot_timing(load_scale_factors(SCALE_FACTORS))
    result = evolve_lhalo_tree(tree, timing, tree_index=61)
    assert result.success
    assert result.groups_evolved == 64
    assert len(result.records_by_snapshot[63]) == 1
    actual = record_to_catalogue(result.records_by_snapshot[63][0], tree, timing)

    expected = {
        "UniqueGalaxyID": 62000000052,
        "Type": 0,
        "Mvir": 166.9674072265625,
        "ColdGas": 0.2642709016799927,
        "HotGas": 10.761331558227539,
        "EjectedGas": 0.0,
        "StellarMass": 16.464038848876953,
        "BulgeMass": 9.11240291595459,
        "ICS": 0.4570813775062561,
        "MetalsStellarMass": 0.41766661405563354,
        "MetalsBulgeMass": 0.1965985745191574,
        "MetalsColdGas": 0.01184783037751913,
        "MetalsHotGas": 0.28210997581481934,
        "MetalsICS": 0.005123186390846968,
        "BlackHoleMass": 0.009141568094491959,
    }
    for name, reference in expected.items():
        observed = np.asarray(actual[name])
        reference = np.asarray(reference)
        if reference.dtype.kind in "iu":
            np.testing.assert_array_equal(observed, reference, err_msg=name)
        else:
            np.testing.assert_allclose(
                observed,
                reference,
                rtol=2.0e-6,
                atol=2.0e-6,
                err_msg=name,
            )
