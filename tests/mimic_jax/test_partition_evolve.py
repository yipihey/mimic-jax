"""Batched multi-tree Mini-Millennium execution regression."""

from pathlib import Path

import numpy as np
import pytest

from mimic_jax.io import open_lhalo_partition
from mimic_jax.sage16 import (
    evolve_lhalo_partition,
    fiducial_parameters,
    linearize_lhalo_partition,
    load_scale_factors,
    process_perturbations,
    record_to_catalogue,
    snapshot_timing,
)

ROOT = Path(__file__).parents[2]


def test_partition_batch_controls_are_validated_before_evolution():
    partition = open_lhalo_partition(Path(__file__).parents[1] / "data/input/trees_063.0")
    timing = snapshot_timing(
        load_scale_factors(ROOT / "simulations/mini-millennium/mini-millennium.a_list")
    )
    with pytest.raises(ValueError, match="member_binning"):
        evolve_lhalo_partition(
            partition,
            timing,
            tree_indices=(1575,),
            member_binning="arbitrary",
        )
    with pytest.raises(ValueError, match="max_batch_members"):
        evolve_lhalo_partition(
            partition,
            timing,
            tree_indices=(1575,),
            max_batch_members=0,
        )


def test_two_linear_trees_batch_to_upstream_z0_catalogue_values():
    partition = open_lhalo_partition(Path(__file__).parents[1] / "data/input/trees_063.0")
    timing = snapshot_timing(
        load_scale_factors(ROOT / "simulations/mini-millennium/mini-millennium.a_list")
    )
    tree_indices = (1170, 1575)
    result = evolve_lhalo_partition(
        partition,
        timing,
        tree_indices=tree_indices,
        output_snapshots=(63,),
        batch_size=2,
        member_binning="power_of_two",
    )

    assert result.success
    assert result.groups_evolved == 14
    assert len(result.records_by_snapshot[63]) == 2
    expected = {
        1170: {
            "UniqueGalaxyID": 1171000000007,
            "Mvir": 2.323773145675659,
            "ColdGas": 0.09874552488327026,
            "HotGas": 0.05546988174319267,
            "EjectedGas": 0.0,
            "StellarMass": 0.0009983601048588753,
            "StarFormationRate": 0.0,
            "BlackHoleMass": 0.0,
        },
        1575: {
            "UniqueGalaxyID": 1576000000005,
            "Mvir": 1.2049193382263184,
            "ColdGas": 0.029922185465693474,
            "HotGas": 0.0,
            "EjectedGas": 0.011029256507754326,
            "StellarMass": 0.00040726797305978835,
            "StarFormationRate": 0.029563724994659424,
            "BlackHoleMass": 0.0,
        },
    }
    for tree_index, records in zip(result.tree_indices, result.records_by_tree):
        assert len(records[63]) == 1
        tree = partition.read_tree(tree_index)
        actual = record_to_catalogue(records[63][0], tree, timing)
        for name, reference in expected[tree_index].items():
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


def test_tree_parameter_tangent_matches_a_symmetric_full_rerun():
    partition = open_lhalo_partition(Path(__file__).parents[1] / "data/input/trees_063.0")
    timing = snapshot_timing(
        load_scale_factors(ROOT / "simulations/mini-millennium/mini-millennium.a_list")
    )
    parameter_name = "SfrEfficiency"
    parameters = fiducial_parameters()
    step = 1.0e-3 * float(parameters.SfrEfficiency)

    linearized = linearize_lhalo_partition(
        partition,
        timing,
        tree_indices=(1575,),
        parameter_names=(parameter_name,),
        output_snapshots=(63,),
        batch_size=2,
        member_binning="power_of_two",
    )
    automatic = float(linearized.records_by_snapshot[63][0].state_tangent.StellarMass[0])
    perturbed = []
    for direction in (-1.0, 1.0):
        current = parameters._replace(SfrEfficiency=parameters.SfrEfficiency + direction * step)
        result = evolve_lhalo_partition(
            partition,
            timing,
            tree_indices=(1575,),
            parameters=current,
            output_snapshots=(63,),
            batch_size=2,
            member_binning="power_of_two",
        )
        perturbed.append(float(result.records_by_snapshot[63][0].state.StellarMass))
    finite_difference = (perturbed[1] - perturbed[0]) / (2.0 * step)

    assert linearized.success
    assert linearized.control_names == (parameter_name,)
    np.testing.assert_allclose(automatic, finite_difference, rtol=6.0e-4, atol=1.0e-7)


def test_tree_process_tangent_matches_a_symmetric_finite_epoch_rerun():
    partition = open_lhalo_partition(Path(__file__).parents[1] / "data/input/trees_063.0")
    timing = snapshot_timing(
        load_scale_factors(ROOT / "simulations/mini-millennium/mini-millennium.a_list")
    )
    edges = np.asarray(
        [np.log(timing.scale_factor[0]), np.log(timing.scale_factor[-1])],
        dtype=np.float64,
    )
    linearized = linearize_lhalo_partition(
        partition,
        timing,
        tree_indices=(1575,),
        process_names=("cooling",),
        ln_scale_factor_edges=edges,
        output_snapshots=(63,),
        batch_size=2,
        member_binning="power_of_two",
    )
    automatic = float(linearized.records_by_snapshot[63][0].state_tangent.StellarMass[0])
    step = 1.0e-3
    perturbed = []
    for direction in (-1.0, 1.0):
        result = evolve_lhalo_partition(
            partition,
            timing,
            tree_indices=(1575,),
            perturbations=process_perturbations(cooling=direction * step),
            output_snapshots=(63,),
            batch_size=2,
            member_binning="power_of_two",
        )
        perturbed.append(float(result.records_by_snapshot[63][0].state.StellarMass))
    finite_difference = (perturbed[1] - perturbed[0]) / (2.0 * step)

    assert linearized.success
    assert linearized.control_names == ("cooling:epoch_0",)
    np.testing.assert_allclose(automatic, finite_difference, rtol=6.0e-4, atol=1.0e-7)
