"""Model-neutral catalogue, observable, and merger-tree comparison contracts."""

from pathlib import Path

import h5py
import numpy as np

from mimic_jax.catalogue import observable_capabilities
from mimic_jax.io import open_lhalo_partition
from mimic_jax.observables import (
    catalogue_black_hole_bulge_relation,
    catalogue_cosmic_sfr_density,
    catalogue_log_relation,
    catalogue_mass_density,
    catalogue_mass_function,
)
from mimic_jax.sage16 import (
    load_sage_comparison_catalogue,
    load_scale_factors,
    snapshot_timing,
)
from mimic_jax.shark import load_shark_catalogue, shark_comparison_catalogue
from mimic_jax.shark.tree import _REQUIRED_NODE_FIELDS, load_shark_tree
from mimic_jax.trees import (
    SAGE16_TREE_REQUIREMENTS,
    SHARK_LAGOS23_TREE_REQUIREMENTS,
    assess_tree_compatibility,
    canonical_tree_from_lhalo,
    canonical_tree_from_shark,
)

TEST_DATA = Path(__file__).parents[1] / "data" / "input"


def _write_sage_catalogue(path):
    dtype = np.dtype(
        [
            ("UniqueGalaxyID", "<i8"),
            ("Type", "<i4"),
            ("StellarMass", "<f4"),
            ("BulgeMass", "<f4"),
            ("ColdGas", "<f4"),
            ("HotGas", "<f4"),
            ("EjectedGas", "<f4"),
            ("ICS", "<f4"),
            ("StarFormationRate", "<f4"),
            ("BlackHoleMass", "<f4"),
            ("CentralMvir", "<f8"),
            ("Mvir", "<f8"),
            ("MetalsColdGas", "<f4"),
            ("MetalsStellarMass", "<f4"),
            ("MetalsBulgeMass", "<f4"),
            ("DiskScaleRadius", "<f4"),
            ("Vmax", "<f4"),
            ("Pos", "<f4", (3,)),
            ("Vel", "<f4", (3,)),
            ("Cooling", "<f8"),
            ("Heating", "<f8"),
            ("SupernovaOutflowRate", "<f4"),
        ]
    )
    galaxies = np.zeros(2, dtype=dtype)
    galaxies["UniqueGalaxyID"] = [1, 2]
    galaxies["Type"] = [0, 1]
    galaxies["StellarMass"] = [1.0, 0.5]
    galaxies["BulgeMass"] = [0.3, 0.2]
    galaxies["ColdGas"] = [0.3, 0.15]
    galaxies["HotGas"] = [2.0, 0.0]
    galaxies["EjectedGas"] = [0.4, 0.0]
    galaxies["ICS"] = [0.1, 0.0]
    galaxies["StarFormationRate"] = [3.0, 1.0]
    galaxies["BlackHoleMass"] = [1.0e-3, 2.0e-4]
    galaxies["CentralMvir"] = [100.0, 100.0]
    galaxies["Mvir"] = [100.0, 10.0]
    galaxies["MetalsColdGas"] = [0.003, 0.0015]
    galaxies["MetalsStellarMass"] = [0.01, 0.005]
    galaxies["MetalsBulgeMass"] = [0.003, 0.002]
    galaxies["DiskScaleRadius"] = [0.007, 0.0035]
    galaxies["Vmax"] = [200.0, 120.0]
    galaxies["Pos"] = [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]
    galaxies["Vel"] = [[100.0, 0.0, 0.0], [80.0, 10.0, 0.0]]
    with h5py.File(path, "w") as handle:
        handle.create_group("Snap063")["Galaxies"] = galaxies


def _write_shark_catalogue(path):
    with h5py.File(path, "w") as handle:
        handle.create_group("cosmology")["h"] = np.float32(0.7)
        run_info = handle.create_group("run_info")
        run_info["effective_volume"] = np.float32(1000.0)
        run_info["redshift"] = np.float64(0.0)
        run_info["shark_git_revision"] = np.bytes_("abc123")
        run_info["shark_version"] = np.bytes_("2.0.0")
        run_info["seed"] = np.uint32(17)
        galaxies = handle.create_group("galaxies")
        galaxies["id_galaxy"] = np.asarray([1, 2], dtype=np.int64)
        galaxies["type"] = np.asarray([0, 1], dtype=np.int32)
        values = {
            "mstars_disk": [7.0e9, 3.0e9],
            "mstars_bulge": [3.0e9, 2.0e9],
            "mgas_disk": [2.0e9, 1.0e9],
            "mgas_bulge": [1.0e9, 0.5e9],
            "matom_disk": [1.2e9, 0.6e9],
            "matom_bulge": [0.4e9, 0.2e9],
            "mmol_disk": [0.8e9, 0.4e9],
            "mmol_bulge": [0.6e9, 0.3e9],
            "sfr_disk": [1.4e9, 0.7e9],
            "sfr_burst": [0.7e9, 0.0],
            "m_bh": [1.0e7, 2.0e6],
            "mvir_hosthalo": [1.0e12, 1.0e12],
            "mgas_metals_disk": [2.0e7, 1.0e7],
            "mgas_metals_bulge": [1.0e7, 0.5e7],
            "mstars_metals_disk": [7.0e7, 3.0e7],
            "mstars_metals_bulge": [3.0e7, 2.0e7],
        }
        for name, value in values.items():
            galaxies[name] = np.asarray(value, dtype=np.float32)
        galaxies["bh_spin"] = np.asarray([0.2, 0.8], dtype=np.float32)
        galaxies["rstar_disk"] = np.asarray([0.007, 0.0035], dtype=np.float32)


def _write_shark_tree(path):
    size = 2
    with h5py.File(path, "w") as handle:
        nodes = handle.create_group("haloTrees")
        nodes.attrs["numberOfTrees"] = 1
        nodes.attrs["treesAreSelfContained"] = 1
        nodes.attrs["treesHaveSubhalos"] = 1
        int64 = {
            "nodeIndex": [10, 11],
            "descendantIndex": [11, -1],
            "mainProgenitorIndex": [-1, 10],
            "hostIndex": [10, 11],
            "enclosingIndex": [-1, -1],
        }
        int32_names = {
            "snapshotNumber",
            "descendantSnapshot",
            "fofIndex",
            "isMainProgenitor",
            "isFoFCentre",
            "isDHaloCentre",
            "isInterpolated",
            "isRemerged",
        }
        vector_names = {"angularMomentum", "position", "velocity"}
        for name in _REQUIRED_NODE_FIELDS:
            if name in int64:
                value = np.asarray(int64[name], dtype=np.int64)
            elif name in int32_names:
                value = np.ones(size, dtype=np.int32)
            elif name in vector_names:
                value = np.ones((size, 3), dtype=np.float32)
            else:
                value = np.ones(size, dtype=np.float32)
            nodes[name] = value
        tree_index = handle.create_group("treeIndex")
        tree_index["firstNode"] = np.asarray([0], dtype=np.int32)
        tree_index["numberOfNodes"] = np.asarray([2], dtype=np.int32)
        tree_index["finalSnapshot"] = np.asarray([1], dtype=np.int32)
        output = handle.create_group("outputTimes")
        output["snapshotNumber"] = np.asarray([0, 1], dtype=np.int32)
        output["redshift"] = np.asarray([1.0, 0.0], dtype=np.float32)
        simulation = handle.create_group("simulation")
        simulation.attrs["boxSize"] = 100.0
        simulation.attrs["particleMass"] = 1.0e9
        info = handle.create_group("fileInfo")
        info.attrs["numberOfFiles"] = 1
        info.attrs["thisFile"] = 0


def test_canonical_catalogues_apply_same_units_and_observables(tmp_path):
    sage_path = tmp_path / "sage.hdf5"
    shark_path = tmp_path / "shark.hdf5"
    _write_sage_catalogue(sage_path)
    _write_shark_catalogue(shark_path)
    sage = load_sage_comparison_catalogue(
        [sage_path],
        snapshot=63,
        hubble_h=0.7,
        effective_volume_mpc_over_h_cubed=1000.0,
        redshift=0.0,
        dataset="controlled",
    )
    shark = shark_comparison_catalogue(
        load_shark_catalogue(shark_path), dataset="controlled", snapshot=199
    )

    np.testing.assert_allclose(sage.values("stellar_mass"), shark.values("stellar_mass"))
    np.testing.assert_allclose(sage.values("cold_gas_mass"), shark.values("cold_gas_mass"))
    np.testing.assert_allclose(sage.values("baryonic_mass"), shark.values("baryonic_mass"))
    # MetalsStellarMass is already total in SAGE; its bulge subset is not added again.
    np.testing.assert_allclose(sage.values("stellar_metal_mass"), [1.0e8 / 0.7, 5.0e7 / 0.7])
    edges = np.asarray([9.0, 10.0, 11.0])
    sage_smf = catalogue_mass_function(sage, "stellar_mass", bin_edges=edges)
    shark_smf = catalogue_mass_function(shark, "stellar_mass", bin_edges=edges)
    np.testing.assert_array_equal(sage_smf.counts, shark_smf.counts)
    np.testing.assert_allclose(sage_smf.number_density, shark_smf.number_density)
    np.testing.assert_allclose(catalogue_cosmic_sfr_density(sage), 0.001372)
    np.testing.assert_allclose(catalogue_mass_density(sage, "stellar_mass"), 7.35e6)
    sfr_relation = catalogue_log_relation(
        sage,
        predictor_field="stellar_mass",
        response_field="star_formation_rate",
        bin_edges=edges,
    )
    np.testing.assert_array_equal(sfr_relation.counts, [1, 1])
    relation = catalogue_black_hole_bulge_relation(shark, bin_edges=np.asarray([9.0, 9.55, 10.0]))
    np.testing.assert_array_equal(relation.counts, [1, 1])

    sage_capability = {item.key: item for item in observable_capabilities(sage)}
    shark_capability = {item.key: item for item in observable_capabilities(shark)}
    assert sage_capability["atomic_gas_mass_function"].status == "unavailable"
    assert shark_capability["atomic_gas_mass_function"].status == "direct"
    assert sage_capability["stellar_metallicity"].status == "qualified"


def test_tree_contract_reports_native_readiness_and_cross_model_blockers(tmp_path):
    partition = open_lhalo_partition(TEST_DATA / "trees_063.0")
    lhalo = canonical_tree_from_lhalo(
        partition.read_tree(1575),
        snapshot_timing(load_scale_factors(TEST_DATA / "mini-millennium.a_list")),
        source_path=partition.path,
        tree_index=1575,
        particle_mass_1e10_msun_over_h=0.0860657,
    )
    sage_native = assess_tree_compatibility(lhalo, SAGE16_TREE_REQUIREMENTS)
    shark_foreign = assess_tree_compatibility(lhalo, SHARK_LAGOS23_TREE_REQUIREMENTS)
    assert sage_native.fully_runnable
    assert not shark_foreign.field_ready
    assert not shark_foreign.population_driver_ready
    assert "concentration" in shark_foreign.missing_fields

    path = tmp_path / "tree.hdf5"
    _write_shark_tree(path)
    shark_tree = canonical_tree_from_shark(load_shark_tree(path), 0)
    shark_native = assess_tree_compatibility(shark_tree, SHARK_LAGOS23_TREE_REQUIREMENTS)
    sage_foreign = assess_tree_compatibility(shark_tree, SAGE16_TREE_REQUIREMENTS)
    assert shark_native.field_ready
    assert shark_native.native_run
    assert not shark_native.population_driver_ready
    assert not sage_foreign.field_ready
    assert not sage_foreign.population_driver_ready
    assert "velocity_dispersion" in sage_foreign.missing_fields
