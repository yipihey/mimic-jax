"""Native SHARK provenance/configuration and tree-schema tests."""

from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mimic_jax.shark.population_parity import (
    SHARK_RHS_TRACE_MAGIC,
    SHARK_RHS_TRACE_WIDTH,
    _population_replay_kernel,
    _StreamingComparison,
    compare_native_shark_catalogues,
    evaluate_shark_population_parity,
    load_shark_rhs_trace,
)
from mimic_jax.shark.reference import prepare_reference_config, sha256_file, verify_sha256
from mimic_jax.shark.tree import _REQUIRED_NODE_FIELDS, load_shark_tree


def test_prepare_reference_config_changes_only_explicit_run_inputs(tmp_path):
    template = tmp_path / "sample.cfg"
    template.write_text(
        """[execution]\nseed = 1\nsimulation_batches = 5\noutput_directory = old\nname_model = old-model\n\n[simulation]\ntree_files_prefix = old-tree\nredshift_file = old-z\n""",
        encoding="utf-8",
    )
    tree = tmp_path / "tree_199.0.hdf5"
    tree.touch()
    redshifts = tmp_path / "redshifts.txt"
    redshifts.touch()
    destination = tmp_path / "effective.cfg"
    prepare_reference_config(
        template,
        destination,
        tree_file=tree,
        redshift_file=redshifts,
        output_directory=tmp_path / "output",
        seed=123456,
        model_name="lagos23-reference",
        simulation_batch=0,
    )
    text = destination.read_text(encoding="utf-8")
    assert "seed = 123456" in text
    assert "name_model = lagos23-reference" in text
    assert f"tree_files_prefix = {tmp_path / 'tree_199'}" in text
    digest = sha256_file(destination)
    assert verify_sha256(destination, digest) == digest
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(destination, "0" * 64)


def _write_tree(path: Path):
    size = 3
    with h5py.File(path, "w") as handle:
        nodes = handle.create_group("haloTrees")
        nodes.attrs["numberOfTrees"] = 2
        nodes.attrs["treesAreSelfContained"] = 1
        nodes.attrs["treesHaveSubhalos"] = 1
        integer_fields = {
            "nodeIndex",
            "descendantIndex",
            "mainProgenitorIndex",
            "hostIndex",
            "enclosingIndex",
        }
        int32_fields = {
            "snapshotNumber",
            "descendantSnapshot",
            "fofIndex",
            "isMainProgenitor",
            "isFoFCentre",
            "isDHaloCentre",
            "isInterpolated",
            "isRemerged",
        }
        vector_fields = {"angularMomentum", "position", "velocity"}
        for name in _REQUIRED_NODE_FIELDS:
            if name == "nodeIndex":
                value = np.asarray([10, 11, 12], dtype=np.int64)
            elif name == "descendantIndex":
                value = np.asarray([11, -1, -1], dtype=np.int64)
            elif name in integer_fields:
                value = np.full(size, -1, dtype=np.int64)
            elif name in int32_fields:
                value = np.zeros(size, dtype=np.int32)
            elif name in vector_fields:
                value = np.zeros((size, 3), dtype=np.float32)
            else:
                value = np.ones(size, dtype=np.float32)
            nodes.create_dataset(name, data=value)
        tree_index = handle.create_group("treeIndex")
        tree_index.create_dataset("firstNode", data=np.asarray([0, 2], dtype=np.int32))
        tree_index.create_dataset("numberOfNodes", data=np.asarray([2, 1], dtype=np.int32))
        tree_index.create_dataset("finalSnapshot", data=np.asarray([199, 199], dtype=np.int32))
        output = handle.create_group("outputTimes")
        output.create_dataset("snapshotNumber", data=np.asarray([198, 199], dtype=np.int32))
        output.create_dataset("redshift", data=np.asarray([0.1, 0.0], dtype=np.float32))
        simulation = handle.create_group("simulation")
        simulation.attrs["boxSize"] = 210.0
        simulation.attrs["particleMass"] = 5.9e9
        info = handle.create_group("fileInfo")
        info.attrs["numberOfFiles"] = 64
        info.attrs["thisFile"] = 0


def test_tree_reader_validates_schema_boundaries_and_links(tmp_path):
    path = tmp_path / "tree_199.0.hdf5"
    _write_tree(path)
    tree = load_shark_tree(path)
    assert tree.number_of_trees == 2
    assert tree.number_of_nodes_total == 3
    assert tree.tree_slice(0) == slice(0, 2)
    np.testing.assert_array_equal(tree.nodes_at_snapshot(0), np.asarray([0, 1, 2]))
    assert tree.self_contained and tree.has_subhalos
    assert tree.number_of_missing_descendants == 0


def test_tree_reader_exposes_upstream_skipped_missing_descendants(tmp_path):
    path = tmp_path / "tree_199.0.hdf5"
    _write_tree(path)
    with h5py.File(path, "r+") as handle:
        handle["haloTrees/descendantIndex"][0] = np.int64(999)

    tree = load_shark_tree(path)
    assert tree.number_of_missing_descendants == 1
    np.testing.assert_array_equal(tree.missing_descendant_rows, np.asarray([0]))
    with pytest.raises(ValueError, match="1 unresolved positive descendant links"):
        load_shark_tree(path, require_all_descendants=True)


def test_population_trace_replay_is_batched_and_machine_readable(tmp_path):
    tree_path = tmp_path / "tree_199.0.hdf5"
    _write_tree(tree_path)
    trace_path = tmp_path / "population.bin"
    record = np.zeros((1, SHARK_RHS_TRACE_WIDTH), dtype="<f8")
    record[0, 0] = SHARK_RHS_TRACE_MAGIC
    record[0, 1:6] = (7, 10, 0, 0, 0)
    record[0, 6:25] = (
        2.0e9,
        1.0e9,
        2.0e8,
        8.0e10,
        1.0e9,
        0.0,
        2.0e7,
        1.0e7,
        2.0e6,
        8.0e8,
        1.0e7,
        0.0,
        0.0,
        0.0,
        1.0e10,
        8.0e9,
        4.0e8,
        3.0e12,
        2.0e10,
    )
    auxiliary = record[0, 44:70]
    auxiliary[:15] = (
        0.006,
        0.004,
        1.0e8,
        0.01,
        4.0,
        0.2,
        0.5,
        180.0,
        200.0,
        1.0e7,
        1.0e5,
        0.0,
        0.0,
        0.0,
        0.3,
    )
    # The zero BH accretion rates make QSO feedback inactive, so the stored
    # SFR placeholder does not feed back into the calculated rate.
    kernel = _population_replay_kernel()
    provisional_rates, _, _, _ = jax.device_get(
        kernel(
            jnp.asarray(record[:, 6:25]),
            jnp.asarray(record[:, 44:70]),
            jnp.asarray([False]),
            jnp.zeros((1, 11), dtype=jnp.float64),
        )
    )
    rates = np.asarray(provisional_rates[0])
    auxiliary[2] = rates[0]
    auxiliary[15:23] = rates[1:9]
    auxiliary[3:5] = rates[9:11]
    calculated_rates, rhs, _, derived = jax.device_get(
        kernel(
            jnp.asarray(record[:, 6:25]),
            jnp.asarray(record[:, 44:70]),
            jnp.asarray([False]),
            jnp.asarray(rates[None, :]),
        )
    )
    assert np.allclose(calculated_rates[0], rates)
    record[0, 25:44] = rhs[0]
    record[0, 67:70] = derived[0]
    record.tofile(trace_path)

    result = evaluate_shark_population_parity(
        trace_path,
        tree_path,
        batch_size=1,
        relative_tolerance=1.0e-12,
        absolute_tolerance=1.0e-12,
    )
    assert result.passed
    assert result.rhs_evaluations == 1
    assert result.unique_galaxies_evaluated == 1
    assert result.strict_passed
    assert result.rhs.failing_values == 0
    assert result.rates.failing_values == 0
    assert result.rates.outside_warning_band_values == 0
    destination = result.write_json(tmp_path / "parity.json")
    assert destination.read_text(encoding="utf-8").endswith("\n")
    assert load_shark_rhs_trace(trace_path).shape == (1, SHARK_RHS_TRACE_WIDTH)

    record[0, 0] = -1.0
    record.tofile(trace_path)
    with pytest.raises(ValueError, match="magic/version"):
        load_shark_rhs_trace(trace_path)


def test_population_parity_retains_strict_exceptions_inside_warning_band():
    comparison = _StreamingComparison(1.0e-4, 2.0e-4, 0.0)
    comparison.update(np.asarray([1.00015]), np.asarray([1.0]))
    result = comparison.result()

    assert result.failing_values == 1
    assert result.outside_warning_band_values == 0


def test_native_trace_does_not_change_catalogue_values(tmp_path):
    roots = [tmp_path / name / "199" / "0" for name in ("traced", "clean")]
    for root in roots:
        root.mkdir(parents=True)
        with h5py.File(root / "galaxies.hdf5", "w") as handle:
            galaxies = handle.create_group("galaxies")
            galaxies.create_dataset("mass", data=np.asarray([1.0, 2.0]))

    comparison = compare_native_shark_catalogues(tmp_path / "traced", tmp_path / "clean")
    assert comparison.passed
    assert comparison.snapshots_compared == 1
    assert comparison.datasets_compared == 1
    assert comparison.values_compared == 2

    with h5py.File(roots[0] / "galaxies.hdf5", "r+") as handle:
        handle["galaxies/mass"][1] = 3.0
    changed = compare_native_shark_catalogues(tmp_path / "traced", tmp_path / "clean")
    assert not changed.passed
    assert changed.mismatching_datasets == 1
    assert changed.mismatching_values == 1
