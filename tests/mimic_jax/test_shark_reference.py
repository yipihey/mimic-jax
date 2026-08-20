"""Native SHARK provenance/configuration and tree-schema tests."""

from pathlib import Path

import h5py
import numpy as np
import pytest

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
