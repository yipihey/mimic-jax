"""Mini-Millennium L-Halo binary reader contracts."""

from pathlib import Path

import numpy as np
import pytest

from mimic_jax.io import LHALO_DTYPE, open_lhalo_partition

DATA = Path(__file__).parents[1] / "data" / "input" / "trees_063.0"


def test_lhalo_dtype_matches_upstream_raw_halo_layout():
    assert LHALO_DTYPE.itemsize == 104
    assert LHALO_DTYPE.fields["MostBoundID"][1] == 80
    assert LHALO_DTYPE.fields["SnapNum"][1] == 88
    assert LHALO_DTYPE.fields["SubHalfMass"][1] == 100


def test_partition_header_and_random_tree_access_match_mini_millennium():
    partition = open_lhalo_partition(DATA)
    assert partition.tree_count == 3432
    assert partition.total_halos == 174845
    assert int(partition.tree_halo_counts[0]) == 4569

    tree = partition.read_tree(1575)
    assert tree.shape == (6,)
    np.testing.assert_array_equal(tree["SnapNum"], np.arange(63, 57, -1))
    np.testing.assert_array_equal(tree["FirstHaloInFOFgroup"], np.arange(6))
    np.testing.assert_array_equal(tree["NextHaloInFOFgroup"], -1)


def test_reader_rejects_bad_tree_index():
    partition = open_lhalo_partition(DATA)
    with pytest.raises(IndexError, match="outside"):
        partition.read_tree(partition.tree_count)
