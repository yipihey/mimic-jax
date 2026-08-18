"""Catalogue-level SAGE16 population summaries used by scientific reports."""

import numpy as np
import pytest

from mimic_jax.sage16 import (
    binned_fraction,
    binned_quantiles,
    group_baryon_inventory,
    safe_fractional_difference,
    stellar_mass_function,
)


def test_stellar_mass_function_matches_mimic_normalization_and_masks_zero_response():
    result = stellar_mass_function(
        np.asarray([0.0, 0.73, 7.3, 73.0]),
        volume_mpc_over_h_cubed=1000.0,
        hubble_h=0.73,
        bin_edges=np.asarray([9.5, 10.5, 11.5, 12.5]),
    )

    assert np.array_equal(result.counts, np.asarray([1, 1, 1]))
    assert np.allclose(result.number_density, np.full(3, 0.73**3 / 1000.0))
    fractional, valid = safe_fractional_difference(
        np.asarray([2.0, 1.0, 0.0]), np.asarray([1.0, 0.0, 0.0])
    )
    assert np.array_equal(valid, np.asarray([True, False, False]))
    assert fractional[0] == pytest.approx(1.0)
    assert np.isnan(fractional[1:]).all()


def test_group_baryon_inventory_uses_type_zero_halo_and_sums_satellites():
    inventory = group_baryon_inventory(
        unique_galaxy_id=np.asarray([10, 11, 20]),
        unique_central_galaxy_id=np.asarray([10, 10, 20]),
        galaxy_type=np.asarray([0, 1, 0]),
        central_halo_mass=np.asarray([100.0, 5.0, 1000.0]),
        reservoirs={
            "stars": np.asarray([5.0, 2.0, 10.0]),
            "hot": np.asarray([7.0, 1.0, 100.0]),
        },
        hubble_h=1.0,
        global_baryon_fraction=0.2,
        halo_mass_bin_edges=np.asarray([11.5, 12.5, 13.5]),
    )

    assert np.array_equal(inventory.group_counts, np.asarray([1, 1]))
    assert inventory.reservoir_names == ("stars", "hot")
    assert np.allclose(inventory.reservoir_mass, np.asarray([[7.0, 8.0], [10.0, 100.0]]))
    assert np.allclose(
        inventory.allotment_fractions,
        np.asarray([[7.0 / 20.0, 8.0 / 20.0], [10.0 / 200.0, 100.0 / 200.0]]),
    )


def test_group_baryon_inventory_rejects_missing_central_identity():
    with pytest.raises(ValueError, match="identify the Type-0 records"):
        group_baryon_inventory(
            unique_galaxy_id=np.asarray([10]),
            unique_central_galaxy_id=np.asarray([20]),
            galaxy_type=np.asarray([0]),
            central_halo_mass=np.asarray([100.0]),
            reservoirs={"stars": np.asarray([1.0])},
            hubble_h=0.73,
            global_baryon_fraction=0.17,
            halo_mass_bin_edges=np.asarray([10.0, 15.0]),
        )


def test_binned_fraction_and_quantiles_leave_empty_bins_undefined():
    values = np.asarray([0.1, 0.2, 1.1])
    edges = np.asarray([0.0, 1.0, 2.0, 3.0])
    total, selected, fraction = binned_fraction(
        values, np.asarray([True, False, True]), bin_edges=edges
    )
    counts, quantiles = binned_quantiles(
        values,
        np.asarray([1.0, 3.0, 5.0]),
        bin_edges=edges,
        quantiles=(0.5,),
    )

    assert np.array_equal(total, np.asarray([2, 1, 0]))
    assert np.array_equal(selected, np.asarray([1, 1, 0]))
    assert np.allclose(fraction[:2], np.asarray([0.5, 1.0]))
    assert np.isnan(fraction[2])
    assert np.array_equal(counts, total)
    assert np.allclose(quantiles[0, :2], np.asarray([2.0, 5.0]))
    assert np.isnan(quantiles[0, 2])
