"""Native SHARK HDF5 adapter and common-observable tests."""

import h5py
import numpy as np

from mimic_jax.shark import (
    load_shark_catalogue,
    shark_atomic_gas_mass_function,
    shark_black_hole_bulge_relation,
    shark_black_hole_spin_relation,
    shark_cold_gas_fraction_relation,
    shark_cosmic_star_formation_rate_density,
    shark_gas_metallicity_relation,
    shark_molecular_gas_mass_function,
    shark_quenched_fraction,
    shark_stellar_mass_function,
    shark_stellar_metallicity_relation,
    shark_stellar_size_relation,
    shark_stellar_to_halo_mass_relation,
)


def _write_catalogue(path):
    with h5py.File(path, "w") as handle:
        cosmology = handle.create_group("cosmology")
        cosmology["h"] = np.float32(0.7)
        run_info = handle.create_group("run_info")
        run_info["effective_volume"] = np.float32(1000.0)
        run_info["redshift"] = np.float64(0.0)
        run_info["shark_git_revision"] = np.bytes_("abc123")
        run_info["shark_version"] = np.bytes_("2.0.0")
        run_info["seed"] = np.uint32(17)
        galaxies = handle.create_group("galaxies")
        galaxies["id_galaxy"] = np.asarray([1, 2], dtype=np.int32)
        galaxies["type"] = np.asarray([0, 1], dtype=np.int32)
        fields = {
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
        for name, values in fields.items():
            galaxies[name] = np.asarray(values, dtype=np.float32)
        galaxies["bh_spin"] = np.asarray([0.2, 0.8], dtype=np.float32)
        galaxies["rstar_disk"] = np.asarray([0.007, 0.0035], dtype=np.float32)


def test_native_catalogue_maps_component_sums_units_and_metadata(tmp_path):
    path = tmp_path / "galaxies.hdf5"
    _write_catalogue(path)
    catalogue = load_shark_catalogue(path)

    np.testing.assert_allclose(catalogue.stellar_mass, [1.0e10, 5.0e9])
    np.testing.assert_allclose(catalogue.cold_gas_mass, [3.0e9, 1.5e9])
    np.testing.assert_allclose(catalogue.atomic_gas_mass, [1.6e9, 0.8e9])
    np.testing.assert_allclose(catalogue.molecular_gas_mass, [1.4e9, 0.7e9])
    np.testing.assert_allclose(catalogue.star_formation_rate_msun_per_year, [3.0, 1.0])
    np.testing.assert_allclose(catalogue.stellar_mass_msun, [1.0e10 / 0.7, 5.0e9 / 0.7])
    np.testing.assert_allclose(shark_cosmic_star_formation_rate_density(catalogue), 0.004)
    assert catalogue.upstream_revision == "abc123"
    assert catalogue.upstream_version == "2.0.0"
    assert catalogue.seed == 17
    assert "bh_spin" in catalogue.extras

    result = shark_stellar_mass_function(
        catalogue,
        bin_edges=np.asarray([9.0, 10.0, 11.0]),
    )
    np.testing.assert_array_equal(result.counts, [1, 1])
    np.testing.assert_allclose(result.number_density, [0.7**3 / 1000.0] * 2)
    atomic = shark_atomic_gas_mass_function(catalogue, bin_edges=np.asarray([8.8, 9.2, 9.6]))
    molecular = shark_molecular_gas_mass_function(catalogue, bin_edges=np.asarray([8.8, 9.15, 9.5]))
    np.testing.assert_array_equal(atomic.counts, [1, 1])
    np.testing.assert_array_equal(molecular.counts, [1, 1])

    relation_edges = np.asarray([9.0, 10.0, 11.0])
    gas_fraction = shark_cold_gas_fraction_relation(catalogue, bin_edges=relation_edges)
    np.testing.assert_array_equal(gas_fraction.counts, [1, 1])
    np.testing.assert_allclose(gas_fraction.median, [1.5 / 6.5, 3.0 / 13.0])

    metallicity = shark_gas_metallicity_relation(catalogue, bin_edges=relation_edges)
    np.testing.assert_allclose(metallicity.median, [np.log10(0.01), np.log10(0.01)])

    bh_bulge = shark_black_hole_bulge_relation(catalogue, bin_edges=np.asarray([9.0, 9.55, 10.0]))
    np.testing.assert_array_equal(bh_bulge.counts, [1, 1])
    np.testing.assert_allclose(
        bh_bulge.median,
        [np.log10(2.0e6 / 0.7), np.log10(1.0e7 / 0.7)],
    )

    quenched = shark_quenched_fraction(
        catalogue,
        bin_edges=relation_edges,
        specific_sfr_threshold_per_year=2.5e-10,
    )
    np.testing.assert_array_equal(quenched.counts, [1, 1])
    np.testing.assert_allclose(quenched.fraction, [1.0, 1.0])
    spin = shark_black_hole_spin_relation(catalogue, bin_edges=np.asarray([6.0, 6.8, 7.5]))
    np.testing.assert_allclose(spin.median, [0.8, 0.2])
    size = shark_stellar_size_relation(catalogue, bin_edges=relation_edges)
    np.testing.assert_allclose(
        size.median,
        [np.log10(0.0035 / 0.7 * 1.0e3), np.log10(0.007 / 0.7 * 1.0e3)],
    )
    stellar_metallicity = shark_stellar_metallicity_relation(catalogue, bin_edges=relation_edges)
    np.testing.assert_allclose(stellar_metallicity.median, [np.log10(0.01), np.log10(0.01)])
    stellar_halo = shark_stellar_to_halo_mass_relation(
        catalogue, bin_edges=np.asarray([12.0, 12.5])
    )
    np.testing.assert_array_equal(stellar_halo.counts, [1])
    np.testing.assert_allclose(stellar_halo.median, [np.log10(1.0e10 / 0.7)])


def test_native_catalogue_rejects_missing_required_fields(tmp_path):
    path = tmp_path / "invalid.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_group("galaxies")["id_galaxy"] = np.asarray([1])
    try:
        load_shark_catalogue(path)
    except ValueError as error:
        assert "missing required fields" in str(error)
    else:
        raise AssertionError("Missing native fields must fail loudly")
