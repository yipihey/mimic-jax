"""Process-level SAGE16 formulas and executable conservation checks."""

import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16 import (
    apply_cooling,
    apply_metal_enrichment,
    apply_reincorporation,
    apply_star_formation_supernova,
    baryonic_mass,
    calculate_star_formation_budget,
    calculate_supernova_feedback_budget,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    metal_mass,
    quiescent_disk_step,
    sage16_units,
    step_context,
)

MASS_ATOL = 2.0e-6
METAL_ATOL = 2.0e-7


def test_star_formation_matches_upstream_threshold_formula():
    state = initial_galaxy_state(ColdGas=10.0, DiskScaleRadius=0.05)
    halo = initial_halo_forcing(Vvir=200.0, dT=0.01)
    context = step_context(num_substeps=1, time_interval=0.01)
    parameters = fiducial_parameters()._replace(SfrEfficiency=jnp.asarray(0.02))

    budget = calculate_star_formation_budget(state, halo, context, parameters)
    # The radius is stored as float, then promoted into SAGE's double local.
    reff = 3.0 * float(np.float32(0.05))
    tdyn = reff / 200.0
    cold_crit = 0.19 * 200.0 * reff
    expected = 0.02 * (10.0 - cold_crit) / tdyn * 0.01

    assert np.isclose(float(budget.NewStellarMass), expected, rtol=1.0e-13, atol=0.0)

    below = state._replace(ColdGas=jnp.asarray(0.1, dtype=jnp.float32))
    # Mirror the C fixture exactly: the caller computes the threshold from the
    # input double radius, while GalaxyData stores both values as float.
    input_cold_crit = 0.19 * 200.0 * (3.0 * 0.05)
    at_threshold = state._replace(ColdGas=jnp.asarray(input_cold_crit, dtype=jnp.float32))
    assert float(calculate_star_formation_budget(below, halo, context, parameters)[0]) == 0.0
    assert float(calculate_star_formation_budget(at_threshold, halo, context, parameters)[0]) == 0.0


def test_supernova_budget_matches_upstream_renormalisation_and_energy_formula():
    state = initial_galaxy_state(ColdGas=5.0)
    central_halo = initial_halo_forcing(Vvir=150.0)
    parameters = fiducial_parameters()
    units = sage16_units()
    raw = calculate_star_formation_budget(
        initial_galaxy_state(ColdGas=10.0, DiskScaleRadius=0.01),
        central_halo,
        step_context(time_interval=0.01),
        parameters,
    )
    raw = raw._replace(NewStellarMass=jnp.asarray(3.0, dtype=jnp.float64))

    budget = calculate_supernova_feedback_budget(state, central_halo, parameters, units, raw)

    assert np.isclose(float(budget.NewStellarMass), 1.25, rtol=0.0, atol=1.0e-14)
    assert np.isclose(float(budget.SupernovaReheatedMass), 3.75, rtol=0.0, atol=1.0e-14)
    specific_energy = (
        0.005
        * (float(units.UnitMass_in_g) / 1.989e33)
        / float(units.Hubble_h)
        * (1.0e51 / float(units.UnitEnergy_in_cgs) * float(units.Hubble_h))
    )
    expected_ejected = max((0.3 * specific_energy / 150.0**2 - 3.0) * 1.25, 0.0)
    assert np.isclose(float(budget.SupernovaEjectedMass), expected_ejected, rtol=1.0e-14, atol=0.0)


def test_central_quiescent_chain_conserves_baryons_and_accounts_for_metal_source():
    state = initial_galaxy_state(
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
    halo = initial_halo_forcing(Vvir=150.0, dT=1.0e-4)
    context = step_context(time_interval=1.0e-4)
    parameters = fiducial_parameters()
    units = sage16_units()

    sf = calculate_star_formation_budget(state, halo, context, parameters)
    sn = calculate_supernova_feedback_budget(state, halo, parameters, units, sf)
    applied = apply_star_formation_supernova(state, state, halo, parameters, sn)
    enriched = apply_metal_enrichment(applied.galaxy, applied.central, halo, True, parameters)

    assert np.isclose(
        float(baryonic_mass(applied.galaxy)),
        float(baryonic_mass(state)),
        rtol=0.0,
        atol=MASS_ATOL,
    )
    assert np.isclose(
        float(metal_mass(applied.galaxy)),
        float(metal_mass(state)),
        rtol=0.0,
        atol=METAL_ATOL,
    )
    assert np.isclose(
        float(metal_mass(enriched.galaxy) - metal_mass(state)),
        float(enriched.transfer.produced_metals),
        rtol=0.0,
        atol=METAL_ATOL,
    )
    assert float(enriched.galaxy.NewStellarMass) == 0.0
    assert float(enriched.transfer.produced_metals) == float(parameters.Yield * sn.NewStellarMass)


def test_satellite_feedback_conserves_combined_local_and_central_reservoirs():
    satellite = initial_galaxy_state(
        ColdGas=4.0,
        HotGas=0.5,
        StellarMass=1.0,
        MetalsColdGas=0.08,
        MetalsHotGas=0.01,
        MetalsStellarMass=0.02,
        DiskScaleRadius=0.01,
    )
    central = initial_galaxy_state(
        ColdGas=2.0,
        HotGas=6.0,
        EjectedGas=1.0,
        StellarMass=3.0,
        MetalsColdGas=0.04,
        MetalsHotGas=0.12,
        MetalsEjectedGas=0.02,
        MetalsStellarMass=0.06,
    )
    satellite_halo = initial_halo_forcing(Type=1, Vvir=100.0, dT=1.0e-4)
    central_halo = initial_halo_forcing(Type=0, Vvir=150.0, dT=1.0e-4)
    context = step_context(time_interval=1.0e-4)
    parameters = fiducial_parameters()
    units = sage16_units()
    initial_mass = baryonic_mass(satellite) + baryonic_mass(central)
    initial_metals = metal_mass(satellite) + metal_mass(central)

    sf = calculate_star_formation_budget(satellite, satellite_halo, context, parameters)
    sn = calculate_supernova_feedback_budget(satellite, central_halo, parameters, units, sf)
    applied = apply_star_formation_supernova(satellite, central, satellite_halo, parameters, sn)

    assert np.isclose(
        float(baryonic_mass(applied.galaxy) + baryonic_mass(applied.central)),
        float(initial_mass),
        rtol=0.0,
        atol=2.0 * MASS_ATOL,
    )
    assert np.isclose(
        float(metal_mass(applied.galaxy) + metal_mass(applied.central)),
        float(initial_metals),
        rtol=0.0,
        atol=2.0 * METAL_ATOL,
    )
    assert float(applied.central.HotGas) != float(central.HotGas)


def test_cooling_transfer_conserves_gas_and_metals():
    state = initial_galaxy_state(
        ColdGas=2.0,
        HotGas=8.0,
        MetalsColdGas=0.04,
        MetalsHotGas=0.16,
        CoolingGas=1.5,
    )
    halo = initial_halo_forcing(Vvir=200.0, dT=0.01)
    result = apply_cooling(state, halo)

    assert np.isclose(float(result.transfer.gas), 1.5)
    assert np.isclose(float(result.transfer.metals), 0.03, rtol=0.0, atol=2.0e-9)
    assert np.isclose(
        float(baryonic_mass(result.state)), float(baryonic_mass(state)), atol=MASS_ATOL
    )
    assert np.isclose(float(metal_mass(result.state)), float(metal_mass(state)), atol=METAL_ATOL)
    assert np.isclose(float(result.state.Cooling), 3.0e6, rtol=1.0e-15)


def test_reincorporation_transfer_conserves_gas_and_metals():
    state = initial_galaxy_state(
        HotGas=2.0,
        EjectedGas=4.0,
        MetalsHotGas=0.04,
        MetalsEjectedGas=0.08,
    )
    halo = initial_halo_forcing(Type=0, Vvir=100.0, Rvir=0.2, dT=0.01)
    context = step_context(num_substeps=10, time_interval=0.01)
    result = apply_reincorporation(state, halo, context, fiducial_parameters())

    expected = (100.0 / (445.48 * 0.15) - 1.0) * 4.0 / (0.2 / 100.0) * 0.001
    assert np.isclose(float(result.transfer.gas), expected, rtol=1.0e-14)
    assert np.isclose(float(result.transfer.metals), 0.02 * expected, rtol=1.0e-7)
    assert np.isclose(
        float(baryonic_mass(result.state)), float(baryonic_mass(state)), atol=MASS_ATOL
    )
    assert np.isclose(float(metal_mass(result.state)), float(metal_mass(state)), atol=METAL_ATOL)


def test_quiescent_convenience_step_matches_explicit_module_sequence():
    state = initial_galaxy_state(ColdGas=10.0, HotGas=5.0, DiskScaleRadius=0.01)
    halo = initial_halo_forcing(Vvir=150.0, dT=1.0e-4)
    context = step_context(time_interval=1.0e-4)
    parameters = fiducial_parameters()
    units = sage16_units()

    combined = quiescent_disk_step(state, state, halo, halo, context, parameters, units)
    sf = calculate_star_formation_budget(state, halo, context, parameters)
    sn = calculate_supernova_feedback_budget(state, halo, parameters, units, sf)
    applied = apply_star_formation_supernova(state, state, halo, parameters, sn)
    explicit = apply_metal_enrichment(applied.galaxy, applied.central, halo, True, parameters)

    np.testing.assert_allclose(
        np.asarray(combined.galaxy), np.asarray(explicit.galaxy), rtol=0.0, atol=0.0
    )
