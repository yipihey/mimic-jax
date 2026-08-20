"""Complete-state finite maps and hybrid-event tests for SHARK."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.shark.components import (
    BaryonComponent,
    RotatingBaryonComponent,
    black_hole_component,
    initial_shark_galaxy_state,
    initial_shark_subhalo_state,
    initial_shark_system_state,
    sized_component,
    system_baryonic_mass,
    system_metal_mass,
)
from mimic_jax.shark.hybrid import (
    apply_black_hole_seed,
    apply_cooling_staging_transfer,
    apply_cosmological_infall,
    apply_disk_instability_event,
    apply_galaxy_merger_event,
    apply_halo_ram_pressure_stripping,
    apply_hot_halo_black_hole_transfer,
    apply_ism_ram_pressure_stripping,
    apply_reincorporation_transfer,
    apply_tidal_stripping_to_target,
    flow_state_from_system,
    project_flow_state_to_system,
    remnant_radius,
    tidal_stellar_retention_fraction,
)


def _system():
    galaxy = initial_shark_galaxy_state(
        disk_stars=sized_component(20.0, 1.0, 80.0, 0.02),
        disk_gas=sized_component(10.0, 0.5, 50.0, 0.03),
        bulge_stars=sized_component(5.0, 0.3, 5.0, 0.005),
        bulge_gas=sized_component(2.0, 0.1, 2.0, 0.005),
        black_hole=black_hole_component(0.1, 0.001, spin=0.4),
        maximum_circular_velocity=jnp.asarray(180.0),
    )
    subhalo = initial_shark_subhalo_state(
        hot_halo_gas=RotatingBaryonComponent(100.0, 2.0, 500.0),
        cold_halo_gas=RotatingBaryonComponent(4.0, 0.08, 20.0),
        ejected_gas=RotatingBaryonComponent(12.0, 0.3, 90.0),
        lost_gas=BaryonComponent(3.0, 0.05),
    )
    return initial_shark_system_state(galaxy=galaxy, subhalo=subhalo)


def test_finite_pre_ode_maps_close_mass_and_metals_and_expose_am_boundary():
    state = _system()
    initial_mass = system_baryonic_mass(state)
    initial_metals = system_metal_mass(state)
    reincorporated = apply_reincorporation_transfer(state, 3.0)
    np.testing.assert_allclose(system_baryonic_mass(reincorporated), initial_mass)
    np.testing.assert_allclose(system_metal_mass(reincorporated), initial_metals)

    cooled, transfer, rate, am_residual = apply_cooling_staging_transfer(
        reincorporated, 20.0, 0.2, 7.0
    )
    np.testing.assert_allclose(transfer.mass, 4.0)
    np.testing.assert_allclose(rate, 20.0)
    np.testing.assert_allclose(system_baryonic_mass(cooled), initial_mass)
    np.testing.assert_allclose(system_metal_mass(cooled), initial_metals)
    assert np.isfinite(am_residual)

    grown, bh_transfer = apply_hot_halo_black_hole_transfer(cooled, 1.5, 0.2)
    np.testing.assert_allclose(bh_transfer.mass, 0.3)
    np.testing.assert_allclose(system_baryonic_mass(grown), initial_mass)
    np.testing.assert_allclose(system_metal_mass(grown), initial_metals)


def test_infall_is_an_explicit_source_and_baryon_cap_is_visible():
    state = _system()
    before = system_baryonic_mass(state)
    updated, realized = apply_cosmological_infall(state, 7.0, 5.0, 1.0e-4, 3.0)
    np.testing.assert_allclose(realized, 5.0)
    np.testing.assert_allclose(system_baryonic_mass(updated) - before, 5.0)
    np.testing.assert_allclose(system_metal_mass(updated) - system_metal_mass(state), 5.0e-4)


def test_black_hole_seed_is_a_threshold_event_and_jittable():
    state = _system()._replace(galaxy=_system().galaxy._replace(black_hole=black_hole_component()))
    seeded, active = jax.jit(apply_black_hole_seed)(state, 2.0e10, 1.0e10, 1.0e4)
    assert active
    assert seeded.galaxy.black_hole.mass == 1.0e4
    unchanged, active = apply_black_hole_seed(seeded, 2.0e10, 1.0e10, 1.0e4)
    assert not active
    assert unchanged.galaxy.black_hole.mass == 1.0e4


def test_flow_projection_round_trip_preserves_physical_reservoirs():
    state = _system()
    flow = flow_state_from_system(state)
    projected, diagnostics = project_flow_state_to_system(state, flow, 0.2)
    np.testing.assert_allclose(system_baryonic_mass(projected), system_baryonic_mass(state))
    np.testing.assert_allclose(system_metal_mass(projected), system_metal_mass(state))
    np.testing.assert_allclose(diagnostics["mean_star_formation_rate"], 0.0)


def test_disk_instability_is_conservative_in_mass_and_metals_on_both_branches():
    unstable = _system()._replace(
        galaxy=_system().galaxy._replace(maximum_circular_velocity=jnp.asarray(1.0e-3))
    )
    before_mass = system_baryonic_mass(unstable)
    before_metals = system_metal_mass(unstable)
    result = jax.jit(apply_disk_instability_event)(unstable)
    assert result.triggered
    assert result.state.galaxy.disk_stars.mass == 0.0
    assert result.state.galaxy.disk_gas.mass == 0.0
    np.testing.assert_allclose(system_baryonic_mass(result.state), before_mass)
    np.testing.assert_allclose(system_metal_mass(result.state), before_metals)

    stable = unstable._replace(
        galaxy=unstable.galaxy._replace(maximum_circular_velocity=jnp.asarray(1000.0))
    )
    result = apply_disk_instability_event(stable)
    assert not result.triggered
    np.testing.assert_allclose(system_baryonic_mass(result.state), system_baryonic_mass(stable))


def test_major_and_minor_merger_maps_conserve_mass_and_metals():
    central = _system().galaxy
    satellite = initial_shark_galaxy_state(
        disk_stars=sized_component(12.0, 0.6, 40.0, 0.01),
        disk_gas=sized_component(8.0, 0.3, 30.0, 0.015),
        bulge_stars=sized_component(1.0, 0.04, 1.0, 0.003),
        black_hole=black_hole_component(0.02, 0.0002, spin=0.2),
        maximum_circular_velocity=jnp.asarray(120.0),
    )

    def physical_mass(g):
        return (
            g.disk_stars.mass
            + g.disk_gas.mass
            + g.bulge_stars.mass
            + g.bulge_gas.mass
            + g.black_hole.mass
        )

    def physical_metals(g):
        return (
            g.disk_stars.metals
            + g.disk_gas.metals
            + g.bulge_stars.metals
            + g.bulge_gas.metals
            + g.black_hole.metals
        )

    radius = remnant_radius(37.0, 21.0, 0.015, 0.01)
    result = apply_galaxy_merger_event(central, satellite, radius, 0.35)
    assert result.major
    np.testing.assert_allclose(
        physical_mass(result.central), physical_mass(central) + physical_mass(satellite)
    )
    np.testing.assert_allclose(
        physical_metals(result.central), physical_metals(central) + physical_metals(satellite)
    )

    tiny = satellite._replace(
        disk_stars=sized_component(0.2, 0.01, 1.0, 0.01),
        disk_gas=sized_component(0.05, 0.002, 0.2, 0.015),
        bulge_stars=sized_component(),
        black_hole=black_hole_component(),
    )
    result = apply_galaxy_merger_event(central, tiny, 0.01, central.black_hole.spin)
    assert not result.major
    np.testing.assert_allclose(
        physical_mass(result.central), physical_mass(central) + physical_mass(tiny)
    )


def test_environment_maps_conserve_pair_mass_and_metals():
    satellite = _system()
    central = initial_shark_subhalo_state(hot_halo_gas=RotatingBaryonComponent(200.0, 4.0, 900.0))

    def pair_mass(sat, cen):
        return system_baryonic_mass(sat) + cen.hot_halo_gas.mass + cen.stellar_halo.mass

    def pair_metals(sat, cen):
        return system_metal_mass(sat) + cen.hot_halo_gas.metals + cen.stellar_halo.metals

    before_mass = pair_mass(satellite, central)
    before_metals = pair_metals(satellite, central)
    halo = apply_halo_ram_pressure_stripping(satellite, central, 0.5, 1.0)
    np.testing.assert_allclose(pair_mass(halo.satellite, halo.central_subhalo), before_mass)
    np.testing.assert_allclose(pair_metals(halo.satellite, halo.central_subhalo), before_metals)

    ism = apply_ism_ram_pressure_stripping(halo.satellite, halo.central_subhalo, 0.02)
    np.testing.assert_allclose(pair_mass(ism.satellite, ism.central_subhalo), before_mass)
    np.testing.assert_allclose(pair_metals(ism.satellite, ism.central_subhalo), before_metals)

    tidal = apply_tidal_stripping_to_target(ism.satellite, ism.central_subhalo, 4.0)
    np.testing.assert_allclose(pair_mass(tidal.satellite, tidal.central_subhalo), before_mass)
    np.testing.assert_allclose(pair_metals(tidal.satellite, tidal.central_subhalo), before_metals)
    assert 0.0 < tidal_stellar_retention_fraction(0.1) < 1.0


def test_fixed_branch_derivative_conservation_for_reincorporation():
    state = _system()

    def total(requested):
        return jnp.asarray(
            [
                system_baryonic_mass(apply_reincorporation_transfer(state, requested)),
                system_metal_mass(apply_reincorporation_transfer(state, requested)),
            ]
        )

    np.testing.assert_allclose(jax.jacfwd(total)(3.0), np.zeros(2), atol=1.0e-14)
