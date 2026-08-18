"""Continuous-flow, projection, forcing, and fixed-event hybrid contracts."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.numerics import integrate_fixed_step
from mimic_jax.sage16 import (
    StrippingPairState,
    apply_heating_radius_projection,
    apply_infall,
    apply_merger_ownership_event,
    apply_radio_mode_heating,
    calculate_cooling_budget,
    continuous_satellite_stripping_rates,
    fiducial_parameters,
    hybrid_baryonic_mass,
    hybrid_metal_mass,
    hybrid_state_from_galaxy,
    initial_galaxy_state,
    initial_halo_forcing,
    load_cooling_tables,
    prepared_infall_forcing,
    sage16_hybrid_rhs_and_rates,
    sage16_units,
    step_context,
)


def _hybrid_case(**overrides):
    values = {
        "HaloBaryonFraction": 0.17,
        "ColdGas": 2.0,
        "HotGas": 10.0,
        "EjectedGas": 1.0,
        "StellarMass": 1.0,
        "BlackHoleMass": 0.01,
        "MetalsColdGas": 0.04,
        "MetalsHotGas": 0.2,
        "MetalsEjectedGas": 0.02,
        "MetalsStellarMass": 0.02,
        "Rheat": 0.01,
        "DiskScaleRadius": 0.01,
    }
    values.update(overrides)
    galaxy = initial_galaxy_state(**values)
    halo = initial_halo_forcing(Mvir=100.0, Rvir=0.2, Vvir=200.0, dT=1.0e-4)
    return galaxy, halo, fiducial_parameters(), sage16_units(), load_cooling_tables()


def test_radio_flow_and_rheat_projection_match_one_uncapped_upstream_update():
    galaxy, halo, parameters, units, tables = _hybrid_case()
    context = step_context(time_interval=float(halo.dT))
    budget = calculate_cooling_budget(galaxy, halo, context, units, tables)
    upstream = apply_radio_mode_heating(budget.state, halo, context, parameters, units)
    state = hybrid_state_from_galaxy(galaxy)
    hybrid = sage16_hybrid_rhs_and_rates(
        0.0,
        state,
        halo,
        parameters,
        units,
        tables,
    )
    dt = float(halo.dT)

    np.testing.assert_allclose(
        hybrid.rates.cooling * dt,
        upstream.transfer.cooling_after_prior_heating,
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        hybrid.rates.radio_black_hole_accretion * dt,
        upstream.transfer.black_hole_accreted,
        rtol=2.0e-6,
    )
    projected = apply_heating_radius_projection(state, hybrid.rates)
    np.testing.assert_allclose(
        projected.state.Rheat,
        upstream.state.Rheat,
        rtol=2.0e-6,
    )
    assert bool(projected.applied)


def test_rheat_is_markov_state_and_suppresses_later_cooling():
    galaxy, halo, parameters, units, tables = _hybrid_case(Rheat=0.0)
    state = hybrid_state_from_galaxy(galaxy)
    first = sage16_hybrid_rhs_and_rates(0.0, state, halo, parameters, units, tables)
    projected = apply_heating_radius_projection(state, first.rates).state
    later = sage16_hybrid_rhs_and_rates(0.0, projected, halo, parameters, units, tables)

    assert float(projected.Rheat) > float(state.Rheat)
    assert float(later.rates.cooling) < float(first.rates.cooling)

    disabled = parameters._replace(AGNrecipe=jnp.asarray(0, dtype=jnp.int32))
    off = sage16_hybrid_rhs_and_rates(0.0, projected, halo, disabled, units, tables)
    np.testing.assert_allclose(off.rates.cooling, first.rates.cooling, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(off.rates.agn_heating, 0.0)


def test_prepared_infall_is_external_forcing_and_matches_upstream_away_from_boundaries():
    galaxy, halo, parameters, units, tables = _hybrid_case(
        InfallingGas=2.0,
        EjectedGas=5.0,
    )
    context = step_context(num_substeps=4, time_interval=0.04)
    forcing = prepared_infall_forcing(galaxy.InfallingGas, context.time_interval)
    result = sage16_hybrid_rhs_and_rates(
        0.0,
        hybrid_state_from_galaxy(galaxy),
        halo,
        parameters,
        units,
        tables,
        infall_forcing=forcing,
    )
    upstream = apply_infall(galaxy, context)

    np.testing.assert_allclose(
        result.rates.infall_to_hot * context.substep_dt,
        upstream.transfer.external_to_hot,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        sum(result.derivative[index] for index in (1, 2, 3, 4, 6, 7)),
        result.rates.external_baryon_rate,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_hybrid_mass_and_metal_ledgers_and_their_agn_derivative_close():
    galaxy, halo, parameters, units, tables = _hybrid_case()
    state = hybrid_state_from_galaxy(galaxy)
    forcing = prepared_infall_forcing(0.5, 0.01)

    def residuals(efficiency):
        varied = parameters._replace(RadioModeEfficiency=efficiency)
        result = sage16_hybrid_rhs_and_rates(
            0.0,
            state,
            halo,
            varied,
            units,
            tables,
            infall_forcing=forcing,
        )
        baryon = hybrid_baryonic_mass(result.derivative) - result.rates.external_baryon_rate
        metal = (
            hybrid_metal_mass(result.derivative)
            - result.rates.produced_metals
            - result.rates.external_metal_rate
            + result.rates.radio_hot_metals_accreted
        )
        return jnp.asarray((baryon, metal))

    residual = residuals(parameters.RadioModeEfficiency)
    jacobian = jax.jacfwd(residuals)(parameters.RadioModeEfficiency)
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(jacobian, 0.0, rtol=0.0, atol=3.0e-13)


def test_continuous_stripping_is_conservative_and_matches_the_upstream_limit():
    satellite_galaxy = initial_galaxy_state(
        HaloBaryonFraction=0.17,
        HotGas=5.0,
        MetalsHotGas=0.1,
    )
    central_galaxy = initial_galaxy_state(HotGas=100.0, MetalsHotGas=2.0)
    pair = StrippingPairState(
        hybrid_state_from_galaxy(satellite_galaxy),
        hybrid_state_from_galaxy(central_galaxy),
    )
    halo = initial_halo_forcing(Type=1, Mvir=10.0)
    parameters = fiducial_parameters()
    duration = jnp.asarray(0.2, dtype=jnp.float64)

    def rhs(_time, current):
        return continuous_satellite_stripping_rates(
            current,
            halo,
            parameters,
            duration,
        ).derivative

    solution = integrate_fixed_step(rhs, pair, duration=duration, num_steps=128, method="rk4")
    initial_excess = 5.0 - 1.7
    expected_hot = 1.7 + initial_excess * np.exp(-1.0)
    np.testing.assert_allclose(
        solution.final_state.satellite.HotGas,
        expected_hot,
        rtol=2.0e-10,
    )
    np.testing.assert_allclose(
        hybrid_baryonic_mass(solution.final_state.satellite)
        + hybrid_baryonic_mass(solution.final_state.central),
        hybrid_baryonic_mass(pair.satellite) + hybrid_baryonic_mass(pair.central),
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        hybrid_metal_mass(solution.final_state.satellite)
        + hybrid_metal_mass(solution.final_state.central),
        hybrid_metal_mass(pair.satellite) + hybrid_metal_mass(pair.central),
        rtol=0.0,
        atol=2.0e-13,
    )


def test_fixed_merger_event_map_is_differentiable_and_conservative():
    target = initial_galaxy_state(
        ColdGas=2.0,
        HotGas=3.0,
        StellarMass=4.0,
        BulgeMass=1.0,
        BlackHoleMass=0.02,
    )

    def descendant_stars(source_stars):
        source = initial_galaxy_state(
            ColdGas=0.5,
            HotGas=1.0,
            StellarMass=source_stars,
            BlackHoleMass=0.01,
        )
        descendant, _ = apply_merger_ownership_event(source, target)
        return descendant.StellarMass + descendant.BulgeMass

    derivative = jax.grad(descendant_stars)(jnp.asarray(2.0, dtype=jnp.float32))
    np.testing.assert_allclose(derivative, 2.0, rtol=0.0, atol=0.0)

    source = initial_galaxy_state(
        ColdGas=0.5,
        HotGas=1.0,
        StellarMass=2.0,
        BlackHoleMass=0.01,
    )
    descendant, _ = apply_merger_ownership_event(source, target)
    np.testing.assert_allclose(
        float(
            descendant.ColdGas
            + descendant.HotGas
            + descendant.StellarMass
            + descendant.BlackHoleMass
        ),
        float(
            source.ColdGas
            + source.HotGas
            + source.StellarMass
            + source.BlackHoleMass
            + target.ColdGas
            + target.HotGas
            + target.StellarMass
            + target.BlackHoleMass
        ),
        rtol=0.0,
        atol=1.1e-6,
    )
