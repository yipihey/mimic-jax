"""Tests for the separately labelled minimal SAGE16 teacher--student model."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16 import (
    ReducedForcing,
    ReducedParameters,
    StaticEfficiencyParameters,
    add_cosmological_infall,
    apply_reduced_merger_event,
    cooling_efficiency,
    evolve_reduced_interval,
    initial_reduced_state,
    merge_reduced_states,
    reduced_baryonic_mass,
    static_stellar_mass,
)
from mimic_jax.sage16.reduced import _evolve_reduced_interval_numpy
from mimic_jax.sensitivity import parameter_response_matrix, validate_parameter_response

jax.config.update("jax_enable_x64", True)


def reduced_parameters():
    return ReducedParameters(
        StarFormationTimescaleGyr=jnp.asarray(0.1),
        CoolingTimescaleGyr=jnp.asarray(1.0),
        FeedbackMassLoadingAtPivot=jnp.asarray(2.0),
        FeedbackHaloMassSlope=jnp.asarray(1.0),
        QuenchingHaloMass=jnp.asarray(50.0),
        QuenchingSlope=jnp.asarray(2.0),
        ColdGasThresholdPerSpin=jnp.asarray(0.5),
        CoolingRedshiftExponent=jnp.asarray(1.5),
        BlackHoleQuenchingMass=jnp.asarray(1.0e-3),
    )


def test_infall_local_evolution_and_merger_preserve_explicit_mass_budget():
    first = add_cosmological_infall(initial_reduced_state(), 10.0)
    forcing = ReducedForcing(
        HaloMass=jnp.asarray(100.0),
        SpinMagnitude=jnp.asarray(1.0),
        OnePlusRedshift=jnp.asarray(2.0),
    )
    evolved, diagnostics = evolve_reduced_interval(
        first, forcing, reduced_parameters(), 0.8, substeps=4
    )
    np.testing.assert_allclose(reduced_baryonic_mass(evolved), 10.0, rtol=0.0, atol=2.0e-15)
    assert diagnostics.CooledMass > 0.0
    assert diagnostics.LockedStellarMass > 0.0
    assert diagnostics.ReheatedMass > 0.0
    assert diagnostics.StarFormationRate >= 0.0
    assert all(float(value) >= 0.0 for value in evolved)

    second = initial_reduced_state(CircumgalacticGas=1.0, ColdGas=2.0, StellarMass=3.0)
    merged = merge_reduced_states(evolved, second)
    np.testing.assert_allclose(reduced_baryonic_mass(merged), 16.0, rtol=0.0, atol=3.0e-15)

    post_merger, black_hole_growth = apply_reduced_merger_event(merged, 0.3)
    assert black_hole_growth > 0.0
    np.testing.assert_allclose(reduced_baryonic_mass(post_merger), 16.0, rtol=0.0, atol=3.0e-15)


def test_high_mass_quenching_reduces_cooling_without_changing_the_physics_api():
    parameters = reduced_parameters()
    low_mass = ReducedForcing(10.0, 1.0, 1.0)
    high_mass = ReducedForcing(1000.0, 1.0, 1.0)
    assert cooling_efficiency(high_mass, parameters) < cooling_efficiency(low_mass, parameters)


def test_numpy_tree_fitting_backend_matches_jax_kernel():
    state = initial_reduced_state(CircumgalacticGas=4.0, ColdGas=1.0, StellarMass=0.5)
    forcing = ReducedForcing(80.0, 0.7, 1.8)
    parameters = reduced_parameters()
    jax_result = evolve_reduced_interval(state, forcing, parameters, 0.4, substeps=3)
    numpy_result = _evolve_reduced_interval_numpy(state, forcing, parameters, 0.4, substeps=3)
    np.testing.assert_allclose(jax_result[0], numpy_result[0], rtol=2.0e-15, atol=2.0e-15)
    np.testing.assert_allclose(jax_result[1], numpy_result[1], rtol=2.0e-15, atol=2.0e-15)


def test_reduced_model_is_jittable_vectorizable_and_differentiable():
    forcing = ReducedForcing(
        HaloMass=jnp.asarray([10.0, 100.0, 1000.0]),
        SpinMagnitude=jnp.asarray([0.2, 0.5, 0.8]),
        OnePlusRedshift=jnp.asarray([1.0, 2.0, 3.0]),
    )
    state = initial_reduced_state(
        CircumgalacticGas=jnp.asarray([2.0, 5.0, 20.0]),
        ColdGas=jnp.asarray([0.5, 1.0, 2.0]),
    )

    def observable(parameters):
        result, _ = evolve_reduced_interval(state, forcing, parameters, 0.5)
        return jnp.sum(result.StellarMass)

    def final_stellar_mass(cooling_timescale):
        parameters = reduced_parameters()._replace(CoolingTimescaleGyr=cooling_timescale)
        return observable(parameters)

    value = jax.jit(final_stellar_mass)(jnp.asarray(1.0))
    derivative = jax.grad(final_stellar_mass)(jnp.asarray(1.0))
    assert np.isfinite(value)
    assert np.isfinite(derivative)
    assert derivative < 0.0

    response = parameter_response_matrix(
        observable,
        reduced_parameters(),
        parameter_names=("CoolingTimescaleGyr",),
        observable_names=("StellarMass",),
    )
    validation = validate_parameter_response(
        response,
        observable,
        reduced_parameters(),
        relative_steps=(1.0e-2, 3.0e-3, 1.0e-3),
    )
    assert response.values[0, 0] < 0.0
    finite_difference_error = np.asarray(validation.absolute_error[:, 0, 0])
    assert finite_difference_error[-1] < finite_difference_error[0]
    np.testing.assert_allclose(
        validation.finite_difference[-1],
        response.values,
        rtol=1.0e-6,
        atol=2.0e-8,
    )


def test_static_efficiency_baseline_is_positive_and_peaks_near_its_mass_scale():
    parameters = StaticEfficiencyParameters(100.0, 0.2, 1.0, 0.5)
    masses = jnp.asarray([1.0, 100.0, 10000.0])
    stellar_mass = static_stellar_mass(masses, parameters)
    efficiency = stellar_mass / masses
    assert np.all(np.asarray(stellar_mass) > 0.0)
    assert efficiency[1] > efficiency[0]
    assert efficiency[1] > efficiency[2]
