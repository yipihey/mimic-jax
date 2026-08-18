"""Fractional parameter and finite-epoch process response tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mimic_jax import (
    LOG_ELASTICITY,
    PROCESS_LOG_RESPONSE,
    REFERENCE_SCALE,
    InvalidNormalizationError,
    finite_epoch_magnitude_weights,
    parameter_response_matrix,
    process_response_tensor,
    redshift_from_ln_scale_factor,
    response_similarity,
    uniform_ln_scale_factor_edges,
    validate_parameter_response,
    validate_process_response,
)
from mimic_jax.sage16 import (
    PROCESS_NAMES,
    evolve_central_history,
    fiducial_parameters,
    initial_galaxy_state,
    initial_halo_forcing,
    perturbations_from_matrix,
    quiescent_disk_step,
    sage16_units,
    step_context,
)


def _single_step_observable(parameters):
    state = initial_galaxy_state(
        ColdGas=10.0,
        HotGas=5.0,
        StellarMass=2.0,
        DiskScaleRadius=0.01,
    )
    halo = initial_halo_forcing(Vvir=150.0, dT=1.0e-4)
    result = quiescent_disk_step(
        state,
        state,
        halo,
        halo,
        step_context(time_interval=1.0e-4),
        parameters,
        sage16_units(),
    )
    return jnp.asarray([result.galaxy.StellarMass, result.galaxy.ColdGas])


def _stack_record(record, count):
    return jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (count,) + value.shape),
        record,
    )


def _history_case(num_epochs=4):
    state = initial_galaxy_state(
        ColdGas=5.0,
        HotGas=8.0,
        EjectedGas=2.0,
        StellarMass=1.0,
        DiskScaleRadius=0.01,
    )
    halo = initial_halo_forcing(Vvir=150.0, Rvir=0.2, dT=1.0e-4)
    context = step_context(time_interval=1.0e-4)
    return (
        state,
        _stack_record(halo, num_epochs),
        _stack_record(context, num_epochs),
        jnp.asarray([0.20, 0.25, 0.30, 0.35], dtype=jnp.float64),
        fiducial_parameters(),
        sage16_units(),
    )


def test_parameter_elasticity_has_physical_normalization_and_fd_validation():
    parameters = fiducial_parameters()
    response = parameter_response_matrix(
        _single_step_observable,
        parameters,
        parameter_names=("SfrEfficiency", "FeedbackReheatingEpsilon"),
        observable_names=("stellar_mass", "cold_gas"),
        observable_units=("1e10 Msun/h", "1e10 Msun/h"),
        parameter_units=("dimensionless", "dimensionless"),
    )

    assert response.normalization == LOG_ELASTICITY
    assert response.values.shape == (2, 2)
    np.testing.assert_allclose(
        response.values,
        response.raw_derivatives
        * response.parameter_values[jnp.newaxis, :]
        / response.observable_values[:, jnp.newaxis],
        rtol=1.0e-13,
    )
    validation = validate_parameter_response(response, _single_step_observable, parameters)
    assert validation.finite_difference.shape == (3, 2, 2)
    np.testing.assert_allclose(
        validation.finite_difference[0],
        response.values,
        rtol=8.0e-3,
        atol=2.0e-5,
    )


def test_invalid_log_normalization_is_explicit_and_reference_scales_are_supported(tmp_path):
    parameters = fiducial_parameters()._replace(FracZleaveDisk=jnp.asarray(0.0))

    with pytest.raises(InvalidNormalizationError):
        parameter_response_matrix(
            lambda current: jnp.asarray([current.FracZleaveDisk]),
            parameters,
            parameter_names=("FracZleaveDisk",),
        )

    masked = parameter_response_matrix(
        lambda current: jnp.asarray([current.FracZleaveDisk]),
        parameters,
        parameter_names=("FracZleaveDisk",),
        invalid="mask",
    )
    assert not bool(masked.valid[0, 0])
    assert bool(jnp.isnan(masked.values[0, 0]))

    referenced = parameter_response_matrix(
        lambda current: jnp.asarray([current.FracZleaveDisk]),
        parameters,
        parameter_names=("FracZleaveDisk",),
        normalization=REFERENCE_SCALE,
        observable_scales=jnp.asarray([0.1]),
        parameter_scales=jnp.asarray([0.1]),
    )
    assert referenced.normalization == REFERENCE_SCALE
    assert float(referenced.values[0, 0]) == 1.0
    np.testing.assert_array_equal(referenced.observable_scales, [0.1])
    np.testing.assert_array_equal(referenced.parameter_scales, [0.1])

    archive = tmp_path / "response.npz"
    referenced.save(archive)
    with np.load(archive) as saved:
        assert saved["normalization"] == REFERENCE_SCALE
        np.testing.assert_array_equal(saved["observable_scales"], [0.1])
        np.testing.assert_array_equal(saved["sign"], [[1.0]])


def test_discrete_parameters_are_rejected_from_derivative_axis():
    with pytest.raises(TypeError):
        parameter_response_matrix(
            lambda current: jnp.asarray([current.SfrEfficiency]),
            fiducial_parameters(),
            parameter_names=("AGNrecipe",),
        )


def test_parameter_response_fingerprint_similarity_is_bounded():
    response = parameter_response_matrix(
        _single_step_observable,
        fiducial_parameters(),
        parameter_names=("SfrEfficiency", "FeedbackReheatingEpsilon"),
    )
    similarity, valid = response_similarity(response)
    assert bool(jnp.all(valid))
    np.testing.assert_allclose(jnp.diag(similarity), 1.0, rtol=1.0e-13)
    assert bool(jnp.all(jnp.abs(similarity) <= 1.0 + 1.0e-12))


def test_history_scan_is_jittable_and_exposes_named_process_transfers():
    state, halos, contexts, cooling, parameters, units = _history_case()
    eager = evolve_central_history(state, halos, contexts, cooling, parameters, units)
    compiled = jax.jit(evolve_central_history)(state, halos, contexts, cooling, parameters, units)

    np.testing.assert_allclose(
        compiled.final_state.StellarMass,
        eager.final_state.StellarMass,
        rtol=0.0,
        atol=0.0,
    )
    assert eager.states.StellarMass.shape == (4,)
    assert eager.diagnostics.cooling.gas.shape == (4,)
    assert eager.diagnostics.star_formation.locked_stars.shape == (4,)


def test_finite_epoch_process_response_is_dimensionless_and_fd_validated():
    state, halos, contexts, cooling, parameters, units = _history_case()
    edges = uniform_ln_scale_factor_edges(4.0, 0.0, 4)

    def observables(epsilon):
        perturbations = perturbations_from_matrix(epsilon)
        history = evolve_central_history(
            state,
            halos,
            contexts,
            cooling,
            parameters,
            units,
            perturbations,
        )
        return jnp.asarray([history.final_state.StellarMass, history.final_state.ColdGas])

    baseline = evolve_central_history(state, halos, contexts, cooling, parameters, units)
    references = jnp.stack(
        [
            baseline.diagnostics.cooling.gas,
            baseline.diagnostics.star_formation.formed_stars,
            baseline.diagnostics.star_formation.cold_to_hot,
            baseline.diagnostics.star_formation.hot_to_ejected,
            baseline.diagnostics.reincorporation.gas,
            jnp.zeros_like(baseline.diagnostics.cooling.gas),
        ]
    )
    response = process_response_tensor(
        observables,
        process_names=PROCESS_NAMES,
        ln_scale_factor_edges=edges,
        observable_names=("final_stellar_mass", "final_cold_gas"),
        observable_units=("1e10 Msun/h", "1e10 Msun/h"),
        process_reference_values=references,
    )

    assert response.normalization == PROCESS_LOG_RESPONSE
    assert response.values.shape == (2, len(PROCESS_NAMES), 4)
    np.testing.assert_allclose(response.redshift_edges, [4.0, 2.3437015, 1.236068, 0.4953488, 0.0])
    validation = validate_process_response(
        response,
        observables,
        log_rate_steps=(1.0e-2,),
    )
    np.testing.assert_allclose(
        validation.finite_difference[0],
        response.values,
        rtol=3.0e-2,
        atol=3.0e-5,
    )
    weights, valid = finite_epoch_magnitude_weights(response)
    np.testing.assert_allclose(jnp.sum(weights[valid], axis=-1), 1.0, rtol=1.0e-13)


def test_ln_scale_factor_edges_round_trip_to_decreasing_redshift():
    edges = uniform_ln_scale_factor_edges(8.0, 0.0, 8)
    redshift = redshift_from_ln_scale_factor(edges)

    assert np.isclose(float(redshift[0]), 8.0)
    assert np.isclose(float(redshift[-1]), 0.0)
    assert bool(jnp.all(jnp.diff(edges) > 0.0))
    assert bool(jnp.all(jnp.diff(redshift) < 0.0))
