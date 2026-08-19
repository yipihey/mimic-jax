import numpy as np
import pytest

from mimic_jax.inference import (
    LocalLogResponseEmulator,
    LogGaussianObservation,
    fit_local_log_response,
    independent_log_covariance,
    local_log_posterior,
    random_walk_metropolis,
    symmetric_log_standard_deviation,
    validate_log_posterior_gradient,
)


def example_problem():
    baseline = np.asarray([10.0, 20.0, 30.0])
    response = np.asarray([[1.0, 0.2], [0.1, -0.8], [0.5, 0.7]])
    fiducial = np.asarray([2.0, 4.0])
    truth = np.asarray([0.08, -0.12])
    emulator = LocalLogResponseEmulator(
        baseline,
        response,
        fiducial,
        ("low", "middle", "high"),
        ("feedback", "reincorporation"),
    )
    observed = np.asarray(emulator.values_from_log_ratios(truth))
    observation = LogGaussianObservation(
        observed,
        np.diag(np.asarray([0.03, 0.04, 0.05]) ** 2),
        emulator.observable_names,
        ("density",) * 3,
    )
    return emulator, observation, truth


def test_log_error_conversion_and_model_count_variance_are_explicit():
    values = np.asarray([10.0, 20.0])
    errors = np.asarray([1.0, 2.0])
    standard_deviation = symmetric_log_standard_deviation(values, errors)
    expected = 0.5 * (np.log(values + errors) - np.log(values - errors))
    np.testing.assert_allclose(standard_deviation, expected)
    covariance = independent_log_covariance(values, errors, model_counts=[100, 25])
    np.testing.assert_allclose(np.diag(covariance), expected**2 + np.asarray([0.01, 0.04]))
    with pytest.raises(ValueError, match="cross zero"):
        symmetric_log_standard_deviation([1.0], [1.0])


def test_local_fit_recovers_known_log_parameter_ratios_and_covariance():
    emulator, observation, truth = example_problem()
    result = fit_local_log_response(emulator, observation)
    np.testing.assert_allclose(result.log_parameter_ratios, truth, rtol=1.0e-10, atol=1.0e-10)
    assert result.chi_square_best < 1.0e-20
    assert result.rank == 2
    assert np.all(np.asarray(result.one_sigma_ratio_lower) < np.exp(truth))
    assert np.all(np.asarray(result.one_sigma_ratio_upper) > np.exp(truth))


def test_rank_deficiency_is_reported_instead_of_hidden_by_a_prior():
    emulator = LocalLogResponseEmulator(
        np.asarray([1.0, 2.0]),
        np.asarray([[1.0, 1.0], [2.0, 2.0]]),
        np.ones(2),
        ("a", "b"),
        ("one", "two"),
    )
    observation = LogGaussianObservation(np.asarray([1.1, 2.2]), np.eye(2), ("a", "b"), ("", ""))
    with pytest.raises(np.linalg.LinAlgError, match="does not identify"):
        fit_local_log_response(emulator, observation)


def test_jax_likelihood_gradient_matches_centered_finite_difference():
    emulator, observation, truth = example_problem()
    automatic, finite, error = validate_log_posterior_gradient(
        truth + np.asarray([0.02, -0.01]),
        emulator,
        observation,
        relative_steps=(1.0e-2, 1.0e-3, 1.0e-4),
    )
    np.testing.assert_allclose(finite[-1], automatic, rtol=2.0e-6, atol=2.0e-6)
    assert float(np.max(error)) < 1.0e-8


def test_reference_metropolis_chain_matches_local_gaussian_moments():
    emulator, observation, _ = example_problem()
    fit = fit_local_log_response(emulator, observation)
    chain = random_walk_metropolis(
        lambda values: local_log_posterior(values, emulator, observation),
        fit.log_parameter_ratios,
        0.5 * np.asarray(fit.covariance),
        num_steps=30_000,
        burn_in=3_000,
        seed=17,
    )
    assert 0.2 < chain.acceptance_fraction < 0.8
    np.testing.assert_allclose(np.mean(chain.samples, axis=0), fit.log_parameter_ratios, atol=0.02)
    np.testing.assert_allclose(
        np.cov(chain.samples, rowvar=False), fit.covariance, rtol=0.18, atol=2.0e-4
    )
