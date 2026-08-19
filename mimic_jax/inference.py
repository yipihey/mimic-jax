"""Small, explicit inference tools for differentiable mimic-jax observables.

The functions in this module deliberately stop short of being a general
Bayesian-inference framework.  They provide the pieces needed to turn an
already validated observable response into a transparent local calibration:

* a Gaussian likelihood in logarithmic observable space;
* a first-order emulator whose coefficients are physical elasticities;
* a maximum-likelihood solution and local Laplace covariance; and
* a simple random-walk Metropolis reference for checking that covariance.

The local emulator is a proposal mechanism, not a replacement for nonlinear
SAGE.  Scientific applications must validate proposed parameter changes with
the full model before using its curvature as an uncertainty estimate.
"""

from dataclasses import dataclass
from typing import Any, Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

Array = Any


@dataclass(frozen=True)
class LogGaussianObservation:
    """Positive observations with a covariance for their natural logarithms."""

    values: Array
    log_covariance: Array
    names: Tuple[str, ...]
    units: Tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        covariance = np.asarray(self.log_covariance, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Observation values must be a finite positive vector")
        if covariance.shape != (values.size, values.size):
            raise ValueError("log_covariance must be square and match the observations")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("log_covariance must be finite")
        if not np.allclose(covariance, covariance.T, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("log_covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
            raise ValueError("log_covariance must be positive definite")
        if len(self.names) != values.size or len(self.units) != values.size:
            raise ValueError("Observation names and units must match the observation vector")

    @property
    def log_values(self):
        return jnp.log(jnp.asarray(self.values, dtype=jnp.float64))

    @property
    def precision(self):
        return jnp.linalg.inv(jnp.asarray(self.log_covariance, dtype=jnp.float64))


@dataclass(frozen=True)
class LocalLogResponseEmulator:
    """First-order observable model built from fractional SAGE responses.

    If ``q_i = ln(theta_i / theta_i,fid)``, the emulator predicts

    ``ln O(q) = ln O_fid + E q``

    where ``E`` is the observable-by-parameter elasticity matrix.  This form
    preserves positivity and keeps every coefficient physically interpretable.
    """

    baseline_values: Array
    elasticities: Array
    fiducial_parameters: Array
    observable_names: Tuple[str, ...]
    parameter_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        baseline = np.asarray(self.baseline_values, dtype=np.float64)
        elasticities = np.asarray(self.elasticities, dtype=np.float64)
        parameters = np.asarray(self.fiducial_parameters, dtype=np.float64)
        if baseline.ndim != 1 or np.any(baseline <= 0.0) or not np.all(np.isfinite(baseline)):
            raise ValueError("baseline_values must be a finite positive vector")
        if parameters.ndim != 1 or np.any(parameters <= 0.0) or not np.all(np.isfinite(parameters)):
            raise ValueError("fiducial_parameters must be a finite positive vector")
        if elasticities.shape != (baseline.size, parameters.size):
            raise ValueError("elasticities must have shape observable x parameter")
        if not np.all(np.isfinite(elasticities)):
            raise ValueError("elasticities must be finite")
        if len(self.observable_names) != baseline.size:
            raise ValueError("observable_names must match baseline_values")
        if len(self.parameter_names) != parameters.size:
            raise ValueError("parameter_names must match fiducial_parameters")

    def log_values_from_log_ratios(self, log_parameter_ratios):
        ratios = jnp.asarray(log_parameter_ratios, dtype=jnp.float64)
        if ratios.shape != np.asarray(self.fiducial_parameters).shape:
            raise ValueError("log_parameter_ratios must match the parameter vector")
        return jnp.log(jnp.asarray(self.baseline_values)) + jnp.asarray(self.elasticities) @ ratios

    def values_from_log_ratios(self, log_parameter_ratios):
        return jnp.exp(self.log_values_from_log_ratios(log_parameter_ratios))

    def parameters_from_log_ratios(self, log_parameter_ratios):
        return jnp.asarray(self.fiducial_parameters) * jnp.exp(
            jnp.asarray(log_parameter_ratios, dtype=jnp.float64)
        )


@dataclass(frozen=True)
class LocalGaussianFit:
    """Maximum-likelihood point and local covariance in log-parameter space."""

    log_parameter_ratios: Array
    covariance: Array
    hessian: Array
    gradient_at_fiducial: Array
    chi_square_fiducial: float
    chi_square_best: float
    rank: int
    condition_number: float
    observable_count: int
    parameter_names: Tuple[str, ...]
    derivative_method: str = "JAX elasticity matrix; analytic Gaussian solve"

    @property
    def parameter_ratios(self):
        return jnp.exp(jnp.asarray(self.log_parameter_ratios))

    @property
    def one_sigma_log(self):
        return jnp.sqrt(jnp.diag(jnp.asarray(self.covariance)))

    @property
    def one_sigma_ratio_lower(self):
        return jnp.exp(jnp.asarray(self.log_parameter_ratios) - self.one_sigma_log)

    @property
    def one_sigma_ratio_upper(self):
        return jnp.exp(jnp.asarray(self.log_parameter_ratios) + self.one_sigma_log)


@dataclass(frozen=True)
class MetropolisResult:
    """Samples and diagnostics from a small random-walk Metropolis reference."""

    samples: Array
    log_probabilities: Array
    accepted: Array
    acceptance_fraction: float
    seed: int
    burn_in: int


def symmetric_log_standard_deviation(values, lower_errors, upper_errors=None):
    """Convert asymmetric linear error extents into a symmetric log scale.

    This is appropriate only when the supplied upper and lower extents are
    intended to define a Gaussian working likelihood.  It does not infer a
    missing covariance or turn systematic envelopes into independent errors.
    """

    values = np.asarray(values, dtype=np.float64)
    lower = np.asarray(lower_errors, dtype=np.float64)
    upper = lower if upper_errors is None else np.asarray(upper_errors, dtype=np.float64)
    if values.shape != lower.shape or values.shape != upper.shape:
        raise ValueError("Values and error extents must have matching shapes")
    if np.any(values <= 0.0) or np.any(lower < 0.0) or np.any(upper < 0.0):
        raise ValueError("Values must be positive and errors non-negative")
    if np.any(values - lower <= 0.0):
        raise ValueError("Lower error extents must not cross zero in log space")
    return 0.5 * (np.log(values + upper) - np.log(values - lower))


def independent_log_covariance(
    values,
    lower_errors,
    upper_errors=None,
    *,
    model_counts=None,
):
    """Build an explicitly diagonal working covariance in log-observable space.

    ``model_counts`` adds the usual ``1/N`` Poisson variance in log number
    density.  No sample-variance or observational-bin covariance is invented;
    callers must describe those omissions in scientific output.
    """

    standard_deviation = symmetric_log_standard_deviation(
        values,
        lower_errors,
        upper_errors,
    )
    variance = standard_deviation**2
    if model_counts is not None:
        counts = np.asarray(model_counts, dtype=np.float64)
        if counts.shape != variance.shape or np.any(counts <= 0.0):
            raise ValueError("model_counts must be positive and match the observations")
        variance = variance + 1.0 / counts
    return np.diag(variance)


def log_gaussian_chi_square(predicted_values, observation: LogGaussianObservation):
    """Return the Gaussian chi-square for positive predictions in log space."""

    predicted = jnp.asarray(predicted_values, dtype=jnp.float64)
    if predicted.shape != np.asarray(observation.values).shape:
        raise ValueError("Predictions must match the observation vector")
    residual = jnp.log(predicted) - observation.log_values
    return residual @ observation.precision @ residual


def local_log_posterior(
    log_parameter_ratios,
    emulator: LocalLogResponseEmulator,
    observation: LogGaussianObservation,
):
    """Unnormalised log likelihood for the local response emulator.

    No implicit prior is included.  Bounds or priors must be supplied by a
    higher-level application and reported explicitly.
    """

    values = emulator.values_from_log_ratios(log_parameter_ratios)
    return -0.5 * log_gaussian_chi_square(values, observation)


def fit_local_log_response(
    emulator: LocalLogResponseEmulator,
    observation: LogGaussianObservation,
    *,
    maximum_condition_number: float = 1.0e12,
) -> LocalGaussianFit:
    """Solve the local Gaussian maximum-likelihood problem exactly.

    A rank-deficient or numerically singular response is rejected rather than
    regularised with an unstated prior.  This makes the distinction between
    parameter influence and parameter identifiability executable.
    """

    baseline = jnp.asarray(emulator.baseline_values, dtype=jnp.float64)
    response = jnp.asarray(emulator.elasticities, dtype=jnp.float64)
    residual = observation.log_values - jnp.log(baseline)
    precision = observation.precision
    hessian = response.T @ precision @ response
    gradient = response.T @ precision @ residual
    hessian_numpy = np.asarray(hessian)
    rank = int(np.linalg.matrix_rank(hessian_numpy))
    parameter_count = hessian_numpy.shape[0]
    condition = float(np.linalg.cond(hessian_numpy))
    if rank != parameter_count or not np.isfinite(condition):
        raise np.linalg.LinAlgError(
            "The observable response does not identify all selected parameters; "
            "reduce the parameter set or supply and report an explicit prior"
        )
    if condition > maximum_condition_number:
        raise np.linalg.LinAlgError(
            f"The response Hessian condition number {condition:.3e} exceeds "
            f"the stated limit {maximum_condition_number:.3e}"
        )
    solution = jnp.linalg.solve(hessian, gradient)
    covariance = jnp.linalg.inv(hessian)
    chi_square_fiducial = float(log_gaussian_chi_square(emulator.baseline_values, observation))
    chi_square_best = float(
        log_gaussian_chi_square(emulator.values_from_log_ratios(solution), observation)
    )
    return LocalGaussianFit(
        log_parameter_ratios=solution,
        covariance=covariance,
        hessian=hessian,
        gradient_at_fiducial=-gradient,
        chi_square_fiducial=chi_square_fiducial,
        chi_square_best=chi_square_best,
        rank=rank,
        condition_number=condition,
        observable_count=np.asarray(observation.values).size,
        parameter_names=emulator.parameter_names,
    )


def random_walk_metropolis(
    log_probability: Callable[[Array], Array],
    initial,
    proposal_covariance,
    *,
    num_steps: int,
    burn_in: int = 0,
    seed: int = 0,
) -> MetropolisResult:
    """Run a deterministic-seed random-walk Metropolis reference chain.

    This intentionally simple sampler is for validating a low-dimensional
    local Gaussian result, not for production posterior exploration.
    """

    if num_steps <= 0 or burn_in < 0 or burn_in >= num_steps:
        raise ValueError("Require num_steps > burn_in >= 0")
    current = np.asarray(initial, dtype=np.float64)
    covariance = np.asarray(proposal_covariance, dtype=np.float64)
    if covariance.shape != (current.size, current.size):
        raise ValueError("proposal_covariance must match the initial vector")
    proposal_cholesky = np.linalg.cholesky(covariance)
    rng = np.random.default_rng(seed)
    samples = np.empty((num_steps, current.size), dtype=np.float64)
    probabilities = np.empty(num_steps, dtype=np.float64)
    accepted = np.zeros(num_steps, dtype=bool)
    current_probability = float(log_probability(jnp.asarray(current)))
    if not np.isfinite(current_probability):
        raise ValueError("Initial log probability must be finite")
    for index in range(num_steps):
        proposal = current + proposal_cholesky @ rng.normal(size=current.size)
        proposal_probability = float(log_probability(jnp.asarray(proposal)))
        log_acceptance = proposal_probability - current_probability
        if np.isfinite(proposal_probability) and np.log(rng.uniform()) < log_acceptance:
            current = proposal
            current_probability = proposal_probability
            accepted[index] = True
        samples[index] = current
        probabilities[index] = current_probability
    return MetropolisResult(
        samples=samples[burn_in:],
        log_probabilities=probabilities[burn_in:],
        accepted=accepted[burn_in:],
        acceptance_fraction=float(np.mean(accepted[burn_in:])),
        seed=seed,
        burn_in=burn_in,
    )


def validate_log_posterior_gradient(
    log_parameter_ratios,
    emulator: LocalLogResponseEmulator,
    observation: LogGaussianObservation,
    *,
    relative_steps: Sequence[float] = (1.0e-2, 3.0e-3, 1.0e-3),
):
    """Compare the JAX gradient with centered finite differences."""

    point = jnp.asarray(log_parameter_ratios, dtype=jnp.float64)
    function = lambda values: local_log_posterior(values, emulator, observation)
    automatic = jax.grad(function)(point)
    rows = []
    for step in relative_steps:
        if step <= 0.0:
            raise ValueError("Finite-difference steps must be positive")
        columns = []
        for index in range(point.size):
            direction = jnp.zeros_like(point).at[index].set(step)
            columns.append((function(point + direction) - function(point - direction)) / (2 * step))
        rows.append(jnp.stack(columns))
    finite = jnp.stack(rows)
    return automatic, finite, jnp.abs(finite - automatic[None, :])
