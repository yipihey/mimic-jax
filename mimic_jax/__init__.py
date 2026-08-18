"""JAX-native physics kernels that preserve MIMIC model semantics."""

from mimic_jax import sage16
from mimic_jax.sensitivity import (
    LOG_ELASTICITY,
    PROCESS_LOG_RESPONSE,
    REFERENCE_SCALE,
    HistoricalProcessResponse,
    InvalidNormalizationError,
    ParameterResponseMatrix,
    finite_epoch_magnitude_weights,
    ln_scale_factor,
    parameter_response_matrix,
    process_response_tensor,
    redshift_from_ln_scale_factor,
    response_similarity,
    uniform_ln_scale_factor_edges,
    validate_parameter_response,
    validate_process_response,
)

__all__ = [
    "LOG_ELASTICITY",
    "PROCESS_LOG_RESPONSE",
    "REFERENCE_SCALE",
    "HistoricalProcessResponse",
    "InvalidNormalizationError",
    "ParameterResponseMatrix",
    "finite_epoch_magnitude_weights",
    "ln_scale_factor",
    "parameter_response_matrix",
    "process_response_tensor",
    "redshift_from_ln_scale_factor",
    "response_similarity",
    "sage16",
    "uniform_ln_scale_factor_edges",
    "validate_parameter_response",
    "validate_process_response",
]
