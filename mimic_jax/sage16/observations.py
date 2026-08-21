"""Backward-compatible SAGE namespace for shared observational products."""

from mimic_jax.observations import (
    BALDRY2008_CHABRIER_SHIFT_DEX,
    BALDRY2008_COMPLETENESS_LOG_MASS,
    BALDRY2008_DOI,
    BinnedObservation,
    load_baldry2008_stellar_mass_function,
)

__all__ = [
    "BALDRY2008_CHABRIER_SHIFT_DEX",
    "BALDRY2008_COMPLETENESS_LOG_MASS",
    "BALDRY2008_DOI",
    "BinnedObservation",
    "load_baldry2008_stellar_mass_function",
]
