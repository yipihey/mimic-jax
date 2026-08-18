"""Differentiable summaries of already-evolved SAGE16 galaxy records."""

from typing import Sequence

import jax.numpy as jnp
from jax.scipy.special import ndtr


def soft_stellar_mass_bin_weights(
    stellar_mass,
    *,
    hubble_h: float,
    bin_edges: Sequence[float],
    bandwidth_dex: float,
):
    """Assign positive stellar masses to logarithmic bins with Gaussian CDFs.

    The familiar SAGE stellar-mass function uses hard 0.1-dex bins.  A hard
    histogram is constant almost everywhere with respect to a galaxy mass and
    therefore has no useful automatic derivative.  This function changes only
    the population estimator: it treats each galaxy as a narrow Gaussian in
    ``log10(M*/Msun)`` and integrates that density over the same bin edges.
    The SAGE16 evolution and its thresholds are untouched.
    """

    mass = jnp.asarray(stellar_mass)
    edges = jnp.asarray(bin_edges, dtype=jnp.float64)
    if mass.ndim != 1:
        raise ValueError("stellar_mass must be one-dimensional")
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must contain at least two values")
    if bandwidth_dex <= 0.0:
        raise ValueError("bandwidth_dex must be positive")
    if hubble_h <= 0.0:
        raise ValueError("hubble_h must be positive")

    positive = mass > 0.0
    safe_mass = jnp.where(positive, mass, 1.0)
    logarithmic_mass = jnp.log10(safe_mass * 1.0e10 / hubble_h)
    standardized = (edges[None, :] - logarithmic_mass[:, None]) / bandwidth_dex
    weights = ndtr(standardized[:, 1:]) - ndtr(standardized[:, :-1])
    return jnp.where(positive[:, None], weights, 0.0)


def soft_stellar_mass_function(
    stellar_mass,
    *,
    volume_mpc_over_h_cubed: float,
    hubble_h: float,
    bin_edges: Sequence[float],
    bandwidth_dex: float,
):
    """Return a differentiable finite-volume estimate of ``phi(M*)``.

    Units and volume normalization match :func:`stellar_mass_function`.  The
    bandwidth is explicit metadata and must be validated against both the hard
    histogram and finite parameter perturbations in any scientific use.
    """

    if volume_mpc_over_h_cubed <= 0.0:
        raise ValueError("volume_mpc_over_h_cubed must be positive")
    edges = jnp.asarray(bin_edges, dtype=jnp.float64)
    weights = soft_stellar_mass_bin_weights(
        stellar_mass,
        hubble_h=hubble_h,
        bin_edges=edges,
        bandwidth_dex=bandwidth_dex,
    )
    return jnp.sum(weights, axis=0) / volume_mpc_over_h_cubed * hubble_h**3 / jnp.diff(edges)
