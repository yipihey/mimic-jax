"""Differentiable population-estimator tests."""

import jax
import jax.numpy as jnp
import numpy as np

from mimic_jax.sage16 import (
    soft_stellar_mass_bin_weights,
    soft_stellar_mass_function,
)


def test_soft_bin_weights_approach_hard_bins_away_from_edges():
    hubble_h = 0.73
    logarithmic_mass = jnp.asarray([8.25, 8.75, 9.25])
    stellar_mass = 10.0**logarithmic_mass * hubble_h / 1.0e10
    edges = jnp.asarray([8.0, 8.5, 9.0, 9.5])
    weights = soft_stellar_mass_bin_weights(
        stellar_mass,
        hubble_h=hubble_h,
        bin_edges=edges,
        bandwidth_dex=0.02,
    )

    np.testing.assert_allclose(weights, np.eye(3), rtol=0.0, atol=1.0e-12)


def test_soft_mass_function_jvp_matches_symmetric_difference():
    masses = jnp.asarray([0.05, 0.1, 0.2, 0.4], dtype=jnp.float64)
    tangent = jnp.asarray([0.01, -0.02, 0.03, -0.01], dtype=jnp.float64)
    edges = jnp.arange(8.0, 10.1, 0.2)

    def observable(values):
        return soft_stellar_mass_function(
            values,
            volume_mpc_over_h_cubed=1000.0,
            hubble_h=0.73,
            bin_edges=edges,
            bandwidth_dex=0.05,
        )

    _, automatic = jax.jvp(observable, (masses,), (tangent,))
    step = 1.0e-5
    finite_difference = (
        observable(masses + step * tangent) - observable(masses - step * tangent)
    ) / (2.0 * step)
    np.testing.assert_allclose(automatic, finite_difference, rtol=2.0e-7, atol=2.0e-10)
