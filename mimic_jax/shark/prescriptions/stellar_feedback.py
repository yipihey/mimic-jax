"""Fiducial Lagos13 stellar-feedback loading factors from upstream SHARK."""

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

Array = Any

_SECONDS_PER_GYR = 3.15576e16
_METRES_PER_MPC = 3.0856775807e22
_KILOMETRES_IN_METRES = 1.0e3
_HUBBLE_100_PER_GYR = 100.0 * _KILOMETRES_IN_METRES * _SECONDS_PER_GYR / _METRES_PER_MPC
_STRICT_LOADING_OFFSET = 1.0e-3


class Lagos13FeedbackParameters(NamedTuple):
    """Parameters selected by upstream ``sample_lagos23.cfg``."""

    hubble_h: Array
    omega_m: Array
    beta_disk: Array
    velocity_normalization_km_per_s: Array
    halo_efficiency: Array
    disk_efficiency: Array
    redshift_power: Array
    minimum_loading: Array
    galaxy_velocity_scaling: Array
    radial_feedback: Array


class StellarFeedbackLoadings(NamedTuple):
    """Mass and angular-momentum loadings multiplying the SFR rates."""

    reheating: Array
    ejection: Array
    angular_momentum_reheating: Array
    angular_momentum_ejection: Array


def lagos13_feedback_parameters(
    *,
    hubble_h: float = 0.6751,
    omega_m: float = 0.3121,
    beta_disk: float = 3.79746174188,
    velocity_normalization_km_per_s: float = 120.0,
    halo_efficiency: float = 2.0,
    disk_efficiency: float = 1.0,
    redshift_power: float = 0.12,
    minimum_loading: float = 0.104050197191,
    galaxy_velocity_scaling: bool = False,
    radial_feedback: bool = False,
) -> Lagos13FeedbackParameters:
    """Construct the fiducial Lagos13 feedback parameter set."""

    return Lagos13FeedbackParameters(
        # ``CosmologicalParameters`` stores these as C++ ``float`` upstream.
        hubble_h=jnp.asarray(np.float32(hubble_h), dtype=jnp.float64),
        omega_m=jnp.asarray(np.float32(omega_m), dtype=jnp.float64),
        beta_disk=jnp.asarray(beta_disk, dtype=jnp.float64),
        velocity_normalization_km_per_s=jnp.asarray(
            velocity_normalization_km_per_s, dtype=jnp.float64
        ),
        halo_efficiency=jnp.asarray(halo_efficiency, dtype=jnp.float64),
        disk_efficiency=jnp.asarray(disk_efficiency, dtype=jnp.float64),
        redshift_power=jnp.asarray(redshift_power, dtype=jnp.float64),
        minimum_loading=jnp.asarray(minimum_loading, dtype=jnp.float64),
        galaxy_velocity_scaling=jnp.asarray(galaxy_velocity_scaling, dtype=jnp.bool_),
        radial_feedback=jnp.asarray(radial_feedback, dtype=jnp.bool_),
    )


def _flat_universe_age(redshift, parameters: Lagos13FeedbackParameters):
    """Match ``Cosmology::convert_redshift_to_age`` for flat Lambda CDM."""

    scale_factor = 1.0 / (1.0 + redshift)
    hubble_time_100 = 1.0 / _HUBBLE_100_PER_GYR
    # In upstream Cosmology, OmegaM and Hubble_h are ``float``.  The literal
    # ``3`` therefore leaves this denominator in float precision, while the
    # literal ``1.0`` in the asinh argument promotes OmegaM to double.
    omega_m_float = jnp.asarray(parameters.omega_m, dtype=jnp.float32)
    hubble_h_float = jnp.asarray(parameters.hubble_h, dtype=jnp.float32)
    denominator_float = (
        jnp.float32(3.0) * hubble_h_float * jnp.sqrt(jnp.float32(1.0) - omega_m_float)
    )
    coefficient_float = jnp.float32(2.0) / denominator_float
    return (
        hubble_time_100
        * jnp.asarray(coefficient_float, dtype=jnp.float64)
        * jnp.arcsinh(
            jnp.sqrt((1.0 / jnp.asarray(omega_m_float, dtype=jnp.float64) - 1.0) * scale_factor)
            * scale_factor
        )
    )


def lagos13_feedback_loadings(
    star_formation_rate,
    subhalo_velocity,
    galaxy_velocity,
    redshift,
    parameters: Lagos13FeedbackParameters,
) -> StellarFeedbackLoadings:
    """Return upstream Lagos13 reheating/ejection loadings.

    The pinned upstream branch accepts the galaxy redshift but evaluates the
    cosmological age at ``parameters.redshift_power``.  This makes the Lagos13
    loading independent of the supplied galaxy redshift.  mimic-jax preserves
    that behavior exactly for reference equivalence and records it explicitly;
    any alternative interpretation must be a separately named experiment.
    """

    del redshift
    star_formation_rate = jnp.asarray(star_formation_rate)
    subhalo_velocity = jnp.asarray(subhalo_velocity)
    galaxy_velocity = jnp.asarray(galaxy_velocity)
    velocity = jnp.where(
        parameters.galaxy_velocity_scaling & (galaxy_velocity > 0.0),
        galaxy_velocity,
        subhalo_velocity,
    )
    active = (star_formation_rate > 0.0) & (velocity > 0.0)
    safe_velocity = jnp.where(active, velocity, 1.0)

    # This deliberately follows the pinned upstream implementation, including
    # its age(redshift_power)**redshift_power normalization.
    universe_age = _flat_universe_age(parameters.redshift_power, parameters)
    hot_velocity = parameters.velocity_normalization_km_per_s * jnp.power(
        universe_age, parameters.redshift_power
    )
    supernova_constant = jnp.power(hot_velocity / safe_velocity, parameters.beta_disk)
    reheating = parameters.disk_efficiency * supernova_constant
    floored = reheating < parameters.minimum_loading
    reheating = jnp.maximum(reheating, parameters.minimum_loading)
    supernova_constant = jnp.where(
        floored, reheating / parameters.disk_efficiency, supernova_constant
    )

    supernova_velocity = 1.9 * jnp.power(safe_velocity, 1.1)
    halo_energy = parameters.halo_efficiency * supernova_constant * 0.5 * supernova_velocity**2
    halo_binding_energy = 0.5 * safe_velocity**2
    candidate_ejection = halo_energy / halo_binding_energy - reheating
    positive_ejection = jnp.maximum(candidate_ejection, 0.0)
    capped = positive_ejection > reheating
    ejection = jnp.where(capped, reheating, positive_ejection)
    reheating = jnp.where(capped, reheating + _STRICT_LOADING_OFFSET, reheating)

    reheating = jnp.where(active, reheating, 0.0)
    ejection = jnp.where(active, ejection, 0.0)
    angular_reheating = jnp.where(parameters.radial_feedback, 0.0, reheating)
    angular_ejection = jnp.where(parameters.radial_feedback, 0.0, ejection)
    return StellarFeedbackLoadings(
        reheating=reheating,
        ejection=ejection,
        angular_momentum_reheating=angular_reheating,
        angular_momentum_ejection=angular_ejection,
    )
