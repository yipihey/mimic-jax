"""Exact threshold semantics for SHARK's shipped reionisation models."""

from typing import Any, NamedTuple

import jax.numpy as jnp

Array = Any


class Sobacchi13ReionisationParameters(NamedTuple):
    """Parameters selected by upstream ``sample_lagos23.cfg``."""

    velocity_cut_km_per_s: Array
    reionisation_redshift: Array
    velocity_redshift_power: Array


def sobacchi13_reionisation_parameters(
    *,
    velocity_cut_km_per_s: float = 35.0,
    reionisation_redshift: float = 10.0,
    velocity_redshift_power: float = -0.2,
) -> Sobacchi13ReionisationParameters:
    """Construct the fiducial Lagos23 Sobacchi13 gate parameters."""

    return Sobacchi13ReionisationParameters(
        velocity_cut_km_per_s=jnp.asarray(velocity_cut_km_per_s, dtype=jnp.float64),
        reionisation_redshift=jnp.asarray(reionisation_redshift, dtype=jnp.float64),
        velocity_redshift_power=jnp.asarray(velocity_redshift_power, dtype=jnp.float64),
    )


def sobacchi13_velocity_threshold(redshift, parameters: Sobacchi13ReionisationParameters):
    """Return the cooling-suppression velocity threshold in km/s.

    The expression is real only after reionisation (``z < zcut``).  We expose
    the undefined pre-reionisation branch as NaN rather than hiding the domain.
    """

    redshift = jnp.asarray(redshift)
    post_reionisation = redshift < parameters.reionisation_redshift
    safe_redshift = jnp.minimum(redshift, parameters.reionisation_redshift)
    reionisation_factor = 1.0 - jnp.power(
        (1.0 + safe_redshift) / (1.0 + parameters.reionisation_redshift), 2.0
    )
    threshold = (
        parameters.velocity_cut_km_per_s
        * jnp.power(1.0 + safe_redshift, parameters.velocity_redshift_power)
        * jnp.power(reionisation_factor, 0.833)
    )
    return jnp.where(post_reionisation, threshold, jnp.nan)


def sobacchi13_reionised_halo(
    virial_velocity, redshift, parameters: Sobacchi13ReionisationParameters
):
    """Return whether upstream suppresses cooling for this halo."""

    threshold = sobacchi13_velocity_threshold(redshift, parameters)
    return (jnp.asarray(redshift) < parameters.reionisation_redshift) & (
        jnp.asarray(virial_velocity) < threshold
    )
