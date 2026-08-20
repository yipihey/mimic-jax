"""Reference and continuous forms of SHARK's gas reincorporation recipe."""

from typing import Any, NamedTuple

import jax.numpy as jnp

Array = Any


class ReincorporationParameters(NamedTuple):
    """Parameters selected by upstream ``sample_lagos23.cfg``."""

    timescale_normalization_gyr: Array
    halo_mass_normalization_msun_over_h: Array
    halo_mass_power: Array


def lagos23_reincorporation_parameters(
    *,
    timescale_normalization_gyr: float = 21.53,
    halo_mass_normalization_msun_over_h: float = 1.383e11,
    halo_mass_power: float = -2.339,
) -> ReincorporationParameters:
    """Construct the fiducial Lagos23 reincorporation parameters."""

    return ReincorporationParameters(
        timescale_normalization_gyr=jnp.asarray(timescale_normalization_gyr, dtype=jnp.float64),
        halo_mass_normalization_msun_over_h=jnp.asarray(
            halo_mass_normalization_msun_over_h, dtype=jnp.float64
        ),
        halo_mass_power=jnp.asarray(halo_mass_power, dtype=jnp.float64),
    )


def reincorporation_timescale(halo_mass, parameters: ReincorporationParameters):
    """Return the SHARK reincorporation timescale in Gyr."""

    return parameters.timescale_normalization_gyr * jnp.power(
        halo_mass / parameters.halo_mass_normalization_msun_over_h,
        parameters.halo_mass_power,
    )


def continuous_reincorporation_rate(
    ejected_gas,
    halo_mass,
    is_satellite,
    parameters: ReincorporationParameters,
):
    """Return the continuous ejected-to-hot flow in ``Msun/h/Gyr``.

    Upstream disables the recipe for satellites and for timescales above
    100 Gyr.  A zero timescale is an instantaneous transfer event rather than
    an infinite ODE rate, so this function returns zero for that branch; the
    reference finite map below applies the event exactly.
    """

    ejected_gas = jnp.asarray(ejected_gas)
    halo_mass = jnp.asarray(halo_mass)
    timescale = reincorporation_timescale(halo_mass, parameters)
    active = (
        (~jnp.asarray(is_satellite))
        & (ejected_gas > 0.0)
        & (timescale > 0.0)
        & (timescale <= 100.0)
    )
    safe_timescale = jnp.where(active, timescale, 1.0)
    return jnp.where(active, ejected_gas / safe_timescale, 0.0)


def reference_reincorporated_mass(
    ejected_gas,
    halo_mass,
    interval_gyr,
    is_satellite,
    parameters: ReincorporationParameters,
):
    """Return the exact finite transfer realized by upstream gas cooling.

    SHARK stores halo and baryon masses as C++ ``float``.  This reference map
    reproduces that storage boundary before applying
    ``Reincorporation::reincorporated_mass`` and its caller's source cap.
    """

    ejected_reference = jnp.asarray(jnp.asarray(ejected_gas, dtype=jnp.float32), dtype=jnp.float64)
    halo_reference = jnp.asarray(jnp.asarray(halo_mass, dtype=jnp.float32), dtype=jnp.float64)
    timescale = reincorporation_timescale(halo_reference, parameters)
    regular = ejected_reference / jnp.where(timescale != 0.0, timescale, 1.0)
    requested = jnp.where(timescale == 0.0, ejected_reference, regular * interval_gyr)
    requested = jnp.where(timescale > 100.0, 0.0, requested)
    requested = jnp.where(jnp.asarray(is_satellite), 0.0, requested)
    return jnp.minimum(jnp.maximum(requested, 0.0), ejected_reference)
