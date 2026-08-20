"""Closed fiducial disk rate layer under explicitly fixed halo forcing."""

from typing import Any, NamedTuple

import jax.numpy as jnp

from mimic_jax.shark.prescriptions.star_formation import (
    Lagos23StarFormationParameters,
    lagos23_br06_star_formation,
)
from mimic_jax.shark.prescriptions.stellar_feedback import (
    Lagos13FeedbackParameters,
    lagos13_feedback_loadings,
)
from mimic_jax.shark.types import SharkFlowRates, SharkState

Array = Any

_DISK_HALF_MASS_TO_SCALE_LENGTH = 1.678346990


class Lagos23DiskForcing(NamedTuple):
    """Halo/structural quantities held outside the 19-variable flow state."""

    gas_half_mass_radius: Array
    stellar_half_mass_radius: Array
    redshift: Array
    burst: Array
    galaxy_velocity: Array
    subhalo_velocity: Array
    cooling_rate: Array
    cooling_metallicity: Array
    cooling_specific_angular_momentum: Array
    qso_reheating_loading: Array
    qso_ejection_loading: Array


def lagos23_disk_forcing(
    *,
    gas_half_mass_radius: float,
    stellar_half_mass_radius: float,
    redshift: float,
    burst: bool = False,
    galaxy_velocity: float,
    subhalo_velocity: float,
    cooling_rate: float = 0.0,
    cooling_metallicity: float = 0.0,
    cooling_specific_angular_momentum: float = 0.0,
    qso_reheating_loading: float = 0.0,
    qso_ejection_loading: float = 0.0,
) -> Lagos23DiskForcing:
    """Construct fixed forcing for a controlled Lagos23 disk interval."""

    return Lagos23DiskForcing(
        gas_half_mass_radius=jnp.asarray(gas_half_mass_radius, dtype=jnp.float64),
        stellar_half_mass_radius=jnp.asarray(stellar_half_mass_radius, dtype=jnp.float64),
        redshift=jnp.asarray(redshift, dtype=jnp.float64),
        burst=jnp.asarray(burst, dtype=jnp.bool_),
        galaxy_velocity=jnp.asarray(galaxy_velocity, dtype=jnp.float64),
        subhalo_velocity=jnp.asarray(subhalo_velocity, dtype=jnp.float64),
        cooling_rate=jnp.asarray(cooling_rate, dtype=jnp.float64),
        cooling_metallicity=jnp.asarray(cooling_metallicity, dtype=jnp.float64),
        cooling_specific_angular_momentum=jnp.asarray(
            cooling_specific_angular_momentum, dtype=jnp.float64
        ),
        qso_reheating_loading=jnp.asarray(qso_reheating_loading, dtype=jnp.float64),
        qso_ejection_loading=jnp.asarray(qso_ejection_loading, dtype=jnp.float64),
    )


def lagos23_disk_flow_rates(
    time,
    state: SharkState,
    forcing: Lagos23DiskForcing,
    star_formation_parameters: Lagos23StarFormationParameters,
    feedback_parameters: Lagos13FeedbackParameters,
) -> SharkFlowRates:
    """Evaluate the implemented Lagos23 disk prescriptions as named rates.

    This is an actual nonlinear SHARK rate layer, but not yet the complete
    population model.  Cooling/AGN preparation, sizes, and halo quantities are
    explicit fixed forcing during the interval.  Upstream's finite preparation
    maps remain outside this function until their direct oracles pass.
    """

    del time
    positive_cold_gas = state.cold_gas > 0.0
    safe_cold_gas = jnp.where(positive_cold_gas, state.cold_gas, 1.0)
    gas_metallicity = jnp.where(positive_cold_gas, state.cold_gas_metals / safe_cold_gas, 0.0)
    geometric_specific_angular_momentum = (
        2.0
        * forcing.galaxy_velocity
        * forcing.gas_half_mass_radius
        / _DISK_HALF_MASS_TO_SCALE_LENGTH
    )
    # Upstream initializes ``jgas`` from the disk geometry, but replaces it
    # by the evolved extensive-AM ratio only on the same branch on which it
    # measures a non-floor cold-gas metallicity.
    measured_specific_angular_momentum = jnp.where(
        positive_cold_gas,
        state.cold_gas_angular_momentum / safe_cold_gas,
        0.0,
    )
    gas_specific_angular_momentum = jnp.where(
        positive_cold_gas & (state.cold_gas_metals > 0.0),
        measured_specific_angular_momentum,
        geometric_specific_angular_momentum,
    )
    star_formation = lagos23_br06_star_formation(
        state.cold_gas,
        state.stellar_mass,
        forcing.gas_half_mass_radius,
        forcing.stellar_half_mass_radius,
        gas_metallicity,
        forcing.redshift,
        forcing.burst,
        forcing.galaxy_velocity,
        gas_specific_angular_momentum,
        star_formation_parameters,
    )
    feedback = lagos13_feedback_loadings(
        star_formation.mass,
        forcing.subhalo_velocity,
        forcing.galaxy_velocity,
        forcing.redshift,
        feedback_parameters,
    )
    return SharkFlowRates(
        cooling=forcing.cooling_rate,
        star_formation=star_formation.mass,
        star_formation_angular_momentum=star_formation.angular_momentum,
        stellar_reheating_loading=feedback.reheating,
        stellar_ejection_loading=feedback.ejection,
        angular_momentum_reheating_loading=feedback.angular_momentum_reheating,
        angular_momentum_ejection_loading=feedback.angular_momentum_ejection,
        qso_reheating_loading=forcing.qso_reheating_loading,
        qso_ejection_loading=forcing.qso_ejection_loading,
        cooling_metallicity=forcing.cooling_metallicity,
        cooling_specific_angular_momentum=forcing.cooling_specific_angular_momentum,
    )
