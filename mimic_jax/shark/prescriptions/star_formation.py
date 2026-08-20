"""Lagos23 BR06 molecular star-formation prescription.

This is a direct JAX expression of ``StarFormation::star_formation_rate`` and
its BR06 surface-density branch in upstream SHARK.  SHARK uses adaptive GSL
quadrature internally; mimic-jax uses deterministic Gauss--Legendre quadrature
so that the prescription can be jitted, vectorized, and differentiated.  Its
quadrature error is validated separately from baryonic time-integration error.
"""

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

Array = Any

_PI = 3.14159265358979323846
_TWO_PI = 2.0 * _PI
_G_SI = 6.67259e-11
_SOLAR_MASS_KG = 1.9891e30
_METRES_PER_MPC = 3.0856775807e22
_CENTIMETRES_PER_MPC = _METRES_PER_MPC * 1.0e2
_KILOMETRES_IN_METRES = 1.0e3
_SOLAR_MASS_G = _SOLAR_MASS_KG * 1.0e3
_BOLTZMANN_ERG = 1.3806503e-23 * 1.0e7
_GRAVITATIONAL_CONSTANT = _G_SI * _SOLAR_MASS_KG / _METRES_PER_MPC / _KILOMETRES_IN_METRES**2
_PRESSURE_CONVERSION = (
    (_PI / 2.0)
    * _GRAVITATIONAL_CONSTANT
    * _SOLAR_MASS_G
    / _CENTIMETRES_PER_MPC**3
    * (1.0e2 * _KILOMETRES_IN_METRES) ** 2
    / _BOLTZMANN_ERG
)
_DISK_HALF_MASS_TO_SCALE_LENGTH = 1.678346990
_MINIMUM_MASS = 1.0e-3
_MINIMUM_RADIUS = 1.0e-10

# Fixed nodes are traceable JAX constants.  128 points make radial-quadrature
# error negligible compared with the 5% relative tolerance of sample_lagos23.
_GL_NODES_NP, _GL_WEIGHTS_NP = np.polynomial.legendre.leggauss(128)
_GL_NODES = jnp.asarray(_GL_NODES_NP)
_GL_WEIGHTS = jnp.asarray(_GL_WEIGHTS_NP)


class Lagos23StarFormationParameters(NamedTuple):
    """Parameters selected by upstream ``sample_lagos23.cfg``."""

    hubble_h: Array
    efficiency_per_gyr: Array
    pressure_normalization_k_per_cm3: Array
    pressure_power: Array
    gas_velocity_dispersion_km_per_s: Array
    starburst_boost: Array
    angular_momentum_transfer: Array


class StarFormationRates(NamedTuple):
    """Star-formation mass and total-angular-momentum rates."""

    mass: Array
    angular_momentum: Array


def lagos23_star_formation_parameters(
    *,
    hubble_h: float = 0.6751,
    efficiency_per_gyr: float = 1.49181009365,
    pressure_normalization_k_per_cm3: float = 34673.0,
    pressure_power: float = 0.92,
    gas_velocity_dispersion_km_per_s: float = 10.0,
    starburst_boost: float = 15.0,
    angular_momentum_transfer: bool = True,
) -> Lagos23StarFormationParameters:
    """Construct the fiducial Lagos23 BR06 parameter set."""

    return Lagos23StarFormationParameters(
        # Upstream stores cosmological parameters as C++ ``float`` even though
        # its rate calculations use ``double``.  Preserve that reference value.
        hubble_h=jnp.asarray(np.float32(hubble_h), dtype=jnp.float64),
        efficiency_per_gyr=jnp.asarray(efficiency_per_gyr, dtype=jnp.float64),
        pressure_normalization_k_per_cm3=jnp.asarray(
            pressure_normalization_k_per_cm3, dtype=jnp.float64
        ),
        pressure_power=jnp.asarray(pressure_power, dtype=jnp.float64),
        gas_velocity_dispersion_km_per_s=jnp.asarray(
            gas_velocity_dispersion_km_per_s, dtype=jnp.float64
        ),
        starburst_boost=jnp.asarray(starburst_boost, dtype=jnp.float64),
        angular_momentum_transfer=jnp.asarray(angular_momentum_transfer, dtype=jnp.bool_),
    )


def lagos23_br06_star_formation(
    cold_gas,
    stellar_mass,
    gas_half_mass_radius,
    stellar_half_mass_radius,
    gas_metallicity,
    redshift,
    burst,
    galaxy_velocity,
    gas_specific_angular_momentum,
    parameters: Lagos23StarFormationParameters,
) -> StarFormationRates:
    """Return the upstream Lagos23 BR06 mass and angular-momentum rates.

    Masses use ``Msun/h``, radii use SHARK's ``Mpc/h`` convention, velocity is
    km/s, and the returned mass rate is ``Msun/h/Gyr``.  ``gas_metallicity`` is
    retained in the interface because all upstream star-formation laws share
    it, although BR06 itself does not use it.  SHARK deliberately does not add
    an extra ``(1 + z)`` size conversion here, so ``redshift`` is also unused
    in this branch.
    """

    del gas_metallicity, redshift
    cold_gas = jnp.asarray(cold_gas)
    stellar_mass = jnp.asarray(stellar_mass)
    gas_half_mass_radius = jnp.asarray(gas_half_mass_radius)
    stellar_half_mass_radius = jnp.asarray(stellar_half_mass_radius)
    burst = jnp.asarray(burst)
    galaxy_velocity = jnp.asarray(galaxy_velocity)
    gas_specific_angular_momentum = jnp.asarray(gas_specific_angular_momentum)

    active = (cold_gas > _MINIMUM_MASS) & (gas_half_mass_radius > _MINIMUM_RADIUS)
    safe_gas_radius = jnp.where(active, gas_half_mass_radius, 1.0)
    gas_scale_radius = safe_gas_radius / _DISK_HALF_MASS_TO_SCALE_LENGTH / parameters.hubble_h
    stellar_radius_active = (stellar_mass > 0.0) & (stellar_half_mass_radius > 0.0)
    safe_stellar_radius = jnp.where(stellar_radius_active, stellar_half_mass_radius, 1.0)
    stellar_scale_radius = (
        safe_stellar_radius / _DISK_HALF_MASS_TO_SCALE_LENGTH / parameters.hubble_h
    )

    gas_surface_density = (
        jnp.where(active, cold_gas, 0.0) / parameters.hubble_h / _TWO_PI / gas_scale_radius**2
    )
    stellar_surface_density = jnp.where(
        stellar_radius_active,
        stellar_mass / parameters.hubble_h / _TWO_PI / stellar_scale_radius**2,
        0.0,
    )

    radii = 2.5 * gas_scale_radius * (_GL_NODES + 1.0)
    radial_weights = 2.5 * gas_scale_radius * _GL_WEIGHTS
    local_gas = gas_surface_density * jnp.exp(-radii / gas_scale_radius)
    local_stars = jnp.where(
        stellar_radius_active,
        stellar_surface_density * jnp.exp(-radii / stellar_scale_radius),
        0.0,
    )
    stellar_scale_height = 0.14 * radii
    stellar_velocity_dispersion = jnp.sqrt(
        _PI * _GRAVITATIONAL_CONSTANT * stellar_scale_height * local_stars
    )
    stellar_pressure_term = jnp.where(
        (local_stars > 0.0) & (stellar_velocity_dispersion > 0.0),
        parameters.gas_velocity_dispersion_km_per_s / stellar_velocity_dispersion * local_stars,
        0.0,
    )
    midplane_pressure = _PRESSURE_CONVERSION * local_gas * (local_gas + stellar_pressure_term)
    molecular_ratio = jnp.power(
        midplane_pressure / parameters.pressure_normalization_k_per_cm3,
        parameters.pressure_power,
    )
    molecular_fraction = molecular_ratio / (1.0 + molecular_ratio)
    surface_integrand = (
        _TWO_PI * parameters.efficiency_per_gyr * molecular_fraction * local_gas * radii
    )
    surface_integrand = jnp.where(
        burst, surface_integrand * parameters.starburst_boost, surface_integrand
    )

    physical_mass_rate = jnp.sum(radial_weights * surface_integrand)
    mass_rate = jnp.maximum(physical_mass_rate * parameters.hubble_h, 0.0)
    physical_radius_weighted_rate = jnp.sum(radial_weights * radii * surface_integrand)
    integrated_angular_momentum_rate = jnp.maximum(
        physical_radius_weighted_rate * parameters.hubble_h * galaxy_velocity,
        0.0,
    )
    equal_specific_angular_momentum_rate = mass_rate * gas_specific_angular_momentum
    angular_momentum_rate = jnp.where(
        parameters.angular_momentum_transfer,
        jnp.where(
            (gas_specific_angular_momentum > 0.0)
            & (integrated_angular_momentum_rate > equal_specific_angular_momentum_rate),
            equal_specific_angular_momentum_rate,
            integrated_angular_momentum_rate,
        ),
        equal_specific_angular_momentum_rate,
    )
    angular_momentum_rate = jnp.where(burst, 0.0, angular_momentum_rate)
    return StarFormationRates(
        mass=jnp.where(active, mass_rate, 0.0),
        angular_momentum=jnp.where(active, angular_momentum_rate, 0.0),
    )
