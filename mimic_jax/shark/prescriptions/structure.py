"""Fiducial SHARK halo structure, sizes, and environmental forces.

The functions are direct, unit-explicit counterparts of the Lagos23-selected
``DarkMatterHalos`` and ``Environment`` branches.  Root finding is separated
from the residual functions so the active stripping constraint remains
visible to the hybrid driver.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

Array = Any

_G = 6.67259e-11 * 1.9891e30 / 3.0856775807e22 / 1.0e6
_MPC_KM_TO_GYR = 3.0856775807e19 / 3.15576e16
_BOLTZMANN_ERG = 1.3806503e-16
_SOLAR_MASS_G = 1.9891e33
_ATOMIC_MASS_G = 1.66053873e-24
_ATOMIC_MASS_HYDROGEN = 1.00794
_ATOMIC_MASS_HELIUM = 4.002602
_HYDROGEN_MASS_FRACTION = 0.778
_HELIUM_MASS_FRACTION = 0.222
_MEAN_ATOMIC_WEIGHT = 1.0 / (
    2.0 * _HYDROGEN_MASS_FRACTION / _ATOMIC_MASS_HYDROGEN
    + 3.0 * _HELIUM_MASS_FRACTION / _ATOMIC_MASS_HELIUM
)
_MASS_ACCRETION_CGS_TO_MSUN_PER_GYR = 3.15576e16 / 1.9891e33
_HUBBLE_TIME_100_GYR = 1.0 / (100.0 * 1.0e3 * 3.15576e16 / 3.0856775807e22)
_MPC_CM = 3.0856775807e24


class SharkCosmology(NamedTuple):
    omega_m: Array
    omega_b: Array
    omega_lambda: Array
    hubble_h: Array


class HaloStructure(NamedTuple):
    mass: Array
    virial_velocity: Array
    virial_radius: Array
    concentration: Array
    spin: Array
    position: Array
    velocity: Array


def lagos23_cosmology(*, omega_m=0.3121, omega_b=0.0491, omega_lambda=0.6879, hubble_h=0.6751):
    return SharkCosmology(
        *(
            jnp.asarray(value, dtype=jnp.float32).astype(jnp.float64)
            for value in (omega_m, omega_b, omega_lambda, hubble_h)
        )
    )


def universal_baryon_fraction(cosmology: SharkCosmology):
    return cosmology.omega_b / cosmology.omega_m


def hubble_parameter_km_s_mpc(redshift, cosmology: SharkCosmology):
    return (
        100.0
        * cosmology.hubble_h
        * jnp.sqrt(cosmology.omega_m * (1.0 + jnp.asarray(redshift)) ** 3 + cosmology.omega_lambda)
    )


def critical_density_msun_per_mpc3(redshift, cosmology: SharkCosmology):
    reduced = hubble_parameter_km_s_mpc(redshift, cosmology) / 100.0
    return 2.7754e11 * reduced**2


def number_density_200crit_per_cm3(redshift, cosmology: SharkCosmology):
    """Return the ion number density at 200 times critical density."""

    return (
        200.0
        * critical_density_msun_per_mpc3(redshift, cosmology)
        * _SOLAR_MASS_G
        / _MPC_CM**3
        / (_ATOMIC_MASS_G * _MEAN_ATOMIC_WEIGHT)
    )


def cosmic_age_gyr(redshift, cosmology: SharkCosmology):
    """Return upstream's analytic flat-LCDM age."""

    scale_factor = 1.0 / (1.0 + jnp.asarray(redshift))
    # Upstream writes ``1 - parameters.OmegaM``; because OmegaM is a C++
    # float that subtraction occurs in float32 before promotion to ``double``.
    one_minus_omega_m = (
        jnp.asarray(1.0, dtype=jnp.float32) - cosmology.omega_m.astype(jnp.float32)
    ).astype(jnp.float64)
    flat_age = (
        _HUBBLE_TIME_100_GYR
        * 2.0
        / (3.0 * cosmology.hubble_h * jnp.sqrt(one_minus_omega_m))
        * jnp.arcsinh(jnp.sqrt((1.0 / cosmology.omega_m - 1.0) * scale_factor) * scale_factor)
    )
    eds_age = (
        _HUBBLE_TIME_100_GYR
        * 2.0
        * scale_factor
        * jnp.sqrt(scale_factor)
        / (3.0 * cosmology.hubble_h)
    )
    return jnp.where(
        (jnp.abs(1.0 - cosmology.omega_m) < 1.0e-4) & (cosmology.omega_lambda == 0.0),
        eds_age,
        flat_age,
    )


def halo_virial_velocity_km_per_s(mass_msun_over_h, redshift, cosmology: SharkCosmology):
    return jnp.cbrt(
        10.0 * _G * jnp.asarray(mass_msun_over_h) * hubble_parameter_km_s_mpc(redshift, cosmology)
    )


def halo_virial_radius_mpc_over_h(mass_msun_over_h, redshift, cosmology: SharkCosmology):
    velocity = halo_virial_velocity_km_per_s(mass_msun_over_h, redshift, cosmology)
    return _G * jnp.asarray(mass_msun_over_h) / velocity**2


def halo_dynamical_time_gyr(mass_msun_over_h, redshift, cosmology: SharkCosmology):
    velocity = halo_virial_velocity_km_per_s(mass_msun_over_h, redshift, cosmology)
    radius = halo_virial_radius_mpc_over_h(mass_msun_over_h, redshift, cosmology)
    physical_radius = radius / cosmology.hubble_h
    return _MPC_KM_TO_GYR * physical_radius / velocity


def duffy08_concentration(mass_msun_over_h, redshift):
    return (
        7.85
        * (1.0 + jnp.asarray(redshift)) ** -0.71
        * (jnp.asarray(mass_msun_over_h) / 2.0e12) ** -0.081
    )


def nfw_enclosed_mass_fraction(normalized_radius, concentration):
    radius = jnp.asarray(normalized_radius)
    concentration = jnp.asarray(concentration)
    numerator = 1.0 / (1.0 + concentration * radius) - 1.0 + jnp.log1p(concentration * radius)
    denominator = 1.0 / (1.0 + concentration) - 1.0 + jnp.log1p(concentration)
    return jnp.minimum(numerator / denominator, 1.0)


def mo98_disk_half_mass_radius(virial_radius_mpc_over_h, spin):
    """Return the exact fiducial MO98 radius normalization used upstream."""

    theoretical = 3.0 / jnp.sqrt(2.0) * jnp.asarray(spin) * virial_radius_mpc_over_h
    return 0.334 * theoretical


def cooling_gas_specific_angular_momentum(
    mass_msun_over_h, spin, redshift, cosmology: SharkCosmology
):
    h0 = 10.0 * hubble_parameter_km_s_mpc(redshift, cosmology)
    return (
        jnp.sqrt(2.0)
        * _G**0.66
        * jnp.asarray(spin)
        * jnp.asarray(mass_msun_over_h) ** 0.66
        / h0**0.33
    )


def halo_rotation_velocity_squared(normalized_radius, mass, concentration, virial_radius):
    x = jnp.asarray(normalized_radius)
    c = jnp.asarray(concentration)
    numerator = jnp.log1p(c * x) - c * x / (1.0 + c * x)
    denominator = x * (jnp.log1p(c) - c / (1.0 + c))
    return _G * mass / virial_radius * numerator / denominator


def disk_rotation_velocity_squared(normalized_radius, mass, concentration, virial_radius):
    x = jnp.asarray(normalized_radius)
    c = jnp.asarray(concentration)
    cx = c * x
    numerator = c + 4.8 * c * jnp.exp(-0.35 * cx - 3.5 / cx)
    denominator = cx + cx**-2.0 + 2.0 * cx**-0.5
    return _G * mass / virial_radius * numerator / denominator


def bulge_rotation_velocity_squared(normalized_radius, mass, concentration, virial_radius):
    x = jnp.asarray(normalized_radius)
    c = jnp.asarray(concentration)
    cx = c * x
    numerator = cx**2 * c
    denominator = (1.0 + cx**2) ** 1.5
    value = _G * mass / virial_radius * numerator / denominator
    return jnp.where(mass > 0.0, value, 0.0)


def disk_specific_angular_momentum(
    disk_radius,
    disk_mass,
    bulge_radius,
    bulge_mass,
    halo_mass,
    halo_concentration,
    virial_radius,
):
    disk_concentration = jnp.where(
        disk_radius > 0.0, virial_radius / (disk_radius / 1.678346990), 0.0
    )
    bulge_concentration = jnp.where(bulge_radius > 0.0, virial_radius / bulge_radius, 0.0)
    x = disk_radius / virial_radius
    velocity_squared = (
        halo_rotation_velocity_squared(x, halo_mass, halo_concentration, virial_radius)
        + disk_rotation_velocity_squared(x, disk_mass, disk_concentration, virial_radius)
        + bulge_rotation_velocity_squared(x, bulge_mass, bulge_concentration, virial_radius)
    )
    return 2.0 * disk_radius / 1.678346990 * jnp.sqrt(velocity_squared)


def exponential_surface_density(radius, mass, half_mass_radius):
    scale = jnp.asarray(half_mass_radius) / 1.67
    value = jnp.asarray(mass) / (2.0 * jnp.pi * scale**2) * jnp.exp(-radius / scale)
    return jnp.where((mass > 0.0) & (scale > 0.0), value, 0.0)


def plummer_surface_density(radius, mass, half_mass_radius):
    scale = jnp.asarray(half_mass_radius) / 1.3
    value = jnp.asarray(mass) * scale**2 / (jnp.pi * (radius**2 + scale**2) ** 2)
    return jnp.where((mass > 0.0) & (scale > 0.0), value, 0.0)


def ism_ram_pressure_residual(
    radius,
    ram_pressure,
    disk_gas_mass,
    disk_gas_radius,
    bulge_gas_mass,
    bulge_gas_radius,
    disk_stellar_mass,
    disk_stellar_radius,
    bulge_stellar_mass,
    bulge_stellar_radius,
    alpha_cold=1.0,
):
    gas_surface_density = (
        exponential_surface_density(radius, disk_gas_mass, disk_gas_radius)
        + exponential_surface_density(radius, bulge_gas_mass, bulge_gas_radius)
    ) / 1.0e12
    galaxy_surface_density = (
        exponential_surface_density(radius, disk_gas_mass, disk_gas_radius)
        + exponential_surface_density(radius, bulge_gas_mass, bulge_gas_radius)
        + exponential_surface_density(radius, disk_stellar_mass, disk_stellar_radius)
        + plummer_surface_density(radius, bulge_stellar_mass, bulge_stellar_radius)
    ) / 1.0e12
    restoring_pressure = (
        alpha_cold * 2.0 * jnp.pi * _G * gas_surface_density * galaxy_surface_density * 1.0e6
    )
    return restoring_pressure - jnp.asarray(ram_pressure)


def halo_ram_pressure_residual(
    radius,
    ram_pressure,
    enclosed_total_mass,
    hot_profile_mass,
    infall_virial_radius,
    alpha_halo=1.0,
):
    restoring_pressure = (
        alpha_halo
        * _G
        * jnp.asarray(enclosed_total_mass)
        * jnp.asarray(hot_profile_mass)
        / (8.0 * infall_virial_radius * jnp.asarray(radius) ** 3)
        / 1.0e18
    )
    return restoring_pressure - jnp.asarray(ram_pressure)


def solve_active_stripping_radius(residual, lower_radius, upper_radius, iterations=64):
    """Deterministic JAX bisection for an already-bracketed active constraint."""

    lower = jnp.asarray(lower_radius)
    upper = jnp.asarray(upper_radius)
    lower_value = residual(lower)

    def body(_, values):
        lo, hi, flo = values
        mid = 0.5 * (lo + hi)
        fmid = residual(mid)
        same = jnp.signbit(fmid) == jnp.signbit(flo)
        return (
            jnp.where(same, mid, lo),
            jnp.where(same, hi, mid),
            jnp.where(same, fmid, flo),
        )

    lower, upper, _ = jax.lax.fori_loop(0, iterations, body, (lower, upper, lower_value))
    return 0.5 * (lower + upper)


def ram_pressure_from_host(
    satellite_position,
    satellite_velocity,
    host_position,
    host_velocity,
    host_hot_mass,
    host_virial_radius,
    redshift,
    cosmology: SharkCosmology,
):
    conversion = cosmology.hubble_h * (1.0 + redshift)
    relative_position = (jnp.asarray(satellite_position) - jnp.asarray(host_position)) / conversion
    distance = jnp.linalg.norm(relative_position)
    hubble_flow = distance * hubble_parameter_km_s_mpc(redshift, cosmology)
    relative_velocity = jnp.asarray(satellite_velocity) - jnp.asarray(host_velocity)
    # Preserve upstream's scalar addition of Hubble flow to each Cartesian
    # component, despite its unusual vector interpretation.
    velocity = jnp.linalg.norm(relative_velocity + hubble_flow)
    density = host_hot_mass / (4.0 * jnp.pi * host_virial_radius * distance**2) / 1.0e18
    return density * velocity**2


def quasi_hydrostatic_halo(
    halo_mass_msun_over_h,
    virial_temperature_k,
    cooling_function_cgs,
    number_density_200crit,
    redshift,
    hot_halo_threshold,
    cosmology: SharkCosmology,
):
    """Return the exact Correa18/Lagos23 hot-halo active-set decision."""

    physical_mass = halo_mass_msun_over_h / cosmology.hubble_h
    normalized_mass = physical_mass / 1.0e12
    log_mass = jnp.log10(normalized_mass)
    growth_rate = 0.47 * normalized_mass**0.15 * (0.333 * (redshift + 1.0)) ** 2.25 * physical_mass
    hot_fraction = 10.0 ** (-0.8 + 0.5 * log_mass - 0.05 * log_mass**2)
    hot_accretion_fraction = 1.0 / (jnp.exp(-4.3 * (log_mass + 0.15)) + 1.0)
    heating = (
        1.5
        * _BOLTZMANN_ERG
        * virial_temperature_k
        / (_ATOMIC_MASS_G * _MEAN_ATOMIC_WEIGHT)
        * universal_baryon_fraction(cosmology)
        * growth_rate
        / _MASS_ACCRETION_CGS_TO_MSUN_PER_GYR
        * (0.666 * hot_fraction + hot_accretion_fraction)
    )
    cooling = (
        hot_fraction
        * physical_mass
        * _SOLAR_MASS_G
        * universal_baryon_fraction(cosmology)
        * cooling_function_cgs
        * number_density_200crit
        / (_ATOMIC_MASS_G * _MEAN_ATOMIC_WEIGHT)
    )
    ratio = cooling / heating
    return (ratio < hot_halo_threshold) | (physical_mass > 3.0e12)
