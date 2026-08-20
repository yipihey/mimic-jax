"""Pure JAX building blocks for SHARK's Croton06 cooling prescription.

The functions in this module are direct expressions of the public utility
methods in upstream ``GasCooling``.  They deliberately stop before the hybrid
parts of ``GasCooling::cooling_rate``: finite reincorporation and infall
budgets, the reionisation gate, black-hole seeding, the AGN heating-radius
projection, and source-reservoir caps.  This keeps the continuous cooling law
usable at arbitrary baryonic substeps without changing the reference event
semantics.

SHARK first converts the relevant masses and radii to physical units.  The
inputs here therefore use physical ``Msun``, physical ``Mpc``, ``km/s``, and
``Gyr`` unless a docstring says otherwise.
"""

import json
from functools import lru_cache
from importlib import resources
from typing import Any, NamedTuple

import jax.numpy as jnp

Array = Any

_PI = 3.14159265358979323846
_FOUR_PI = 4.0 * _PI
_SPHERE_VOLUME_FACTOR = 4.0 / 3.0 * _PI
_GYR_TO_SECONDS = 3.15576e16
_METRES_PER_MPC = 3.0856775807e22
_CENTIMETRES_PER_MPC = _METRES_PER_MPC * 1.0e2
_SOLAR_MASS_KG = 1.9891e30
_SOLAR_MASS_G = _SOLAR_MASS_KG * 1.0e3
_BOLTZMANN_ERG = 1.3806503e-23 * 1.0e7
_ATOMIC_MASS_G = 1.66053873e-27 * 1.0e3
_ATOMIC_MASS_HYDROGEN = 1.00794
_ATOMIC_MASS_HELIUM = 4.002602
_HYDROGEN_MASS_FRACTION = 0.778
_HELIUM_MASS_FRACTION = 0.222
_PRIMORDIAL_MEAN_ATOMIC_WEIGHT = 1.0 / (
    2.0 * _HYDROGEN_MASS_FRACTION / _ATOMIC_MASS_HYDROGEN
    + 3.0 * _HELIUM_MASS_FRACTION / _ATOMIC_MASS_HELIUM
)
_COOLING_LUMINOSITY_CONVERSION = (
    1.0e-6
    * (1.0e-20 * _SOLAR_MASS_KG / _METRES_PER_MPC / _ATOMIC_MASS_HYDROGEN / 1.66053873e-27) ** 2
    / _METRES_PER_MPC
)


class Croton06CoolingParameters(NamedTuple):
    """Numerical parameters of SHARK's fiducial Croton06 cooling branch."""

    tau_cooling: Array
    core_radius_fraction: Array
    maximum_log10_cooling_function: Array


class CoolingFunctionTable(NamedTuple):
    """Rectangular SHARK cooling-function grid."""

    log10_temperature_k: Array
    metallicity: Array
    log10_cooling_function: Array


class Croton06CoolingSolution(NamedTuple):
    """Unheated cooling solution before AGN and finite availability caps."""

    virial_temperature: Array
    log10_cooling_function: Array
    mean_number_density: Array
    cooling_time: Array
    characteristic_time_seconds: Array
    cooling_radius: Array
    cooling_rate: Array


def lagos23_croton06_cooling_parameters(
    *,
    tau_cooling: float = 1.0,
    core_radius_fraction: float = 0.1,
    maximum_log10_cooling_function: float = -23.0,
) -> Croton06CoolingParameters:
    """Construct the cooling parameters in upstream ``sample_lagos23.cfg``."""

    return Croton06CoolingParameters(
        tau_cooling=jnp.asarray(tau_cooling, dtype=jnp.float64),
        core_radius_fraction=jnp.asarray(core_radius_fraction, dtype=jnp.float64),
        maximum_log10_cooling_function=jnp.asarray(
            maximum_log10_cooling_function, dtype=jnp.float64
        ),
    )


@lru_cache(maxsize=1)
def cloudy_cie_cooling_table() -> CoolingFunctionTable:
    """Load the pinned upstream Cloudy CIE table as immutable JAX arrays."""

    path = resources.files("mimic_jax.shark").joinpath("data/cloudy_cie.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CoolingFunctionTable(
        log10_temperature_k=jnp.asarray(payload["log10_temperature_k"], dtype=jnp.float64),
        metallicity=jnp.asarray(payload["metallicity"], dtype=jnp.float64),
        log10_cooling_function=jnp.asarray(payload["log10_cooling_function"], dtype=jnp.float64),
    )


def interpolate_log10_cooling_function(
    log10_temperature_k, metallicity, table: CoolingFunctionTable
):
    """Bilinearly interpolate with the same endpoint clamping as upstream.

    The interpolation is piecewise smooth.  Gradients are exact inside a
    table cell and take the selected one-sided branch at cell boundaries.
    """

    temperature_grid = table.log10_temperature_k
    metallicity_grid = table.metallicity
    temperature = jnp.clip(
        jnp.asarray(log10_temperature_k), temperature_grid[0], temperature_grid[-1]
    )
    abundance = jnp.clip(jnp.asarray(metallicity), metallicity_grid[0], metallicity_grid[-1])
    temperature_index = jnp.clip(
        jnp.searchsorted(temperature_grid, temperature, side="right") - 1,
        0,
        temperature_grid.size - 2,
    )
    metallicity_index = jnp.clip(
        jnp.searchsorted(metallicity_grid, abundance, side="right") - 1,
        0,
        metallicity_grid.size - 2,
    )
    temperature_low = temperature_grid[temperature_index]
    temperature_high = temperature_grid[temperature_index + 1]
    metallicity_low = metallicity_grid[metallicity_index]
    metallicity_high = metallicity_grid[metallicity_index + 1]
    temperature_fraction = (temperature - temperature_low) / (temperature_high - temperature_low)
    metallicity_fraction = (abundance - metallicity_low) / (metallicity_high - metallicity_low)
    values = table.log10_cooling_function
    low_low = values[metallicity_index, temperature_index]
    low_high = values[metallicity_index, temperature_index + 1]
    high_low = values[metallicity_index + 1, temperature_index]
    high_high = values[metallicity_index + 1, temperature_index + 1]
    low_temperature_interpolation = low_low + temperature_fraction * (low_high - low_low)
    high_temperature_interpolation = high_low + temperature_fraction * (high_high - high_low)
    return low_temperature_interpolation + metallicity_fraction * (
        high_temperature_interpolation - low_temperature_interpolation
    )


def virial_temperature(virial_velocity_km_per_s):
    """Return upstream's virial temperature, ``35.9 V_vir^2``, in kelvin."""

    velocity = jnp.asarray(virial_velocity_km_per_s)
    return 35.9 * jnp.power(velocity, 2.0)


def pseudo_cooling_luminosity(temperature_k, log10_cooling_function):
    """Return the Lagos23 hot-mode accretion proxy in upstream units.

    This is the quantity called ``Lpseudo_cool`` in
    ``GasCooling::cooling_rate``.  It is a validity-gated input to the
    Croton06 black-hole accretion law, not a radiated luminosity.
    """

    return (
        _BOLTZMANN_ERG
        * jnp.asarray(temperature_k)
        / jnp.power(10.0, jnp.asarray(log10_cooling_function))
        / 1.0e40
    )


def cooling_time_gyr(temperature_k, log10_cooling_function, number_density_per_cm3):
    """Return the notional cooling time from ``GasCooling::cooling_time``."""

    return (
        3.0
        * _BOLTZMANN_ERG
        * jnp.asarray(temperature_k)
        / (
            2.0
            * jnp.power(10.0, jnp.asarray(log10_cooling_function))
            * jnp.asarray(number_density_per_cm3)
        )
        / _GYR_TO_SECONDS
    )


def mean_hot_halo_number_density(hot_gas_mass_msun, virial_radius_mpc):
    """Return SHARK's notional mean ion number density in ``cm^-3``."""

    mass = jnp.asarray(hot_gas_mass_msun)
    radius = jnp.asarray(virial_radius_mpc)
    return (
        mass
        * _SOLAR_MASS_G
        / (_SPHERE_VOLUME_FACTOR * jnp.power(radius * _CENTIMETRES_PER_MPC, 3.0))
        / (_ATOMIC_MASS_G * _PRIMORDIAL_MEAN_ATOMIC_WEIGHT)
    )


def cooling_radius_mpc(
    hot_gas_mass_msun,
    virial_radius_mpc,
    characteristic_time_seconds,
    log10_cooling_function,
    temperature_k,
):
    """Return the isothermal-profile cooling radius in physical Mpc."""

    mass = jnp.asarray(hot_gas_mass_msun)
    radius = jnp.asarray(virial_radius_mpc)
    pseudo_density = mass * _SOLAR_MASS_G / (_FOUR_PI * radius * _CENTIMETRES_PER_MPC)
    denominator = (
        1.5
        * _BOLTZMANN_ERG
        * jnp.asarray(temperature_k)
        * (_ATOMIC_MASS_G * _PRIMORDIAL_MEAN_ATOMIC_WEIGHT)
        / jnp.power(10.0, jnp.asarray(log10_cooling_function))
    )
    return (
        jnp.sqrt(pseudo_density / denominator * jnp.asarray(characteristic_time_seconds))
        / _CENTIMETRES_PER_MPC
    )


def isothermal_shell_number_density(hot_gas_mass_msun, virial_radius_mpc, radius_mpc):
    """Return SHARK's isothermal-shell ion number density in ``cm^-3``."""

    return (
        jnp.asarray(hot_gas_mass_msun)
        * _SOLAR_MASS_G
        / _FOUR_PI
        / (jnp.asarray(virial_radius_mpc) * _CENTIMETRES_PER_MPC)
        / jnp.power(jnp.asarray(radius_mpc) * _CENTIMETRES_PER_MPC, 2.0)
        / (_ATOMIC_MASS_G * _PRIMORDIAL_MEAN_ATOMIC_WEIGHT)
    )


def cooling_luminosity_1e40_erg_per_s(
    log10_cooling_function,
    cooling_radius_mpc,
    virial_radius_mpc,
    hot_gas_mass_msun,
    core_radius_fraction,
):
    """Return upstream's integrated cooling luminosity in ``10^40 erg/s``."""

    cooling_radius = jnp.asarray(cooling_radius_mpc)
    virial_radius = jnp.asarray(virial_radius_mpc)
    core_radius = jnp.asarray(core_radius_fraction) * virial_radius
    virial_ratio = virial_radius / core_radius
    cooling_ratio = cooling_radius / core_radius
    virial_geometry = jnp.arctan(virial_ratio) - virial_ratio / (virial_ratio**2 + 1.0)
    cooling_geometry = jnp.arctan(cooling_ratio) - cooling_ratio / (cooling_ratio**2 + 1.0)
    average_pseudo_density = jnp.power(jnp.asarray(hot_gas_mass_msun), 2.0) / jnp.power(
        core_radius, 3.0
    )
    geometry_factor = (virial_geometry - cooling_geometry) / jnp.power(
        virial_ratio - jnp.arctan(virial_ratio), 2.0
    )
    luminosity = (
        _COOLING_LUMINOSITY_CONVERSION
        / (8.0 * _PI)
        * jnp.power(10.0, jnp.asarray(log10_cooling_function))
        * average_pseudo_density
        * geometry_factor
    )
    return jnp.where(cooling_radius < virial_radius, luminosity, 0.0)


def croton06_unheated_cooling(
    hot_gas_mass_msun,
    density_mass_msun,
    virial_radius_mpc,
    virial_velocity_km_per_s,
    halo_dynamical_time_gyr,
    log10_cooling_function,
    parameters: Croton06CoolingParameters,
) -> Croton06CoolingSolution:
    """Evaluate SHARK's Croton06 rate before AGN and finite maps.

    ``density_mass_msun`` equals the total hot+cold halo mass for centrals.
    For type-1 satellites upstream also includes already stripped hot gas in
    this density normalization.  Making it explicit prevents an environmental
    bookkeeping choice from being mistaken for part of the cooling law.

    The returned rate is physical ``Msun/Gyr``.  Upstream subsequently applies
    AGN regulation, converts it to comoving units, and caps the finite transfer
    by the available hot reservoir.
    """

    hot_mass = jnp.asarray(hot_gas_mass_msun)
    virial_radius = jnp.asarray(virial_radius_mpc)
    temperature = virial_temperature(virial_velocity_km_per_s)
    capped_log_lambda = jnp.minimum(
        jnp.asarray(log10_cooling_function), parameters.maximum_log10_cooling_function
    )
    mean_density = mean_hot_halo_number_density(density_mass_msun, virial_radius)
    cooling_time = parameters.tau_cooling * jnp.asarray(halo_dynamical_time_gyr)
    characteristic_time = cooling_time * _GYR_TO_SECONDS
    raw_radius = cooling_radius_mpc(
        hot_mass,
        virial_radius,
        characteristic_time,
        capped_log_lambda,
        temperature,
    )
    rapid_cooling = raw_radius >= virial_radius
    bounded_radius = jnp.minimum(raw_radius, virial_radius)
    rate = jnp.where(
        rapid_cooling,
        hot_mass / cooling_time,
        0.5 * (raw_radius / virial_radius) * (hot_mass / cooling_time),
    )
    active = (hot_mass > 0.0) & (virial_radius > 0.0) & (cooling_time > 0.0) & jnp.isfinite(rate)
    zero = jnp.zeros_like(rate)
    return Croton06CoolingSolution(
        virial_temperature=jnp.where(active, temperature, zero),
        log10_cooling_function=capped_log_lambda,
        mean_number_density=jnp.where(active, mean_density, zero),
        cooling_time=jnp.where(active, cooling_time, zero),
        characteristic_time_seconds=jnp.where(active, characteristic_time, zero),
        cooling_radius=jnp.where(active, bounded_radius, zero),
        cooling_rate=jnp.where(active, rate, zero),
    )


def croton06_unheated_cooling_from_table(
    hot_gas_mass_msun,
    hot_gas_metal_mass_msun,
    density_mass_msun,
    virial_radius_mpc,
    virial_velocity_km_per_s,
    halo_dynamical_time_gyr,
    parameters: Croton06CoolingParameters,
    table: CoolingFunctionTable,
) -> Croton06CoolingSolution:
    """Evaluate the fiducial rate including SHARK's tabulated cooling law."""

    hot_mass = jnp.asarray(hot_gas_mass_msun)
    metallicity = jnp.where(hot_mass > 0.0, jnp.asarray(hot_gas_metal_mass_msun) / hot_mass, 0.0)
    temperature = virial_temperature(virial_velocity_km_per_s)
    log_lambda = interpolate_log10_cooling_function(jnp.log10(temperature), metallicity, table)
    return croton06_unheated_cooling(
        hot_mass,
        density_mass_msun,
        virial_radius_mpc,
        virial_velocity_km_per_s,
        halo_dynamical_time_gyr,
        log_lambda,
        parameters,
    )
