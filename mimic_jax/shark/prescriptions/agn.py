"""Deterministic Lagos23 hot-halo AGN response and memory projection.

SHARK's fiducial AGN implementation combines continuous rates with hybrid
state updates.  Hot-halo accretion and mechanical power are ordinary local
functions.  The historical heating radius is a running-maximum projection,
while Griffin19 spin evolution contains seeded discrete/random choices.  This
module implements the deterministic rate layer and the exact Markovian memory
projection without pretending the latter is a smooth ODE.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

Array = Any

_PI = 3.14159265358979323846
_GYR_TO_SECONDS = 3.15576e16
_SOLAR_MASS_KG = 1.9891e30
_SOLAR_MASS_G = _SOLAR_MASS_KG * 1.0e3
_GRAVITATIONAL_CONSTANT_SI = 6.67259e-11
_SPEED_OF_LIGHT_M_PER_S = 2.99792458e8
_SPEED_OF_LIGHT_CM_PER_S = _SPEED_OF_LIGHT_M_PER_S * 100.0
_THOMSON_CROSS_SECTION_M2 = 6.65245854e-29
_ATOMIC_MASS_KG = 1.66053873e-27
_ATOMIC_MASS_HYDROGEN = 1.00794
_ERG_TO_JOULE = 1.0e-7
_MASS_ACCRETION_CGS_TO_MSUN_PER_GYR = _GYR_TO_SECONDS / _SOLAR_MASS_G
_EDDINGTON_SCALE = (
    4.0
    * _PI
    * _SPEED_OF_LIGHT_M_PER_S
    * _GRAVITATIONAL_CONSTANT_SI
    * _SOLAR_MASS_KG
    * 1.0e-20
    * _ATOMIC_MASS_KG
    * _ATOMIC_MASS_HYDROGEN
    / (_THOMSON_CROSS_SECTION_M2 * 1.0e20)
)
_MAXIMUM_COOLING_LUMINOSITY = 1.0e30
_SOLAR_LUMINOSITY_1E40_ERG_PER_S = 3.828e-7


class Lagos23AgnParameters(NamedTuple):
    """Deterministic parameters selected by ``sample_lagos23.cfg``."""

    hubble_h: Array
    kappa_agn: Array
    kappa_jet: Array
    alpha_cool: Array
    hot_halo_threshold: Array
    radiative_efficiency: Array
    mdotcrit_adaf: Array
    eta_superedd: Array
    low_accretion_adaf: Array
    constant_lowlum_adaf: Array
    constant_highlum_adaf: Array
    memory_start_redshift: Array
    wind_feedback: Array
    epsilon_wind: Array
    solar_metallicity: Array
    alpha_thin_disk: Array
    viscosity_ratio: Array
    spin_iteration_limit: Array


class HeatingRadiusState(NamedTuple):
    """Minimal Markov state for SHARK's history-dependent AGN heating."""

    heating_radius_mpc: Array


class Lagos23AgnCoolingResponse(NamedTuple):
    """AGN-regulated cooling diagnostics before the memory projection."""

    black_hole_accretion_rate: Array
    mechanical_luminosity: Array
    heating_to_cooling_ratio: Array
    heating_rate: Array
    candidate_heating_radius: Array


class QsoOutflowLoadings(NamedTuple):
    """QSO reheating and halo-loss mass loadings per unit SFR."""

    reheating: Array
    ejection: Array


def lagos23_agn_parameters(
    *,
    hubble_h: float = 0.6751,
    kappa_agn: float = 10.307944916121,
    kappa_jet: float = 0.0228516176369,
    alpha_cool: float = 0.5,
    hot_halo_threshold: float = 10.0,
    radiative_efficiency: float = 0.1,
    mdotcrit_adaf: float = 0.01,
    eta_superedd: float = 4.0,
    alpha_adaf: float = 0.1,
    alpha_thin_disk: float = 0.1,
    delta_adaf: float = 0.2,
    spin_iteration_limit: int = 50,
    memory_start_redshift: float = 20.0,
    wind_feedback: bool = True,
    epsilon_wind: float = 10.0,
    solar_metallicity: float = 0.018,
) -> Lagos23AgnParameters:
    """Construct deterministic fiducial Lagos23 AGN parameters.

    Parameters stored as C++ ``float`` upstream are rounded through float32 so
    direct-oracle comparisons preserve the actual reference configuration.
    """

    alpha_adaf_value = jnp.asarray(alpha_adaf, dtype=jnp.float32).astype(jnp.float64)
    delta_adaf_value = jnp.asarray(delta_adaf, dtype=jnp.float32).astype(jnp.float64)
    beta = 1.0 - alpha_adaf_value / 0.55
    return Lagos23AgnParameters(
        hubble_h=jnp.asarray(hubble_h, dtype=jnp.float32).astype(jnp.float64),
        kappa_agn=jnp.asarray(kappa_agn, dtype=jnp.float64),
        kappa_jet=jnp.asarray(kappa_jet, dtype=jnp.float32).astype(jnp.float64),
        alpha_cool=jnp.asarray(alpha_cool, dtype=jnp.float64),
        hot_halo_threshold=jnp.asarray(hot_halo_threshold, dtype=jnp.float32).astype(jnp.float64),
        radiative_efficiency=jnp.asarray(radiative_efficiency, dtype=jnp.float64),
        mdotcrit_adaf=jnp.asarray(mdotcrit_adaf, dtype=jnp.float32).astype(jnp.float64),
        eta_superedd=jnp.asarray(eta_superedd, dtype=jnp.float32).astype(jnp.float64),
        low_accretion_adaf=jnp.asarray(
            0.001 * (delta_adaf_value / 0.0005) * (1.0 - beta) / beta * alpha_adaf_value**2,
            dtype=jnp.float32,
        ).astype(jnp.float64),
        constant_lowlum_adaf=jnp.asarray(
            (delta_adaf_value / 0.0005) * (1.0 - beta) / 0.5 * 6.0,
            dtype=jnp.float32,
        ).astype(jnp.float64),
        constant_highlum_adaf=jnp.asarray(
            beta / 0.5 / alpha_adaf_value**2 * 6.0, dtype=jnp.float32
        ).astype(jnp.float64),
        memory_start_redshift=jnp.asarray(memory_start_redshift, dtype=jnp.float32).astype(
            jnp.float64
        ),
        wind_feedback=jnp.asarray(wind_feedback, dtype=jnp.bool_),
        epsilon_wind=jnp.asarray(epsilon_wind, dtype=jnp.float64),
        solar_metallicity=jnp.asarray(solar_metallicity, dtype=jnp.float64),
        alpha_thin_disk=jnp.asarray(alpha_thin_disk, dtype=jnp.float32).astype(jnp.float64),
        viscosity_ratio=jnp.asarray(alpha_thin_disk, dtype=jnp.float32).astype(jnp.float64) ** 2,
        spin_iteration_limit=jnp.asarray(spin_iteration_limit, dtype=jnp.int32),
    )


def upstream_minstd_uniform_sequence(seed, count):
    """Reproduce libc++ ``default_random_engine`` uniform doubles.

    The pinned macOS oracle resolves ``default_random_engine`` to
    ``minstd_rand``.  ``uniform_real_distribution<double>`` consumes two
    31-bit draws per value through ``generate_canonical``.  Keeping this
    compatibility function separate from JAX's portable PRNG makes reference
    realization and new stochastic experiments explicit choices.
    """

    multiplier = jnp.asarray(48271, dtype=jnp.int64)
    modulus = jnp.asarray(2147483647, dtype=jnp.int64)
    range_size = jnp.asarray(2147483646, dtype=jnp.float64)
    initial = jnp.mod(jnp.asarray(seed, dtype=jnp.int64), modulus)
    initial = jnp.where(initial == 0, 1, initial)

    def draw(current, _):
        first = jnp.mod(multiplier * current, modulus)
        second = jnp.mod(multiplier * first, modulus)
        canonical = (
            (first.astype(jnp.float64) - 1.0) + (second.astype(jnp.float64) - 1.0) * range_size
        ) / range_size**2
        return second, 2.0 * canonical - 1.0

    _, values = jax.lax.scan(draw, initial, xs=None, length=count)
    return values


def final_thin_disk_spin(initial_mass, final_mass, isco_radius):
    """Return upstream's Bardeen thin-disk spin-up map."""

    mass_ratio = jnp.asarray(initial_mass) / jnp.asarray(final_mass)
    threshold = jnp.sqrt(2.0 / (3.0 * jnp.asarray(isco_radius)))
    value = (
        0.333
        * jnp.sqrt(isco_radius)
        * mass_ratio
        * (4.0 - jnp.sqrt(3.0 * isco_radius * mass_ratio**2 - 2.0))
    )
    spin = jnp.where(mass_ratio > threshold, jnp.minimum(value, 0.998), 0.998)
    spin = jnp.where((spin < 0.0) & (spin > -0.01), -0.01, spin)
    return jnp.where((spin > 0.0) & (spin < 0.01), 0.01, spin)


def griffin19_accretion_spin_from_cosine(
    black_hole_mass_msun_over_h,
    initial_spin,
    accreted_mass_msun_over_h,
    accretion_time_gyr,
    orientation_cosine,
    parameters: Lagos23AgnParameters,
):
    """Return the fiducial warped-disk Griffin19 accretion spin update.

    The random orientation is an explicit input.  This makes the event
    conditionally differentiable and permits exact upstream RNG replay or a
    portable JAX-key experiment without conflating the two realizations.
    """

    stored_mass = jnp.asarray(black_hole_mass_msun_over_h, dtype=jnp.float32).astype(jnp.float64)
    stored_delta = jnp.asarray(accreted_mass_msun_over_h, dtype=jnp.float64)
    spin0 = jnp.asarray(initial_spin, dtype=jnp.float32)
    active = (stored_delta > 0.0) & (stored_mass > 0.0) & (accretion_time_gyr > 0.0)
    black_hole_mass = stored_mass / parameters.hubble_h
    accretion_rate = stored_delta / accretion_time_gyr / parameters.hubble_h
    normalized_rate = eddington_accretion_ratio(
        accretion_rate, black_hole_mass, parameters.radiative_efficiency
    )
    self_gravity_radius = (
        4790.0
        * parameters.alpha_thin_disk**0.5185
        * normalized_rate**-0.2962
        * (black_hole_mass / 1.0e8) ** -0.9629
    )
    self_gravity_mass = jnp.where(
        normalized_rate < parameters.mdotcrit_adaf,
        black_hole_mass,
        1.35
        * parameters.alpha_thin_disk**-0.8
        * normalized_rate**0.6
        * (black_hole_mass / 1.0e8) ** 2.2
        * self_gravity_radius**1.4,
    )
    # This mixed-unit comparison is present in the pinned upstream branch.
    available_disk_mass = jnp.minimum(self_gravity_mass, stored_delta)

    def zero_spin_chunks(values):
        mass, spin = values

        def chunk(_, chunk_values):
            current_mass, current_spin = chunk_values
            efficiency, isco = thin_disk_efficiency_and_isco(current_spin)
            final_mass = current_mass + (1.0 - efficiency) * available_disk_mass / 10.0
            final_spin = final_thin_disk_spin(current_mass, final_mass, isco).astype(jnp.float32)
            return final_mass, final_spin

        return jax.lax.fori_loop(0, 10, chunk, (mass, spin))

    def nonzero_spin_chunks(values):
        initial_mass, initial_spin_value = values

        def chunk(_, chunk_values):
            mass, spin, remaining, running = chunk_values
            efficiency, isco = thin_disk_efficiency_and_isco(spin)
            warp_radius = (
                3410.0
                * jnp.abs(spin.astype(jnp.float64)) ** 0.625
                * (mass / 1.0e8) ** 0.125
                * normalized_rate**-0.25
                * parameters.viscosity_ratio**-0.625
                * parameters.alpha_thin_disk**-0.5
            )
            warp_mass = (
                1.35
                * parameters.alpha_thin_disk**-0.8
                * normalized_rate**0.6
                * (mass / 1.0e8) ** 2.2
                * warp_radius**1.4
            )
            delta = jnp.minimum(warp_mass, remaining)
            angular_radius = jnp.where(warp_mass > remaining, self_gravity_radius, warp_radius)
            final_mass = mass + (1.0 - efficiency.astype(jnp.float64)) * delta
            angular_ratio = (
                delta
                / (jnp.sqrt(2.0) * mass * jnp.abs(spin.astype(jnp.float64)))
                * jnp.sqrt(angular_radius)
            )
            black_hole_angular_momentum = (
                jnp.abs(spin.astype(jnp.float64))
                * mass**2
                * _GRAVITATIONAL_CONSTANT_SI
                / 1.0e3
                / (_SPEED_OF_LIGHT_M_PER_S / 1.0e3)
            )
            disk_angular_momentum = 2.0 * angular_ratio * black_hole_angular_momentum
            cosine_final = (
                disk_angular_momentum
                + black_hole_angular_momentum * jnp.asarray(orientation_cosine)
            ) / jnp.sqrt(
                black_hole_angular_momentum**2
                + disk_angular_momentum**2
                + 2.0
                * black_hole_angular_momentum
                * disk_angular_momentum
                * jnp.asarray(orientation_cosine)
            )
            cosine_final = jnp.clip(cosine_final, -1.0, 1.0)
            retrograde = cosine_final <= -angular_ratio
            selected_spin = jnp.where(retrograde, -spin, spin)
            efficiency_selected, isco_selected = thin_disk_efficiency_and_isco(selected_spin)
            candidate = jnp.abs(final_thin_disk_spin(mass, final_mass, isco_selected))
            candidate = jnp.where(retrograde, -candidate, candidate).astype(jnp.float32)
            do_step = running & (remaining > 0.0)
            return (
                jnp.where(do_step, final_mass, mass),
                jnp.where(do_step, candidate, spin),
                jnp.where(do_step, remaining - delta, remaining),
                do_step & ((remaining - delta) > 0.0),
            )

        mass, spin, _, _ = jax.lax.fori_loop(
            0,
            50,
            chunk,
            (
                initial_mass,
                initial_spin_value,
                available_disk_mass,
                jnp.asarray(True),
            ),
        )
        return mass, spin

    _, spin = jax.lax.cond(
        spin0 == 0.0,
        zero_spin_chunks,
        nonzero_spin_chunks,
        (black_hole_mass, spin0),
    )
    return jnp.where(active, spin, spin0).astype(jnp.float32)


def griffin19_accretion_spin_upstream_rng(
    black_hole_mass_msun_over_h,
    initial_spin,
    accreted_mass_msun_over_h,
    accretion_time_gyr,
    galaxy_id,
    execution_seed,
    parameters: Lagos23AgnParameters,
):
    """Replay the pinned upstream random orientation for an accretion event."""

    cosine = upstream_minstd_uniform_sequence(
        jnp.asarray(execution_seed, dtype=jnp.int64) + jnp.asarray(galaxy_id, dtype=jnp.int64), 1
    )[0]
    return griffin19_accretion_spin_from_cosine(
        black_hole_mass_msun_over_h,
        initial_spin,
        accreted_mass_msun_over_h,
        accretion_time_gyr,
        cosine,
        parameters,
    )


def griffin19_merger_spin_from_orientations(
    primary_mass,
    secondary_mass,
    primary_spin,
    secondary_spin,
    cosine_theta_primary,
    cosine_theta_secondary,
    phi,
):
    """Return the Rezzolla08/Berti08 BH-merger spin on a fixed RNG branch."""

    m1 = jnp.asarray(primary_mass, dtype=jnp.float32)
    m2 = jnp.asarray(secondary_mass, dtype=jnp.float32)
    s1_input = jnp.asarray(primary_spin, dtype=jnp.float32)
    s2_input = jnp.asarray(secondary_spin, dtype=jnp.float32)
    both = (m1 > 0.0) & (m2 > 0.0)
    both_spinless = both & (s1_input == 0.0) & (s2_input == 0.0)
    general = both & ~both_spinless

    cos1 = jnp.asarray(cosine_theta_primary)
    cos2 = jnp.asarray(cosine_theta_secondary)
    sin1 = jnp.sqrt(jnp.maximum(1.0 - cos1**2, 0.0))
    sin2 = jnp.sqrt(jnp.maximum(1.0 - cos2**2, 0.0))
    alpha = jnp.abs(sin1 * sin2 * jnp.cos(phi) + cos1 * cos2)
    beta = jnp.abs(cos1)
    gamma = jnp.abs(cos2)
    symmetric_ratio = m1 * m2 / (m1 + m2) ** 2
    mass_ratio_original = m2 / m1
    ratio_squared = mass_ratio_original**2
    ratio_fourth = mass_ratio_original**4
    mass_ratio = jnp.where(
        mass_ratio_original > 1.0, 1.0 / mass_ratio_original, mass_ratio_original
    )
    s1 = jnp.where(mass_ratio_original > 1.0, s2_input, s1_input)
    s2 = jnp.where(mass_ratio_original > 1.0, s1_input, s2_input)
    s1 = jnp.abs(s1)
    s2 = jnp.abs(s2)
    angular_momentum = jnp.abs(
        -0.129
        / (1.0 + ratio_squared) ** 2
        * (s1**2 + s2**2 * ratio_fourth + 2.0 * s1 * s2 * ratio_squared * alpha)
        + (-0.384 * symmetric_ratio - 2.686 + 2.0)
        / (1.0 + ratio_squared)
        * (s1 * beta + s2 * ratio_squared * gamma)
        + 2.0 * jnp.sqrt(3.0)
        - 3.454 * symmetric_ratio
        + 2.353 * symmetric_ratio**2
    )
    spin_argument = (
        s1**2
        + s2**2 * ratio_fourth
        + 2.0 * s1 * s2 * ratio_squared * alpha
        + 2.0 * (s2 * beta + s2 * ratio_squared * gamma) * angular_momentum * mass_ratio
        + angular_momentum**2 * ratio_squared
    )
    general_spin = jnp.minimum(
        jnp.sqrt(jnp.maximum(spin_argument, 0.0)) / (1.0 + mass_ratio) ** 2, 0.998
    )
    q = jnp.where(mass_ratio_original > 1.0, 1.0 / mass_ratio_original, mass_ratio_original)
    spinless = jnp.clip(
        2.0 * jnp.sqrt(3.0) * q / (1.0 + q) ** 2 - 2.029 * q**2 / (1.0 + q) ** 4,
        0.0,
        1.0,
    )
    result = jnp.where(general, general_spin, jnp.where(both_spinless, spinless, 0.0))
    return result.astype(jnp.float32)


def griffin19_merger_spin_upstream_rng(
    primary_mass,
    secondary_mass,
    primary_spin,
    secondary_spin,
    galaxy_id,
    execution_seed,
):
    """Replay the three upstream orientation draws for a BH merger."""

    draws = upstream_minstd_uniform_sequence(
        jnp.asarray(execution_seed, dtype=jnp.int64) + jnp.asarray(galaxy_id, dtype=jnp.int64), 3
    )
    return griffin19_merger_spin_from_orientations(
        primary_mass,
        secondary_mass,
        primary_spin,
        secondary_spin,
        draws[0],
        draws[1],
        2.0 * jnp.pi * (1.0 + draws[2]),
    )


def eddington_luminosity_1e40_erg_per_s(black_hole_mass_msun):
    """Return SHARK's Eddington luminosity in ``10^40 erg/s``."""

    mass = jnp.asarray(black_hole_mass_msun)
    luminosity = _EDDINGTON_SCALE * mass / _ERG_TO_JOULE
    return jnp.where(mass > 0.0, luminosity, 0.0)


def lagos23_hot_halo_accretion_rate(
    pseudo_cooling_luminosity,
    black_hole_mass_msun_over_h,
    hot_gas_fraction,
    virial_velocity_km_per_s,
    parameters: Lagos23AgnParameters,
):
    """Return the fiducial Croton06/Lagos23 hot-mode rate in ``Msun/Gyr``.

    The luminosity is only an upstream validity gate in this branch; its value
    does not otherwise enter the Croton06 accretion formula.  The BH mass is
    SHARK's stored ``Msun/h`` value at this stage, matching the reference call
    order exactly.
    """

    luminosity = jnp.asarray(pseudo_cooling_luminosity)
    raw_rate = (
        parameters.kappa_agn
        * (jnp.asarray(black_hole_mass_msun_over_h) / 1.0e8)
        * (jnp.asarray(hot_gas_fraction) / 0.1)
        * jnp.power(jnp.asarray(virial_velocity_km_per_s) / 200.0, 3.0)
    )
    active = (luminosity > 0.0) & (luminosity < _MAXIMUM_COOLING_LUMINOSITY)
    return jnp.where(active, raw_rate * _MASS_ACCRETION_CGS_TO_MSUN_PER_GYR, 0.0)


def hot_halo_accretion_rate_for_saturated_heating(
    heating_rate_msun_per_gyr,
    virial_velocity_km_per_s,
    black_hole_spin,
):
    """Return upstream's accretion rate when AGN fully offsets cooling.

    ``GasCooling::cooling_rate`` replaces the nominal Croton06 rate on the
    saturated heating-radius branch.  The returned value is physical
    ``Msun/Gyr`` and is converted to SHARK's stored ``Msun/h/Gyr`` by the
    interval driver.
    """

    efficiency, _ = thin_disk_efficiency_and_isco(black_hole_spin)
    luminosity = (
        jnp.asarray(heating_rate_msun_per_gyr)
        * 0.5
        * jnp.power(jnp.asarray(virial_velocity_km_per_s) * 1.0e5, 2.0)
        / _MASS_ACCRETION_CGS_TO_MSUN_PER_GYR
        / 1.0e40
    )
    return (
        luminosity
        / efficiency.astype(jnp.float64)
        / _SPEED_OF_LIGHT_CM_PER_S**2
        * 1.0e40
        * _MASS_ACCRETION_CGS_TO_MSUN_PER_GYR
    )


def thin_disk_efficiency_and_isco(spin):
    """Return SHARK's radiative efficiency and ISCO radius.

    Upstream applies ``abs(spin)`` before its sign branch, so its current
    implementation always evaluates the prograde expression.  That behavior
    is intentionally preserved here and is covered by the direct oracle.
    """

    absolute_spin = jnp.abs(jnp.asarray(spin, dtype=jnp.float32)).astype(jnp.float64)
    spin_squared = absolute_spin**2
    z1 = 1.0 + jnp.power(1.0 - spin_squared, 0.333) * (
        jnp.power(1.0 + absolute_spin, 0.333) + jnp.power(1.0 - absolute_spin, 0.333)
    )
    z2 = jnp.sqrt(3.0 * spin_squared + z1**2)
    isco = 3.0 + z2 - jnp.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    efficiency = 1.0 - jnp.sqrt(1.0 - 2.0 / (3.0 * isco))
    efficiency = jnp.where(efficiency < 0.0, 0.07, efficiency)
    efficiency = jnp.where(efficiency > 0.5, 0.5, efficiency)
    # The public upstream API returns std::vector<float>.
    return efficiency.astype(jnp.float32), isco.astype(jnp.float32)


def eddington_accretion_ratio(
    accretion_rate_msun_per_gyr,
    black_hole_mass_msun,
    radiative_efficiency,
):
    """Return physical accretion rate divided by the Eddington rate."""

    rate = jnp.asarray(accretion_rate_msun_per_gyr)
    eddington = eddington_luminosity_1e40_erg_per_s(black_hole_mass_msun)
    eddington_mass_rate_cgs = (
        1.0e40 * eddington / (jnp.asarray(radiative_efficiency) * _SPEED_OF_LIGHT_CM_PER_S**2)
    )
    ratio = (rate / _MASS_ACCRETION_CGS_TO_MSUN_PER_GYR) / eddington_mass_rate_cgs
    return jnp.where(rate > 0.0, ratio, 0.0)


def lagos23_mechanical_luminosity_1e40_erg_per_s(
    black_hole_mass_msun_over_h,
    hot_halo_accretion_rate_msun_over_h_per_gyr,
    starburst_accretion_rate_msun_over_h_per_gyr,
    spin,
    parameters: Lagos23AgnParameters,
):
    """Return Griffin19 mechanical luminosity in ``10^40 erg/s``."""

    physical_mass = jnp.asarray(black_hole_mass_msun_over_h) / parameters.hubble_h
    physical_rate = (
        jnp.asarray(hot_halo_accretion_rate_msun_over_h_per_gyr)
        + jnp.asarray(starburst_accretion_rate_msun_over_h_per_gyr)
    ) / parameters.hubble_h
    normalized_rate = (
        eddington_accretion_ratio(physical_rate, physical_mass, parameters.radiative_efficiency)
        / 0.01
    )
    high = (
        2.5e3
        * jnp.power(physical_mass / 1.0e9, 1.1)
        * jnp.power(normalized_rate, 1.2)
        * jnp.power(jnp.asarray(spin), 2.0)
    )
    low = 2.0e5 * (physical_mass / 1.0e9) * normalized_rate * jnp.power(jnp.asarray(spin), 2.0)
    return jnp.where(normalized_rate >= 1.0, high, low)


def lagos23_bolometric_luminosity_1e40_erg_per_s(
    black_hole_mass_msun_over_h,
    hot_halo_accretion_rate_msun_over_h_per_gyr,
    starburst_accretion_rate_msun_over_h_per_gyr,
    spin,
    parameters: Lagos23AgnParameters,
):
    """Return upstream's spin-dependent Lagos23 bolometric luminosity."""

    physical_mass = jnp.asarray(black_hole_mass_msun_over_h) / parameters.hubble_h
    physical_rate = (
        jnp.asarray(hot_halo_accretion_rate_msun_over_h_per_gyr)
        + jnp.asarray(starburst_accretion_rate_msun_over_h_per_gyr)
    ) / parameters.hubble_h
    normalized_rate = eddington_accretion_ratio(
        physical_rate, physical_mass, parameters.radiative_efficiency
    )
    efficiency, isco = thin_disk_efficiency_and_isco(spin)
    efficiency = efficiency.astype(jnp.float64)
    isco = isco.astype(jnp.float64)
    thin_disk = (
        efficiency
        * (physical_rate / _MASS_ACCRETION_CGS_TO_MSUN_PER_GYR)
        * _SPEED_OF_LIGHT_CM_PER_S**2
        / 1.0e40
    )
    eddington_luminosity = eddington_luminosity_1e40_erg_per_s(physical_mass)
    super_eddington_threshold = parameters.eta_superedd * (0.1 / efficiency)
    super_eddington = (
        parameters.eta_superedd
        * (1.0 + jnp.log(normalized_rate / parameters.eta_superedd * efficiency / 0.1))
        * eddington_luminosity
    )
    luminous_adaf = (
        0.2
        * efficiency
        * (physical_rate / _MASS_ACCRETION_CGS_TO_MSUN_PER_GYR)
        * _SPEED_OF_LIGHT_CM_PER_S**2
        * normalized_rate
        * parameters.constant_highlum_adaf
        / isco
        / 1.0e40
    )
    faint_adaf = (
        0.0002
        * efficiency
        * (physical_rate / _MASS_ACCRETION_CGS_TO_MSUN_PER_GYR)
        * _SPEED_OF_LIGHT_CM_PER_S**2
        * parameters.constant_lowlum_adaf
        / isco
        / 1.0e40
    )
    high_accretion = jnp.where(
        normalized_rate > super_eddington_threshold, super_eddington, thin_disk
    )
    low_accretion = jnp.where(
        normalized_rate > parameters.low_accretion_adaf, luminous_adaf, faint_adaf
    )
    return jnp.where(normalized_rate >= parameters.mdotcrit_adaf, high_accretion, low_accretion)


def qso_critical_luminosity_1e40_erg_per_s(gas_mass, baryonic_mass, bulge_radius_mpc):
    """Return the SHARK QSO wind threshold in ``10^40 erg/s``."""

    gas_fraction = jnp.asarray(gas_mass) / jnp.asarray(baryonic_mass)
    gravitational_constant = _GRAVITATIONAL_CONSTANT_SI * _SOLAR_MASS_KG / 3.0856775807e22 / 1.0e6
    velocity_dispersion = jnp.sqrt(
        gravitational_constant * jnp.asarray(baryonic_mass) / jnp.asarray(bulge_radius_mpc)
    )
    return 3.0e6 * (gas_fraction / 0.1) * jnp.power(velocity_dispersion / 200.0, 4.0)


def salpeter_timescale_gyr(bolometric_luminosity, black_hole_mass_msun):
    """Return upstream's luminosity-dependent Salpeter time in Gyr."""

    eddington_ratio = jnp.asarray(bolometric_luminosity) / eddington_luminosity_1e40_erg_per_s(
        black_hole_mass_msun
    )
    return 43.0 / eddington_ratio / 1.0e3


def qso_outflow_velocity_km_per_s(
    bolometric_luminosity,
    gas_metallicity,
    gas_mass,
    parameters: Lagos23AgnParameters,
):
    """Return SHARK's QSO wind velocity in km/s."""

    return (
        320.0
        * jnp.power(
            jnp.asarray(bolometric_luminosity) / (1.0e7 * _SOLAR_LUMINOSITY_1E40_ERG_PER_S),
            0.5,
        )
        * jnp.power(jnp.asarray(gas_metallicity) / parameters.solar_metallicity, 0.25)
        * jnp.power(jnp.asarray(gas_mass), -0.25)
    )


def lagos23_qso_outflow_loadings(
    *,
    gas_mass,
    black_hole_mass_msun_over_h,
    hot_halo_accretion_rate_msun_over_h_per_gyr,
    starburst_accretion_rate_msun_over_h_per_gyr,
    spin,
    gas_metallicity,
    circular_velocity_km_per_s,
    star_formation_rate,
    bulge_baryonic_mass,
    bulge_radius_mpc,
    parameters: Lagos23AgnParameters,
) -> QsoOutflowLoadings:
    """Return exact deterministic QSO reheating/ejection loadings."""

    physical_black_hole_mass = jnp.asarray(black_hole_mass_msun_over_h) / parameters.hubble_h
    physical_accretion_rate = (
        jnp.asarray(hot_halo_accretion_rate_msun_over_h_per_gyr)
        + jnp.asarray(starburst_accretion_rate_msun_over_h_per_gyr)
    ) / parameters.hubble_h
    bolometric = lagos23_bolometric_luminosity_1e40_erg_per_s(
        black_hole_mass_msun_over_h,
        hot_halo_accretion_rate_msun_over_h_per_gyr,
        starburst_accretion_rate_msun_over_h_per_gyr,
        spin,
        parameters,
    )
    critical = qso_critical_luminosity_1e40_erg_per_s(
        gas_mass, bulge_baryonic_mass, bulge_radius_mpc
    )
    active = (
        (physical_accretion_rate > 0.0)
        & (physical_black_hole_mass > 0.0)
        & (jnp.asarray(star_formation_rate) > 0.0)
        & (jnp.asarray(gas_mass) > 0.0)
        & parameters.wind_feedback
        & (bolometric > critical)
    )
    salpeter_time = salpeter_timescale_gyr(bolometric, physical_black_hole_mass)
    outflow_velocity = qso_outflow_velocity_km_per_s(
        bolometric, gas_metallicity, gas_mass, parameters
    )
    reheating_rate = jnp.asarray(gas_mass) / salpeter_time
    ejection_rate = (
        parameters.epsilon_wind
        * jnp.power(outflow_velocity / jnp.asarray(circular_velocity_km_per_s), 2.0)
        - 1.0
    ) * reheating_rate
    valid_reheating = (reheating_rate >= 0.0) & jnp.isfinite(reheating_rate)
    valid_ejection = (ejection_rate >= 0.0) & jnp.isfinite(ejection_rate)
    reheating_rate = jnp.where(valid_reheating, reheating_rate, 0.0)
    ejection_rate = jnp.where(valid_ejection, ejection_rate, 0.0)
    safe_sfr = jnp.where(jnp.asarray(star_formation_rate) > 0.0, star_formation_rate, 1.0)
    return QsoOutflowLoadings(
        reheating=jnp.where(active, reheating_rate / safe_sfr, 0.0),
        ejection=jnp.where(active, ejection_rate / safe_sfr, 0.0),
    )


def lagos23_agn_cooling_response(
    *,
    pseudo_cooling_luminosity,
    cooling_luminosity,
    unheated_cooling_rate,
    cooling_radius_mpc,
    black_hole_mass_msun_over_h,
    black_hole_starburst_accretion_rate_msun_over_h_per_gyr,
    black_hole_spin,
    hot_gas_fraction,
    virial_velocity_km_per_s,
    hydrostatic,
    parameters: Lagos23AgnParameters,
) -> Lagos23AgnCoolingResponse:
    """Return the instantaneous Lagos23 AGN heating candidate.

    This function does not apply the historical maximum or cap the resulting
    cooling transfer.  Those are explicit projections downstream.
    """

    accretion = lagos23_hot_halo_accretion_rate(
        pseudo_cooling_luminosity,
        black_hole_mass_msun_over_h,
        hot_gas_fraction,
        virial_velocity_km_per_s,
        parameters,
    )
    mechanical = lagos23_mechanical_luminosity_1e40_erg_per_s(
        black_hole_mass_msun_over_h,
        accretion * parameters.hubble_h,
        black_hole_starburst_accretion_rate_msun_over_h_per_gyr,
        black_hole_spin,
        parameters,
    )
    cooling_luminosity_value = jnp.asarray(cooling_luminosity)
    ratio = jnp.where(
        jnp.asarray(hydrostatic) & (cooling_luminosity_value > 0.0),
        parameters.kappa_jet * mechanical / cooling_luminosity_value,
        0.0,
    )
    heating_rate = ratio * jnp.asarray(unheated_cooling_rate)
    candidate_radius = ratio * jnp.asarray(cooling_radius_mpc)
    return Lagos23AgnCoolingResponse(
        black_hole_accretion_rate=accretion,
        mechanical_luminosity=mechanical,
        heating_to_cooling_ratio=ratio,
        heating_rate=heating_rate,
        candidate_heating_radius=candidate_radius,
    )


def project_lagos23_heating_radius(
    state: HeatingRadiusState,
    candidate_heating_radius_mpc,
    redshift,
    parameters: Lagos23AgnParameters,
) -> HeatingRadiusState:
    """Apply the exact running-maximum history update used by Lagos23.

    This is a Markovian projection once ``heating_radius_mpc`` is state.  It is
    piecewise differentiable but is not recast as a smooth ODE.
    """

    candidate = jnp.asarray(candidate_heating_radius_mpc)
    updated = jnp.maximum(state.heating_radius_mpc, candidate)
    remember = jnp.asarray(redshift) < parameters.memory_start_redshift
    return HeatingRadiusState(jnp.where(remember, updated, state.heating_radius_mpc))


def cooling_rate_after_heating_radius(
    unheated_cooling_rate,
    cooling_radius_mpc,
    heating_state: HeatingRadiusState,
    parameters: Lagos23AgnParameters,
):
    """Apply SHARK's stored-heating-radius suppression to a cooling rate."""

    cooling_radius = jnp.asarray(cooling_radius_mpc)
    safe_radius = jnp.where(cooling_radius > 0.0, cooling_radius, 1.0)
    raw_ratio = heating_state.heating_radius_mpc / safe_radius
    saturated = raw_ratio > parameters.alpha_cool
    bounded_ratio = jnp.where(saturated, 1.0, raw_ratio)
    regulated = (1.0 - bounded_ratio) * jnp.asarray(unheated_cooling_rate)
    return jnp.maximum(regulated, 0.0), bounded_ratio, saturated
