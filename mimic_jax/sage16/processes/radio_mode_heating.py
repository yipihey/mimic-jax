"""Fiducial SAGE16 radio-mode AGN heating and black-hole accretion."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.perturbations import log_fractionally_perturb
from mimic_jax.sage16.precision import as_float32, as_float64, require_x64
from mimic_jax.sage16.processes.common import metallicity, object_substep_dt
from mimic_jax.sage16.transfers import RadioModeHeatingResult, RadioModeHeatingTransfer
from mimic_jax.sage16.types import (
    GalaxyState,
    HaloForcing,
    Sage16Parameters,
    Sage16Units,
    StepContext,
)

SECONDS_PER_YEAR = 3.155e7
SOLAR_MASS_G = 1.989e33
PROTON_MASS_G = 1.6726e-24
BOLTZMANN_CGS = 1.3806e-16
VIRIAL_TEMPERATURE_COEFFICIENT = 35.9
HEATING_VELOCITY_KM_S = 1.34e5


def _empirical_accretion_rate(state, halo, parameters, units):
    unit_conversion = units.UnitMass_in_g / units.UnitTime_in_s * SECONDS_PER_YEAR / SOLAR_MASS_G
    velocity_ratio = halo.Vvir / 200.0
    normalized_rate = (
        parameters.RadioModeEfficiency
        / unit_conversion
        * (as_float64(state.BlackHoleMass) / 0.01)
        * velocity_ratio
        * velocity_ratio
        * velocity_ratio
    )
    return jnp.where(
        halo.Mvir > 0.0,
        normalized_rate * ((as_float64(state.HotGas) / halo.Mvir) / 0.1),
        normalized_rate,
    )


def _bondi_accretion_rate(state, halo, parameters, units):
    temperature = VIRIAL_TEMPERATURE_COEFFICIENT * halo.Vvir * halo.Vvir
    density_time = PROTON_MASS_G * BOLTZMANN_CGS * temperature / state.CoolingLambda
    density_time /= units.UnitDensity_in_cgs * units.UnitTime_in_s
    return (
        (2.5 * jnp.pi * units.G)
        * (0.375 * 0.6 * density_time)
        * as_float64(state.BlackHoleMass)
        * parameters.RadioModeEfficiency
    )


def _cold_cloud_accretion_rate(state, halo, dt, cooling_after_prior_heating):
    radius_ratio = state.Rcool / halo.Rvir
    threshold = 0.0001 * halo.Mvir * radius_ratio * radius_ratio * radius_ratio
    return jnp.where(
        as_float64(state.BlackHoleMass) > threshold,
        0.0001 * cooling_after_prior_heating / dt,
        0.0,
    )


def _zero_transfer(state):
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    radius = as_float64(state.Rheat)
    return RadioModeHeatingTransfer(zero, zero, zero, zero, zero, zero, radius, radius)


def apply_radio_mode_heating(
    state: GalaxyState,
    halo: HaloForcing,
    context: StepContext,
    parameters: Sage16Parameters,
    units: Sage16Units,
    log_fractional_perturbation=0.0,
) -> RadioModeHeatingResult:
    """Apply SAGE's finite radio-mode update without changing its time ordering.

    Stored ``Rheat`` suppresses the current cooling budget. New heating grows
    ``Rheat`` for later substeps; it is not subtracted from ``CoolingGas`` a
    second time in this call. The optional perturbation multiplies the selected
    upstream accretion rate before the unchanged Eddington and physical caps.
    """

    require_x64()
    active = (state.CoolingGas > 0.0) & (parameters.AGNrecipe > 0)

    def apply_active(_):
        dt = object_substep_dt(halo, context)
        cooling_before = as_float64(state.CoolingGas)
        rheat = as_float64(state.Rheat)
        rcool = as_float64(state.Rcool)
        cooling_after_prior_heating = jnp.where(
            rheat < rcool,
            (1.0 - rheat / rcool) * cooling_before,
            0.0,
        )

        def accrete_hot_gas(_):
            empirical = lambda _: _empirical_accretion_rate(state, halo, parameters, units)
            bondi = lambda _: _bondi_accretion_rate(state, halo, parameters, units)
            cold_cloud = lambda _: _cold_cloud_accretion_rate(
                state, halo, dt, cooling_after_prior_heating
            )
            rate = jax.lax.cond(
                parameters.AGNrecipe == 2,
                bondi,
                lambda operand: jax.lax.cond(
                    parameters.AGNrecipe == 3, cold_cloud, empirical, operand
                ),
                operand=None,
            )
            rate = log_fractionally_perturb(rate, log_fractional_perturbation)
            eddington_rate = (
                1.3e38
                * as_float64(state.BlackHoleMass)
                * 1.0e10
                / units.Hubble_h
                / (units.UnitEnergy_in_cgs / units.UnitTime_in_s)
                / (0.1 * 9.0e10)
            )
            rate = jnp.minimum(rate, eddington_rate)
            accreted = jnp.minimum(rate * dt, as_float64(state.HotGas))
            velocity_ratio = HEATING_VELOCITY_KM_S / halo.Vvir
            heating_coefficient = velocity_ratio * velocity_ratio
            heating_mass = heating_coefficient * accreted
            heating_limited = heating_mass > cooling_after_prior_heating
            accreted = jnp.where(
                heating_limited,
                cooling_after_prior_heating / heating_coefficient,
                accreted,
            )
            heating_mass = jnp.where(heating_limited, cooling_after_prior_heating, heating_mass)
            hot_metals_accreted = metallicity(state.HotGas, state.MetalsHotGas) * accreted
            cooling_denominator = jnp.where(
                cooling_after_prior_heating > 0.0,
                cooling_after_prior_heating,
                1.0,
            )
            candidate_rheat = heating_mass / cooling_denominator * rcool
            rheat_after = jnp.where(
                (rheat < rcool) & (cooling_after_prior_heating > 0.0) & (candidate_rheat > rheat),
                candidate_rheat,
                rheat,
            )
            updated = state._replace(
                CoolingGas=cooling_after_prior_heating,
                BlackHoleMass=as_float32(as_float64(state.BlackHoleMass) + accreted),
                HotGas=as_float32(as_float64(state.HotGas) - accreted),
                MetalsHotGas=as_float32(as_float64(state.MetalsHotGas) - hot_metals_accreted),
                Rheat=as_float32(rheat_after),
                Heating=as_float64(state.Heating)
                + jnp.where(
                    (heating_mass > 0.0) & (halo.dT > 0.0),
                    0.5 * heating_mass * halo.Vvir * halo.Vvir / halo.dT,
                    0.0,
                ),
            )
            return RadioModeHeatingResult(
                updated,
                RadioModeHeatingTransfer(
                    cooling_before,
                    cooling_after_prior_heating,
                    rate,
                    accreted,
                    hot_metals_accreted,
                    heating_mass,
                    rheat,
                    as_float64(updated.Rheat),
                ),
            )

        def suppress_without_accretion(_):
            updated = state._replace(CoolingGas=cooling_after_prior_heating)
            transfer = _zero_transfer(state)._replace(
                cooling_before=cooling_before,
                cooling_after_prior_heating=cooling_after_prior_heating,
            )
            return RadioModeHeatingResult(updated, transfer)

        # Upstream still evaluates the accretion formula when prior heating has
        # reduced cooling to zero, but the heating cap then sets both accreted
        # and heated mass to zero. This behavior-equivalent guard avoids
        # undefined inactive-branch derivatives such as 0 / CoolingLambda.
        return jax.lax.cond(
            (state.HotGas > 0.0) & (cooling_after_prior_heating > 0.0),
            accrete_hot_gas,
            suppress_without_accretion,
            operand=None,
        )

    def apply_inactive(_):
        return RadioModeHeatingResult(state, _zero_transfer(state))

    return jax.lax.cond(active, apply_active, apply_inactive, operand=None)
