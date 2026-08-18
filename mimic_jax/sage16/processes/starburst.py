"""Fiducial SAGE16 collisional starburst, SN feedback, and yield."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.perturbations import log_fractionally_perturb
from mimic_jax.sage16.precision import as_float32, as_float64, require_x64
from mimic_jax.sage16.processes.common import metallicity
from mimic_jax.sage16.processes.star_formation import (
    ENERGY_SN,
    ETA_SN,
    SAGE_COLD_GAS_YIELD_THRESHOLD,
    SAGE_METAL_EJECTION_MVIR_SCALE,
    SOLAR_MASS,
)
from mimic_jax.sage16.transfers import StarburstResult, StarburstTransfer
from mimic_jax.sage16.types import GalaxyState, HaloForcing, Sage16Parameters, Sage16Units


def _zero_transfer(trigger, burst_efficiency):
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return StarburstTransfer(
        trigger,
        burst_efficiency,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
    )


def apply_collisional_starburst(
    galaxy: GalaxyState,
    central: GalaxyState,
    galaxy_halo: HaloForcing,
    central_halo: HaloForcing,
    efficiency_factor,
    mode,
    rate_dt,
    parameters: Sage16Parameters,
    units: Sage16Units,
    log_starburst_perturbation=0.0,
    log_reheating_perturbation=0.0,
    log_ejection_perturbation=0.0,
) -> StarburstResult:
    """Apply one finite disk-instability (mode 1) or merger (mode 0) burst."""

    require_x64()
    trigger = as_float64(efficiency_factor)
    mode = jnp.asarray(mode, dtype=jnp.int32)
    burst_efficiency = jnp.where(mode == 1, trigger, 0.56 * trigger**0.7)
    galaxy_is_central = galaxy_halo.Type == 0

    def apply_active(_):
        stars = log_fractionally_perturb(
            burst_efficiency * as_float64(galaxy.ColdGas),
            log_starburst_perturbation,
        )
        stars = jnp.maximum(stars, 0.0)
        reheated = log_fractionally_perturb(
            parameters.FeedbackReheatingEpsilon * stars,
            log_reheating_perturbation,
        )
        combined = stars + reheated
        renormalize = (combined > as_float64(galaxy.ColdGas)) & (combined > 0.0)
        factor = jnp.where(
            renormalize,
            as_float64(galaxy.ColdGas) / jnp.where(combined > 0.0, combined, 1.0),
            1.0,
        )
        stars = stars * factor
        reheated = reheated * factor

        energy_sn_code = ENERGY_SN / units.UnitEnergy_in_cgs * units.Hubble_h
        eta_sn_code = ETA_SN * (units.UnitMass_in_g / SOLAR_MASS) / units.Hubble_h
        proposed_ejected = jnp.where(
            central_halo.Vvir > 0.0,
            (
                parameters.FeedbackEjectionEfficiency
                * (eta_sn_code * energy_sn_code)
                / central_halo.Vvir**2
                - parameters.FeedbackReheatingEpsilon
            )
            * stars,
            0.0,
        )
        proposed_ejected = jnp.maximum(
            log_fractionally_perturb(proposed_ejected, log_ejection_perturbation),
            0.0,
        )

        locked_stars = (1.0 - parameters.RecycleFraction) * stars
        cold_metallicity = metallicity(galaxy.ColdGas, galaxy.MetalsColdGas)
        metals_to_stars = cold_metallicity * locked_stars
        local = galaxy._replace(
            ColdGas=as_float32(as_float64(galaxy.ColdGas) - locked_stars),
            MetalsColdGas=as_float32(as_float64(galaxy.MetalsColdGas) - metals_to_stars),
            StellarMass=as_float32(as_float64(galaxy.StellarMass) + locked_stars),
            MetalsStellarMass=as_float32(as_float64(galaxy.MetalsStellarMass) + metals_to_stars),
            BulgeMass=as_float32(as_float64(galaxy.BulgeMass) + locked_stars),
            MetalsBulgeMass=as_float32(as_float64(galaxy.MetalsBulgeMass) + metals_to_stars),
            StarFormationRate=as_float32(
                as_float64(galaxy.StarFormationRate)
                + jnp.where(as_float64(rate_dt) > 0.0, stars / as_float64(rate_dt), 0.0)
            ),
        )

        post_sf_metallicity = metallicity(local.ColdGas, local.MetalsColdGas)
        metals_to_hot = post_sf_metallicity * reheated
        local = local._replace(
            ColdGas=as_float32(as_float64(local.ColdGas) - reheated),
            MetalsColdGas=as_float32(as_float64(local.MetalsColdGas) - metals_to_hot),
        )
        destination = jax.lax.cond(
            galaxy_is_central,
            lambda _: local,
            lambda _: central,
            operand=None,
        )
        destination = destination._replace(
            HotGas=as_float32(as_float64(destination.HotGas) + reheated),
            MetalsHotGas=as_float32(as_float64(destination.MetalsHotGas) + metals_to_hot),
        )
        ejected = jnp.minimum(proposed_ejected, as_float64(destination.HotGas))
        hot_metallicity = metallicity(destination.HotGas, destination.MetalsHotGas)
        metals_to_ejected = hot_metallicity * ejected
        destination = destination._replace(
            HotGas=as_float32(as_float64(destination.HotGas) - ejected),
            MetalsHotGas=as_float32(as_float64(destination.MetalsHotGas) - metals_to_ejected),
            EjectedGas=as_float32(as_float64(destination.EjectedGas) + ejected),
            MetalsEjectedGas=as_float32(
                as_float64(destination.MetalsEjectedGas) + metals_to_ejected
            ),
        )
        local = jax.lax.cond(
            galaxy_is_central,
            lambda _: destination,
            lambda _: local,
            operand=None,
        )
        local = local._replace(
            SupernovaOutflowRate=as_float32(
                as_float64(local.SupernovaOutflowRate)
                + jnp.where(as_float64(rate_dt) > 0.0, reheated / as_float64(rate_dt), 0.0)
            )
        )
        destination = jax.lax.cond(
            galaxy_is_central,
            lambda _: local,
            lambda _: destination,
            operand=None,
        )

        fraction_leaving = parameters.FracZleaveDisk * jnp.exp(
            -central_halo.Mvir / SAGE_METAL_EJECTION_MVIR_SCALE
        )
        metals_to_cold_new = jnp.where(
            (local.ColdGas > SAGE_COLD_GAS_YIELD_THRESHOLD)
            & (trigger < parameters.ThresholdMajorMerger),
            parameters.Yield * (1.0 - fraction_leaving) * stars,
            0.0,
        )
        produced_metals = parameters.Yield * stars
        metals_to_hot_new = produced_metals - metals_to_cold_new
        local = local._replace(
            MetalsColdGas=as_float32(as_float64(local.MetalsColdGas) + metals_to_cold_new)
        )
        destination = jax.lax.cond(
            galaxy_is_central,
            lambda _: local,
            lambda _: destination,
            operand=None,
        )
        destination = destination._replace(
            MetalsHotGas=as_float32(as_float64(destination.MetalsHotGas) + metals_to_hot_new)
        )
        local = jax.lax.cond(
            galaxy_is_central,
            lambda _: destination,
            lambda _: local,
            operand=None,
        )
        destination = jax.lax.cond(
            galaxy_is_central,
            lambda _: local,
            lambda _: destination,
            operand=None,
        )
        transfer = StarburstTransfer(
            trigger,
            burst_efficiency,
            stars,
            locked_stars,
            reheated,
            ejected,
            metals_to_stars,
            metals_to_hot,
            metals_to_ejected,
            produced_metals,
            metals_to_cold_new,
            metals_to_hot_new,
        )
        return StarburstResult(local, destination, transfer)

    def apply_zero(_):
        same_central = jax.lax.cond(
            galaxy_is_central,
            lambda _: galaxy,
            lambda _: central,
            operand=None,
        )
        return StarburstResult(
            galaxy,
            same_central,
            _zero_transfer(trigger, burst_efficiency),
        )

    return jax.lax.cond(trigger > 0.0, apply_active, apply_zero, operand=None)


def apply_disk_instability_starburst(
    galaxy: GalaxyState,
    central: GalaxyState,
    halo: HaloForcing,
    central_halo: HaloForcing,
    parameters: Sage16Parameters,
    units: Sage16Units,
    perturbations,
) -> StarburstResult:
    """Consume the live instability trigger using the upstream full-interval rate."""

    valid_time = halo.dT > 0.0

    def apply_valid(_):
        return apply_collisional_starburst(
            galaxy,
            central,
            halo,
            central_halo,
            galaxy.UnstableDiskGasFraction,
            1,
            halo.dT,
            parameters,
            units,
            perturbations.starburst,
            perturbations.sn_reheating,
            perturbations.sn_ejection,
        )

    def apply_invalid(_):
        return StarburstResult(
            galaxy,
            central,
            _zero_transfer(as_float64(galaxy.UnstableDiskGasFraction), zero),
        )

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return jax.lax.cond(valid_time, apply_valid, apply_invalid, operand=None)
