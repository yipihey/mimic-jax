"""Fiducial quiescent star-formation, SN-feedback, and disk-yield chain."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.perturbations import (
    log_fractionally_perturb,
    process_perturbations,
)
from mimic_jax.sage16.precision import as_float32, as_float64, require_x64
from mimic_jax.sage16.processes.common import metallicity, object_substep_dt
from mimic_jax.sage16.transfers import (
    MetalEnrichmentResult,
    MetalEnrichmentTransfer,
    QuiescentStepResult,
    StarFormationApplyResult,
    StarFormationBudget,
    StarFormationTransfer,
)
from mimic_jax.sage16.types import (
    GalaxyState,
    HaloForcing,
    Sage16Parameters,
    Sage16Units,
    StepContext,
)

SOLAR_MASS = 1.989e33
ENERGY_SN = 1.0e51
ETA_SN = 5.0e-3
SAGE_COLD_GAS_YIELD_THRESHOLD = 1.0e-8
SAGE_METAL_EJECTION_MVIR_SCALE = 30.0


def calculate_star_formation_budget(
    state: GalaxyState,
    halo: HaloForcing,
    context: StepContext,
    parameters: Sage16Parameters,
    log_fractional_perturbation=0.0,
) -> StarFormationBudget:
    """Calculate ``NewStellarMass`` using the upstream threshold and finite substep."""

    require_x64()
    dt = object_substep_dt(halo, context)
    reff = parameters.StarFormingDiskFactor * as_float64(state.DiskScaleRadius)
    tdyn = reff / halo.Vvir
    cold_crit = 0.19 * halo.Vvir * reff
    cold = as_float64(state.ColdGas)
    strdot = jnp.where(
        (cold > cold_crit) & (tdyn > 0.0),
        parameters.SfrEfficiency * (cold - cold_crit) / tdyn,
        0.0,
    )
    stars = jnp.maximum(
        log_fractionally_perturb(strdot, log_fractional_perturbation) * dt,
        0.0,
    )
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return StarFormationBudget(stars, zero, zero)


def calculate_supernova_feedback_budget(
    state: GalaxyState,
    central_halo: HaloForcing,
    parameters: Sage16Parameters,
    units: Sage16Units,
    budget: StarFormationBudget,
    log_reheating_perturbation=0.0,
    log_ejection_perturbation=0.0,
) -> StarFormationBudget:
    """Calculate and renormalise the SAGE16 reheating/ejection transport budgets."""

    require_x64()
    stars = as_float64(budget.NewStellarMass)
    reheated = log_fractionally_perturb(
        parameters.FeedbackReheatingEpsilon * stars,
        log_reheating_perturbation,
    )
    combined = stars + reheated
    renormalise = (combined > as_float64(state.ColdGas)) & (combined > 0.0)
    factor = jnp.where(renormalise, as_float64(state.ColdGas) / combined, 1.0)
    stars = stars * factor
    reheated = reheated * factor

    energy_sn_code = ENERGY_SN / units.UnitEnergy_in_cgs * units.Hubble_h
    eta_sn_code = ETA_SN * (units.UnitMass_in_g / SOLAR_MASS) / units.Hubble_h
    ejected = jnp.where(
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
    ejected = jnp.maximum(
        log_fractionally_perturb(ejected, log_ejection_perturbation),
        0.0,
    )
    return StarFormationBudget(stars, reheated, ejected)


def _zero_star_formation_transfer():
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return StarFormationTransfer(zero, zero, zero, zero, zero, zero, zero)


def _apply_positive_star_formation(
    galaxy: GalaxyState,
    central: GalaxyState,
    halo: HaloForcing,
    parameters: Sage16Parameters,
    budget: StarFormationBudget,
    is_central: bool,
) -> StarFormationApplyResult:
    stars = as_float64(budget.NewStellarMass)
    reheated = as_float64(budget.SupernovaReheatedMass)
    proposed_ejected = as_float64(budget.SupernovaEjectedMass)
    locked_stars = (1.0 - parameters.RecycleFraction) * stars

    cold_metallicity = metallicity(galaxy.ColdGas, galaxy.MetalsColdGas)
    metals_to_stars = cold_metallicity * locked_stars
    local = galaxy._replace(
        ColdGas=as_float32(as_float64(galaxy.ColdGas) - locked_stars),
        MetalsColdGas=as_float32(as_float64(galaxy.MetalsColdGas) - metals_to_stars),
        StellarMass=as_float32(as_float64(galaxy.StellarMass) + locked_stars),
        MetalsStellarMass=as_float32(as_float64(galaxy.MetalsStellarMass) + metals_to_stars),
        StarFormationRate=as_float32(
            as_float64(galaxy.StarFormationRate) + jnp.where(halo.dT > 0.0, stars / halo.dT, 0.0)
        ),
        NewStellarMass=stars,
        SupernovaReheatedMass=as_float64(budget.SupernovaReheatedMass),
        SupernovaEjectedMass=as_float64(budget.SupernovaEjectedMass),
    )

    post_sf_metallicity = metallicity(local.ColdGas, local.MetalsColdGas)
    metals_to_hot = post_sf_metallicity * reheated
    local = local._replace(
        ColdGas=as_float32(as_float64(local.ColdGas) - reheated),
        MetalsColdGas=as_float32(as_float64(local.MetalsColdGas) - metals_to_hot),
        SupernovaOutflowRate=as_float32(
            as_float64(local.SupernovaOutflowRate)
            + jnp.where(halo.dT > 0.0, reheated / halo.dT, 0.0)
        ),
    )

    destination = jax.lax.cond(is_central, lambda _: local, lambda _: central, operand=None)
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
        MetalsEjectedGas=as_float32(as_float64(destination.MetalsEjectedGas) + metals_to_ejected),
    )

    local = local._replace(
        SupernovaReheatedMass=jnp.asarray(0.0, dtype=jnp.float64),
        SupernovaEjectedMass=jnp.asarray(0.0, dtype=jnp.float64),
    )
    local = jax.lax.cond(is_central, lambda _: destination, lambda _: local, operand=None)
    destination = jax.lax.cond(is_central, lambda _: local, lambda _: destination, operand=None)
    transfer = StarFormationTransfer(
        formed_stars=stars,
        locked_stars=locked_stars,
        cold_to_hot=reheated,
        hot_to_ejected=ejected,
        cold_metals_to_stars=metals_to_stars,
        cold_metals_to_hot=metals_to_hot,
        hot_metals_to_ejected=metals_to_ejected,
    )
    return StarFormationApplyResult(local, destination, transfer)


def apply_star_formation_supernova(
    galaxy: GalaxyState,
    central: GalaxyState,
    halo: HaloForcing,
    parameters: Sage16Parameters,
    budget: StarFormationBudget,
) -> StarFormationApplyResult:
    """Commit the calculated SF/SN budgets to local and FoF-central reservoirs."""

    require_x64()

    def apply_positive(_):
        return _apply_positive_star_formation(
            galaxy, central, halo, parameters, budget, halo.Type == 0
        )

    def apply_zero(_):
        updated = galaxy._replace(
            NewStellarMass=as_float64(budget.NewStellarMass),
            SupernovaReheatedMass=jnp.asarray(0.0, dtype=jnp.float64),
            SupernovaEjectedMass=jnp.asarray(0.0, dtype=jnp.float64),
        )
        same_central = jax.lax.cond(halo.Type == 0, lambda _: updated, lambda _: central, None)
        return StarFormationApplyResult(updated, same_central, _zero_star_formation_transfer())

    return jax.lax.cond(budget.NewStellarMass > 0.0, apply_positive, apply_zero, operand=None)


def apply_metal_enrichment(
    galaxy: GalaxyState,
    central: GalaxyState,
    central_halo: HaloForcing,
    galaxy_is_central,
    parameters: Sage16Parameters,
) -> MetalEnrichmentResult:
    """Apply SAGE's delayed disk-SF yield and expose it as an explicit metal source."""

    require_x64()
    stars = as_float64(galaxy.NewStellarMass)
    consumed = galaxy._replace(NewStellarMass=jnp.asarray(0.0, dtype=jnp.float64))

    def apply_positive(_):
        fraction_leaving = parameters.FracZleaveDisk * jnp.exp(
            -central_halo.Mvir / SAGE_METAL_EJECTION_MVIR_SCALE
        )
        cold_above_floor = consumed.ColdGas > SAGE_COLD_GAS_YIELD_THRESHOLD
        metals_to_cold = jnp.where(
            cold_above_floor, parameters.Yield * (1.0 - fraction_leaving) * stars, 0.0
        )
        metals_to_hot = parameters.Yield * stars - metals_to_cold
        local = consumed._replace(
            MetalsColdGas=as_float32(as_float64(consumed.MetalsColdGas) + metals_to_cold)
        )
        destination = jax.lax.cond(
            galaxy_is_central, lambda _: local, lambda _: central, operand=None
        )
        destination = destination._replace(
            MetalsHotGas=as_float32(as_float64(destination.MetalsHotGas) + metals_to_hot)
        )
        local = jax.lax.cond(galaxy_is_central, lambda _: destination, lambda _: local, None)
        destination = jax.lax.cond(
            galaxy_is_central, lambda _: local, lambda _: destination, operand=None
        )
        transfer = MetalEnrichmentTransfer(parameters.Yield * stars, metals_to_cold, metals_to_hot)
        return MetalEnrichmentResult(local, destination, transfer)

    def apply_zero(_):
        destination = jax.lax.cond(
            galaxy_is_central, lambda _: consumed, lambda _: central, operand=None
        )
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        return MetalEnrichmentResult(
            consumed, destination, MetalEnrichmentTransfer(zero, zero, zero)
        )

    return jax.lax.cond(stars > 0.0, apply_positive, apply_zero, operand=None)


def quiescent_disk_step(
    galaxy: GalaxyState,
    central: GalaxyState,
    halo: HaloForcing,
    central_halo: HaloForcing,
    context: StepContext,
    parameters: Sage16Parameters,
    units: Sage16Units,
    perturbations=None,
) -> QuiescentStepResult:
    """Run the faithful disk-SF/SN/yield chain for a disk-stable substep."""

    if perturbations is None:
        perturbations = process_perturbations()
    sf_budget = calculate_star_formation_budget(
        galaxy,
        halo,
        context,
        parameters,
        perturbations.star_formation,
    )
    sn_budget = calculate_supernova_feedback_budget(
        galaxy,
        central_halo,
        parameters,
        units,
        sf_budget,
        perturbations.sn_reheating,
        perturbations.sn_ejection,
    )
    applied = apply_star_formation_supernova(galaxy, central, halo, parameters, sn_budget)
    enriched = apply_metal_enrichment(
        applied.galaxy,
        applied.central,
        central_halo,
        halo.Type == 0,
        parameters,
    )
    return QuiescentStepResult(
        galaxy=enriched.galaxy,
        central=enriched.central,
        budget=sn_budget,
        transfer=applied.transfer,
        enrichment=enriched.transfer,
    )
