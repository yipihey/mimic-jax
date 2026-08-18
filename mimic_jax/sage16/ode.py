"""Continuous-rate limit of the quiescent SAGE16 central baryon cycle."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from mimic_jax.numerics import FixedStepSolution, integrate_fixed_step
from mimic_jax.sage16.cooling_tables import CoolingTables, metal_dependent_cooling_rate
from mimic_jax.sage16.perturbations import log_fractionally_perturb, process_perturbations
from mimic_jax.sage16.precision import as_float64, require_x64
from mimic_jax.sage16.processes.common import metallicity
from mimic_jax.sage16.processes.cooling import apply_cooling
from mimic_jax.sage16.processes.cooling_budget import (
    BOLTZMANN,
    IONIZED_GAS_DENSITY_FACTOR,
    PROTON_MASS,
    VIRIAL_TEMPERATURE_COEFFICIENT,
    calculate_cooling_budget,
)
from mimic_jax.sage16.processes.reincorporation import apply_reincorporation
from mimic_jax.sage16.processes.star_formation import (
    ENERGY_SN,
    ETA_SN,
    SAGE_COLD_GAS_YIELD_THRESHOLD,
    SAGE_METAL_EJECTION_MVIR_SCALE,
    SOLAR_MASS,
    apply_metal_enrichment,
    apply_star_formation_supernova,
    calculate_star_formation_budget,
    calculate_supernova_feedback_budget,
)
from mimic_jax.sage16.transfers import CentralHistoryResult, CentralStepDiagnostics
from mimic_jax.sage16.types import (
    GalaxyState,
    HaloForcing,
    Sage16Parameters,
    Sage16Units,
    StepContext,
)

Array = Any

SAGE16_ODE_RATE_SUBSET = "sage16_ode_rate_subset"
UPSTREAM_RATE_SUBSET = "upstream_rate_subset"
ODE_STATE_NAMES = (
    "ColdGas",
    "HotGas",
    "EjectedGas",
    "StellarMass",
    "MetalsColdGas",
    "MetalsHotGas",
    "MetalsEjectedGas",
    "MetalsStellarMass",
)


class Sage16OdeState(NamedTuple):
    """Continuous reservoirs in the initial quiescent central-galaxy ODE."""

    ColdGas: Array
    HotGas: Array
    EjectedGas: Array
    StellarMass: Array
    MetalsColdGas: Array
    MetalsHotGas: Array
    MetalsEjectedGas: Array
    MetalsStellarMass: Array


class Sage16OdeRates(NamedTuple):
    """Named physical rates whose stoichiometry defines the reservoir RHS."""

    cooling: Array
    star_formation: Array
    locked_stars: Array
    sn_reheating: Array
    sn_ejection: Array
    reincorporation: Array
    produced_metals: Array
    new_metals_to_cold: Array
    new_metals_to_hot: Array


class Sage16OdeRhsResult(NamedTuple):
    """Reservoir derivative and the physical rates that generated it."""

    derivative: Sage16OdeState
    rates: Sage16OdeRates


def ode_state_from_galaxy(state: GalaxyState) -> Sage16OdeState:
    """Project stored SAGE reservoirs into a float64 continuous state."""

    require_x64()
    return Sage16OdeState(*(as_float64(getattr(state, name)) for name in ODE_STATE_NAMES))


def galaxy_from_ode_state(template: GalaxyState, state: Sage16OdeState) -> GalaxyState:
    """Insert an ODE state into a galaxy template without changing other fields."""

    return template._replace(**state._asdict())


def calculate_continuous_cooling_rate(
    state: Sage16OdeState,
    halo: HaloForcing,
    units: Sage16Units,
    tables: CoolingTables,
):
    """Return the uncapped SAGE cooling rate and its instantaneous radius."""

    active = (state.HotGas > 0.0) & (halo.Vvir > 0.0)

    def calculate(_):
        cooling_time = halo.Rvir / halo.Vvir
        temperature = VIRIAL_TEMPERATURE_COEFFICIENT * halo.Vvir**2
        log_metallicity = jnp.where(
            state.MetalsHotGas > 0.0,
            jnp.log10(state.MetalsHotGas / state.HotGas),
            -10.0,
        )
        cooling_lambda = metal_dependent_cooling_rate(
            jnp.log10(temperature),
            log_metallicity,
            tables,
        )
        density_conversion = PROTON_MASS * BOLTZMANN * temperature / cooling_lambda
        density_conversion /= units.UnitDensity_in_cgs * units.UnitTime_in_s
        density_at_cooling_radius = density_conversion / cooling_time * IONIZED_GAS_DENSITY_FACTOR
        central_density = state.HotGas / (4.0 * jnp.pi * halo.Rvir)
        cooling_radius = jnp.sqrt(central_density / density_at_cooling_radius)
        cold_accretion = state.HotGas / cooling_time
        hot_halo_cooling = state.HotGas / halo.Rvir * cooling_radius / (2.0 * cooling_time)
        rate = jnp.where(cooling_radius > halo.Rvir, cold_accretion, hot_halo_cooling)
        return jnp.maximum(rate, 0.0), cooling_radius, cooling_lambda

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return jax.lax.cond(active, calculate, lambda _: (zero, zero, zero), operand=None)


def _star_formation_rate(
    state: Sage16OdeState,
    disk_scale_radius,
    halo: HaloForcing,
    parameters: Sage16Parameters,
):
    reff = parameters.StarFormingDiskFactor * as_float64(disk_scale_radius)
    tdyn = reff / halo.Vvir
    cold_critical = 0.19 * halo.Vvir * reff
    safe_tdyn = jnp.where(tdyn > 0.0, tdyn, 1.0)
    return jnp.where(
        (state.ColdGas > cold_critical) & (tdyn > 0.0),
        parameters.SfrEfficiency * (state.ColdGas - cold_critical) / safe_tdyn,
        0.0,
    )


def _reincorporation_rate(
    state: Sage16OdeState,
    halo: HaloForcing,
    parameters: Sage16Parameters,
):
    critical_velocity = 445.48 * parameters.ReIncorporationFactor
    active = (halo.Type == 0) & (state.EjectedGas > 0.0) & (halo.Vvir > critical_velocity)
    rate = (halo.Vvir / critical_velocity - 1.0) * state.EjectedGas / (halo.Rvir / halo.Vvir)
    return jnp.where(active, rate, 0.0)


def _supernova_rates(
    state: Sage16OdeState,
    halo: HaloForcing,
    parameters: Sage16Parameters,
    units: Sage16Units,
    star_formation_rate,
    reheating_perturbation,
    ejection_perturbation,
):
    reheating = log_fractionally_perturb(
        parameters.FeedbackReheatingEpsilon * star_formation_rate,
        reheating_perturbation,
    )
    energy_sn_code = ENERGY_SN / units.UnitEnergy_in_cgs * units.Hubble_h
    eta_sn_code = ETA_SN * (units.UnitMass_in_g / SOLAR_MASS) / units.Hubble_h
    proposed_ejection = jnp.where(
        halo.Vvir > 0.0,
        (
            parameters.FeedbackEjectionEfficiency * eta_sn_code * energy_sn_code / halo.Vvir**2
            - parameters.FeedbackReheatingEpsilon
        )
        * star_formation_rate,
        0.0,
    )
    proposed_ejection = jnp.maximum(
        log_fractionally_perturb(proposed_ejection, ejection_perturbation),
        0.0,
    )
    # This is the infinitesimal limit of upstream's post-reheating hot-gas cap.
    # Away from the HotGas=0 boundary the requested rate is unchanged.
    ejection = jnp.where(
        state.HotGas > 0.0,
        proposed_ejection,
        jnp.minimum(proposed_ejection, reheating),
    )
    return reheating, ejection


def sage16_ode_rhs_and_rates(
    time,
    state: Sage16OdeState,
    halo: HaloForcing,
    disk_scale_radius,
    parameters: Sage16Parameters,
    units: Sage16Units,
    cooling_tables: CoolingTables,
    perturbations=None,
) -> Sage16OdeRhsResult:
    """Evaluate the piecewise-smooth continuous limit of the rate-based subset.

    Halo forcing and disk size are held fixed over the interval. Thresholds are
    reproduced exactly; no smoothing is introduced. ``time`` is accepted for a
    standard non-autonomous RHS signature and reserved for interpolated forcing.
    """

    del time
    require_x64()
    if perturbations is None:
        perturbations = process_perturbations()

    cooling, _, _ = calculate_continuous_cooling_rate(state, halo, units, cooling_tables)
    cooling = log_fractionally_perturb(cooling, perturbations.cooling)
    star_formation = _star_formation_rate(state, disk_scale_radius, halo, parameters)
    star_formation = log_fractionally_perturb(
        star_formation,
        perturbations.star_formation,
    )
    locked_stars = (1.0 - parameters.RecycleFraction) * star_formation
    reheating, ejection = _supernova_rates(
        state,
        halo,
        parameters,
        units,
        star_formation,
        perturbations.sn_reheating,
        perturbations.sn_ejection,
    )
    reincorporation = _reincorporation_rate(state, halo, parameters)
    reincorporation = log_fractionally_perturb(
        reincorporation,
        perturbations.reincorporation,
    )

    fraction_leaving = parameters.FracZleaveDisk * jnp.exp(
        -halo.Mvir / SAGE_METAL_EJECTION_MVIR_SCALE
    )
    produced_metals = parameters.Yield * star_formation
    new_metals_to_cold = jnp.where(
        state.ColdGas > SAGE_COLD_GAS_YIELD_THRESHOLD,
        produced_metals * (1.0 - fraction_leaving),
        0.0,
    )
    new_metals_to_hot = produced_metals - new_metals_to_cold

    cold_metallicity = metallicity(state.ColdGas, state.MetalsColdGas)
    hot_metallicity = metallicity(state.HotGas, state.MetalsHotGas)
    ejected_metallicity = metallicity(state.EjectedGas, state.MetalsEjectedGas)
    ejection_metallicity = jnp.where(
        state.HotGas > 0.0,
        hot_metallicity,
        cold_metallicity,
    )

    derivative = Sage16OdeState(
        ColdGas=cooling - locked_stars - reheating,
        HotGas=-cooling + reheating - ejection + reincorporation,
        EjectedGas=ejection - reincorporation,
        StellarMass=locked_stars,
        MetalsColdGas=(
            hot_metallicity * cooling
            - cold_metallicity * (locked_stars + reheating)
            + new_metals_to_cold
        ),
        MetalsHotGas=(
            -hot_metallicity * cooling
            + cold_metallicity * reheating
            - ejection_metallicity * ejection
            + ejected_metallicity * reincorporation
            + new_metals_to_hot
        ),
        MetalsEjectedGas=(ejection_metallicity * ejection - ejected_metallicity * reincorporation),
        MetalsStellarMass=cold_metallicity * locked_stars,
    )
    rates = Sage16OdeRates(
        cooling=cooling,
        star_formation=star_formation,
        locked_stars=locked_stars,
        sn_reheating=reheating,
        sn_ejection=ejection,
        reincorporation=reincorporation,
        produced_metals=produced_metals,
        new_metals_to_cold=new_metals_to_cold,
        new_metals_to_hot=new_metals_to_hot,
    )
    return Sage16OdeRhsResult(derivative=derivative, rates=rates)


def sage16_ode_rhs(
    time,
    state: Sage16OdeState,
    halo: HaloForcing,
    disk_scale_radius,
    parameters: Sage16Parameters,
    units: Sage16Units,
    cooling_tables: CoolingTables,
    perturbations=None,
) -> Sage16OdeState:
    """Return only ``dx/dt`` for use by numerical integrators and AD."""

    return sage16_ode_rhs_and_rates(
        time,
        state,
        halo,
        disk_scale_radius,
        parameters,
        units,
        cooling_tables,
        perturbations,
    ).derivative


def integrate_sage16_ode(
    initial_state: Sage16OdeState,
    halo: HaloForcing,
    disk_scale_radius,
    parameters: Sage16Parameters,
    units: Sage16Units,
    cooling_tables: CoolingTables,
    *,
    duration=None,
    num_steps: int,
    method: str,
    perturbations=None,
) -> FixedStepSolution:
    """Integrate the fixed-forcing quiescent central SAGE16 ODE subset."""

    if duration is None:
        duration = halo.dT

    def rhs(time, state):
        return sage16_ode_rhs(
            time,
            state,
            halo,
            disk_scale_radius,
            parameters,
            units,
            cooling_tables,
            perturbations,
        )

    return integrate_fixed_step(
        rhs,
        initial_state,
        duration=duration,
        num_steps=num_steps,
        method=method,
    )


def upstream_rate_subset_step(
    state: GalaxyState,
    halo: HaloForcing,
    context: StepContext,
    parameters: Sage16Parameters,
    units: Sage16Units,
    cooling_tables: CoolingTables,
    perturbations=None,
):
    """Apply only the upstream modules represented in the continuous RHS."""

    if perturbations is None:
        perturbations = process_perturbations()
    reincorporated = apply_reincorporation(
        state,
        halo,
        context,
        parameters,
        perturbations.reincorporation,
    )
    cooling_budget = calculate_cooling_budget(
        reincorporated.state,
        halo,
        context,
        units,
        cooling_tables,
    )
    cooled = apply_cooling(
        cooling_budget.state,
        halo,
        log_fractional_perturbation=perturbations.cooling,
    )
    star_formation_budget = calculate_star_formation_budget(
        cooled.state,
        halo,
        context,
        parameters,
        perturbations.star_formation,
    )
    supernova_budget = calculate_supernova_feedback_budget(
        cooled.state,
        halo,
        parameters,
        units,
        star_formation_budget,
        perturbations.sn_reheating,
        perturbations.sn_ejection,
    )
    applied = apply_star_formation_supernova(
        cooled.state,
        cooled.state,
        halo,
        parameters,
        supernova_budget,
    )
    enriched = apply_metal_enrichment(
        applied.galaxy,
        applied.central,
        halo,
        True,
        parameters,
    )
    diagnostics = CentralStepDiagnostics(
        cooling=cooled.transfer,
        reincorporation=reincorporated.transfer,
        star_formation=applied.transfer,
        enrichment=enriched.transfer,
    )
    return enriched.galaxy, diagnostics


def subcycle_upstream_rate_subset(
    initial_state: GalaxyState,
    halo: HaloForcing,
    context: StepContext,
    parameters: Sage16Parameters,
    units: Sage16Units,
    cooling_tables: CoolingTables,
    *,
    num_substeps: int,
    perturbations=None,
) -> CentralHistoryResult:
    """Repeat the equivalent upstream sequential rate modules at fixed forcing."""

    if not isinstance(num_substeps, int) or num_substeps <= 0:
        raise ValueError("num_substeps must be a positive Python integer")
    if perturbations is None:
        perturbations = process_perturbations()
    substep_numbers = jnp.arange(num_substeps, dtype=jnp.int32)
    interval_dt = context.time_interval / num_substeps

    def scan_step(state, substep_number):
        substep_context = context._replace(
            substep_number=substep_number,
            num_substeps=jnp.asarray(num_substeps, dtype=jnp.int32),
            substep_dt=interval_dt,
            substep_time=(context.time + context.time_interval)
            - (as_float64(substep_number) + 0.5) * interval_dt,
        )
        new_state, diagnostics = upstream_rate_subset_step(
            state,
            halo,
            substep_context,
            parameters,
            units,
            cooling_tables,
            perturbations,
        )
        return new_state, (new_state, diagnostics)

    final_state, (states, diagnostics) = jax.lax.scan(
        scan_step,
        initial_state,
        substep_numbers,
    )
    return CentralHistoryResult(final_state, states, diagnostics)
