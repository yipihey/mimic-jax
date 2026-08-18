"""Fiducial SAGE16 metallicity-dependent cooling-budget calculation."""

import jax
import jax.numpy as jnp

from mimic_jax.sage16.cooling_tables import CoolingTables, metal_dependent_cooling_rate
from mimic_jax.sage16.precision import as_float64, require_x64
from mimic_jax.sage16.processes.common import object_substep_dt
from mimic_jax.sage16.transfers import CoolingBudget, CoolingBudgetResult
from mimic_jax.sage16.types import GalaxyState, HaloForcing, Sage16Units, StepContext

PROTON_MASS = 1.6726e-24
BOLTZMANN = 1.3806e-16
VIRIAL_TEMPERATURE_COEFFICIENT = 35.9
IONIZED_GAS_DENSITY_FACTOR = 0.885


def calculate_cooling_budget(
    state: GalaxyState,
    halo: HaloForcing,
    context: StepContext,
    units: Sage16Units,
    tables: CoolingTables,
) -> CoolingBudgetResult:
    """Calculate SAGE's finite-substep cooling amount without moving reservoirs."""

    require_x64()
    active = (state.HotGas > 0.0) & (halo.Vvir > 0.0)

    def calculate(_):
        dt = object_substep_dt(halo, context)
        cooling_time = halo.Rvir / halo.Vvir
        temperature = VIRIAL_TEMPERATURE_COEFFICIENT * halo.Vvir * halo.Vvir
        log_metallicity = jax.lax.cond(
            state.MetalsHotGas > 0.0,
            lambda _: jnp.log10(as_float64(state.MetalsHotGas) / as_float64(state.HotGas)),
            lambda _: jnp.asarray(-10.0, dtype=jnp.float64),
            operand=None,
        )
        cooling_lambda = metal_dependent_cooling_rate(
            jnp.log10(temperature),
            log_metallicity,
            tables,
        )
        density_conversion = PROTON_MASS * BOLTZMANN * temperature / cooling_lambda
        density_conversion /= units.UnitDensity_in_cgs * units.UnitTime_in_s
        density_at_cooling_radius = density_conversion / cooling_time * IONIZED_GAS_DENSITY_FACTOR
        central_density = as_float64(state.HotGas) / (4.0 * jnp.pi * halo.Rvir)
        cooling_radius = jnp.sqrt(central_density / density_at_cooling_radius)
        cold_accretion = as_float64(state.HotGas) / cooling_time * dt
        hot_halo_cooling = (
            as_float64(state.HotGas) / halo.Rvir * (cooling_radius / (2.0 * cooling_time)) * dt
        )
        cooling_gas = jnp.where(
            cooling_radius > halo.Rvir,
            cold_accretion,
            hot_halo_cooling,
        )
        cooling_gas = jnp.where(
            cooling_gas > as_float64(state.HotGas),
            as_float64(state.HotGas),
            jnp.maximum(cooling_gas, 0.0),
        )
        budget = CoolingBudget(cooling_gas, cooling_radius, cooling_lambda)
        return CoolingBudgetResult(
            state._replace(
                CoolingGas=cooling_gas,
                Rcool=cooling_radius,
                CoolingLambda=cooling_lambda,
            ),
            budget,
        )

    def inactive(_):
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        budget = CoolingBudget(zero, zero, zero)
        return CoolingBudgetResult(
            state._replace(CoolingGas=zero, Rcool=zero, CoolingLambda=zero),
            budget,
        )

    return jax.lax.cond(active, calculate, inactive, operand=None)
