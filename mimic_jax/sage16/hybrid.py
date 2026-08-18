"""Hybrid continuous-flow, projection, and event-facing SAGE16 primitives.

This module is deliberately separate from the exact upstream-sequential path.
It exposes a Markov state for persistent reservoirs and history variables, a
continuous limit for prescriptions with a defensible rate interpretation, and
explicit projections for upstream quantities such as the AGN heating radius.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from mimic_jax.sage16.cooling_tables import CoolingTables
from mimic_jax.sage16.ode import (
    Sage16OdeState,
    calculate_continuous_cooling_rate,
    sage16_ode_rhs_and_rates,
)
from mimic_jax.sage16.perturbations import log_fractionally_perturb, process_perturbations
from mimic_jax.sage16.precision import as_float64, require_x64
from mimic_jax.sage16.processes.common import metallicity
from mimic_jax.sage16.types import GalaxyState, HaloForcing, Sage16Parameters, Sage16Units

Array = Any

SECONDS_PER_YEAR = 3.155e7
SOLAR_MASS_G = 1.989e33
PROTON_MASS_G = 1.6726e-24
BOLTZMANN_CGS = 1.3806e-16
VIRIAL_TEMPERATURE_COEFFICIENT = 35.9
HEATING_VELOCITY_KM_S = 1.34e5

HYBRID_STATE_NAMES = (
    "HaloBaryonFraction",
    "ColdGas",
    "HotGas",
    "EjectedGas",
    "StellarMass",
    "BulgeMass",
    "ICS",
    "BlackHoleMass",
    "MetalsColdGas",
    "MetalsHotGas",
    "MetalsEjectedGas",
    "MetalsStellarMass",
    "MetalsBulgeMass",
    "MetalsICS",
    "Rheat",
    "DiskScaleRadius",
    "MergTime",
    "TimeOfLastMajorMerger",
    "TimeOfLastMinorMerger",
)


class Sage16HybridState(NamedTuple):
    """Persistent state needed by SAGE16 flows, projections, and event maps.

    Snapshot transport budgets and accumulated output diagnostics are omitted:
    they are controls or quadratures, not physical state. ``Rheat`` and the
    merger clock are included because they make their prescriptions Markovian.
    """

    HaloBaryonFraction: Array
    ColdGas: Array
    HotGas: Array
    EjectedGas: Array
    StellarMass: Array
    BulgeMass: Array
    ICS: Array
    BlackHoleMass: Array
    MetalsColdGas: Array
    MetalsHotGas: Array
    MetalsEjectedGas: Array
    MetalsStellarMass: Array
    MetalsBulgeMass: Array
    MetalsICS: Array
    Rheat: Array
    DiskScaleRadius: Array
    MergTime: Array
    TimeOfLastMajorMerger: Array
    TimeOfLastMinorMerger: Array


class PreparedInfallForcing(NamedTuple):
    """Snapshot infall budget expressed as external forcing over an interval."""

    budget: Array
    interval_duration: Array
    net_gas_rate: Array


class ContinuousInfallRates(NamedTuple):
    """Piecewise source/sink rates produced by a prepared infall forcing."""

    external_to_hot: Array
    ejected_to_external: Array
    hot_to_external: Array
    ejected_metals_to_external: Array
    hot_metals_to_external: Array
    external_baryon_rate: Array
    external_metal_rate: Array


class RadioModeFlowRates(NamedTuple):
    """AGN flow rates plus the candidate value for the separate ``Rheat`` map."""

    raw_cooling: Array
    cooling_after_prior_heating: Array
    black_hole_accretion: Array
    hot_metals_accreted: Array
    heating: Array
    cooling_radius: Array
    cooling_lambda: Array
    candidate_heating_radius: Array


class Sage16HybridRates(NamedTuple):
    """Named rates used to assemble the single-galaxy hybrid RHS."""

    cooling: Array
    star_formation: Array
    locked_stars: Array
    sn_reheating: Array
    sn_ejection: Array
    reincorporation: Array
    produced_metals: Array
    new_metals_to_cold: Array
    new_metals_to_hot: Array
    radio_black_hole_accretion: Array
    radio_hot_metals_accreted: Array
    agn_heating: Array
    cooling_radius: Array
    cooling_lambda: Array
    candidate_heating_radius: Array
    infall_to_hot: Array
    infall_from_ejected: Array
    infall_from_hot: Array
    external_baryon_rate: Array
    external_metal_rate: Array


class Sage16HybridRhsResult(NamedTuple):
    """Continuous derivative, rates, and explicit external-source ledger."""

    derivative: Sage16HybridState
    rates: Sage16HybridRates


class HeatingRadiusProjectionResult(NamedTuple):
    """Result of the monotone, history-carrying SAGE16 AGN projection."""

    state: Sage16HybridState
    previous_radius: Array
    candidate_radius: Array
    projected_radius: Array
    applied: Array


class StrippingPairState(NamedTuple):
    """Continuous state of one Type-1 satellite and its receiving central."""

    satellite: Sage16HybridState
    central: Sage16HybridState


class ContinuousStrippingRates(NamedTuple):
    """Group-coupled hot-gas stripping flow and its structural ledger."""

    derivative: StrippingPairState
    gas: Array
    metals: Array
    allowed_baryons: Array
    satellite_baryons: Array


def hybrid_state_from_galaxy(state: GalaxyState) -> Sage16HybridState:
    """Project the canonical SAGE record into its persistent Markov state."""

    require_x64()
    return Sage16HybridState(*(as_float64(getattr(state, name)) for name in HYBRID_STATE_NAMES))


def galaxy_from_hybrid_state(template: GalaxyState, state: Sage16HybridState) -> GalaxyState:
    """Insert hybrid-state values into a canonical record, retaining diagnostics."""

    return template._replace(**state._asdict())


def zero_hybrid_derivative(state: Sage16HybridState) -> Sage16HybridState:
    """Return a shape- and dtype-compatible zero tangent for a hybrid state."""

    return jax.tree_util.tree_map(jnp.zeros_like, state)


def prepared_infall_forcing(budget, interval_duration) -> PreparedInfallForcing:
    """Convert upstream's signed snapshot budget to piecewise-constant forcing.

    This is a continuous counterpart of ``InfallingGas / num_substeps``. It is
    not claimed to reproduce an alternative smoothly interpolated halo history.
    """

    budget = as_float64(budget)
    interval_duration = as_float64(interval_duration)
    rate = jnp.where(interval_duration > 0.0, budget / interval_duration, 0.0)
    return PreparedInfallForcing(budget, interval_duration, rate)


def continuous_infall_rates(
    state: Sage16HybridState,
    forcing: PreparedInfallForcing,
    log_fractional_perturbation=0.0,
) -> ContinuousInfallRates:
    """Resolve prepared infall into pristine supply or ordered external removal.

    Negative forcing removes ejected gas first and hot gas after the ejected
    reservoir reaches zero. Reservoir exhaustion is therefore a boundary event
    for a finite-step integrator; this RHS defines the flow on either side.
    """

    rate = log_fractionally_perturb(
        forcing.net_gas_rate,
        log_fractional_perturbation,
    )
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    external_to_hot = jnp.maximum(rate, zero)
    removal = jnp.maximum(-rate, zero)
    ejected_to_external = jnp.where(state.EjectedGas > 0.0, removal, zero)
    hot_to_external = jnp.where(
        (state.EjectedGas <= 0.0) & (state.HotGas > 0.0),
        removal,
        zero,
    )
    ejected_metals = metallicity(state.EjectedGas, state.MetalsEjectedGas)
    hot_metals = metallicity(state.HotGas, state.MetalsHotGas)
    ejected_metals_to_external = ejected_metals * ejected_to_external
    hot_metals_to_external = hot_metals * hot_to_external
    fulfilled_removal = ejected_to_external + hot_to_external
    return ContinuousInfallRates(
        external_to_hot=external_to_hot,
        ejected_to_external=ejected_to_external,
        hot_to_external=hot_to_external,
        ejected_metals_to_external=ejected_metals_to_external,
        hot_metals_to_external=hot_metals_to_external,
        external_baryon_rate=external_to_hot - fulfilled_removal,
        external_metal_rate=-(ejected_metals_to_external + hot_metals_to_external),
    )


def _empirical_radio_rate(state, halo, parameters, units):
    unit_conversion = units.UnitMass_in_g / units.UnitTime_in_s * SECONDS_PER_YEAR / SOLAR_MASS_G
    velocity_ratio = halo.Vvir / 200.0
    normalized = (
        parameters.RadioModeEfficiency
        / unit_conversion
        * (state.BlackHoleMass / 0.01)
        * velocity_ratio**3
    )
    return jnp.where(
        halo.Mvir > 0.0,
        normalized * ((state.HotGas / halo.Mvir) / 0.1),
        normalized,
    )


def _bondi_radio_rate(state, halo, parameters, units, cooling_lambda):
    temperature = VIRIAL_TEMPERATURE_COEFFICIENT * halo.Vvir**2
    density_time = PROTON_MASS_G * BOLTZMANN_CGS * temperature / cooling_lambda
    density_time /= units.UnitDensity_in_cgs * units.UnitTime_in_s
    return (
        (2.5 * jnp.pi * units.G)
        * (0.375 * 0.6 * density_time)
        * state.BlackHoleMass
        * parameters.RadioModeEfficiency
    )


def _radio_mode_flow_rates(
    state,
    halo,
    parameters,
    units,
    raw_cooling,
    cooling_radius,
    cooling_lambda,
    log_fractional_perturbation,
):
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    safe_cooling_radius = jnp.where(cooling_radius > 0.0, cooling_radius, 1.0)
    cooling_after_prior = jnp.where(
        parameters.AGNrecipe > 0,
        jnp.where(
            state.Rheat < cooling_radius,
            (1.0 - state.Rheat / safe_cooling_radius) * raw_cooling,
            zero,
        ),
        raw_cooling,
    )
    active = (
        (parameters.AGNrecipe > 0)
        & (state.HotGas > 0.0)
        & (cooling_after_prior > 0.0)
        & (halo.Vvir > 0.0)
        & (cooling_lambda > 0.0)
    )

    def calculate(_):
        empirical = lambda __: _empirical_radio_rate(state, halo, parameters, units)
        bondi = lambda __: _bondi_radio_rate(
            state,
            halo,
            parameters,
            units,
            cooling_lambda,
        )

        def cold_cloud(__):
            ratio = cooling_radius / halo.Rvir
            threshold = 0.0001 * halo.Mvir * ratio**3
            return jnp.where(state.BlackHoleMass > threshold, 0.0001 * cooling_after_prior, 0.0)

        rate = jax.lax.cond(
            parameters.AGNrecipe == 2,
            bondi,
            lambda operand: jax.lax.cond(
                parameters.AGNrecipe == 3,
                cold_cloud,
                empirical,
                operand,
            ),
            operand=None,
        )
        rate = log_fractionally_perturb(rate, log_fractional_perturbation)
        eddington = (
            1.3e38
            * state.BlackHoleMass
            * 1.0e10
            / units.Hubble_h
            / (units.UnitEnergy_in_cgs / units.UnitTime_in_s)
            / (0.1 * 9.0e10)
        )
        rate = jnp.minimum(rate, eddington)
        coefficient = (HEATING_VELOCITY_KM_S / halo.Vvir) ** 2
        heating = coefficient * rate
        heating_limited = heating > cooling_after_prior
        rate = jnp.where(heating_limited, cooling_after_prior / coefficient, rate)
        heating = jnp.where(heating_limited, cooling_after_prior, heating)
        hot_metals_accreted = metallicity(state.HotGas, state.MetalsHotGas) * rate
        candidate = heating / cooling_after_prior * cooling_radius
        return rate, hot_metals_accreted, heating, candidate

    rate, metals, heating, candidate = jax.lax.cond(
        active,
        calculate,
        lambda _: (zero, zero, zero, state.Rheat),
        operand=None,
    )
    return RadioModeFlowRates(
        raw_cooling=raw_cooling,
        cooling_after_prior_heating=cooling_after_prior,
        black_hole_accretion=rate,
        hot_metals_accreted=metals,
        heating=heating,
        cooling_radius=cooling_radius,
        cooling_lambda=cooling_lambda,
        candidate_heating_radius=candidate,
    )


def _ode_state(state: Sage16HybridState) -> Sage16OdeState:
    return Sage16OdeState(
        ColdGas=state.ColdGas,
        HotGas=state.HotGas,
        EjectedGas=state.EjectedGas,
        StellarMass=state.StellarMass,
        MetalsColdGas=state.MetalsColdGas,
        MetalsHotGas=state.MetalsHotGas,
        MetalsEjectedGas=state.MetalsEjectedGas,
        MetalsStellarMass=state.MetalsStellarMass,
    )


def sage16_hybrid_rhs_and_rates(
    time,
    state: Sage16HybridState,
    halo: HaloForcing,
    parameters: Sage16Parameters,
    units: Sage16Units,
    cooling_tables: CoolingTables,
    *,
    infall_forcing: PreparedInfallForcing = None,
    perturbations=None,
) -> Sage16HybridRhsResult:
    """Evaluate the continuous part of the SAGE16 hybrid system.

    Halo properties are external forcing. ``Rheat`` is read as Markov state but
    changed only by :func:`apply_heating_radius_projection`. Thresholds and
    boundary switches are preserved without smoothing.
    """

    require_x64()
    if perturbations is None:
        perturbations = process_perturbations()
    if infall_forcing is None:
        zero = jnp.asarray(0.0, dtype=jnp.float64)
        infall_forcing = PreparedInfallForcing(zero, zero, zero)

    ode_state = _ode_state(state)
    base = sage16_ode_rhs_and_rates(
        time,
        ode_state,
        halo,
        state.DiskScaleRadius,
        parameters,
        units,
        cooling_tables,
        perturbations,
    )
    raw_cooling, cooling_radius, cooling_lambda = calculate_continuous_cooling_rate(
        ode_state,
        halo,
        units,
        cooling_tables,
    )
    raw_cooling = log_fractionally_perturb(raw_cooling, perturbations.cooling)
    radio = _radio_mode_flow_rates(
        state,
        halo,
        parameters,
        units,
        raw_cooling,
        cooling_radius,
        cooling_lambda,
        perturbations.agn_heating,
    )
    infall = continuous_infall_rates(state, infall_forcing, perturbations.infall)

    cooling_correction = radio.cooling_after_prior_heating - raw_cooling
    hot_metallicity = metallicity(state.HotGas, state.MetalsHotGas)
    base_derivative = base.derivative
    zero_derivative = zero_hybrid_derivative(state)
    merger_clock_rate = jnp.where((halo.Type == 1) | (halo.Type == 2), -1.0, 0.0)
    derivative = zero_derivative._replace(
        ColdGas=base_derivative.ColdGas + cooling_correction,
        HotGas=(
            base_derivative.HotGas
            - cooling_correction
            - radio.black_hole_accretion
            + infall.external_to_hot
            - infall.hot_to_external
        ),
        EjectedGas=base_derivative.EjectedGas - infall.ejected_to_external,
        StellarMass=base_derivative.StellarMass,
        BlackHoleMass=radio.black_hole_accretion,
        MetalsColdGas=(base_derivative.MetalsColdGas + hot_metallicity * cooling_correction),
        MetalsHotGas=(
            base_derivative.MetalsHotGas
            - hot_metallicity * cooling_correction
            - radio.hot_metals_accreted
            - infall.hot_metals_to_external
        ),
        MetalsEjectedGas=(base_derivative.MetalsEjectedGas - infall.ejected_metals_to_external),
        MetalsStellarMass=base_derivative.MetalsStellarMass,
        MergTime=merger_clock_rate,
    )
    rates = Sage16HybridRates(
        cooling=radio.cooling_after_prior_heating,
        star_formation=base.rates.star_formation,
        locked_stars=base.rates.locked_stars,
        sn_reheating=base.rates.sn_reheating,
        sn_ejection=base.rates.sn_ejection,
        reincorporation=base.rates.reincorporation,
        produced_metals=base.rates.produced_metals,
        new_metals_to_cold=base.rates.new_metals_to_cold,
        new_metals_to_hot=base.rates.new_metals_to_hot,
        radio_black_hole_accretion=radio.black_hole_accretion,
        radio_hot_metals_accreted=radio.hot_metals_accreted,
        agn_heating=radio.heating,
        cooling_radius=cooling_radius,
        cooling_lambda=cooling_lambda,
        candidate_heating_radius=radio.candidate_heating_radius,
        infall_to_hot=infall.external_to_hot,
        infall_from_ejected=infall.ejected_to_external,
        infall_from_hot=infall.hot_to_external,
        external_baryon_rate=infall.external_baryon_rate,
        external_metal_rate=infall.external_metal_rate,
    )
    return Sage16HybridRhsResult(derivative, rates)


def sage16_hybrid_rhs(*args, **kwargs) -> Sage16HybridState:
    """Return only the continuous derivative for integration and AD."""

    return sage16_hybrid_rhs_and_rates(*args, **kwargs).derivative


def apply_heating_radius_projection(
    state: Sage16HybridState,
    rates: Sage16HybridRates,
) -> HeatingRadiusProjectionResult:
    """Apply upstream's monotone AGN-history map to the augmented state.

    New heating affects later cooling. It does not reduce the current cooling
    flow a second time. The map is piecewise differentiable on a fixed branch.
    """

    previous = state.Rheat
    candidate = rates.candidate_heating_radius
    active = (previous < rates.cooling_radius) & (rates.cooling > 0.0) & (candidate > previous)
    projected = jnp.where(active, candidate, previous)
    return HeatingRadiusProjectionResult(
        state=state._replace(Rheat=projected),
        previous_radius=previous,
        candidate_radius=candidate,
        projected_radius=projected,
        applied=active,
    )


def continuous_satellite_stripping_rates(
    pair: StrippingPairState,
    satellite_halo: HaloForcing,
    parameters: Sage16Parameters,
    interval_duration,
    log_fractional_perturbation=0.0,
) -> ContinuousStrippingRates:
    """Return the continuous limit of SAGE's recomputed Type-1 excess removal.

    Upstream removes ``excess / N`` each of ``N`` substeps. With fixed forcing,
    the limit is ``dM_hot/dt = -excess / interval_duration`` and strips a
    fraction ``1-exp(-1)`` of an initially positive excess over one interval.
    """

    satellite = pair.satellite
    central = pair.central
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    baryons = (
        satellite.StellarMass
        + satellite.ColdGas
        + satellite.HotGas
        + satellite.EjectedGas
        + satellite.BlackHoleMass
        + satellite.ICS
    )
    fraction = jnp.where(
        satellite.HaloBaryonFraction > 0.0,
        satellite.HaloBaryonFraction,
        parameters.GlobalBaryonFraction,
    )
    allowed = fraction * satellite_halo.Mvir
    duration = as_float64(interval_duration)
    raw_rate = jnp.where(duration > 0.0, jnp.maximum(baryons - allowed, 0.0) / duration, zero)
    rate = log_fractionally_perturb(raw_rate, log_fractional_perturbation)
    active = (satellite_halo.Type == 1) & (satellite.HotGas > 0.0)
    rate = jnp.where(active, rate, zero)
    metals_rate = metallicity(satellite.HotGas, satellite.MetalsHotGas) * rate
    satellite_derivative = zero_hybrid_derivative(satellite)._replace(
        HotGas=-rate,
        MetalsHotGas=-metals_rate,
    )
    central_derivative = zero_hybrid_derivative(central)._replace(
        HotGas=rate,
        MetalsHotGas=metals_rate,
    )
    return ContinuousStrippingRates(
        derivative=StrippingPairState(satellite_derivative, central_derivative),
        gas=rate,
        metals=metals_rate,
        allowed_baryons=allowed,
        satellite_baryons=baryons,
    )


def hybrid_baryonic_mass(state: Sage16HybridState):
    """Total modeled baryonic mass in the persistent hybrid reservoirs."""

    return (
        state.ColdGas
        + state.HotGas
        + state.EjectedGas
        + state.StellarMass
        + state.ICS
        + state.BlackHoleMass
    )


def hybrid_metal_mass(state: Sage16HybridState):
    """Tracked metal mass; SAGE does not carry a black-hole metal reservoir."""

    return (
        state.MetalsColdGas
        + state.MetalsHotGas
        + state.MetalsEjectedGas
        + state.MetalsStellarMass
        + state.MetalsICS
    )
